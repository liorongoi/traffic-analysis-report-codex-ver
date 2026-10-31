"""Conservative, single-recording vehicle association across video frames.

The tracker deliberately has no default thresholds. Callers must pass an
explicit policy, and ambiguous associations start uncertain track fragments
instead of forcing identities together. Keep one tracker instance across
processing chunks and checkpoint its JSON-compatible state to preserve IDs.
"""
from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True, slots=True)
class TrackerPolicy:
    max_gap_seconds: Fraction
    minimum_iou: float
    max_center_distance: float
    association_margin: float
    compatible_label_groups: tuple[frozenset[str], ...] = ()
    scene_cut_threshold: float | None = None

    def __post_init__(self):
        if isinstance(self.max_gap_seconds, bool):
            raise ValueError('max_gap_seconds must be a rational duration, not a boolean')
        gap = Fraction(self.max_gap_seconds)
        if gap <= 0:
            raise ValueError('max_gap_seconds must be positive')
        object.__setattr__(self, 'max_gap_seconds', gap)
        for name, value in (('minimum_iou', self.minimum_iou),
                            ('max_center_distance', self.max_center_distance),
                            ('association_margin', self.association_margin)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if not 0 <= self.minimum_iou <= 1:
            raise ValueError('minimum_iou must be between zero and one')
        if self.max_center_distance <= 0:
            raise ValueError('max_center_distance must be positive')
        if not 0 <= self.association_margin <= 1:
            raise ValueError('association_margin must be between zero and one')
        if self.scene_cut_threshold is not None:
            value=self.scene_cut_threshold
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError('scene_cut_threshold must be null or a finite number in (0, 1]')
            object.__setattr__(self,'scene_cut_threshold',float(value))
        groups = tuple(frozenset(group) for group in self.compatible_label_groups)
        if any(len(group) < 2 or any(not isinstance(label, str) or not label for label in group)
               for group in groups):
            raise ValueError('Each compatible label group must contain at least two non-empty labels')
        object.__setattr__(self, 'compatible_label_groups', groups)

    def labels_compatible(self, previous, current):
        return previous == current or any(previous in group and current in group
                                          for group in self.compatible_label_groups)

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            raise ValueError('Tracker policy config must be a mapping')
        required = ('max_gap_seconds', 'minimum_iou', 'max_center_distance', 'association_margin')
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError('Tracker policy config is missing: ' + ', '.join(missing))
        gap = config['max_gap_seconds']
        if isinstance(gap, (list, tuple)):
            gap = Fraction(*gap)
        groups = tuple(frozenset(group) for group in config.get('compatible_label_groups', ()))
        return cls(gap, config['minimum_iou'], config['max_center_distance'],
            config['association_margin'], groups, config.get('scene_cut_threshold'))


def checkpoint_policy_matches(saved_policy, expected_policy):
    """Allow pre-scene-cut checkpoints to mean the new optional setting is off."""
    if not isinstance(saved_policy,dict) or not isinstance(expected_policy,dict):
        return False
    if 'scene_cut_threshold' not in saved_policy:
        saved_policy={**saved_policy,'scene_cut_threshold':None}
    if 'scene_cut_threshold' not in expected_policy:
        expected_policy={**expected_policy,'scene_cut_threshold':None}
    return saved_policy==expected_policy


@dataclass(frozen=True, slots=True)
class TrackSummary:
    track_id: str
    first_pts: int
    first_time_base_num: int
    first_time_base_den: int
    last_pts: int
    last_time_base_num: int
    last_time_base_den: int
    first_offset_seconds: Fraction
    last_offset_seconds: Fraction
    last_box: tuple[float, float, float, float]
    labels: tuple[str, ...]
    observation_count: int
    association_uncertain: bool
    uncertainty_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrackAssignment:
    detection_index: int
    track: TrackSummary
    matched_existing: bool
    association_uncertain: bool


@dataclass(frozen=True, slots=True)
class FrameTrackingResult:
    assignments: tuple[TrackAssignment, ...]
    closed_tracks: tuple[TrackSummary, ...]


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Detection box must contain normalized x, y, width, height')
    x, y, width, height = value
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError('Detection box values must be finite numbers')
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
        raise ValueError('Detection box must be positive and inside normalized image bounds')
    return tuple(float(v) for v in value)


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def _center_distance(a, b):
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    return math.hypot(ax - bx, ay - by)


class VideoTrackManager:
    """Associate frame detections within one camera view and recording only."""

    STATE_VERSION = 2

    def __init__(self, policy, *, run_id, state=None):
        if not isinstance(policy, TrackerPolicy):
            raise TypeError('policy must be a TrackerPolicy with explicit thresholds')
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError('run_id must be a non-empty recording/run identifier')
        self.policy = policy
        self.run_id = run_id.strip()
        self.next_id = 1
        self.active = {}
        self.last_offset = None
        # Night-light hypothesis identities are an independent review-only
        # namespace. They never enter vehicle events or detection counts.
        self.next_review_id = 1
        self.active_review_hypotheses = {}
        self.last_review_offset = None
        if state is not None:
            self._restore(state)

    @staticmethod
    def _time(timestamp):
        try:
            pts = timestamp.pts
            num = timestamp.time_base_num
            den = timestamp.time_base_den
            offset = timestamp.offset_seconds
        except AttributeError as exc:
            raise TypeError('timestamp must be a VideoTimestamp') from exc
        if den <= 0 or num <= 0:
            raise ValueError('timestamp must have a positive rational time base')
        return int(pts), int(num), int(den), Fraction(offset)

    def _summary(self, item):
        return TrackSummary(item['track_id'], item['first_pts'], item['first_time_base_num'],
            item['first_time_base_den'], item['last_pts'], item['last_time_base_num'],
            item['last_time_base_den'], item['first_offset_seconds'], item['last_offset_seconds'],
            tuple(item['last_box']), tuple(sorted(item['labels'])), item['observation_count'],
            item['association_uncertain'], tuple(sorted(item['uncertainty_reasons'])))

    def _policy_state(self):
        gap = self.policy.max_gap_seconds
        return {'max_gap_seconds': [gap.numerator, gap.denominator],
            'minimum_iou': self.policy.minimum_iou,
            'max_center_distance': self.policy.max_center_distance,
            'association_margin': self.policy.association_margin,
            'scene_cut_threshold':self.policy.scene_cut_threshold,
            'compatible_label_groups': [sorted(group) for group in self.policy.compatible_label_groups]}

    def _new(self, timestamp, detection, uncertain_reason=None):
        pts, num, den, offset = self._time(timestamp)
        track_id = f'{self.run_id}:{self.next_id:08d}'
        self.next_id += 1
        reason_set = {uncertain_reason} if uncertain_reason else set()
        item = {'track_id': track_id, 'first_pts': pts, 'first_time_base_num': num,
            'first_time_base_den': den, 'last_pts': pts, 'last_time_base_num': num,
            'last_time_base_den': den, 'first_offset_seconds': offset,
            'last_offset_seconds': offset, 'last_box': detection['box'],
            'labels': {detection['label']}, 'observation_count': 1,
            'association_uncertain': bool(reason_set), 'uncertainty_reasons': reason_set}
        self.active[track_id] = item
        return item

    def _update(self, item, timestamp, detection, uncertain_reason=None):
        pts, num, den, offset = self._time(timestamp)
        item['last_pts'], item['last_time_base_num'], item['last_time_base_den'] = pts, num, den
        item['last_offset_seconds'] = offset
        item['last_box'] = detection['box']
        item['labels'].add(detection['label'])
        item['observation_count'] += 1
        if uncertain_reason:
            item['association_uncertain'] = True
            item['uncertainty_reasons'].add(uncertain_reason)

    def update(self, timestamp, detections):
        """Consume one presentation-time frame; keep state across chunk boundaries."""
        _, _, _, offset = self._time(timestamp)
        if self.last_offset is not None and offset < self.last_offset:
            raise ValueError('Tracker timestamps must be monotonic within one recording')

        normalized = []
        for detection in detections:
            if not isinstance(detection, dict):
                raise ValueError('Each detection must be a mapping')
            label = detection.get('label') or detection.get('suggested_kind') or detection.get('kind')
            if not isinstance(label, str) or not label.strip():
                raise ValueError('Each detection must have a non-empty label')
            normalized.append({'label': label.strip(), 'box': _box(detection.get('box'))})

        self.last_offset = offset

        closed = []
        active_items = []
        for track_id, item in list(self.active.items()):
            gap = offset - item['last_offset_seconds']
            if gap > self.policy.max_gap_seconds:
                closed.append(self._summary(item))
                del self.active[track_id]
            else:
                active_items.append(item)

        options_by_detection = []
        for detection in normalized:
            options = []
            for item in active_items:
                if not any(self.policy.labels_compatible(old, detection['label']) for old in item['labels']):
                    continue
                overlap = _iou(item['last_box'], detection['box'])
                distance = _center_distance(item['last_box'], detection['box'])
                if overlap < self.policy.minimum_iou and distance > self.policy.max_center_distance:
                    continue
                proximity = max(0.0, 1.0 - distance / self.policy.max_center_distance)
                options.append((max(overlap, proximity), item['track_id']))
            options.sort(key=lambda pair: (-pair[0], pair[1]))
            options_by_detection.append(options)

        best_owners = {}
        for index, options in enumerate(options_by_detection):
            if options:
                best_owners.setdefault(options[0][1], []).append(index)

        assignments = []
        for index, (detection, options) in enumerate(zip(normalized, options_by_detection)):
            selected = None
            reason = None
            if options:
                best_score, best_id = options[0]
                runner_up = options[1][0] if len(options) > 1 else None
                ambiguous = runner_up is not None and best_score - runner_up <= self.policy.association_margin
                collision = len(best_owners[best_id]) > 1
                if ambiguous:
                    reason = 'competing_tracks'
                elif collision:
                    reason = 'competing_detections'
                else:
                    selected = self.active[best_id]
            if selected is None:
                item = self._new(timestamp, detection, reason)
                matched = False
            else:
                self._update(selected, timestamp, detection)
                item = selected
                matched = True
            assignments.append(TrackAssignment(index, self._summary(item), matched, bool(reason)))
        return FrameTrackingResult(tuple(assignments), tuple(closed))

    def finish(self, *, boundary_reason=None):
        """Close tracks at recording end or a known scene boundary."""
        if boundary_reason is not None and (not isinstance(boundary_reason,str) or not boundary_reason.strip()):
            raise ValueError('boundary_reason must be a non-empty string')
        if boundary_reason:
            for item in self.active.values():
                item['association_uncertain']=True
                item['uncertainty_reasons'].add(boundary_reason.strip())
        closed = tuple(self._summary(item) for item in self.active.values())
        self.active.clear()
        return closed

    @property
    def active_track_count(self):
        return len(self.active)

    def state(self):
        """Return a JSON-compatible checkpoint suitable for chunk continuation."""
        def fraction(value):
            return [value.numerator, value.denominator]
        return {'version': self.STATE_VERSION, 'run_id': self.run_id, 'policy': self._policy_state(), 'next_id': self.next_id,
            'last_offset_seconds': fraction(self.last_offset) if self.last_offset is not None else None,
            'active': [{**item,
                'first_offset_seconds': fraction(item['first_offset_seconds']),
                'last_offset_seconds': fraction(item['last_offset_seconds']),
                'last_box': list(item['last_box']), 'labels': sorted(item['labels']),
                'uncertainty_reasons': sorted(item['uncertainty_reasons'])}
                for item in self.active.values()],
            'review_hypotheses': {'next_id': self.next_review_id,
                'last_offset_seconds': fraction(self.last_review_offset) if self.last_review_offset is not None else None,
                'active': [{**item,
                    'last_offset_seconds': fraction(item['last_offset_seconds']),
                    'last_box': list(item['last_box']),
                    'states': [list(state) for state in sorted(item['states'])],
                    'uncertainty_reasons': sorted(item['uncertainty_reasons'])}
                    for item in self.active_review_hypotheses.values()]}}

    def update_review_hypotheses(self, timestamp, hypotheses, *, scene_cut=False):
        """Link nearby light proposals into uncertain, review-only temporal candidates.

        These candidate IDs are intentionally separate from vehicle track IDs:
        they are saved with frame overlays but never create vehicle events or
        contribute to detection/track counts.
        """
        _, _, _, offset = self._time(timestamp)
        if self.last_review_offset is not None and offset < self.last_review_offset:
            raise ValueError('Review-hypothesis timestamps must be monotonic within one recording')
        if not isinstance(hypotheses, (list, tuple)):
            raise ValueError('Review hypotheses must be a sequence')
        normalized = []
        for raw in hypotheses:
            if not isinstance(raw, dict) or raw.get('kind') != 'light':
                raise ValueError('Only light hypotheses may receive review candidate IDs')
            label, subtype, pairing = raw.get('label'), raw.get('light_subtype'), raw.get('pairing')
            reasons = raw.get('reasons', [])
            if (not isinstance(label, str) or not label.strip()
                    or subtype not in ('headlight_candidate','streetlight_suspect','reflection_suspect','uncertain_light')
                    or pairing not in ('single','hypothesis')
                    or not isinstance(reasons, (list, tuple))
                    or any(not isinstance(reason, str) for reason in reasons)):
                raise ValueError('Review hypothesis fields are invalid')
            normalized.append({'kind':'light','label':label.strip(),'box':_box(raw.get('box')),
                'light_subtype':subtype,'pairing':pairing,'reasons':list(reasons)})

        active_items = []
        if scene_cut:
            self.active_review_hypotheses.clear()
        else:
            for candidate_id, item in list(self.active_review_hypotheses.items()):
                if offset - item['last_offset_seconds'] > self.policy.max_gap_seconds:
                    del self.active_review_hypotheses[candidate_id]
                else:
                    active_items.append(item)

        options_by_hypothesis = []
        for hypothesis in normalized:
            options = []
            for item in active_items:
                overlap = _iou(item['last_box'], hypothesis['box'])
                distance = _center_distance(item['last_box'], hypothesis['box'])
                if overlap < self.policy.minimum_iou and distance > self.policy.max_center_distance:
                    continue
                proximity = max(0.0, 1.0 - distance / self.policy.max_center_distance)
                options.append((max(overlap, proximity), item['review_track_id']))
            options.sort(key=lambda pair: (-pair[0], pair[1]))
            options_by_hypothesis.append(options)
        best_owners = {}
        for options in options_by_hypothesis:
            if options:
                best_owners.setdefault(options[0][1], []).append(options)

        enriched = []
        for hypothesis, options in zip(normalized, options_by_hypothesis):
            selected = None
            uncertain = False
            if options:
                best_score, best_id = options[0]
                runner_up = options[1][0] if len(options) > 1 else None
                ambiguous = runner_up is not None and best_score - runner_up <= self.policy.association_margin
                collision = len(best_owners[best_id]) > 1
                if ambiguous or collision:
                    uncertain = True
                else:
                    selected = self.active_review_hypotheses[best_id]

            current_state = (hypothesis['light_subtype'], hypothesis['pairing'])
            if selected is None:
                candidate_id = f'{self.run_id}:night-{self.next_review_id:08d}'
                self.next_review_id += 1
                selected = {'review_track_id':candidate_id,'last_offset_seconds':offset,
                    'last_box':hypothesis['box'],'last_state':current_state,'states':{current_state},
                    'observation_count':0,'uncertainty_reasons':set()}
                self.active_review_hypotheses[candidate_id] = selected
                if uncertain:
                    selected['uncertainty_reasons'].add('competing_light_hypotheses')
                state_change = None
            else:
                previous_state = selected['last_state']
                state_change = None
                if previous_state != current_state:
                    state_change = {'from_subtype':previous_state[0],'from_pairing':previous_state[1],
                        'to_subtype':current_state[0],'to_pairing':current_state[1]}
                    selected['states'].add(current_state)
                selected['last_offset_seconds'] = offset
                selected['last_box'] = hypothesis['box']
                selected['last_state'] = current_state
                if uncertain:
                    selected['uncertainty_reasons'].add('ambiguous_light_association')
            selected['observation_count'] += 1
            enriched.append({**hypothesis,'review_track_id':selected['review_track_id'],
                'review_observation_index':selected['observation_count'],
                'review_association_uncertain':bool(selected['uncertainty_reasons']),
                'review_uncertainty_reasons':sorted(selected['uncertainty_reasons']),
                'review_state_change':state_change})
        self.last_review_offset = offset
        return enriched

    def possible_light_candidate_links(self, timestamp, detections, assignments):
        """Suggest review-only links from recent light candidates to vehicle detections.

        This does not merge identities: the vehicle tracker and event records
        remain unchanged. It only annotates a contemporaneous frame when the
        spatial match to a recent light candidate is unique under the frozen
        tracker policy.
        """
        _, _, _, offset = self._time(timestamp)
        if not isinstance(detections,(list,tuple)) or not isinstance(assignments,(list,tuple)):
            raise ValueError('Vehicle detections and assignments must be sequences')
        active = [item for item in self.active_review_hypotheses.values()
            if 0 <= offset-item['last_offset_seconds'] <= self.policy.max_gap_seconds]
        assignment_by_index = {}
        for assignment in assignments:
            index=getattr(assignment,'detection_index',None)
            track=getattr(assignment,'track',None)
            track_id=getattr(track,'track_id',None)
            if (isinstance(index,bool) or not isinstance(index,int) or index<0 or index>=len(detections)
                    or not isinstance(track_id,str) or not track_id or index in assignment_by_index):
                raise ValueError('Vehicle assignment does not match its detection')
            assignment_by_index[index]=track_id
        options_by_detection=[]
        for index,detection in enumerate(detections):
            if not isinstance(detection,dict):
                raise ValueError('Vehicle detection must be a mapping')
            box=_box(detection.get('box'))
            options=[]
            for item in active:
                overlap=_iou(item['last_box'],box)
                distance=_center_distance(item['last_box'],box)
                if overlap<self.policy.minimum_iou and distance>self.policy.max_center_distance:
                    continue
                proximity=max(0.0,1.0-distance/self.policy.max_center_distance)
                options.append((max(overlap,proximity),item['review_track_id']))
            options.sort(key=lambda pair:(-pair[0],pair[1]))
            options_by_detection.append(options)
        owners={}
        for options in options_by_detection:
            if options:
                owners.setdefault(options[0][1],[]).append(options)
        links={}
        for index,options in enumerate(options_by_detection):
            if not options:
                continue
            best_score,best_id=options[0]
            runner_up=options[1][0] if len(options)>1 else None
            ambiguous=runner_up is not None and best_score-runner_up<=self.policy.association_margin
            collision=len(owners[best_id])>1
            if ambiguous or collision:
                links[index]={'possible_light_candidate_link_uncertain':True,
                    'possible_light_candidate_link_reason':'competing_light_candidates'
                        if ambiguous else 'competing_vehicle_detections'}
            elif index in assignment_by_index:
                links[index]={'possible_light_candidate_id':best_id,
                    'possible_light_candidate_vehicle_track_id':assignment_by_index[index],
                    'possible_light_candidate_link_basis':'unique_spatial_match_within_tracker_policy'}
        return links

    def _restore(self, state):
        if (not isinstance(state, dict) or state.get('version') not in (1, self.STATE_VERSION)
                or state.get('run_id') != self.run_id):
            raise ValueError('Incompatible video tracker checkpoint')
        saved_policy=state.get('policy')
        if not checkpoint_policy_matches(saved_policy,self._policy_state()):
            raise ValueError('Tracker policy changed since this video checkpoint')
        next_id = state.get('next_id')
        if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 1:
            raise ValueError('Invalid tracker checkpoint next ID')
        def fraction(pair):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('Invalid tracker checkpoint timestamp')
            if any(isinstance(value, bool) or not isinstance(value, int) for value in pair) or pair[1] <= 0:
                raise ValueError('Invalid tracker checkpoint timestamp')
            return Fraction(pair[0], pair[1])
        active = {}
        largest_id = 0
        for raw in state.get('active', []):
            item = dict(raw)
            item['first_offset_seconds'] = fraction(item['first_offset_seconds'])
            item['last_offset_seconds'] = fraction(item['last_offset_seconds'])
            item['last_box'] = _box(item['last_box'])
            labels = item.get('labels')
            reasons = item.get('uncertainty_reasons')
            if (not isinstance(labels, list) or not labels
                    or any(not isinstance(label, str) or not label for label in labels)
                    or not isinstance(reasons, list)
                    or any(not isinstance(reason, str) or not reason for reason in reasons)):
                raise ValueError('Invalid labels or uncertainty in tracker checkpoint')
            item['labels'] = set(labels)
            item['uncertainty_reasons'] = set(reasons)
            track_id = item.get('track_id')
            prefix = self.run_id + ':'
            try:
                numeric_id = int(track_id[len(prefix):]) if isinstance(track_id, str) and track_id.startswith(prefix) else 0
            except ValueError:
                numeric_id = 0
            if numeric_id < 1 or track_id in active:
                raise ValueError('Invalid or duplicate track ID in checkpoint')
            largest_id = max(largest_id, numeric_id)
            for key in ('first_pts', 'last_pts', 'first_time_base_num', 'first_time_base_den',
                        'last_time_base_num', 'last_time_base_den', 'observation_count'):
                if isinstance(item.get(key), bool) or not isinstance(item.get(key), int):
                    raise ValueError('Invalid track counters or PTS in checkpoint')
            if (item['first_time_base_num'] <= 0 or item['first_time_base_den'] <= 0
                    or item['last_time_base_num'] <= 0 or item['last_time_base_den'] <= 0
                    or item['observation_count'] < 1
                    or Fraction(item['first_pts'] * item['first_time_base_num'], item['first_time_base_den'])
                       != item['first_offset_seconds']
                    or Fraction(item['last_pts'] * item['last_time_base_num'], item['last_time_base_den'])
                       != item['last_offset_seconds']
                    or item['first_offset_seconds'] > item['last_offset_seconds']
                    or not isinstance(item.get('association_uncertain'), bool)):
                raise ValueError('Inconsistent track timeline in checkpoint')
            active[item['track_id']] = item
        if largest_id >= next_id:
            raise ValueError('Tracker checkpoint next ID would reuse an existing track ID')
        last = state.get('last_offset_seconds')
        self.last_offset = fraction(last) if last is not None else None
        if active and (self.last_offset is None or any(item['last_offset_seconds'] > self.last_offset for item in active.values())):
            raise ValueError('Tracker checkpoint time is inconsistent')
        self.next_id, self.active = next_id, active
        if state.get('version') == 1:
            self.next_review_id, self.active_review_hypotheses, self.last_review_offset = 1, {}, None
            return
        review = state.get('review_hypotheses')
        if not isinstance(review, dict):
            raise ValueError('Invalid review-hypothesis checkpoint')
        next_review_id = review.get('next_id')
        if isinstance(next_review_id, bool) or not isinstance(next_review_id, int) or next_review_id < 1:
            raise ValueError('Invalid review-hypothesis checkpoint next ID')
        review_active = {}
        largest_review_id = 0
        for raw in review.get('active', []):
            item = dict(raw)
            candidate_id = item.get('review_track_id')
            prefix = self.run_id + ':night-'
            try:
                numeric_id = int(candidate_id[len(prefix):]) if isinstance(candidate_id,str) and candidate_id.startswith(prefix) else 0
            except ValueError:
                numeric_id = 0
            if numeric_id < 1 or candidate_id in review_active:
                raise ValueError('Invalid or duplicate review-hypothesis ID in checkpoint')
            item['last_offset_seconds'] = fraction(item.get('last_offset_seconds'))
            item['last_box'] = _box(item.get('last_box'))
            states = item.get('states')
            reasons = item.get('uncertainty_reasons')
            if (not isinstance(states,list) or not states
                    or any(not isinstance(state,list) or len(state)!=2
                        or state[0] not in ('headlight_candidate','streetlight_suspect','reflection_suspect','uncertain_light')
                        or state[1] not in ('single','hypothesis') for state in states)
                    or not isinstance(reasons,list)
                    or any(not isinstance(reason,str) or not reason for reason in reasons)
                    or not isinstance(item.get('last_state'),(list,tuple))
                    or tuple(item.get('last_state')) not in [tuple(value) for value in states]
                    or isinstance(item.get('observation_count'),bool)
                    or not isinstance(item.get('observation_count'),int) or item['observation_count']<1):
                raise ValueError('Invalid review-hypothesis state in checkpoint')
            item['states'] = {tuple(value) for value in states}
            item['last_state'] = tuple(item['last_state'])
            item['uncertainty_reasons'] = set(reasons)
            review_active[candidate_id] = item
            largest_review_id = max(largest_review_id,numeric_id)
        if largest_review_id >= next_review_id:
            raise ValueError('Review-hypothesis checkpoint next ID would reuse an existing ID')
        last_review = review.get('last_offset_seconds')
        self.last_review_offset = fraction(last_review) if last_review is not None else None
        if review_active and (self.last_review_offset is None or
                any(item['last_offset_seconds'] > self.last_review_offset for item in review_active.values())):
            raise ValueError('Review-hypothesis checkpoint time is inconsistent')
        self.next_review_id, self.active_review_hypotheses = next_review_id, review_active
