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
        groups = tuple(frozenset(group) for group in self.compatible_label_groups)
        if any(len(group) < 2 or any(not isinstance(label, str) or not label for label in group)
               for group in groups):
            raise ValueError('Each compatible label group must contain at least two non-empty labels')
        object.__setattr__(self, 'compatible_label_groups', groups)

    def labels_compatible(self, previous, current):
        return previous == current or any(previous in group and current in group
                                          for group in self.compatible_label_groups)


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

    STATE_VERSION = 1

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

    def finish(self):
        """Close all active tracks at recording end or an explicit scene cut."""
        closed = tuple(self._summary(item) for item in self.active.values())
        self.active.clear()
        return closed

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
                for item in self.active.values()]}

    def _restore(self, state):
        if not isinstance(state, dict) or state.get('version') != self.STATE_VERSION or state.get('run_id') != self.run_id:
            raise ValueError('Incompatible video tracker checkpoint')
        if state.get('policy') != self._policy_state():
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
