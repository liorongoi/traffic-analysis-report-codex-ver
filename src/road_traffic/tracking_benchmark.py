"""Offline sequence-level tracking metrics for paired human labels and predictions.

Input is an explicit JSON frame/object manifest. This evaluator measures the
provided tracks; it neither infers identity nor defines acceptance thresholds.
"""
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3


GROUP_FIELDS = {
    'country': ('country', 'target_country'),
    'condition': ('condition', 'day_night', 'period'),
    'camera_view': ('camera_view', 'view_id', 'view'),
    'hardware_profile': ('hardware_profile', 'hardware'),
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
        raise ValueError(f'{name} must be a non-empty string or integer')
    return value


def _read(path):
    path = Path(path)
    data = path.read_bytes()
    try:
        value = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{path} must contain valid UTF-8 JSON') from exc
    return value, hashlib.sha256(data).hexdigest()


def _sequences(value, label):
    if not isinstance(value, dict) or not isinstance(value.get('sequences'), list):
        raise ValueError(f'{label} must contain a sequences array')
    result = {}
    for sequence in value['sequences']:
        if not isinstance(sequence, dict):
            raise ValueError(f'Each {label} sequence must be an object')
        sequence_id = _identifier(sequence.get('sequence_id'), f'{label} sequence_id')
        if sequence_id in result:
            raise ValueError(f'{label} sequence IDs must be unique')
        if not isinstance(sequence.get('frames'), list):
            raise ValueError(f'{label} sequence {sequence_id} must contain a frames array')
        frames = {}
        for frame in sequence['frames']:
            if not isinstance(frame, dict):
                raise ValueError(f'{label} frames must be objects')
            frame_id = _identifier(frame.get('frame_id'), f'{label} frame_id')
            if frame_id in frames:
                raise ValueError(f'{label} frame IDs must be unique within each sequence')
            if not isinstance(frame.get('objects'), list):
                raise ValueError(f'{label} frame {frame_id} must contain an objects array')
            objects = []
            for obj in frame['objects']:
                if not isinstance(obj, dict):
                    raise ValueError(f'{label} objects must be JSON objects')
                track_id = _identifier(obj.get('track_id'), f'{label} track_id')
                category_id = _identifier(obj.get('category_id'), f'{label} category_id')
                bbox = obj.get('bbox')
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f'{label} bbox must be [x, y, width, height]')
                x, y, width, height = (_finite(item, f'{label} bbox') for item in bbox)
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    raise ValueError(f'{label} bbox must have nonnegative origin and positive dimensions')
                objects.append({'track_id': track_id, 'category_id': category_id,
                                'bbox': (x, y, width, height)})
            frames[frame_id] = objects
        metadata = sequence.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError(f'{label} sequence metadata must be an object')
        result[sequence_id] = {'frames': frames, 'metadata': metadata}
    return result


def _iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2 = min(left[0] + left[2], right[0] + right[2])
    y2 = min(left[1] + left[3], right[1] + right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left[2] * left[3] + right[2] * right[3] - inter
    return inter / union if union else 0.0


def _maximum_assignment(weights):
    """Return maximum-weight row/column pairs with deterministic Hungarian matching."""
    rows = len(weights)
    cols = max((len(row) for row in weights), default=0)
    size = max(rows, cols)
    if not size:
        return []
    maximum = max((weight for row in weights for weight in row), default=0.0)
    costs = [[maximum - (weights[i][j] if i < rows and j < len(weights[i]) else 0.0)
              for j in range(size)] for i in range(size)]
    u, v, p, way = [0.0] * (size + 1), [0.0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minimum = [float('inf')] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = float('inf'), 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], way[j] = current, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = []
    for j in range(1, size + 1):
        i = p[j] - 1
        column = j - 1
        if i < rows and column < len(weights[i]) and weights[i][column] > 0:
            result.append((i, column))
    return result


def _frame_matches(ground_truth, predictions, threshold):
    matches = []
    categories = {obj['category_id'] for obj in ground_truth} | {obj['category_id'] for obj in predictions}
    for category in sorted(categories, key=lambda item: (type(item).__name__, str(item))):
        gt_indices = [i for i, obj in enumerate(ground_truth) if obj['category_id'] == category]
        pred_indices = [i for i, obj in enumerate(predictions) if obj['category_id'] == category]
        weights = []
        for gi in gt_indices:
            row = []
            for pi in pred_indices:
                overlap = _iou(ground_truth[gi]['bbox'], predictions[pi]['bbox'])
                row.append(overlap if overlap >= threshold else 0.0)
            weights.append(row)
        for gi, pi in _maximum_assignment(weights):
            matches.append((gt_indices[gi], pred_indices[pi]))
    return matches


def _sequence_metrics(gt_sequence, pred_sequence, iou_threshold):
    gt_frames, pred_frames = gt_sequence['frames'], pred_sequence['frames']
    if set(gt_frames) != set(pred_frames):
        missing_pred = len(set(gt_frames) - set(pred_frames))
        extra_pred = len(set(pred_frames) - set(gt_frames))
        raise ValueError(f'Frame mismatch: {missing_pred} ground-truth frames lack predictions; '
                         f'{extra_pred} prediction frames lack ground truth')
    gt_total = pred_total = tp = id_switches = fragmentations = 0
    cooccurrence = Counter()
    gt_track_ids, pred_track_ids = set(), set()
    previous_pred_for_gt = {}
    previously_matched = set()
    was_unmatched = set()
    matched_pred_ids_by_gt = defaultdict(set)
    matched_gt_ids_by_pred = defaultdict(set)
    for frame_id in sorted(gt_frames, key=lambda value: (type(value).__name__, str(value))):
        gt_objects, pred_objects = gt_frames[frame_id], pred_frames[frame_id]
        gt_total += len(gt_objects); pred_total += len(pred_objects)
        gt_track_ids.update(obj['track_id'] for obj in gt_objects)
        pred_track_ids.update(obj['track_id'] for obj in pred_objects)
        matches = _frame_matches(gt_objects, pred_objects, iou_threshold)
        tp += len(matches)
        matched_gt_now = set()
        for gi, pi in matches:
            gt_id, pred_id = gt_objects[gi]['track_id'], pred_objects[pi]['track_id']
            matched_gt_now.add(gt_id)
            cooccurrence[(gt_id, pred_id)] += 1
            matched_pred_ids_by_gt[gt_id].add(pred_id)
            matched_gt_ids_by_pred[pred_id].add(gt_id)
            if gt_id in previous_pred_for_gt and previous_pred_for_gt[gt_id] != pred_id:
                id_switches += 1
            if gt_id in was_unmatched and gt_id in previously_matched:
                fragmentations += 1
            previous_pred_for_gt[gt_id] = pred_id
        for obj in gt_objects:
            gt_id = obj['track_id']
            if gt_id not in matched_gt_now and gt_id in previously_matched:
                was_unmatched.add(gt_id)
        previously_matched.update(matched_gt_now)
        was_unmatched.difference_update(matched_gt_now)

    fp, fn = pred_total - tp, gt_total - tp
    # Global identity assignment uses frame-local spatial matches as pair weights.
    gt_ids = sorted({key[0] for key in cooccurrence}, key=lambda item: (type(item).__name__, str(item)))
    pred_ids = sorted({key[1] for key in cooccurrence}, key=lambda item: (type(item).__name__, str(item)))
    weights = [[cooccurrence[(gt_id, pred_id)] for pred_id in pred_ids] for gt_id in gt_ids]
    idtp = sum(weights[i][j] for i, j in _maximum_assignment(weights))
    idfp, idfn = pred_total - idtp, gt_total - idtp
    id_precision = idtp / (idtp + idfp) if idtp + idfp else None
    id_recall = idtp / (idtp + idfn) if idtp + idfn else None
    idf1 = (2 * idtp / (2 * idtp + idfp + idfn)
            if 2 * idtp + idfp + idfn else None)
    mota = 1 - (fn + fp + id_switches) / gt_total if gt_total else None
    gt_observed_ids = {obj['track_id'] for frame in gt_frames.values() for obj in frame}
    duplicate_gt_tracks = sum(len(matched_pred_ids_by_gt[track_id]) > 1 for track_id in gt_observed_ids)
    cross_identity_pred_tracks = sum(len(matched_gt_ids_by_pred[track_id]) > 1 for track_id in pred_track_ids)
    dominant_matches = sum(max((cooccurrence[(gt_id, pred_id)] for pred_id in pred_ids), default=0)
                           for gt_id in gt_ids)
    return {
        'frames': len(gt_frames), 'ground_truth_detections': gt_total,
        'predicted_detections': pred_total, 'tp': tp, 'fp': fp, 'fn': fn,
        'detection_precision': tp / (tp + fp) if tp + fp else None,
        'detection_recall': tp / gt_total if gt_total else None,
        'id_switches': id_switches, 'fragmentations': fragmentations,
        'mota': mota, 'idtp': idtp, 'idfp': idfp, 'idfn': idfn,
        'id_precision': id_precision, 'id_recall': id_recall, 'idf1': idf1,
        'gt_tracks': len(gt_track_ids), 'predicted_tracks': len(pred_track_ids),
        'track_count_error': len(pred_track_ids) - len(gt_track_ids),
        'gt_tracks_split_across_predicted_ids': duplicate_gt_tracks,
        'fraction_gt_tracks_split_across_predicted_ids': duplicate_gt_tracks / len(gt_track_ids)
            if gt_track_ids else None,
        'predicted_ids_matching_multiple_gt_tracks': cross_identity_pred_tracks,
        'dominant_identity_consistency': dominant_matches / gt_total if gt_total else None,
    }


def _aggregate(rows):
    sums = Counter()
    for row in rows:
        for key in ('frames', 'ground_truth_detections', 'predicted_detections', 'tp', 'fp', 'fn',
                    'id_switches', 'fragmentations', 'idtp', 'idfp', 'idfn', 'gt_tracks',
                    'predicted_tracks', 'gt_tracks_split_across_predicted_ids',
                    'predicted_ids_matching_multiple_gt_tracks'):
            sums[key] += row[key]
    gt, pred, tp = sums['ground_truth_detections'], sums['predicted_detections'], sums['tp']
    idtp, idfp, idfn = sums['idtp'], sums['idfp'], sums['idfn']
    return {
        **dict(sums),
        'detection_precision': tp / (tp + sums['fp']) if tp + sums['fp'] else None,
        'detection_recall': tp / gt if gt else None,
        'mota': 1 - (sums['fn'] + sums['fp'] + sums['id_switches']) / gt if gt else None,
        'id_precision': idtp / (idtp + idfp) if idtp + idfp else None,
        'id_recall': idtp / (idtp + idfn) if idtp + idfn else None,
        'idf1': 2 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else None,
        'track_count_error': sums['predicted_tracks'] - sums['gt_tracks'],
        'fraction_gt_tracks_split_across_predicted_ids':
            sums['gt_tracks_split_across_predicted_ids'] / sums['gt_tracks'] if sums['gt_tracks'] else None,
    }


def evaluate_tracking(ground_truth, predictions, *, iou_threshold=0.5):
    """Compute frame detection and sequence identity metrics from paired JSON objects."""
    iou_threshold = _finite(iou_threshold, 'iou_threshold')
    if not 0 < iou_threshold <= 1:
        raise ValueError('iou_threshold must be greater than 0 and at most 1')
    gt_sequences = _sequences(ground_truth, 'ground truth')
    pred_sequences = _sequences(predictions, 'predictions')
    if set(gt_sequences) != set(pred_sequences):
        raise ValueError('Ground-truth and prediction sequence IDs must match exactly')
    results, group_sequences = {}, defaultdict(list)
    for sequence_id in sorted(gt_sequences, key=lambda value: (type(value).__name__, str(value))):
        results[str(sequence_id)] = _sequence_metrics(gt_sequences[sequence_id],
            pred_sequences[sequence_id], iou_threshold)
        metadata = gt_sequences[sequence_id]['metadata']
        for field, aliases in GROUP_FIELDS.items():
            value = next((metadata[key] for key in aliases if metadata.get(key) not in (None, '')), None)
            if value is not None:
                group_sequences[(field, str(value))].append(sequence_id)
    overall = _aggregate(list(results.values()))
    strata = {}
    for (field, value), ids in sorted(group_sequences.items()):
        strata.setdefault(field, {})[value] = _aggregate([results[str(sequence_id)] for sequence_id in ids])
    return {'metric_version': 'roadloom-mot-v1', 'iou_threshold': iou_threshold,
            'matching': 'per-frame, class-aware maximum-total-IoU assignment; below-threshold pairs are unmatched',
            'identity_assignment': 'global maximum co-occurrence assignment over frame-local spatial matches',
            'sequences': results, 'overall': overall, 'strata': strata}


def evaluate_files(ground_truth_path, predictions_path, report_path, *, iou_threshold=0.5):
    ground_truth, gt_hash = _read(ground_truth_path)
    predictions, prediction_hash = _read(predictions_path)
    report = evaluate_tracking(ground_truth, predictions, iou_threshold=iou_threshold)
    report['provenance'] = {'ground_truth_sha256': gt_hash,
        'predictions_sha256': prediction_hash, 'evaluated_utc': datetime.now(timezone.utc).isoformat()}
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)
    return report


def export_video_run(database_path, run_id, output_path, *, sequence_id=None):
    """Export saved RoadLoom frame detections as a tracking-prediction manifest."""
    database = Path(database_path).resolve(strict=True)
    uri = database.as_uri() + '?mode=ro'
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            run = connection.execute('''SELECT r.*,s.original_name,s.sha256 AS source_sha256,
                s.metadata AS source_metadata,m.country FROM video_runs r
                JOIN video_sources s ON s.id=r.source_id AND s.mission_id=r.mission_id
                JOIN mission m ON m.id=r.mission_id WHERE r.id=?''', (run_id,)).fetchone()
            if not run:
                raise ValueError('Video run was not found in the selected mission database')
            if run['status'] != 'completed':
                raise ValueError('Only completed video runs can be exported for benchmark scoring')
            try:
                source_metadata = json.loads(run['source_metadata'])
                run_config = json.loads(run['config'])
                runtime = json.loads(run['runtime_details'] or '{}')
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError('Saved video run or source metadata is corrupt') from exc
            probe = source_metadata.get('media_probe', {})
            dimensions = probe.get('metadata', {}) if isinstance(probe, dict) else {}
            width, height = dimensions.get('width'), dimensions.get('height')
            if (isinstance(width, bool) or not isinstance(width, int) or width <= 0
                    or isinstance(height, bool) or not isinstance(height, int) or height <= 0):
                raise ValueError('Video source dimensions are unavailable; cannot convert normalized boxes')
            frames = []
            rows = connection.execute('''SELECT frame_index,pts,time_base_num,time_base_den,detections
                FROM video_frame_detections WHERE run_id=? ORDER BY frame_index''', (run_id,)).fetchall()
            if not rows:
                raise ValueError('Completed video run has no saved frame detections')
            for row in rows:
                try:
                    detections = json.loads(row['detections'])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Saved detections are corrupt at frame {row['frame_index']}") from exc
                if not isinstance(detections, list):
                    raise ValueError(f"Saved detections are corrupt at frame {row['frame_index']}")
                objects = []
                for index, detection in enumerate(detections):
                    if isinstance(detection, dict) and detection.get('kind') == 'light':
                        continue  # Review-only light hypotheses are never vehicle benchmark predictions.
                    box = detection.get('box') if isinstance(detection, dict) else None
                    label = detection.get('label') if isinstance(detection, dict) else None
                    if not isinstance(box, list) or len(box) != 4 or not isinstance(label, str) or not label:
                        raise ValueError(f"Saved detection is malformed at frame {row['frame_index']}")
                    x, y, box_width, box_height = (_finite(value, 'normalized bbox') for value in box)
                    if (x < 0 or y < 0 or box_width <= 0 or box_height <= 0
                            or x + box_width > 1 + 1e-6 or y + box_height > 1 + 1e-6):
                        raise ValueError(f"Saved detection box is outside frame {row['frame_index']}")
                    track_id = detection.get('track_id')
                    if track_id is None:
                        track_id = f"untracked-{row['frame_index']}-{index}"
                    else:
                        track_id = _identifier(track_id, 'saved prediction track_id')
                    objects.append({'track_id': track_id, 'category_id': label,
                        'bbox': [x * width, y * height, box_width * width, box_height * height]})
                frames.append({'frame_id': row['frame_index'], 'pts': row['pts'],
                    'time_base_num': row['time_base_num'], 'time_base_den': row['time_base_den'],
                    'objects': objects})
    except sqlite3.Error as exc:
        raise ValueError(f'Could not read a RoadLoom mission database: {exc}') from exc
    detector_config = run_config.get('detector') or run_config.get('plate_detector') or {}
    resource = detector_config.get('resource_profile') or {}
    metadata = {
        'country': run['country'], 'camera_view': source_metadata.get('camera_view'),
        'condition': source_metadata.get('condition'),
        'hardware_profile': resource.get('selected') if isinstance(resource, dict) else None,
        'execution_provider': runtime.get('execution_provider'),
        'video_run_id': run_id, 'source_id': run['source_id'],
        'source_name': run['original_name'], 'source_sha256': run['source_sha256'],
        'model_id': detector_config.get('version'), 'model_sha256': detector_config.get('sha256'),
    }
    document = {'sequences': [{'sequence_id': sequence_id or run['original_name'],
        'metadata': metadata, 'frames': frames}]}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(destination)
    return document
