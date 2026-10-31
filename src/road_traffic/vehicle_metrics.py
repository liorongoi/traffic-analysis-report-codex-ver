"""Precision/recall measured against complete, run-bound manual image reviews."""
import json
import math

from . import inference_store
from .store import now


IOU_THRESHOLD = 0.50
METHOD_VERSION = 'manual-review-box-matching-v1'


def _iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x_overlap = max(0.0, min(lx + lw, rx + rw) - max(lx, rx))
    y_overlap = max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = x_overlap * y_overlap
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _counts(predictions, truth, iou_threshold):
    matched = set()
    true_positives = 0
    false_positives = 0
    ordered = sorted(predictions, key=lambda item: (-float(item.get('score') or 0), item['index']))
    for prediction in ordered:
        best_index = None
        best_iou = -1.0
        for index, reference in enumerate(truth):
            if index in matched:
                continue
            overlap = _iou(prediction['box'], reference['box'])
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index is not None and best_iou >= iou_threshold:
            matched.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(truth) - len(matched)
    return {'true_positives': true_positives, 'false_positives': false_positives,
        'false_negatives': false_negatives}


def _summary(counts, image_count):
    tp, fp, fn = counts['true_positives'], counts['false_positives'], counts['false_negatives']
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall else None)
    return {**counts, 'precision': precision, 'recall': recall, 'f1': f1,
        'reviewed_images': image_count}


def evaluate(library, mission, run, iou_threshold=IOU_THRESHOLD):
    """Score vehicle boxes at a saved run's operating point on complete bound reviews."""
    if not isinstance(run, str) or not run:
        raise ValueError('Select a saved still-image inference run')
    if type(iou_threshold) not in (int, float) or not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError('IoU matching threshold must be greater than 0 and at most 1')
    iou_threshold = float(iou_threshold)
    inference_store.prepare(library, mission)
    with library.connect(mission) as conn:
        run_row = conn.execute('SELECT id,status,total,config FROM inference_runs WHERE id=?', (run,)).fetchone()
        if not run_row:
            raise ValueError('Inference run not found in this mission')
        config = json.loads(run_row['config'])
        rows = [dict(row) for row in conn.execute('''SELECT ii.image_id,ii.sha256,
                p.annotations prediction_annotations,p.error prediction_error,
                r.id review_id,r.revision review_revision,r.role evidence_role,
                r.status review_status,r.annotations review_annotations,
                ri.inference_id review_run
            FROM inference_inputs ii
            LEFT JOIN predictions p ON p.inference_id=ii.inference_id AND p.image_id=ii.image_id
            LEFT JOIN reviews r ON r.image_id=ii.image_id AND r.revision=(
                SELECT MAX(latest.revision) FROM reviews latest WHERE latest.image_id=ii.image_id)
            LEFT JOIN review_inference ri ON ri.review_id=r.id
            WHERE ii.inference_id=? ORDER BY ii.image_id''', (run,))]
        image_names = {row['id']: row['name'] for row in conn.execute(
            'SELECT id,name FROM images WHERE id IN (SELECT image_id FROM inference_inputs WHERE inference_id=?)', (run,))}

    totals = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    roles = {}
    reviewed = []
    full_review_count = 0
    partial_review_count = 0
    wrong_run_review_count = 0
    missing_review_count = 0
    failed_prediction_count = 0
    completed_prediction_count = 0
    eligible_prediction_count = 0
    for row in rows:
        if row['prediction_annotations'] is None or row['prediction_error']:
            failed_prediction_count += 1
            continue
        completed_prediction_count += 1
        if row['review_id'] is None:
            missing_review_count += 1
            continue
        if row['review_run'] != run:
            wrong_run_review_count += 1
            continue
        if row['review_status'] != 'reviewed':
            partial_review_count += 1
            continue
        full_review_count += 1
        predictions = []
        for index, item in enumerate(json.loads(row['prediction_annotations'])):
            if item.get('kind') not in ('vehicle', 'candidate'):
                continue
            if not str(item.get('evidence') or '').startswith('yolox-s'):
                continue
            predictions.append({'index': index, 'box': item['box'], 'score': item.get('score')})
        annotations = json.loads(row['review_annotations'])
        truth = [{'box': item['box']} for item in annotations
            if item.get('kind') == 'vehicle' and not item.get('rejected')]
        counts = _counts(predictions, truth, iou_threshold)
        for key in totals:
            totals[key] += counts[key]
        role = row['evidence_role'] or 'none'
        bucket = roles.setdefault(role, {'true_positives': 0, 'false_positives': 0,
            'false_negatives': 0, 'reviewed_images': 0})
        for key in ('true_positives', 'false_positives', 'false_negatives'):
            bucket[key] += counts[key]
        bucket['reviewed_images'] += 1
        eligible_prediction_count += len(predictions)
        reviewed.append({'image_id': row['image_id'], 'name': image_names.get(row['image_id']),
            'sha256': row['sha256'], 'review_id': row['review_id'],
            'review_revision': row['review_revision'], 'evidence_role': role, **counts})

    by_role = {role: _summary(counts, counts['reviewed_images'])
        for role, counts in sorted(roles.items())}
    holdout_roles = [role for role in ('validation', 'test') if role in roles]
    holdout_counts = {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0,
        'reviewed_images': 0}
    for role in holdout_roles:
        for key in holdout_counts:
            holdout_counts[key] += roles[role][key]
    return {
        'method': METHOD_VERSION,
        'mission_id': mission,
        'run_id': run,
        'run_status': run_row['status'],
        'model_id': config.get('vehicle_model_id', 'base'),
        'model_sha256': config.get('model_sha256'),
        'detector_score_threshold': config.get('threshold'),
        'iou_threshold': iou_threshold,
        'matching': 'score-descending greedy one-to-one box matching; IoU >= threshold; class-agnostic vehicle localization',
        'review_requirement': 'latest full-image review bound to this exact inference run',
        'input_images': len(rows),
        'completed_prediction_images': completed_prediction_count,
        'failed_or_missing_prediction_images': failed_prediction_count,
        'fully_reviewed_images': full_review_count,
        'partial_review_images': partial_review_count,
        'unreviewed_images': missing_review_count,
        'reviews_bound_to_other_runs': wrong_run_review_count,
        'eligible_detector_predictions': eligible_prediction_count,
        'overall': _summary(totals, full_review_count),
        'by_evidence_role': by_role,
        'validation_test_holdout': _summary(holdout_counts, holdout_counts['reviewed_images']) if holdout_roles else None,
        'reviewed_image_revisions': reviewed,
        'interpretation': 'Measured on the listed full manual reviews only. This diagnostic does not set a pass/fail accuracy threshold and is not an independent benchmark unless the selected evidence roles and review procedure justify that claim.',
        'computed_at': now(),
    }
