"""Offline COCO-style detection evaluation for a frozen reviewed test split."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
from posixpath import normpath
import zipfile


GROUP_FIELDS = {
    'country': ('country', 'target_country', 'mission_country_context'),
    'condition': ('condition', 'day_night', 'period'),
    'camera_view': ('camera_view', 'view_id', 'view'),
    'hardware_profile': ('hardware_profile', 'hardware'),
}
AP_IOUS = tuple(round(.50 + i * .05, 2) for i in range(10))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _load_ground_truth(path, split):
    path = Path(path)
    if zipfile.is_zipfile(path):
        member = f'{split}/annotations.json'
        with zipfile.ZipFile(path) as archive:
            try:
                value = json.loads(archive.read(member))
            except KeyError as exc:
                raise ValueError(f'Dataset ZIP does not contain {member}') from exc
    else:
        value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not all(isinstance(value.get(key), list)
                                               for key in ('images', 'annotations', 'categories')):
        raise ValueError('Ground truth must be a COCO object with images, annotations, and categories arrays')
    return value


def _load_predictions(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(value, list):
        return value, {}
    if isinstance(value, dict) and isinstance(value.get('predictions'), list):
        metadata = value.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('Prediction metadata must be an object')
        return value['predictions'], metadata
    raise ValueError('Predictions must be a COCO result array or an object with predictions and metadata')


def _bbox(value, name, width=None, height=None):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f'{name} must be [x, y, width, height]')
    x, y, w, h = (_finite_number(v, name) for v in value)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f'{name} must have nonnegative origin and positive dimensions')
    if width is not None and (x + w > width + 1e-6 or y + h > height + 1e-6):
        raise ValueError(f'{name} extends outside its image')
    return (x, y, w, h)


def _iou(left, right):
    x1 = max(left[0], right[0]); y1 = max(left[1], right[1])
    x2 = min(left[0] + left[2], right[0] + right[2])
    y2 = min(left[1] + left[3], right[1] + right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def _ap(tp, fp, positives):
    if positives == 0:
        return None
    if not tp:
        return 0.0
    true_positive = false_positive = 0
    recalls, precisions = [], []
    for is_tp, is_fp in zip(tp, fp):
        true_positive += is_tp
        false_positive += is_fp
        recalls.append(true_positive / positives)
        precisions.append(true_positive / (true_positive + false_positive))
    samples = []
    for target_recall in (index / 100 for index in range(101)):
        samples.append(max((precision for recall, precision in zip(recalls, precisions)
                            if recall >= target_recall), default=0.0))
    return sum(samples) / len(samples)


def _class_evaluation(image_ids, category_id, truths, predictions, iou_threshold):
    gt_by_image = defaultdict(list)
    for image_id in image_ids:
        gt_by_image[image_id] = list(truths.get((image_id, category_id), ()))
    matched = {image_id: set() for image_id in image_ids}
    ordered = sorted((item for item in predictions if item['image_id'] in image_ids
                      and item['category_id'] == category_id),
                     key=lambda item: (-item['score'], item['_index']))
    tp_flags, fp_flags = [], []
    for item in ordered:
        candidates = gt_by_image[item['image_id']]
        best_index, best_iou = None, -1.0
        for index, truth in enumerate(candidates):
            if index in matched[item['image_id']]:
                continue
            overlap = _iou(item['bbox'], truth)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= iou_threshold:
            matched[item['image_id']].add(best_index)
            tp_flags.append(1); fp_flags.append(0)
        else:
            tp_flags.append(0); fp_flags.append(1)
    positives = sum(len(gt_by_image[image_id]) for image_id in image_ids)
    tp = sum(tp_flags); fp = sum(fp_flags); fn = positives - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0
          if precision is not None and recall is not None else None)
    aps = [_ap(*_class_flags(image_ids, category_id, truths, predictions, threshold), positives)
           for threshold in AP_IOUS]
    valid_aps = [value for value in aps if value is not None]
    return {'ground_truth': positives, 'predictions': len(ordered), 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'false_alarms_per_image': fp / len(image_ids) if image_ids else None,
        'ap50': aps[0], 'map50_95': sum(valid_aps) / len(valid_aps) if valid_aps else None}


def _class_flags(image_ids, category_id, truths, predictions, threshold):
    gt_by_image = {image_id: list(truths.get((image_id, category_id), ()))
                   for image_id in image_ids}
    matched = {image_id: set() for image_id in image_ids}
    ordered = sorted((item for item in predictions if item['image_id'] in image_ids
                      and item['category_id'] == category_id),
                     key=lambda item: (-item['score'], item['_index']))
    tp, fp = [], []
    for item in ordered:
        best_index, best_iou = None, -1.0
        for index, truth in enumerate(gt_by_image[item['image_id']]):
            if index not in matched[item['image_id']]:
                overlap = _iou(item['bbox'], truth)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= threshold:
            matched[item['image_id']].add(best_index); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    return tp, fp


def _evaluate(image_ids, categories, truths, predictions, images, operating_iou,
              image_metadata=None, run_metadata=None):
    if not image_ids:
        return {'images': 0, 'categories': {}, 'micro': {}, 'count_error': None}
    per_category = {categories[category_id]: _class_evaluation(image_ids, category_id, truths,
        predictions, operating_iou) for category_id in categories}
    tp = sum(item['tp'] for item in per_category.values())
    fp = sum(item['fp'] for item in per_category.values())
    fn = sum(item['fn'] for item in per_category.values())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0
          if precision is not None and recall is not None else None)
    valid_ap50 = [item['ap50'] for item in per_category.values() if item['ap50'] is not None]
    valid_map = [item['map50_95'] for item in per_category.values() if item['map50_95'] is not None]
    abs_errors, signed_errors = [], []
    for image_id in image_ids:
        gt_count = sum(len(truths.get((image_id, category_id), ())) for category_id in categories)
        pred_count = sum(1 for item in predictions if item['image_id'] == image_id
                         and item['category_id'] in categories)
        abs_errors.append(abs(pred_count - gt_count)); signed_errors.append(pred_count - gt_count)
    return {'images': len(image_ids), 'categories': per_category,
        'micro': {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1,
            'false_alarms_per_image': fp / len(image_ids),
            'ap50_macro': sum(valid_ap50) / len(valid_ap50) if valid_ap50 else None,
            'map50_95_macro': sum(valid_map) / len(valid_map) if valid_map else None},
        'count_error': {'mae_objects_per_image': sum(abs_errors) / len(abs_errors),
            'mean_signed_objects_per_image': sum(signed_errors) / len(signed_errors),
            'exact_count_images_fraction': sum(value == 0 for value in signed_errors) / len(signed_errors)}}


def evaluate_coco(ground_truth, predictions, *, operating_iou=.5, metadata=None,
                  prediction_categories=()):
    """Evaluate COCO XYWH boxes and report stratified, reproducible metrics.

    ``ground_truth`` is a COCO dataset object. ``predictions`` is a COCO result
    array. Metadata groups are read from each COCO image row; ground-truth image
    labels must be complete for every included image. This function does not
    infer labels from unmarked regions or make any acceptance decision.
    """
    operating_iou = _finite_number(operating_iou, 'operating_iou')
    if not .1 <= operating_iou <= .95:
        raise ValueError('operating_iou must be between 0.1 and 0.95')
    if not isinstance(ground_truth, dict) or not all(isinstance(ground_truth.get(k), list)
                                                     for k in ('images', 'annotations', 'categories')):
        raise ValueError('Ground truth must contain COCO images, annotations, and categories arrays')
    images = {}
    for row in ground_truth['images']:
        image_id = row.get('id')
        if image_id is None or image_id in images:
            raise ValueError('Ground-truth image IDs must be present and unique')
        width = _finite_number(row.get('width'), 'image width')
        height = _finite_number(row.get('height'), 'image height')
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions must be positive')
        images[image_id] = row
    categories = {}
    for row in ground_truth['categories']:
        category_id = row.get('id')
        if category_id is None or category_id in categories or not isinstance(row.get('name'), str):
            raise ValueError('Ground-truth categories must have unique IDs and names')
        categories[category_id] = row['name']
    for row in prediction_categories:
        if not isinstance(row, dict) or row.get('id') is None or not isinstance(row.get('name'), str):
            raise ValueError('Prediction-only categories must have an ID and name')
        if row['id'] in categories and categories[row['id']] != row['name']:
            raise ValueError('Prediction-only category ID conflicts with ground-truth taxonomy')
        categories[row['id']] = row['name']
    if not images or not categories:
        raise ValueError('Ground truth must contain at least one image and one category')
    truths = defaultdict(list)
    for row in ground_truth['annotations']:
        image_id, category_id = row.get('image_id'), row.get('category_id')
        if image_id not in images or category_id not in categories:
            raise ValueError('Every ground-truth annotation must reference a known image and category')
        image = images[image_id]
        truths[(image_id, category_id)].append(_bbox(row.get('bbox'), 'ground-truth bbox',
            _finite_number(image['width'], 'image width'), _finite_number(image['height'], 'image height')))
    clean_predictions = []
    for index, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise ValueError('Each prediction must be an object')
        image_id, category_id = row.get('image_id'), row.get('category_id')
        if image_id not in images or category_id not in categories:
            raise ValueError('Every prediction must reference a known image and category')
        image = images[image_id]
        bbox = _bbox(row.get('bbox'), 'prediction bbox',
            _finite_number(image['width'], 'image width'), _finite_number(image['height'], 'image height'))
        score = _finite_number(row.get('score'), 'prediction score')
        if not 0 <= score <= 1:
            raise ValueError('Prediction scores must be between 0 and 1')
        clean_predictions.append({'image_id': image_id, 'category_id': category_id,
                                  'bbox': bbox, 'score': score, '_index': index})
    image_ids = set(images)
    report_metadata = metadata or {}
    prediction_metadata = report_metadata.get('prediction', {}) if isinstance(report_metadata, dict) else {}
    if not isinstance(prediction_metadata, dict):
        prediction_metadata = {}
    image_metadata = prediction_metadata.get('image_metadata', {})
    run_metadata = prediction_metadata.get('strata', {})
    if not isinstance(image_metadata, dict) or not isinstance(run_metadata, dict):
        raise ValueError('Prediction image strata must be JSON objects')
    overall = _evaluate(image_ids, categories, truths, clean_predictions, images,
        operating_iou, image_metadata, run_metadata)
    named_categories = overall['categories']
    grouped = {}
    image_metadata = image_metadata or {}
    run_metadata = run_metadata or {}
    for field, aliases in GROUP_FIELDS.items():
        groups = defaultdict(set)
        for image_id, row in images.items():
            sidecar = image_metadata.get(str(image_id), image_metadata.get(image_id, {}))
            value = next((row.get(alias) for alias in aliases if row.get(alias) not in (None, '')), None)
            if value is None and isinstance(sidecar, dict):
                value = next((sidecar.get(alias) for alias in aliases
                    if sidecar.get(alias) not in (None, '')), None)
            if value is None:
                value = next((run_metadata.get(alias) for alias in aliases
                    if run_metadata.get(alias) not in (None, '')), 'unspecified')
            groups[str(value)].add(image_id)
        if len(groups) > 1 or 'unspecified' not in groups:
            grouped[field] = {value: _evaluate(ids, categories, truths, clean_predictions,
                images, operating_iou, image_metadata, run_metadata)
                for value, ids in sorted(groups.items())}
    return {'schema': 'roadloom-detection-benchmark-v1', 'metrics': {
            'operating_iou': operating_iou, 'ap_iou_thresholds': list(AP_IOUS),
            'matching': 'class-aware greedy matching by descending prediction score, one-to-one within each image',
            'ap': '101-point interpolated AP; macro mean over categories with ground-truth instances',
            'count': 'all predicted and ground-truth instances across categories per image',
            'scope_note': 'Metrics are only meaningful when every image has complete independent labels; unmarked objects are not assumed negative.'},
        'metadata': report_metadata, 'images': len(images), 'categories': named_categories,
        'overall': overall, 'groups': grouped}


def evaluate_files(ground_truth_path, prediction_path, output_path, *, split='test', operating_iou=.5):
    ground_truth_path, prediction_path, output_path = map(Path,
        (ground_truth_path, prediction_path, output_path))
    ground_truth = _load_ground_truth(ground_truth_path, split)
    predictions, prediction_metadata = _load_predictions(prediction_path)
    prediction_categories = prediction_metadata.get('categories', ())
    if prediction_categories is None:
        prediction_categories = ()
    if not isinstance(prediction_categories, (list, tuple)):
        raise ValueError('Prediction categories must be an array')
    report = evaluate_coco(ground_truth, predictions, operating_iou=operating_iou,
        prediction_categories=prediction_categories,
        metadata={'prediction': prediction_metadata,
            'ground_truth_sha256': _sha256(ground_truth_path),
            'predictions_sha256': _sha256(prediction_path),
            'ground_truth_file': ground_truth_path.name, 'predictions_file': prediction_path.name,
            'split': split})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name('.' + output_path.name + '.tmp')
    try:
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                        encoding='utf-8')
        temp.replace(output_path)
    finally:
        temp.unlink(missing_ok=True)
    return report


def predict_coco_zip(dataset_path, model_pack, *, split='test', model_kind='vehicle',
                     threshold=None, tiled=True, detector_factory=None,
                     image_metadata=None, strata=None):
    """Run a local detector once over a frozen COCO ZIP split.

    Only one source image is decoded at a time. Vehicle output is the YOLOX
    detector baseline; mission-specific light/background/ROI branches are not
    applied. Plate output contains boxes only and never recognizes text.
    """
    dataset_path = Path(dataset_path)
    if not zipfile.is_zipfile(dataset_path):
        raise ValueError('Prediction input must be a RoadLoom COCO dataset ZIP')
    gt = _load_ground_truth(dataset_path, split)
    if model_kind not in ('vehicle', 'plate'):
        raise ValueError('model_kind must be vehicle or plate')
    if threshold is None:
        threshold = .18 if model_kind == 'vehicle' else .3
    threshold = _finite_number(threshold, 'threshold')
    if not .01 <= threshold <= .9:
        raise ValueError('threshold must be between 0.01 and 0.9')
    import numpy as np
    from PIL import Image
    if detector_factory:
        detector = detector_factory(model_kind, model_pack, threshold, tiled)
    elif model_kind == 'vehicle':
        from .detector import VehicleDetector
        detector = VehicleDetector(model_pack, threshold=threshold, tiled=tiled)
    else:
        from .plate_review import PlateDetector
        detector = PlateDetector(model_pack, threshold=threshold)
    categories = {row['id']: row for row in gt['categories']}
    next_id = max((key for key in categories if isinstance(key, int)), default=0) + 1
    added_categories = []

    def category_id_for(name, kind, label):
        nonlocal next_id
        existing = next((row['id'] for row in categories.values()
            if (row.get('kind') == kind and row.get('label') == label) or row.get('name') == name), None)
        if existing is not None:
            return existing
        category_id = next_id
        next_id += 1
        row = {'id': category_id, 'name': name, 'kind': kind, 'label': label}
        categories[category_id] = row
        added_categories.append(row)
        return category_id

    mapped_classes = {}
    if model_kind == 'vehicle':
        for label in detector.class_labels.values():
            mapped_classes[label] = category_id_for(f'vehicle:{label}', 'vehicle', label)
    else:
        mapped_classes['plate:region'] = category_id_for('plate:region', 'plate', 'region')
    predictions = []
    with zipfile.ZipFile(dataset_path) as archive:
        for row in gt['images']:
            if _finite_number(row.get('width'), 'image width') * _finite_number(row.get('height'), 'image height') > 24_000_000:
                raise ValueError('Benchmark image exceeds the 24-million-pixel adapter limit')
            filename = row.get('file_name')
            member = normpath(f'{split}/{filename}') if isinstance(filename, str) else ''
            if (not member.startswith(f'{split}/') or member.startswith('../')
                    or '/..' in member or member.startswith('/')):
                raise ValueError('COCO image path escapes its split folder')
            try:
                with archive.open(member) as stream:
                    with Image.open(BytesIO(stream.read())) as image:
                        rgb = np.asarray(image.convert('RGB'))
            except KeyError as exc:
                raise ValueError(f'Dataset ZIP is missing {member}') from exc
            if rgb.shape[1] != row['width'] or rgb.shape[0] != row['height']:
                raise ValueError(f'Dataset image dimensions disagree with COCO metadata: {filename}')
            detections = detector.infer(rgb) if model_kind == 'vehicle' else detector.detect(rgb)
            for detection in detections:
                x, y, width, height = detection['box']
                label = detection.get('label')
                if model_kind == 'vehicle':
                    category_id = mapped_classes.get(label)
                    if category_id is None:
                        raise ValueError(f'Model returned unsupported class {label!r}')
                else:
                    category_id = mapped_classes['plate:region']
                predictions.append({'image_id': row['id'], 'category_id': category_id,
                    'bbox': [x * rgb.shape[1], y * rgb.shape[0],
                             width * rgb.shape[1], height * rgb.shape[0]],
                    'score': float(detection.get('score', detection.get('confidence')))})
    allowed_strata = {alias for aliases in GROUP_FIELDS.values() for alias in aliases}
    if image_metadata is not None and not isinstance(image_metadata, dict):
        raise ValueError('image_metadata must map COCO image IDs to metadata objects')
    for values in (image_metadata or {}).values():
        if not isinstance(values, dict) or any(key not in allowed_strata for key in values):
            raise ValueError('Image metadata may contain only country, condition, camera view, or hardware profile fields')
    if strata is not None and (not isinstance(strata, dict)
            or any(key not in allowed_strata for key in strata)):
        raise ValueError('strata may contain only country, condition, camera view, or hardware profile fields')
    return {'predictions': predictions, 'metadata': {
        'model_kind': model_kind, 'model_sha256': detector.sha,
        'model_pack': Path(model_pack).name, 'threshold': threshold,
        'tiled': bool(tiled) if model_kind == 'vehicle' else False,
        'inference_provider': getattr(getattr(detector, 'session', None), 'provider', None),
        'pipeline_scope': ('YOLOX vehicle detector only; excludes night-light, persistent-background, '
                           'mission ROI and road-mask logic' if model_kind == 'vehicle' else
                           'plate-region detector only; crop localization with no text recognition'),
        'split': split, 'source_dataset_sha256': _sha256(dataset_path),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'categories': added_categories, 'image_metadata': image_metadata or {},
        'strata': strata or {}}}
