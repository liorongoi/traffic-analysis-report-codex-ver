"""Optional local SAM 3.1 semantic road-mask proposals; independent of YOLO."""
from functools import lru_cache
from pathlib import Path
import os

MAX_SIDE = 1600
MAX_CANDIDATES = 8


def _model_dir(project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[2])
    override = os.environ.get('ROAD_TRAFFIC_SAM3_DIR')
    if override:
        return Path(override)
    local_app_data = os.environ.get('LOCALAPPDATA')
    if local_app_data:
        user_pack = Path(local_app_data) / 'RoadTrafficAnalysis' / 'Models' / 'sam3.1-unsloth'
        if user_pack.is_dir():
            return user_pack
    return root / 'model-packs' / 'sam3.1-unsloth'


@lru_cache(maxsize=1)
def _load_sam(model_dir):
    optional_runtime = Path(model_dir) / 'runtime'
    if optional_runtime.is_dir():
        runtime_path = str(optional_runtime.resolve())
        if runtime_path not in os.sys.path:
            os.sys.path.append(runtime_path)
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise RuntimeError('Optional SAM 3.1 Transformers runtime is not installed. YOLO vehicle detection remains available.') from exc
    model_path = Path(model_dir)
    if not (model_path / 'config.json').is_file():
        raise RuntimeError('Optional SAM 3.1 model pack is missing config.json. YOLO vehicle detection remains available.')
    if not (model_path / 'model.safetensors').is_file():
        quantized = next((name for name in (
            'sam3.1-INT8.safetensors',
            'sam3.1-INT8-ConvRot.safetensors',
            'sam3.1-FP8.safetensors',
        ) if (model_path / name).is_file()), None)
        if quantized:
            raise RuntimeError(
                f'The optional SAM 3.1 pack contains {quantized}, which this app cannot load yet. '
                'Use Unsloth model.safetensors (BF16) for the current CPU-compatible loader. '
                'YOLO vehicle detection remains available.')
        raise RuntimeError(
            'Optional SAM 3.1 weights are missing. Add Unsloth model.safetensors (BF16); '
            'YOLO vehicle detection remains available.')
    try:
        model = Sam3Model.from_pretrained(str(model_path), dtype=torch.float32, local_files_only=True)
        processor = Sam3Processor.from_pretrained(str(model_path), local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f'Could not load the local SAM 3.1 model pack: {type(exc).__name__}: {exc}') from exc
    model.to('cpu').eval()
    return torch, model, processor


def propose_road_masks(rgb, project_root=None):
    """Use a road concept prompt to return ranked polygons for operator review."""
    import cv2
    import numpy as np
    from PIL import Image
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an RGB image')
    height, width = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    if scale < 1:
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    torch, model, processor = _load_sam(str(_model_dir(project_root).resolve()))
    image = Image.fromarray(np.ascontiguousarray(rgb))
    inputs = processor(images=image, text='road surface', return_tensors='pt')
    inputs = {key: value.to('cpu') if hasattr(value, 'to') else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(outputs, threshold=.25, mask_threshold=.5,
        target_sizes=inputs['original_sizes'].tolist())[0]
    masks = result.get('masks')
    scores = result.get('scores')
    if masks is None:
        return {'candidates': [], 'model': 'SAM 3.1 Unsloth Transformers local model pack',
            'prompt': 'road surface', 'automatic_vehicle_detection_changed': False, 'review_required': True}
    if hasattr(masks, 'detach'):
        masks = masks.detach().cpu().numpy()
    if hasattr(scores, 'detach'):
        scores = scores.detach().cpu().numpy()
    h, w = rgb.shape[:2]
    candidates = []
    for index, raw_mask in enumerate(masks):
        segmentation = np.asarray(raw_mask).squeeze().astype(np.uint8)
        if segmentation.shape != (h, w):
            segmentation = cv2.resize(segmentation, (w, h), interpolation=cv2.INTER_NEAREST)
        area = float(segmentation.sum()) / segmentation.size
        if not .01 <= area <= .92:
            continue
        ys, xs = np.nonzero(segmentation)
        if len(xs) < 3:
            continue
        bottom = float(ys.max()) / max(1, h - 1)
        center_y = float(ys.mean()) / max(1, h - 1)
        contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, max(1.0, cv2.arcLength(contour, True) * .003), True).reshape(-1, 2)
        if len(contour) < 3:
            continue
        confidence = float(scores[index]) if scores is not None and index < len(scores) else 0.0
        # The text-conditioned model identifies road; geometry just breaks ties.
        score = confidence * (.75 + .25 * min(1.0, bottom))
        candidates.append({'score': round(score, 4), 'model_score': round(confidence, 4),
            'area_fraction': round(area, 4), 'polygon': [[round(float(x) / w, 6), round(float(y) / h, 6)] for x, y in contour],
            'source': 'sam3.1-road-surface-text-prompt',
            'note': 'Text-prompted road mask; operator approval and role assignment required.'})
    candidates.sort(key=lambda item: item['score'], reverse=True)
    selected = []
    for item in candidates:
        xs = [p[0] for p in item['polygon']]; ys = [p[1] for p in item['polygon']]
        box = (min(xs), min(ys), max(xs), max(ys))
        if any(_bbox_iou(box, other['_box']) > .88 for other in selected):
            continue
        item['_box'] = box
        selected.append(item)
        if len(selected) >= MAX_CANDIDATES:
            break
    for item in selected:
        item.pop('_box', None)
    return {'candidates': selected, 'model': 'SAM 3.1 Unsloth Transformers', 'prompt': 'road surface',
        'automatic_vehicle_detection_changed': False, 'review_required': True, 'input_scale': round(scale, 4),
        'execution_provider': 'CPU'}


def _bbox_iou(a, b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter) if aa+bb>inter else 0.0
