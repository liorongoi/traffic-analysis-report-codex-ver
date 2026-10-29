"""Prepare reviewed COCO evidence and run the optional upstream YOLOX trainer."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile


VEHICLE_LABELS = ('bicycle', 'car', 'motorcycle', 'bus', 'truck')
BASELINE_SHA256 = 'f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in path.parts):
        raise ValueError(f'Unsafe dataset path: {name}')
    return path


def _read_split(archive, role):
    member = f'{role}/annotations.json'
    try:
        return json.loads(archive.read(member))
    except KeyError as exc:
        raise ValueError(f'Dataset is missing {member}') from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Dataset file is not valid JSON: {member}') from exc


def prepare_yolox_coco(dataset_zip, destination):
    """Stage only explicit vehicle boxes from the app's protected COCO export.

    Complete reviewed images with no vehicle boxes remain negative examples.
    The held-out test role is counted in the returned summary but is never copied
    into YOLOX training/evaluation directories.
    """
    dataset_zip = Path(dataset_zip).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Training data destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_zip) as archive:
        names = set(archive.namelist())
        if 'dataset-manifest.json' not in names:
            raise ValueError('This is not a RoadLoom reviewed dataset export')
        manifest = json.loads(archive.read('dataset-manifest.json'))
        if manifest.get('format') != 'RoadLoom COCO dataset v1':
            raise ValueError('Unsupported RoadLoom dataset export version')
        splits = {role: _read_split(archive, role) for role in ('training', 'validation', 'test')}
        category_by_id = {}
        for category in splits['training'].get('categories', []):
            category_by_id[category.get('id')] = category
        selected_labels = set()
        for annotation in splits['training'].get('annotations', []):
            category = category_by_id.get(annotation.get('category_id'), {})
            if category.get('kind') == 'vehicle' and category.get('label') in VEHICLE_LABELS:
                selected_labels.add(category['label'])
        classes = [label for label in VEHICLE_LABELS if label in selected_labels]
        if not classes:
            raise ValueError('Training split has no explicitly reviewed vehicle boxes for supported classes')
        class_ids = {label: index + 1 for index, label in enumerate(classes)}

        summary = {'classes': classes, 'splits': {},
            'dataset_manifest_sha256': hashlib.sha256(archive.read('dataset-manifest.json')).hexdigest(),
            'test_used_for_training_or_validation': False}
        for role, yolox_name in (('training', 'train2017'), ('validation', 'val2017')):
            data = splits[role]
            source_images = data.get('images', [])
            source_annotations = data.get('annotations', [])
            ids_with_vehicle = set()
            output_annotations = []
            cat_lookup = {item.get('id'): item for item in data.get('categories', [])}
            for item in source_annotations:
                category = cat_lookup.get(item.get('category_id'), {})
                label = category.get('label')
                if category.get('kind') != 'vehicle' or label not in class_ids:
                    continue
                bbox = item.get('bbox')
                if (not isinstance(bbox, list) or len(bbox) != 4 or
                        any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in bbox)):
                    raise ValueError(f'Invalid COCO box in {role} split')
                x, y, width, height = bbox
                if width <= 0 or height <= 0 or x < 0 or y < 0:
                    raise ValueError(f'Out-of-range COCO box in {role} split')
                image_id = item.get('image_id')
                ids_with_vehicle.add(image_id)
                output_annotations.append({**item, 'category_id': class_ids[label]})

            image_ids = {item.get('id') for item in source_images}
            if any(item.get('image_id') not in image_ids for item in output_annotations):
                raise ValueError(f'{role} annotations refer to missing images')
            if not source_images:
                raise ValueError(f'{role} split is empty; a train and validation split are both required')
            if role == 'validation' and not output_annotations:
                raise ValueError('Validation split has no reviewed vehicle boxes')

            image_root = destination / yolox_name / 'images'
            image_root.mkdir(parents=True, exist_ok=True)
            output_images = []
            for item in source_images:
                original_name = item.get('file_name')
                if not isinstance(original_name, str):
                    raise ValueError(f'Invalid image path in {role} split')
                relative = _safe_member(original_name)
                source_member = f'{role}/{relative.as_posix()}'
                if source_member not in names:
                    raise ValueError(f'Dataset is missing image {source_member}')
                target_name = relative.name
                output_path = image_root / target_name
                if output_path.exists():
                    raise ValueError(f'Duplicate image filename in {role} split: {target_name}')
                with archive.open(source_member) as source, output_path.open('wb') as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                output_images.append({**item, 'file_name': f'images/{target_name}'})

            coco = {'images': output_images, 'annotations': output_annotations,
                'categories': [{'id': class_ids[label], 'name': label} for label in classes]}
            ann_dir = destination / 'annotations'
            ann_dir.mkdir(parents=True, exist_ok=True)
            (ann_dir / f'instances_{yolox_name}.json').write_text(
                json.dumps(coco, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
            summary['splits'][role] = {'images': len(output_images),
                'positive_images': len(ids_with_vehicle), 'vehicle_annotations': len(output_annotations)}

        test_data = splits['test']
        summary['splits']['test'] = {'images': len(test_data.get('images', [])),
            'annotations': len(test_data.get('annotations', [])), 'used': False}
        (destination / 'roadloom-training-manifest.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        return summary


def _write_exp(path, data_root, classes, epochs, workers):
    source = f'''import os
from yolox.exp import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = {len(classes)}
        self.data_dir = {str(data_root)!r}
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.exp_name = "roadloom_yolox_finetune"
        self.max_epoch = {epochs}
        self.data_num_workers = {workers}
        self.eval_interval = 1
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.0
        self.enable_mixup = False
'''
    path.write_text(source, encoding='utf-8')


def finetune_yolox(dataset_zip, output, *, yolox_root=None, baseline=None,
        epochs=50, batch_size=4, workers=0, devices=1):
    """Fine-tune the staged YOLOX-S COCO checkpoint via the official YOLOX trainer."""
    if type(epochs) is not int or not 1 <= epochs <= 300:
        raise ValueError('Epochs must be between 1 and 300')
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError('Batch size must be between 1 and 128')
    if type(workers) is not int or not 0 <= workers <= 16:
        raise ValueError('Data workers must be between 0 and 16')
    if devices != 1:
        raise ValueError('This guided training path currently uses one GPU device')
    project = Path(__file__).resolve().parents[2]
    baseline = Path(baseline or project / 'model-packs/yolox-s/yolox_s_coco_pretrained.pth').resolve(strict=True)
    if _sha256(baseline) != BASELINE_SHA256:
        raise ValueError('The YOLOX-S baseline does not match the bundled COCO checkpoint SHA-256')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Fine-tuning needs the optional PyTorch/YOLOX training runtime. The main app remains available.') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('YOLOX fine-tuning requires a CUDA-capable NVIDIA GPU in this training profile')
    if yolox_root:
        os.environ['PYTHONPATH'] = str(Path(yolox_root).resolve(strict=True)) + os.pathsep + os.environ.get('PYTHONPATH', '')
    try:
        import yolox  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install the optional YOLOX 0.3 training package or point --yolox-root at its local source checkout') from exc

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = output / 'dataset'
    if data_root.exists():
        shutil.rmtree(data_root)
    summary = prepare_yolox_coco(dataset_zip, data_root)
    exp_path = output / 'roadloom_yolox_exp.py'
    _write_exp(exp_path, data_root, summary['classes'], epochs, workers)
    command = [sys.executable, '-m', 'yolox.tools.train', '-f', str(exp_path), '-d', str(devices),
        '-b', str(batch_size), '--ckpt', str(baseline)]
    subprocess.run(command, cwd=output, check=True)
    run_dir = output / 'YOLOX_outputs' / 'roadloom_yolox_finetune'
    checkpoints = [run_dir / 'best_ckpt.pth', run_dir / 'last_epoch_ckpt.pth']
    checkpoint = next((path for path in checkpoints if path.is_file()), None)
    if checkpoint is None:
        raise RuntimeError('YOLOX training exited successfully but produced no best or last-epoch checkpoint')
    model_pack = output / 'roadloom-model-pack'
    model_pack.mkdir(exist_ok=True)
    onnx_path = model_pack / 'model.onnx'
    export = [sys.executable, '-m', 'yolox.tools.export_onnx', '-f', str(exp_path), '-c', str(checkpoint),
        '--output-name', str(onnx_path), '--no-onnxsim']
    subprocess.run(export, cwd=output, check=True)
    digest = _sha256(onnx_path)
    manifest = {'name': 'RoadLoom fine-tuned YOLOX-S', 'architecture': 'YOLOX-S',
        'classes': summary['classes'], 'input_size': [640, 640], 'files': {'model.onnx': {
            'sha256': digest, 'bytes': onnx_path.stat().st_size}},
        'training': {'method': 'fine_tuning', 'baseline': 'OpenCV Zoo YOLOX-S COCO',
            'baseline_sha256': BASELINE_SHA256, 'checkpoint_sha256': _sha256(checkpoint),
            'dataset_manifest_sha256': summary['dataset_manifest_sha256'], 'epochs': epochs,
            'batch_size': batch_size, 'data_workers': workers,
            'validation_role_used': True, 'held_out_test_used': False,
            'best_checkpoint_selected': checkpoint.name == 'best_ckpt.pth'}}
    (model_pack / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'output': str(output), 'model_pack': str(model_pack), 'model': str(onnx_path), 'model_sha256': digest,
        'checkpoint': str(checkpoint), 'classes': summary['classes'], 'dataset': summary}
