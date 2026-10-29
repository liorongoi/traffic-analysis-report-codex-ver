import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

from road_traffic.yolox_training import finetune_yolox, prepare_yolox_coco


def _archive(path, *, validation_boxes=True, unsafe=False):
    categories = [
        {'id': 1, 'name': 'vehicle:car', 'kind': 'vehicle', 'label': 'car'},
        {'id': 2, 'name': 'light:headlight', 'kind': 'light', 'label': 'headlight'},
    ]
    files = {'dataset-manifest.json': json.dumps({'format': 'RoadLoom COCO dataset v1'}),
        'training/annotations.json': json.dumps({'images': [
            {'id': 1, 'file_name': '../escape.jpg', 'width': 10, 'height': 10},
            {'id': 2, 'file_name': 'images/negative.jpg', 'width': 10, 'height': 10}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                'bbox': [1, 1, 3, 3]}, {'id': 2, 'image_id': 1, 'category_id': 2,
                'bbox': [1, 1, 2, 2]}], 'categories': categories}),
        'validation/annotations.json': json.dumps({'images': [
            {'id': 3, 'file_name': 'images/val.jpg', 'width': 10, 'height': 10}],
            'annotations': ([{'id': 3, 'image_id': 3, 'category_id': 1,
                'bbox': [2, 2, 3, 3]}] if validation_boxes else []), 'categories': categories}),
        'test/annotations.json': json.dumps({'images': [
            {'id': 4, 'file_name': 'images/test.jpg', 'width': 10, 'height': 10}],
            'annotations': [{'id': 4, 'image_id': 4, 'category_id': 1,
                'bbox': [1, 1, 3, 3]}], 'categories': categories})}
    files.update({'training/images/negative.jpg': b'neg', 'validation/images/val.jpg': b'val',
        'test/images/test.jpg': b'test'})
    if not unsafe:
        files['training/images/escape.jpg'] = b'pos'
        files['training/annotations.json'] = files['training/annotations.json'].replace('../escape.jpg', 'images/positive.jpg')
        files['training/images/positive.jpg'] = files.pop('training/images/escape.jpg')
    with zipfile.ZipFile(path, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, value)


class YoloxTrainingDatasetTests(unittest.TestCase):
    def test_stages_vehicle_only_and_keeps_reviewed_negative_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'dataset.zip'
            _archive(archive)
            output = root / 'stage'
            summary = prepare_yolox_coco(archive, output)
            self.assertEqual(summary['classes'], ['car'])
            self.assertEqual(summary['splits']['training']['images'], 2)
            self.assertEqual(summary['splits']['training']['positive_images'], 1)
            self.assertEqual(summary['splits']['training']['vehicle_annotations'], 1)
            self.assertFalse((output / 'test2017').exists())
            train = json.loads((output / 'annotations/instances_train2017.json').read_text())
            self.assertEqual(train['categories'], [{'id': 1, 'name': 'car'}])
            self.assertEqual([item['category_id'] for item in train['annotations']], [1])
            self.assertEqual({item['file_name'] for item in train['images']},
                {'images/positive.jpg', 'images/negative.jpg'})

    def test_rejects_unsafe_archive_image_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'dataset.zip'
            _archive(archive, unsafe=True)
            with self.assertRaisesRegex(ValueError, 'Unsafe dataset path'):
                prepare_yolox_coco(archive, root / 'stage')

    def test_requires_reviewed_vehicle_labels_in_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'dataset.zip'
            _archive(archive, validation_boxes=False)
            with self.assertRaisesRegex(ValueError, 'Validation split has no reviewed vehicle boxes'):
                prepare_yolox_coco(archive, root / 'stage')

    def test_training_orchestration_exports_importable_pack_without_using_test_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'dataset.zip'
            _archive(archive)
            output = root / 'training-output'
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                if '-m' in command and command[command.index('-m') + 1] == 'yolox.tools.train':
                    checkpoint = output / 'YOLOX_outputs' / 'roadloom_yolox_finetune' / 'best_ckpt.pth'
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b'best checkpoint')
                else:
                    model = Path(command[command.index('--output-name') + 1])
                    model.write_bytes(b'exported onnx')

            with (patch.dict(sys.modules, {'torch': torch, 'yolox': SimpleNamespace()}),
                    patch('road_traffic.yolox_training.subprocess.run', side_effect=run)):
                result = finetune_yolox(archive, output, epochs=2, batch_size=1)

            pack = Path(result['model_pack'])
            manifest = json.loads((pack / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(result['classes'], ['car'])
            self.assertEqual(manifest['classes'], ['car'])
            self.assertEqual(len(calls), 2)
            self.assertIn('--ckpt', calls[0])
            self.assertIn('--no-onnxsim', calls[1])
            self.assertFalse((output / 'test2017').exists())
            self.assertFalse(result['dataset']['splits']['test']['used'])


if __name__ == '__main__':
    unittest.main()
