import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from road_traffic.ir_lpr_corners import convert_ir_lpr_corners
from road_traffic.benchmark import _load_ground_truth, predict_coco_zip


class IrLprCornersConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        Image.new('RGB', (200, 100), 'white').save(self.images / 'day_001.jpg')
        Image.new('RGB', (200, 100), 'black').save(self.images / 'night_002.jpg')
        self.csv_path = self.root / 'annotations.csv'
        self.rows = [
            ['day_001.jpg', '.1', '.2', '.5', '.2', '.5', '.4', '.1', '.4'],
            ['night_002.jpg', '.3', '.4', '.7', '.4', '.7', '.6', '.3', '.6'],
        ]
        self._write_csv()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_csv(self):
        with self.csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
            writer.writerows(self.rows)

    def test_builds_single_diagnostic_split_from_corner_envelopes(self):
        output = self.root / 'plate-diagnostic.zip'
        result = convert_ir_lpr_corners(self.root, output)
        self.assertEqual((result['split'], result['image_count']), ('diagnostic', 2))
        self.assertFalse(result['eligible_for_training'])
        self.assertFalse(result['eligible_for_acceptance'])
        with zipfile.ZipFile(output) as archive:
            truth = _load_ground_truth(output, 'diagnostic')
            manifest = json.loads(archive.read('diagnostic-manifest.json'))
            self.assertNotIn('test/annotations.json', archive.namelist())
            self.assertEqual(manifest['split_policy'].split(';')[0],
                'The source provides no train/validation/held-out partition')
        self.assertEqual(truth['annotations'][0]['bbox'], [20, 20, 80, 20])
        self.assertEqual([row.get('day_night') for row in truth['images']], ['day', 'night'])
        self.assertTrue(all(row['country'] == 'Iran' for row in truth['images']))
        second = self.root / 'second-diagnostic.zip'
        convert_ir_lpr_corners(self.root, second)
        self.assertEqual(output.read_bytes(), second.read_bytes())

    def test_plate_benchmark_runs_diagnostic_split_and_preserves_tiling_setting(self):
        output = self.root / 'plate-diagnostic.zip'
        convert_ir_lpr_corners(self.root, output)
        calls = []

        class Detector:
            sha = 'a' * 64

            def detect(self, rgb):
                return [{'box': [.1, .2, .4, .2], 'confidence': .9}]

        def factory(kind, pack, threshold, tiled):
            calls.append((kind, threshold, tiled))
            return Detector()

        predictions = predict_coco_zip(output, self.root / 'unused-model-pack',
            split='diagnostic', model_kind='plate', threshold=.18, tiled=False,
            detector_factory=factory)
        self.assertEqual(calls, [('plate', .18, False)])
        self.assertEqual(predictions['metadata']['split'], 'diagnostic')
        self.assertFalse(predictions['metadata']['tiled'])
        self.assertEqual(len(predictions['predictions']), 2)

    def test_does_not_overwrite_existing_output(self):
        output = self.root / 'existing.zip'
        output.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'never overwritten|already exists'):
            convert_ir_lpr_corners(self.root, output)
        self.assertEqual(output.read_bytes(), b'keep')

    def test_rejects_path_traversal_and_missing_images(self):
        self.rows[0][0] = '../outside.jpg'
        self._write_csv()
        with self.assertRaisesRegex(ValueError, 'Unsafe image path'):
            convert_ir_lpr_corners(self.root, self.root / 'bad.zip')

    def test_rejects_invalid_normalized_corner_values(self):
        self.rows[0][1] = '1.1'
        self._write_csv()
        with self.assertRaisesRegex(ValueError, 'normalized finite values'):
            convert_ir_lpr_corners(self.root, self.root / 'bad.zip')

    def test_rejects_duplicate_annotation_rows(self):
        self.rows.append(list(self.rows[0]))
        self._write_csv()
        with self.assertRaisesRegex(ValueError, 'Duplicate image annotation'):
            convert_ir_lpr_corners(self.root, self.root / 'bad.zip')


if __name__ == '__main__':
    unittest.main()
