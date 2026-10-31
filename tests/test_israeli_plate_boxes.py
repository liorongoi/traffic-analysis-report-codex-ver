import json
import csv
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from road_traffic.benchmark import _load_ground_truth, predict_coco_zip
from road_traffic.israeli_plate_boxes import convert_israeli_plate_boxes


class IsraeliPlateBoxesConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root / 'img'
        self.images.mkdir()
        Image.new('RGB', (200, 100), 'white').save(self.images / 'sample.jpg')
        self.rows = [
            ['img/sample.jpg', 'sample.jpg', 'plate', 20, 100, 30, 50],
            ['img/sample.jpg', 'sample.jpg', 'plate', 110, 160, 40, 55],
        ]
        self._write_csv()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_csv(self):
        with (self.root / 'annotations.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image', 'id', 'label', 'xMin', 'xMax', 'yMin', 'yMax'])
            writer.writerows(self.rows)

    def test_builds_diagnostic_boxes_with_country_and_no_text(self):
        output = self.root / 'israel.zip'
        result = convert_israeli_plate_boxes(self.root, output)
        self.assertEqual((result['country'], result['image_count'], result['box_count']),
            ('Israel', 1, 2))
        self.assertEqual(result['split'], 'diagnostic')
        self.assertFalse(result['eligible_for_training'])
        self.assertFalse(result['eligible_for_acceptance'])
        with zipfile.ZipFile(output) as archive:
            truth = _load_ground_truth(output, 'diagnostic')
            manifest = json.loads(archive.read('diagnostic-manifest.json'))
            self.assertEqual([item['bbox'] for item in truth['annotations']],
                [[20, 30, 80, 20], [110, 40, 50, 15]])
            self.assertEqual(truth['images'][0]['country'], 'Israel')
            self.assertNotIn('plate_text', str(manifest).lower())
        second = self.root / 'second.zip'
        convert_israeli_plate_boxes(self.root, second)
        self.assertEqual(output.read_bytes(), second.read_bytes())

    def test_plate_benchmark_accepts_diagnostic_split(self):
        output = self.root / 'israel.zip'
        convert_israeli_plate_boxes(self.root, output)

        class Detector:
            sha = 'b' * 64

            def detect(self, rgb):
                return [{'box': [0.1, 0.3, 0.4, 0.2], 'confidence': .9}]

        def factory(kind, pack, threshold, tiled):
            self.assertEqual(kind, 'plate')
            return Detector()

        report = predict_coco_zip(output, self.root / 'unused-model-pack',
            split='diagnostic', model_kind='plate', threshold=.18, tiled=False,
            detector_factory=factory)
        self.assertEqual(report['metadata']['split'], 'diagnostic')
        self.assertEqual(len(report['predictions']), 1)

    def test_rejects_unsafe_paths_and_duplicate_exclusions(self):
        self.rows[0][0] = '../outside.jpg'
        self._write_csv()
        with self.assertRaisesRegex(ValueError, 'Unsafe image path'):
            convert_israeli_plate_boxes(self.root, self.root / 'bad.zip')
        exclusion = self.root / 'duplicate-exclude.json'
        entry = {'imagefilename': 'img/sample.jpg', 'reason': 'quality'}
        exclusion.write_text(json.dumps([entry, entry]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Invalid or duplicate excluded image'):
            convert_israeli_plate_boxes(self.root, self.root / 'bad2.zip',
                exclusions_file=exclusion)

    def test_rejects_out_of_bounds_box_and_existing_output(self):
        self.rows[0][3] = 150
        self._write_csv()
        with self.assertRaisesRegex(ValueError, 'falls outside'):
            convert_israeli_plate_boxes(self.root, self.root / 'bad.zip')
        output = self.root / 'existing.zip'
        output.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            convert_israeli_plate_boxes(self.root, output)
        self.assertEqual(output.read_bytes(), b'keep')

    def test_explicit_exclusions_are_recorded_and_unknown_names_fail(self):
        exclusion = self.root / 'exclude.json'
        exclusion.write_text(json.dumps([{'imagefilename': 'img/sample.jpg',
            'reason': 'source annotation quality review'}]), encoding='utf-8')
        output = self.root / 'excluded.zip'
        with self.assertRaisesRegex(ValueError, 'No images remain'):
            convert_israeli_plate_boxes(self.root, output, exclusions_file=exclusion)
        self.assertFalse(output.exists())
        self.rows.append(['img/second.jpg', 'second.jpg', 'plate', 10, 60, 10, 30])
        Image.new('RGB', (200, 100), 'white').save(self.images / 'second.jpg')
        self._write_csv()
        result = convert_israeli_plate_boxes(self.root, output, exclusions_file=exclusion)
        with zipfile.ZipFile(output) as archive:
            manifest = json.loads(archive.read('diagnostic-manifest.json'))
        self.assertEqual((result['image_count'], manifest['excluded_image_count']), (1, 1))
        self.assertEqual(manifest['excluded_images'][0]['reason'], 'source annotation quality review')
        exclusion.write_text(json.dumps([{'imagefilename': 'img/missing.jpg',
            'reason': 'not in source'}]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'unknown images'):
            convert_israeli_plate_boxes(self.root, self.root / 'unknown.zip',
                exclusions_file=exclusion)


if __name__ == '__main__':
    unittest.main()
