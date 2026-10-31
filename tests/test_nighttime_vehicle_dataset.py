import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from road_traffic.nighttime_vehicle_dataset import build_dataset, parse_split_range

ROOT = Path(__file__).resolve().parents[1]


class NighttimeVehicleDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PIL import Image
        except ImportError:
            raise unittest.SkipTest('nighttime dataset packaging tests require Pillow')
        cls.Image = Image

    def _source(self, directory, labels='2007 1 1 2 8 4\n2008 0\n2010 1 10 3 5 4\n'):
        root = Path(directory)
        images = root / 'images'
        images.mkdir()
        for image_id in (2007, 2008, 2010):
            image = images / f'img_{image_id:05d}.jpg'
            self.Image.new('L', (20, 10), image_id % 255).save(image)
        label_path = root / 'gt10913.txt'
        label_path.write_text(labels, encoding='utf-8')
        return root, images, label_path

    def test_packages_explicit_disjoint_roles_and_complete_zero_box_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, images, labels = self._source(temporary)
            destination = root / 'nvd.zip'
            result = build_dataset(labels, images,
                ['training=2007-2008', 'test=2010-2010'], destination)
            self.assertEqual(result['images'], 3)
            self.assertEqual(result['annotations'], 2)
            with zipfile.ZipFile(destination) as archive:
                training = json.loads(archive.read('training/annotations.json'))
                test = json.loads(archive.read('test/annotations.json'))
                self.assertEqual([row['id'] for row in training['images']], [2007, 2008])
                self.assertEqual(len(training['annotations']), 1)
                self.assertEqual(len(test['annotations']), 1)
                self.assertEqual(training['images'][0]['condition'], 'night')
                self.assertEqual(training['images'][0]['source_image_id'], 2007)
                self.assertEqual(training['categories'][0]['name'], 'vehicle')
                self.assertEqual(training['info']['source_annotation_sha256'], result['annotation_sha256'])
                self.assertIn('training/images/img_02007.jpg', archive.namelist())
                self.assertIn('test/images/img_02010.jpg', archive.namelist())

    def test_roles_may_not_overlap_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, images, labels = self._source(temporary)
            with self.assertRaisesRegex(ValueError, 'assigned to multiple evidence roles'):
                build_dataset(labels, images, ['training=2007-2008', 'test=2008-2010'], root / 'bad.zip')
            output = root / 'existing.zip'
            output.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                build_dataset(labels, images, ['test=2007-2007'], output)
            self.assertEqual(output.read_bytes(), b'keep')

    def test_out_of_bounds_boxes_reject_the_image_until_explicitly_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, images, labels = self._source(temporary,
                '2007 1 19 2 5 4\n2008 0\n2010 0\n')
            with self.assertRaisesRegex(ValueError, 'out-of-bounds box'):
                build_dataset(labels, images, ['test=2007-2008'], root / 'invalid.zip')
            output = root / 'excluded.zip'
            result = build_dataset(labels, images, ['test=2007-2008'], output,
                exclude_image_ids=[2007])
            self.assertEqual(result['images'], 1)
            with zipfile.ZipFile(output) as archive:
                ground_truth = json.loads(archive.read('test/annotations.json'))
            self.assertEqual([row['id'] for row in ground_truth['images']], [2008])
            self.assertEqual(ground_truth['info']['excluded_image_ids'], [2007])

    def test_rejects_cross_role_range_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, images, labels = self._source(temporary)
            with self.assertRaisesRegex(ValueError, 'assigned to multiple evidence roles'):
                build_dataset(labels, images,
                    ['training=2007-2008', 'validation=2008-2010'], root / 'overlap.zip')

    def test_split_range_parser_requires_explicit_supported_role(self):
        self.assertEqual(parse_split_range('validation=8248-8277'), ('validation', 8248, 8277))
        for invalid in ('2007-2036', 'test=2036-2007', 'holdout=1-2', 'test=a-b'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_split_range(invalid)

    def test_cli_builds_dataset_zip_from_selected_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, images, labels = self._source(temporary)
            destination = root / 'from-cli.zip'
            result = subprocess.run([sys.executable,
                str(ROOT / 'scripts' / 'convert-nighttime-vehicle-dataset.py'),
                '--labels', str(labels), '--images-dir', str(images),
                '--split-range', 'test=2007-2008', '--output', str(destination)],
                check=True, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertEqual(report['images'], 2)
            self.assertEqual(report['annotations'], 1)
            with zipfile.ZipFile(destination) as archive:
                self.assertIn('test/annotations.json', archive.namelist())


if __name__ == '__main__':
    unittest.main()
