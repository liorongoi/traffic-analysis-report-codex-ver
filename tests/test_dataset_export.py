import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from road_traffic.dataset_export import build
from road_traffic.store import Library


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        source = root / 'source'
        source.mkdir()
        Image.new('RGB', (24, 16), (10, 20, 30)).save(source / '17052026-1604-train.jpg')
        Image.new('RGB', (24, 16), (80, 40, 20)).save(source / '17052026-1659-validation.jpg')
        self.library = Library(root / 'workspace')
        self.mission = self.library.create('Learning export', 'Israel', 'Asia/Jerusalem')
        self.library.import_folder(self.mission, source, 'camera')
        self.images = self.library.images(self.mission)

    def tearDown(self):
        self.temp.cleanup()

    def _label(self, row, role='training', status='reviewed', annotations=None):
        self.library.review(self.mission, row['id'], 0, annotations if annotations is not None else [
            {'kind': 'vehicle', 'label': 'car', 'box': [.1, .2, .3, .4],
             'origin': 'human', 'reviewed': True},
            {'kind': 'vehicle', 'label': 'bus', 'box': [.4, .2, .2, .4],
             'origin': 'model', 'reviewed': True},
            {'kind': 'vehicle', 'label': 'bus', 'box': [.1, .1, .2, .2],
             'origin': 'legacy-model-proposal', 'reviewed': True},
        ], 'Reviewer', role=role, status=status)

    def test_coco_export_preserves_roles_and_only_exports_human_boxes(self):
        self._label(self.images[0], 'training')
        self._label(self.images[1], 'validation')
        view = self.library.create_view(self.mission, 'North entrance', 'Reviewer')
        self.library.assign_view(self.mission, self.images[0]['id'], view['id'], 'Reviewer')
        temporary, archive_path = build(self.library, self.mission)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read('dataset-manifest.json'))
                training = json.loads(archive.read('training/annotations.json'))
                validation = json.loads(archive.read('validation/annotations.json'))
                self.assertEqual(training['images'][0]['country'], 'Israel')
                self.assertEqual(training['images'][0]['country_context_basis'], 'mission_configuration')
                self.assertEqual(training['images'][0]['camera_view'], view['id'])
                self.assertIn('training/images/', '\n'.join(archive.namelist()))
                self.assertEqual(manifest['roles']['training']['images'], 1)
                self.assertEqual(manifest['roles']['training']['annotations'], 1)
                self.assertEqual(manifest['roles']['validation']['annotations'], 1)
                self.assertEqual([item['name'] for item in training['categories']], ['vehicle:car'])
                for actual, expected in zip(training['annotations'][0]['bbox'], [2.4, 3.2, 7.2, 6.4]):
                    self.assertAlmostEqual(actual, expected)
                self.assertEqual(validation['annotations'][0]['category_id'], 1)
                self.assertEqual(manifest['partial_reviews_excluded'], 0)
        finally:
            temporary.cleanup()

    def test_partial_reviews_are_excluded_instead_of_exported_as_negatives(self):
        self._label(self.images[0], 'training', status='partial')
        self._label(self.images[1], 'validation')
        temporary, archive_path = build(self.library, self.mission)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                manifest = json.loads(archive.read('dataset-manifest.json'))
                training = json.loads(archive.read('training/annotations.json'))
                self.assertEqual(training['images'], [])
                self.assertEqual(manifest['partial_reviews_excluded'], 1)
        finally:
            temporary.cleanup()

    def test_cross_split_duplicate_hashes_stop_export(self):
        duplicate = self.library.import_folder(self.mission,
            Path(self.temp.name) / 'source', 'camera')
        rows = self.library.images(self.mission)
        self.assertEqual(duplicate['images'], 2)
        by_hash = {}
        for row in rows:
            by_hash.setdefault(row['sha256'], []).append(row)
        repeated = next(items for items in by_hash.values() if len(items) == 2)
        self._label(repeated[0], 'training')
        self._label(repeated[1], 'test')
        other = next(row for row in rows if row['sha256'] != repeated[0]['sha256'])
        self._label(other, 'validation')
        with self.assertRaisesRegex(ValueError, 'identical image bytes occur across review splits'):
            build(self.library, self.mission)

    def test_known_camera_hour_overlap_stops_export(self):
        view = self.library.create_view(self.mission, 'Same camera', 'Reviewer')
        rows = self.library.images(self.mission)
        self.library.assign_view(self.mission, rows[0]['id'], view['id'], 'Reviewer')
        self.library.assign_view(self.mission, rows[1]['id'], view['id'], 'Reviewer')
        self._label(next(row for row in rows if row['id'] == rows[0]['id']), 'training')
        self._label(next(row for row in rows if row['id'] == rows[1]['id']), 'test')
        with self.assertRaisesRegex(ValueError, 'known camera/hour group occurs across review splits'):
            build(self.library, self.mission)


if __name__ == '__main__':
    unittest.main()
