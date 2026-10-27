import tempfile
import unittest
from pathlib import Path

from PIL import Image

from road_traffic.learning import reviewed_evidence_inventory
from road_traffic.store import Library


class ReviewedEvidenceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / 'source'
        self.source.mkdir()
        Image.new('RGB', (16, 16), (20, 40, 60)).save(self.source / '17052026-1604-a.jpg')
        (self.source / '17052026-1604-b.jpg').write_bytes(
            (self.source / '17052026-1604-a.jpg').read_bytes())
        self.library = Library(root / 'workspace')
        self.mission = self.library.create('Evidence inventory', 'Israel', 'Asia/Jerusalem')
        self.library.import_folder(self.mission, self.source, 'camera')
        self.images = self.library.images(self.mission)
        self.view = self.library.create_view(self.mission, 'Known camera', 'Reviewer')
        for image in self.images:
            self.library.assign_view(self.mission, image['id'], self.view['id'], 'Reviewer')

    def tearDown(self):
        self.temp.cleanup()

    def test_counts_only_explicit_human_regions_and_reports_split_overlap(self):
        self.library.review(self.mission, self.images[0]['id'], 0, [
            {'kind': 'vehicle', 'label': 'car', 'box': [.1, .1, .2, .2],
                'origin': 'human', 'reviewed': True},
            {'kind': 'vehicle', 'label': 'car', 'box': [.4, .4, .2, .2],
                'origin': 'model', 'reviewed': True, 'rejected': True},
            {'kind': 'vehicle', 'label': 'truck', 'box': [.6, .6, .2, .2],
                'origin': 'model'},
        ], 'Reviewer', role='training', status='partial')
        self.library.review(self.mission, self.images[1]['id'], 0, [
            {'kind': 'vehicle', 'label': 'bus', 'box': [.2, .2, .2, .2],
                'origin': 'model'},
        ], 'Reviewer', role='test', status='reviewed')

        result = reviewed_evidence_inventory(self.library, self.mission)

        self.assertEqual(result['source'], 'current mission only')
        self.assertFalse(result['model_predictions_used_as_labels'])
        self.assertFalse(result['bulk_approvals_used_as_labels'])
        self.assertFalse(result['partial_unmarked_regions_used_as_negatives'])
        self.assertEqual(result['cross_role_duplicate_hashes'], 1)
        self.assertEqual(result['cross_role_camera_hour_groups'], 1)
        self.assertEqual(result['roles']['training']['images'], 1)
        self.assertEqual(result['roles']['training']['partially_reviewed_images'], 1)
        self.assertEqual(result['roles']['training']['explicit_human_regions'], 1)
        self.assertEqual(result['roles']['training']['regions_by_label'], {'car': 1})
        self.assertEqual(result['roles']['training']['explicitly_rejected_model_regions'], 1)
        self.assertEqual(result['roles']['test']['fully_reviewed_images'], 1)
        self.assertEqual(result['roles']['test']['explicit_human_regions'], 0)
        self.assertEqual(result['roles']['test']['regions_by_kind'], {})


if __name__ == '__main__':
    unittest.main()
