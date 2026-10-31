import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from road_traffic.inference_store import create_run
from road_traffic.store import Library
from road_traffic.vehicle_metrics import evaluate


class VehicleMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.library = Library(root / 'workspace')
        self.mission = self.library.create('Metrics', 'Israel', 'Asia/Jerusalem')
        self.source = root / 'source'
        self.source.mkdir()
        for index in range(3):
            Image.new('RGB', (16, 16), (index * 40, 80, 100)).save(
                self.source / f'17052026-160{index}-image{index}.jpg', 'JPEG')
        self.library.import_folder(self.mission, self.source, 'camera')
        self.images = self.library.images(self.mission)
        self.run = create_run(self.library, self.mission, {
            'adapter': 'test', 'vehicle_model_id': 'base', 'model_sha256': 'a' * 64,
            'threshold': .18})
        predictions = [
            [
                {'kind': 'candidate', 'label': 'car', 'box': [.10, .10, .20, .20],
                    'score': .90, 'origin': 'model', 'evidence': 'yolox-s'},
                {'kind': 'candidate', 'label': 'truck', 'box': [.12, .10, .20, .20],
                    'score': .80, 'origin': 'model', 'evidence': 'yolox-s'},
            ],
            [{'kind': 'candidate', 'label': 'car', 'box': [.50, .50, .10, .10],
                'score': .70, 'origin': 'model', 'evidence': 'yolox-s'}],
            [{'kind': 'candidate', 'label': 'car', 'box': [.20, .20, .10, .10],
                'score': .60, 'origin': 'model', 'evidence': 'yolox-s'}],
        ]
        with self.library.connect(self.mission) as conn:
            for row, annotations in zip(self.images, predictions):
                conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',
                    (self.run, row['id'], json.dumps(annotations), '{}', .01, None))
            conn.execute("UPDATE inference_runs SET status='completed' WHERE id=?", (self.run,))
        self.library.review(self.mission, self.images[0]['id'], 0, [
            {'kind': 'vehicle', 'label': 'car', 'box': [.10, .10, .20, .20],
                'origin': 'model', 'reviewed': True},
        ], 'Reviewer', role='training', status='reviewed', prediction_run=self.run)
        self.library.review(self.mission, self.images[1]['id'], 0, [
            {'kind': 'vehicle', 'label': 'truck', 'box': [.20, .20, .10, .10],
                'origin': 'human', 'reviewed': True},
        ], 'Reviewer', role='test', status='reviewed', prediction_run=self.run)

    def tearDown(self):
        self.temp.cleanup()

    def test_precision_recall_use_only_full_run_bound_reviews_and_one_to_one_boxes(self):
        report = evaluate(self.library, self.mission, self.run)
        self.assertEqual(report['input_images'], 3)
        self.assertEqual(report['fully_reviewed_images'], 2)
        self.assertEqual(report['unreviewed_images'], 1)
        self.assertEqual(report['overall']['true_positives'], 1)
        self.assertEqual(report['overall']['false_positives'], 2)
        self.assertEqual(report['overall']['false_negatives'], 1)
        self.assertAlmostEqual(report['overall']['precision'], 1 / 3)
        self.assertAlmostEqual(report['overall']['recall'], .5)
        self.assertEqual(report['by_evidence_role']['training']['true_positives'], 1)
        self.assertEqual(report['validation_test_holdout']['false_positives'], 1)
        self.assertEqual(report['validation_test_holdout']['false_negatives'], 1)
        self.assertEqual(len(report['reviewed_image_revisions']), 2)
        self.assertEqual(report['iou_threshold'], .5)

    def test_partial_review_is_not_scored_and_other_run_review_is_not_reused(self):
        self.library.review(self.mission, self.images[2]['id'], 0, [
            {'kind': 'vehicle', 'label': 'car', 'box': [.20, .20, .10, .10]},
        ], 'Reviewer', role='none', status='partial', prediction_run=self.run)
        partial = evaluate(self.library, self.mission, self.run)
        self.assertEqual(partial['fully_reviewed_images'], 2)
        self.assertEqual(partial['partial_review_images'], 1)
        other = create_run(self.library, self.mission, {'adapter': 'second'})
        with self.library.connect(self.mission) as conn:
            for row in self.images:
                conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',
                    (other, row['id'], '[]', '{}', .01, None))
            conn.execute("UPDATE inference_runs SET status='completed' WHERE id=?", (other,))
        other_report = evaluate(self.library, self.mission, other)
        self.assertEqual(other_report['fully_reviewed_images'], 0)
        self.assertEqual(other_report['reviews_bound_to_other_runs'], 3)
        self.assertIsNone(other_report['overall']['precision'])
        self.assertIsNone(other_report['overall']['recall'])

    def test_rejects_unknown_run_and_invalid_iou(self):
        with self.assertRaisesRegex(ValueError, 'not found'):
            evaluate(self.library, self.mission, 'unknown')
        with self.assertRaisesRegex(ValueError, 'IoU'):
            evaluate(self.library, self.mission, self.run, 0)
