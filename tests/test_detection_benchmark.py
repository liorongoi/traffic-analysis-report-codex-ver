import json
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

from road_traffic.benchmark import (compare_detection_reports, evaluate_coco,
    evaluate_files, predict_coco_zip)


def sample_truth():
    return {'images': [
        {'id': 1, 'width': 100, 'height': 100, 'country': 'Lebanon', 'day_night': 'night',
         'view_id': 'junction-a', 'hardware_profile': '16GB'},
        {'id': 2, 'width': 100, 'height': 100, 'country': 'Lebanon', 'day_night': 'day',
         'view_id': 'junction-a', 'hardware_profile': '16GB'},
    ], 'categories': [{'id': 1, 'name': 'vehicle:car'}], 'annotations': [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20]},
        {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [50, 50, 20, 20]},
    ]}


class DetectionBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _comparison_report(predictions, model_hash, *, dataset_hash='a'*64, split='test'):
        prediction_metadata={'model_kind':'vehicle','model_sha256':model_hash,
            'model_pack':model_hash[:8],'threshold':.18,'tiled':True,
            'inference_provider':'CPUExecutionProvider'}
        report=evaluate_coco(sample_truth(),predictions,metadata={'prediction':prediction_metadata})
        report['metadata']={'prediction':prediction_metadata,
            'ground_truth_sha256':dataset_hash,'split':split}
        return report

    def test_candidate_comparison_reports_overall_class_and_condition_regressions(self):
        perfect=[
            {'image_id':1,'category_id':1,'bbox':[10,10,20,20],'score':.95},
            {'image_id':2,'category_id':1,'bbox':[50,50,20,20],'score':.9}]
        degraded=[
            {'image_id':1,'category_id':1,'bbox':[10,10,20,20],'score':.95},
            {'image_id':1,'category_id':1,'bbox':[10,10,20,20],'score':.8}]
        baseline=self._comparison_report(perfect,'a'*64)
        candidate=self._comparison_report(degraded,'b'*64)
        result=compare_detection_reports(baseline,candidate)
        self.assertEqual(result['decision'],'review_required')
        self.assertEqual(result['split'],'test')
        self.assertEqual(result['test_dataset_sha256'],'a'*64)
        self.assertEqual(result['changes']['overall']['micro']['recall']['status'],'regressed')
        self.assertEqual(result['changes']['overall']['micro']['false_alarms_per_image']['status'],'regressed')
        self.assertEqual(result['changes']['overall']['count_error']['mae_objects_per_image']['status'],'regressed')
        self.assertEqual(result['changes']['overall']['categories']['vehicle:car']['f1']['status'],'regressed')
        self.assertEqual(result['changes']['groups']['condition']['day']['micro']['recall']['status'],'regressed')
        self.assertTrue(any(item['status']=='regressed' for item in result['regressions']))
        self.assertNotIn('accepted',result)

    def test_candidate_comparison_rejects_noncomparable_reports(self):
        empty=[]
        baseline=self._comparison_report(empty,'a'*64)
        candidate=self._comparison_report(empty,'b'*64)
        with self.assertRaisesRegex(ValueError,'same frozen test dataset'):
            compare_detection_reports(baseline,self._comparison_report(empty,'b'*64,dataset_hash='c'*64))
        with self.assertRaisesRegex(ValueError,'held-out test split'):
            compare_detection_reports(baseline,self._comparison_report(empty,'b'*64,split='validation'))
        with self.assertRaisesRegex(ValueError,'different detector model hash'):
            compare_detection_reports(baseline,self._comparison_report(empty,'a'*64))

    def test_one_to_one_matching_counts_duplicates_and_misses(self):
        predictions = [
            {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'score': .95},
            {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'score': .8},
        ]
        report = evaluate_coco(sample_truth(), predictions)
        micro = report['overall']['micro']
        self.assertEqual((micro['tp'], micro['fp'], micro['fn']), (1, 1, 1))
        self.assertEqual((micro['precision'], micro['recall'], micro['f1']), (.5, .5, .5))
        self.assertEqual(micro['false_alarms_per_image'], .5)
        self.assertEqual(report['overall']['count_error']['mae_objects_per_image'], 1.)
        self.assertAlmostEqual(report['overall']['categories']['vehicle:car']['ap50'], 51 / 101)

    def test_ap_and_count_metrics_for_perfect_predictions(self):
        predictions = [
            {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'score': .95},
            {'image_id': 2, 'category_id': 1, 'bbox': [50, 50, 20, 20], 'score': .7},
        ]
        report = evaluate_coco(sample_truth(), predictions)
        self.assertEqual(report['overall']['micro']['map50_95_macro'], 1.)
        self.assertEqual(report['overall']['count_error']['mae_objects_per_image'], 0.)
        self.assertEqual(report['groups']['condition']['day']['images'], 1)
        self.assertEqual(report['groups']['country']['Lebanon']['micro']['recall'], 1.)

    def test_empty_prediction_set_reports_zero_recall_not_fake_accuracy(self):
        report = evaluate_coco(sample_truth(), [])
        metrics = report['overall']['micro']
        self.assertEqual(metrics['recall'], 0.)
        self.assertEqual(metrics['precision'], None)
        self.assertEqual(metrics['f1'], None)
        self.assertEqual(metrics['map50_95_macro'], 0.)

    def test_per_image_and_run_strata_are_included_in_metric_breakdowns(self):
        report = evaluate_coco(sample_truth(), [], metadata={'prediction': {
            'image_metadata': {'1': {'condition': 'night'}, '2': {'condition': 'day'}},
            'strata': {'hardware_profile': '48GB'}}})
        self.assertEqual(set(report['groups']['condition']), {'day', 'night'})
        self.assertEqual(report['groups']['hardware_profile']['16GB']['images'], 2)

    def test_invalid_boxes_and_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'extends outside'):
            evaluate_coco(sample_truth(), [{'image_id': 1, 'category_id': 1,
                'bbox': [90, 90, 20, 20], 'score': .5}])
        with self.assertRaisesRegex(ValueError, 'scores must be between 0 and 1'):
            evaluate_coco(sample_truth(), [{'image_id': 1, 'category_id': 1,
                'bbox': [10, 10, 20, 20], 'score': 1.2}])

    def test_zip_dataset_input_and_provenance_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'reviewed.zip'
            with zipfile.ZipFile(dataset, 'w') as archive:
                archive.writestr('test/annotations.json', json.dumps(sample_truth()))
            predictions = root / 'predictions.json'
            predictions.write_text(json.dumps({'metadata': {'model_sha256': 'a' * 64,
                'threshold': .2}, 'predictions': [
                    {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'score': .9},
                    {'image_id': 2, 'category_id': 1, 'bbox': [50, 50, 20, 20], 'score': .9},
                ]}), encoding='utf-8')
            output = root / 'report.json'
            report = evaluate_files(dataset, predictions, output)
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(saved, report)
            self.assertEqual(report['metadata']['prediction']['model_sha256'], 'a' * 64)
            self.assertEqual(len(report['metadata']['ground_truth_sha256']), 64)
            self.assertEqual(report['metadata']['storage_bytes']['predictions'], predictions.stat().st_size)

    @unittest.skipUnless(importlib.util.find_spec('PIL') and importlib.util.find_spec('numpy'),
                         'prediction runner needs the installed image runtime')
    def test_prediction_runner_streams_split_and_adds_model_only_categories(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'reviewed.zip'
            image_file = root / 'test.png'
            Image.new('RGB', (20, 10), (30, 60, 90)).save(image_file)
            with zipfile.ZipFile(dataset, 'w') as archive:
                archive.write(image_file, 'test/images/1.png')
                truth = {'images': [{'id': 1, 'file_name': 'images/1.png',
                    'width': 20, 'height': 10, 'country': 'Lebanon'}],
                    'categories': [{'id': 1, 'name': 'vehicle:car', 'kind': 'vehicle', 'label': 'car'}],
                    'annotations': []}
                archive.writestr('test/annotations.json', json.dumps(truth))
            class Detector:
                sha = 'b' * 64
                class_labels = {0: 'car', 1: 'truck'}
                def infer(self, rgb):
                    self.shape = rgb.shape
                    return [{'box': [.1, .2, .3, .4], 'label': 'car', 'score': .9},
                            {'box': [.5, .2, .2, .3], 'label': 'truck', 'score': .4}]
            fake = Detector()
            result = predict_coco_zip(dataset, root, detector_factory=lambda *args: fake)
            self.assertEqual(fake.shape, (10, 20, 3))
            self.assertEqual(len(result['predictions']), 2)
            self.assertEqual(result['predictions'][0]['bbox'], [2., 2., 6., 4.])
            self.assertEqual(result['metadata']['categories'][0]['name'], 'vehicle:truck')
            performance = result['metadata']['performance']
            self.assertEqual(performance['images_processed'], 1)
            self.assertGreater(performance['processing_wall_seconds'], 0)
            self.assertGreater(performance['images_per_second'], 0)
            self.assertGreater(performance['per_image_latency']['p95_seconds'], 0)
            self.assertIn('peak_process_rss_bytes', performance)
            self.assertIsNone(performance['peak_vram_bytes'])
            self.assertEqual(result['metadata']['storage_bytes']['source_dataset'], dataset.stat().st_size)
            report = evaluate_coco(truth, result['predictions'],
                prediction_categories=result['metadata']['categories'])
            self.assertEqual(report['overall']['micro']['tp'], 0)
            self.assertEqual(report['overall']['micro']['fp'], 2)


if __name__ == '__main__':
    unittest.main()
