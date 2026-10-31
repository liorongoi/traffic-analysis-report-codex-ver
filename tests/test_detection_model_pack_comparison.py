import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from road_traffic.benchmark import evaluate_coco


SCRIPT_PATH=Path(__file__).resolve().parents[1]/'scripts'/'compare-detection-model-packs.py'
SPEC=importlib.util.spec_from_file_location('compare_detection_model_packs',SCRIPT_PATH)
comparison_cli=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison_cli)


def _bundle(predictions,model_hash,threshold):
    return {'predictions':predictions,'metadata':{
        'model_kind':'vehicle','model_sha256':model_hash,'model_pack':model_hash[:8],
        'threshold':threshold,'tiled':True,'inference_provider':'CPUExecutionProvider',
        'pipeline_scope':'YOLOX vehicle detector only','split':'test',
        'categories':[],'image_metadata':{},'strata':{},
        'storage_bytes':{'source_dataset':100,'model_pack':200}}}


class DetectionModelPackComparisonTests(unittest.TestCase):
    def test_runs_both_packs_on_the_same_test_zip_and_writes_all_reports(self):
        truth={'images':[{'id':1,'width':100,'height':100,'country':'Lebanon','day_night':'night'}],
            'categories':[{'id':1,'name':'vehicle:car','kind':'vehicle','label':'car'}],
            'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[10,10,20,20]}]}
        perfect=[{'image_id':1,'category_id':1,'bbox':[10,10,20,20],'score':.9}]
        empty=[]
        bundles=[_bundle(perfect,'a'*64,.18),_bundle(empty,'b'*64,.22)]

        def write_evaluation(dataset, predictions_path, report_path, *, split, operating_iou):
            bundle=json.loads(Path(predictions_path).read_text(encoding='utf-8'))
            report=evaluate_coco(truth,bundle['predictions'],operating_iou=operating_iou,
                metadata={'prediction':bundle['metadata'],'ground_truth_sha256':'c'*64,'split':split})
            Path(report_path).write_text(json.dumps(report),encoding='utf-8')
            return report

        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'comparison'
            with patch.object(comparison_cli,'predict_coco_zip',side_effect=bundles) as predict, \
                    patch.object(comparison_cli,'evaluate_files',side_effect=write_evaluation) as evaluate:
                result=comparison_cli.compare_model_packs(root/'dataset.zip',root/'baseline',
                    root/'candidate',output,baseline_threshold=.18,candidate_threshold=.22,
                    hardware_profile='16GB')
            self.assertEqual(predict.call_count,2)
            for call in predict.call_args_list:
                self.assertEqual(call.args[0],root/'dataset.zip')
                self.assertEqual(call.kwargs['split'],'test')
                self.assertEqual(call.kwargs['model_kind'],'vehicle')
            self.assertEqual([call.kwargs['threshold'] for call in predict.call_args_list],[.18,.22])
            self.assertEqual(evaluate.call_count,2)
            self.assertTrue((output/'baseline-predictions.json').is_file())
            self.assertTrue((output/'baseline-report.json').is_file())
            self.assertTrue((output/'candidate-predictions.json').is_file())
            self.assertTrue((output/'candidate-report.json').is_file())
            saved=json.loads((output/'comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(saved,result)
            self.assertEqual(result['decision'],'review_required')
            self.assertTrue(result['regressions'])

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'reports';output.mkdir()
            (output/'prior.json').write_text('{}',encoding='utf-8')
            with patch.object(comparison_cli,'predict_coco_zip') as predict:
                with self.assertRaisesRegex(FileExistsError,'must be empty'):
                    comparison_cli.compare_model_packs(Path('data.zip'),Path('base'),
                        Path('candidate'),output)
            predict.assert_not_called()


if __name__=='__main__':
    unittest.main()
