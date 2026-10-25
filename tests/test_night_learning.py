import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from road_traffic.night import propose,patch_features
from road_traffic.learning import (inventory,train,load_head,candidates,evaluate_test,
    evaluation_status,record_model_decision,governance_status,_test_fingerprint,_legacy_test_fingerprint)
from road_traffic.store import Library


class ContextLightTests(unittest.TestCase):
    def frame(self):return np.full((100,200,3),25,np.uint8)
    def context(self,frequency=0,novelty=True):return {'frequency':np.full((100,200),frequency,np.float32),'novelty':np.full((100,200),novelty,bool)}
    def test_persistent_lamps_are_kept_but_not_paired_as_headlights(self):
        rgb=self.frame();rgb[40:43,70:73]=255;rgb[40:43,80:83]=255
        proposals,details=propose(rgb,self.context(1,False),[])
        self.assertEqual(len(proposals),2)
        self.assertTrue(all(a['light_subtype']=='streetlight_suspect' and a['pairing']=='single' and a['kind']=='light' for a in proposals))
    def test_new_pair_is_one_hypothesis_not_two_vehicles(self):
        rgb=self.frame();rgb[40:43,70:73]=255;rgb[40:43,80:83]=255
        proposals,_=propose(rgb,self.context(),[])
        self.assertEqual(len(proposals),1);self.assertEqual(proposals[0]['pairing'],'hypothesis')
        self.assertEqual(proposals[0]['light_subtype'],'headlight_candidate');self.assertEqual(proposals[0]['kind'],'light')
    def test_single_novel_light_is_eligible(self):
        rgb=self.frame();rgb[40:43,70:73]=255
        proposals,_=propose(rgb,self.context(),[])
        self.assertEqual(len(proposals),1);self.assertEqual(proposals[0]['light_subtype'],'headlight_candidate')
    def test_elongated_reflection_is_not_headlight_pair(self):
        rgb=self.frame();rgb[35:55,70:73]=255
        proposals,_=propose(rgb,self.context(),[])
        self.assertEqual(proposals[0]['light_subtype'],'reflection_suspect')
    def test_missing_background_stays_uncertain(self):
        rgb=self.frame();rgb[40:43,70:73]=255
        proposals,_=propose(rgb,None,[])
        self.assertEqual(proposals[0]['light_subtype'],'uncertain_light')
    def test_glare_next_to_recurring_fixture_stays_uncertain(self):
        rgb=self.frame();rgb[40:43,70:73]=255
        context=self.context();context['frequency'][40:43,66:69]=1
        proposals,_=propose(rgb,context,[])
        self.assertEqual(proposals[0]['light_subtype'],'uncertain_light')
    def test_feature_vector_is_finite(self):
        features=patch_features(self.frame(),[.3,.3,.1,.1])
        self.assertEqual(len(features),13);self.assertTrue(np.isfinite(features).all())


class LightLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name);source=root/'input';source.mkdir()
        for hour in (16,17,18):
            pixels=np.full((100,100,3),20,np.uint8);pixels[10:20,10:20]=255;pixels[10:20,70:80]=255;pixels[80:90,80:90]=hour*10
            Image.fromarray(pixels).save(source/f'C9990001_20260517T{hour}0400Z.jpg')
        self.lib=Library(root/'workspace');self.m=self.lib.create('Explicit synthetic learning test','Israel','Asia/Jerusalem')
        (source/'manifest.json').write_text(json.dumps({'downloads':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source.glob('*.jpg'))]}))
        self.lib.import_folder(self.m,source,'fintraffic');self.rows=self.lib.images(self.m)
    def tearDown(self):self.temp.cleanup()
    def labels(self):return [{'kind':'light','label':label,'box':[x,.1,.1,.1],'night_class':label,'reviewed':True,'review_confidence':4,'origin':'human'} for x,label in ((.1,'headlight'),(.7,'streetlight'))]
    def test_no_labels_does_not_create_checkpoint(self):
        result=train(self.lib,self.m)
        self.assertEqual(result['status'],'needs_labels');self.assertFalse(result['checkpoint_created']);self.assertEqual(candidates(self.lib,self.m),[])
    def test_training_profile_rejects_oversized_validation_set_before_checkpoint(self):
        oversized={'eligible_training_images':2,'classes':['headlight','streetlight'],
            'training':[{} for _ in range(4)],'validation':[{} for _ in range(5001)]}
        with patch('road_traffic.learning.inventory',return_value=oversized):
            with self.assertRaisesRegex(ValueError,'Validation set exceeds'):
                train(self.lib,self.m)
        self.assertEqual(candidates(self.lib,self.m),[])
    def test_explicit_fit_versions_head_and_reports_validation(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='training')
        self.lib.review(self.m,self.rows[2]['id'],0,self.labels(),'Synthetic test',role='validation')
        result=train(self.lib,self.m);self.assertEqual(result['status'],'completed');self.assertEqual(result['report']['training_regions'],4)
        self.assertEqual(result['report']['validation']['regions'],2);self.assertFalse(result['report']['test_labels_used'])
        head,sha=load_head(self.lib,self.m,result['id']);self.assertEqual(len(sha),64)
        self.assertEqual(result['report']['dataset_version'],
            next(row for row in candidates(self.lib,self.m) if row['id']==result['id'])['evidence_hash'])
        self.assertIsNone(result['report']['base_checkpoint'])
        self.assertEqual(result['report']['resource_profile']['workers'],1)
        self.assertEqual(result['report']['resource_profile']['max_training_regions'],5000)
        manifest=json.loads(next(row for row in candidates(self.lib,self.m) if row['id']==result['id'])['evidence'])
        self.assertEqual(manifest['class_schema'],['headlight','streetlight'])
        self.assertEqual(manifest['preprocessing_config']['feature_schema'],'light-patch-v1')
        self.assertIsNone(head.data['base_checkpoint'])
        self.assertEqual(train(self.lib,self.m)['id'],result['id'])
        other=self.lib.create('Other','Latvia','Europe/Riga')
        with self.assertRaises(ValueError):load_head(self.lib,other,result['id'])
    def test_independent_test_evaluation_is_recorded_once_without_training_side_effects(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='training')
        self.lib.review(self.m,self.rows[2]['id'],0,self.labels(),'Synthetic test',role='test')
        candidate=train(self.lib,self.m);before=next(c for c in candidates(self.lib,self.m) if c['id']==candidate['id'])
        status=evaluation_status(self.lib,self.m)
        self.assertEqual(status['test_regions'],2);self.assertFalse(status['already_evaluated'])
        result=evaluate_test(self.lib,self.m,candidate['id'],'Independent reviewer')
        self.assertEqual(result['status'],'completed');self.assertEqual(result['report']['regions'],2)
        self.assertFalse(result['report']['test_labels_used_for_fit_or_validation'])
        self.assertFalse(result['report']['automatic_promotion'])
        self.assertEqual(set(result['report']['per_class']),{'headlight','streetlight'})
        after=next(c for c in candidates(self.lib,self.m) if c['id']==candidate['id'])
        self.assertEqual(after['artifact_hash'],before['artifact_hash'])
        self.assertEqual(evaluation_status(self.lib,self.m)['already_evaluated'],True)
        with self.assertRaisesRegex(ValueError,'already been used'):
            evaluate_test(self.lib,self.m,candidate['id'],'Second reviewer')

    def test_test_fingerprint_includes_human_class_labels(self):
        base=[{'sha256':'a'*64,'box':[.1,.1,.2,.2],'class':'headlight'}]
        relabeled=[{**base[0],'class':'streetlight'}]
        self.assertNotEqual(_test_fingerprint(base),_test_fingerprint(relabeled))

    def test_legacy_fingerprint_still_prevents_reusing_an_already_scored_test_set(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Reviewer',role='training')
        self.lib.review(self.m,self.rows[2]['id'],0,self.labels(),'Reviewer',role='test')
        candidate=train(self.lib,self.m)
        candidate_row=next(row for row in candidates(self.lib,self.m) if row['id']==candidate['id'])
        fingerprint=_legacy_test_fingerprint(inventory(self.lib,self.m)['test'])
        with self.lib.connect(self.m) as conn:
            conn.execute('''INSERT INTO night_evaluations(id,mission_id,candidate_id,candidate_hash,
                test_fingerprint,operator,status,report,created,finished)
                VALUES (?,?,?,?,?,?,'completed','{}','2026-01-01T00:00:00+00:00','2026-01-01T00:00:01+00:00')''',
                ('legacy-evaluation',self.m,candidate['id'],candidate_row['artifact_hash'],fingerprint,'Reviewer'))
        self.assertTrue(evaluation_status(self.lib,self.m)['already_evaluated'])
        with self.assertRaisesRegex(ValueError,'already been used'):
            evaluate_test(self.lib,self.m,candidate['id'],'Another reviewer')

    def test_heldout_evaluation_compares_new_candidate_with_current_promoted_head(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Reviewer',role='training')
        test=self.rows[2]
        self.lib.review(self.m,test['id'],0,self.labels(),'Reviewer',role='test')
        current=train(self.lib,self.m)
        current_row=next(row for row in candidates(self.lib,self.m) if row['id']==current['id'])
        with self.lib.connect(self.m) as conn:
            conn.execute('''INSERT INTO night_model_decisions(id,mission_id,candidate_id,candidate_hash,
                previous_candidate_id,action,evaluation_id,operator,reason,created)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',('decision-current',self.m,current['id'],current_row['artifact_hash'],
                None,'promote',None,'Reviewer','Synthetic current-head fixture','2026-01-01T00:00:00+00:00'))
        reversed_labels=[{**item,'night_class':{'headlight':'streetlight','streetlight':'headlight'}[item['night_class']]}
            for item in self.labels()]
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],1,reversed_labels,'Reviewer',role='training')
        challenger=train(self.lib,self.m)
        evaluation=evaluate_test(self.lib,self.m,challenger['id'],'Independent reviewer')
        comparison=evaluation['report']['baseline_comparison']
        self.assertEqual(comparison['status'],'compared')
        self.assertEqual(comparison['current_candidate_id'],current['id'])
        self.assertEqual(comparison['regions'],2)
        self.assertFalse(comparison['test_labels_used_for_fit_or_validation'])
        self.assertIn('macro_f1',comparison['delta'])
    def test_heldout_evaluation_compares_with_linked_base_context_proposals(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Reviewer',role='training')
        base_classes={'headlight':'headlight_candidate','streetlight':'streetlight_suspect'}
        test_labels=[{**item,'origin':'night-light-heuristic','prediction_id':f"prediction-{i}",
            'light_subtype':base_classes[item['night_class']]} for i,item in enumerate(self.labels())]
        self.lib.review(self.m,self.rows[2]['id'],0,test_labels,'Reviewer',role='test')
        candidate=train(self.lib,self.m)
        evaluation=evaluate_test(self.lib,self.m,candidate['id'],'Independent reviewer')
        comparison=evaluation['report']['base_context_comparison']
        self.assertEqual(comparison['status'],'compared')
        self.assertEqual(comparison['baseline'],'saved_contextual_light_proposals')
        self.assertEqual(comparison['paired_regions'],2)
        self.assertEqual(comparison['test_regions'],2)
        self.assertEqual(comparison['coverage'],1.0)
        self.assertFalse(comparison['test_labels_used_for_fit_or_validation'])
        self.assertIn('macro_f1',comparison['delta'])
    def test_promotion_requires_current_test_evaluation_and_rollback_is_audited(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='training')
        self.lib.review(self.m,self.rows[2]['id'],0,self.labels(),'Synthetic test',role='test')
        candidate=train(self.lib,self.m)
        with self.assertRaisesRegex(ValueError,'Run the current held-out test evaluation'):
            record_model_decision(self.lib,self.m,'promote',candidate['id'],'Reviewer','Manual review')
        evaluate_test(self.lib,self.m,candidate['id'],'Independent reviewer')
        promoted=record_model_decision(self.lib,self.m,'promote',candidate['id'],'Reviewer','Reviewed held-out report')
        self.assertEqual(promoted['action'],'promote')
        state=governance_status(self.lib,self.m)
        self.assertEqual(state['promoted_candidate_id'],candidate['id'])
        self.assertEqual(state['promoted_candidate_hash'],next(item for item in candidates(self.lib,self.m) if item['id']==candidate['id'])['artifact_hash'])
        rollback=record_model_decision(self.lib,self.m,'rollback',None,'Reviewer','Return to base rules')
        self.assertEqual(rollback['previous_candidate_id'],candidate['id'])
        self.assertIsNone(governance_status(self.lib,self.m)['promoted_candidate_id'])
        self.assertEqual([row['action'] for row in governance_status(self.lib,self.m)['history']],['promote','rollback'])
        with self.lib.connect(self.m) as conn:
            with self.assertRaisesRegex(Exception,'append-only'):
                conn.execute('DELETE FROM night_model_decisions WHERE id=?',(promoted['id'],))
    def test_unmodified_unclassified_or_low_confidence_do_not_train(self):
        labels=self.labels();labels[0]['review_confidence']=2;labels[1]['reviewed']=False
        self.lib.review(self.m,self.rows[0]['id'],0,labels,'Synthetic test',role='training')
        self.assertEqual(inventory(self.lib,self.m)['training'],[])
    def test_holdout_image_never_fits(self):
        self.lib.review(self.m,self.rows[0]['id'],0,self.labels(),'Synthetic test',role='test')
        self.lib.review(self.m,self.rows[1]['id'],0,self.labels(),'Synthetic test',role='training')
        data=inventory(self.lib,self.m)
        self.assertEqual(len(data['test']),2);self.assertEqual(len(data['training']),2)
    def test_latest_explicit_decisions_own_evidence(self):
        row=self.rows[0];self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='training')
        self.lib.review(self.m,row['id'],1,[],'Synthetic test',role='training')
        self.assertEqual(inventory(self.lib,self.m)['training'],[])

    def test_candidate_is_applied_by_real_inference_worker(self):
        from road_traffic.detector import model_pack,VERSION
        from road_traffic.inference_store import create_run
        from road_traffic.worker import execute
        pack=Path(__file__).resolve().parents[1]/'model-packs'/'yolox-s'
        if not (pack/'model.onnx').is_file():self.skipTest('Real model pack not staged')
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='training')
        result=train(self.lib,self.m);head,head_hash=load_head(self.lib,self.m,result['id']);_,sha=model_pack(pack)
        run=create_run(self.lib,self.m,{'adapter':VERSION,'threshold':.18,'tiled':False,'night_lights':True,
            'model_sha256':sha,'night_head_id':result['id'],'night_head_hash':head_hash})
        execute(self.lib,self.m,run,pack)
        predictions=[a for row in self.lib.images(self.m,run) for a in row['predictions'] if a['kind']=='light']
        self.assertTrue(predictions)
        self.assertTrue(all(a['correction_head']['model_id']==result['id'] for a in predictions))
        self.assertTrue(all(a['kind']=='light' for a in predictions))

    def test_checkpoint_tampering_is_rejected(self):
        for row in self.rows[:2]:self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='training')
        result=train(self.lib,self.m)
        row=next(c for c in candidates(self.lib,self.m) if c['id']==result['id'])
        (self.lib.location(self.m)/row['artifact']).write_text('{}')
        with self.assertRaises(ValueError):load_head(self.lib,self.m,result['id'])

    def test_review_only_labels_are_not_training(self):
        for row in self.rows:self.lib.review(self.m,row['id'],0,self.labels(),'Synthetic test',role='none')
        self.assertEqual(inventory(self.lib,self.m)['training'],[])
        self.assertEqual(train(self.lib,self.m)['status'],'needs_labels')
