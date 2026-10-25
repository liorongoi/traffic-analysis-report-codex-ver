import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image
from unittest.mock import patch
from road_traffic.store import Library
from road_traffic.inference_store import create_run,enrich,runs
from road_traffic.detector import iou,suppress,illumination,night_proposals,model_pack
from road_traffic.inference import InferenceService


class DetectorMathTests(unittest.TestCase):
    def test_cpu_profile_thread_limit_is_applied_to_onnx_session(self):
        class FakeInput:
            name='images'
        class FakeSession:
            def __init__(self,path,*,sess_options,providers):
                self.options=sess_options;self.providers=providers
            def get_providers(self):return self.providers
            def get_inputs(self):return [FakeInput()]
        with (patch('road_traffic.detector.model_pack',return_value=('fixture','fixture-sha')),
                patch('onnxruntime.InferenceSession',FakeSession)):
            from road_traffic.detector import VehicleDetector
            detector=VehicleDetector('unused',cpu_threads=3)
        self.assertEqual(detector.session.options.intra_op_num_threads,3)
        self.assertEqual(detector.session.options.inter_op_num_threads,1)
        with self.assertRaisesRegex(ValueError,'CPU thread limit'):
            VehicleDetector('unused',cpu_threads=0)

    def test_duplicate_vehicle_classes_do_not_double_count(self):
        a={'box':[.1,.1,.2,.2],'label':'car','score':.8}
        b={'box':[.11,.11,.2,.2],'label':'truck','score':.6}
        self.assertGreater(iou(a['box'],b['box']),.7)
        self.assertEqual(suppress([b,a]),[a])
    def test_bright_monochrome_night_is_not_skipped(self):
        import numpy as np
        self.assertTrue(illumination(np.full((30,30,3),120,np.uint8))['low_light'])
        colorful=np.zeros((30,30,3),np.uint8);colorful[:,:,1]=220;colorful[:,:,0]=120
        self.assertFalse(illumination(colorful)['low_light'])
    def test_single_to_pair_is_not_two_confirmed_vehicles(self):
        import numpy as np
        rgb=np.zeros((100,200,3),np.uint8);rgb[50:53,80:83]=255
        single,info=night_proposals(rgb,None,[])
        self.assertEqual(len(single),1);self.assertEqual(single[0]['pairing'],'single')
        rgb[50:53,90:93]=255
        pair,info=night_proposals(rgb,None,[])
        self.assertEqual(len(pair),1);self.assertEqual(pair[0]['pairing'],'hypothesis')
        self.assertEqual(len(pair[0]['light_components']),2)
        self.assertEqual(pair[0]['kind'],'light')
    def test_persistent_lights_and_detector_overlap_excluded(self):
        import numpy as np
        rgb=np.zeros((100,200,3),np.uint8);rgb[50:53,80:83]=255
        self.assertEqual(night_proposals(rgb,np.ones((100,200),np.uint8),[])[0],[])
        box={'box':[.3,.4,.2,.2]}
        self.assertEqual(night_proposals(rgb,None,[box])[0],[])
    def test_missing_model_fails_without_fetch(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):model_pack(root)


class InferenceLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.lib=Library(root/'workspace');self.m=self.lib.create('Test','Israel','Asia/Jerusalem')
        source=root/'source';source.mkdir();Image.new('RGB',(8,8),(25,50,75)).save(source/'17052026-1604-test.jpg','JPEG')
        self.lib.import_folder(self.m,source,'camera');self.image=self.lib.images(self.m)[0]['id']
        self.run=create_run(self.lib,self.m,{'model':'explicit-test-double'})
        self.pred=[{'kind':'candidate','label':'car','box':[.1,.1,.2,.2],'score':.7,'prediction_id':'fixture-1','origin':'model'}]
        with self.lib.connect(self.m) as conn:
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(self.run,self.image,json.dumps(self.pred),'{}',.8,None))
            conn.execute("UPDATE inference_runs SET status='completed',done=1 WHERE id=?",(self.run,))
    def tearDown(self):self.temp.cleanup()
    def test_model_proposals_appear_without_becoming_reviewed(self):
        row=self.lib.images(self.m)[0]
        self.assertEqual(row['revision'],0);self.assertEqual(row['review_status'],'unreviewed')
        self.assertEqual(json.loads(row['annotations']),self.pred)
        progress=next(item for item in runs(self.lib,self.m) if item['id']==self.run)
        self.assertEqual(progress['average_item_seconds'],.8)

    def test_run_configuration_includes_immutable_mission_and_source_snapshots(self):
        row=self.lib.images(self.m)[0]
        with self.lib.connect(self.m) as conn:
            raw=conn.execute('SELECT config FROM inference_runs WHERE id=?',(self.run,)).fetchone()['config']
            config=json.loads(raw)
            source=conn.execute('SELECT preset,source_root,status,created,error FROM runs WHERE id=?',(row['run_id'],)).fetchone()
            self.assertEqual(config['run_config_schema_version'],1)
            self.assertEqual((config['mission_snapshot']['id'],config['mission_snapshot']['name']),
                (self.m,'Test'))
            self.assertEqual((config['mission_snapshot']['country'],config['mission_snapshot']['zone']),
                ('Israel','Asia/Jerusalem'))
            self.assertEqual(config['mission_snapshot']['input_type'],'images')
            self.assertEqual(config['mission_snapshot']['export_formats'],['csv'])
            self.assertEqual(config['source_runs_snapshot'],[dict(id=row['run_id'],**dict(source))])
            with self.assertRaisesRegex(sqlite3.IntegrityError,'immutable run configuration'):
                conn.execute('UPDATE inference_runs SET config=? WHERE id=?',('{}',self.run))
    def test_correction_survives_restart_and_keeps_original(self):
        changed=[{**self.pred[0],'kind':'vehicle','label':'truck','reviewed':True}]
        self.lib.review(self.m,self.image,0,changed,'Test operator',prediction_run=self.run)
        reopened=Library(self.lib.root);row=reopened.images(self.m)[0]
        self.assertEqual(json.loads(row['annotations'])[0]['label'],'truck')
        self.assertEqual(row['predictions'][0]['label'],'car')
        self.assertEqual(row['review_prediction_run'],self.run)
        with self.lib.connect(self.m) as conn:
            changes=json.loads(conn.execute('SELECT changes FROM review_inference').fetchone()[0])
            self.assertEqual(changes['before'],self.pred);self.assertEqual(changes['after'],changed)
            with self.assertRaises(sqlite3.IntegrityError):conn.execute('DELETE FROM predictions')
    def test_rejection_preserves_original(self):
        changed=[{**self.pred[0],'rejected':True}]
        self.lib.review(self.m,self.image,0,changed,'Test operator',prediction_run=self.run)
        row=self.lib.images(self.m)[0]
        self.assertTrue(json.loads(row['annotations'])[0]['rejected'])
        self.assertNotIn('rejected',row['predictions'][0])
    def test_failed_lineage_save_is_atomic(self):
        with self.assertRaises(ValueError):self.lib.review(self.m,self.image,0,self.pred,'Operator',prediction_run='not-a-real-run')
        self.assertEqual(self.lib.images(self.m)[0]['revision'],0)
    def test_new_run_does_not_replace_review(self):
        self.lib.review(self.m,self.image,0,[],'Operator',prediction_run=self.run)
        newer=create_run(self.lib,self.m,{'model':'second-fixture'})
        with self.lib.connect(self.m) as conn:
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(newer,self.image,'[]','{}',.1,None))
            conn.execute("UPDATE inference_runs SET status='completed' WHERE id=?",(newer,))
        row=self.lib.images(self.m)[0]
        self.assertEqual(row['prediction_run'],self.run);self.assertEqual(row['annotations'],'[]')
        selected=self.lib.images(self.m,newer)[0]
        self.assertEqual(selected['review_prediction_run'],self.run);self.assertEqual(selected['prediction_run'],newer)
    def test_duplicate_worker_blocked(self):
        create_run(self.lib,self.m,{})
        with self.assertRaises(ValueError):create_run(self.lib,self.m,{})
    def test_mission_run_isolation(self):
        other=self.lib.create('Other','Latvia','Europe/Riga')
        with self.assertRaises(ValueError):self.lib.images(other,self.run)


class InferenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.lib=Library(self.root/'workspace');self.mission=self.lib.create('Recover','Israel','Asia/Jerusalem')
        source=self.root/'source';source.mkdir()
        Image.new('RGB',(8,8),(10,30,50)).save(source/'17052026-1604-a.jpg','JPEG')
        Image.new('RGB',(8,8),(50,30,10)).save(source/'17052026-1605-b.jpg','JPEG')
        self.lib.import_folder(self.mission,source,'camera')

    def tearDown(self):self.temp.cleanup()

    def test_continuous_batch_uses_saved_mission_profile(self):
        mission=self.lib.create('Continuous profile','Israel','Asia/Jerusalem',{
            'processing_mode':'continuous','resource_profile':'expanded'})
        service=InferenceService(self.lib,self.root/'model-pack')
        with patch.object(service,'start',return_value={'run':'new-run'}) as start:
            result=service.start_continuous_batch(mission,'source-run')
        self.assertEqual(result,{'run':'new-run'})
        start.assert_called_once_with(mission,source_run='source-run',resource_profile='expanded')

    def test_active_image_run_status_includes_live_resource_usage(self):
        from unittest.mock import Mock
        run=create_run(self.lib,self.mission,{'adapter':'fixture'})
        with self.lib.connect(self.mission) as conn:
            conn.execute("UPDATE inference_runs SET status='running',stage='detecting',worker_pid=? WHERE id=?",(os.getpid(),run))
        service=InferenceService(self.lib,self.root/'model-pack')
        process=Mock();process.poll.return_value=None
        service.processes[(self.mission,run)]=process
        measured={'state':'measured','resident_memory_bytes':4096,'cpu_percent_host_capacity':12.5,'cpu_sample_seconds':2.0}
        with patch('road_traffic.hardware.process_usage',return_value=measured):
            status=next(job for job in service.status(self.mission) if job['id']==run)
        self.assertEqual(status['resource_usage'],measured)

    def test_active_video_worker_occupies_shared_training_slot_after_service_restart(self):
        service=InferenceService(self.lib,self.root/'model-pack')
        orphan={'id':'video-run','status':'running','worker_pid':os.getpid(),'frames_done':4}
        with patch('road_traffic.video_store.runs',return_value=[orphan]):
            self.assertTrue(service._has_active_worker())

    def test_start_snapshots_operator_resource_profile_and_applies_cpu_thread_limit(self):
        from road_traffic.detector import VERSION
        gib=1024**3
        service=InferenceService(self.lib,self.root/'model-pack')
        with (patch('road_traffic.detector.model_pack',return_value=('fixture','fixture-sha')),
                patch('road_traffic.hardware._memory',return_value=(16*gib,12*gib)),
                patch.object(service,'_has_active_worker',return_value=False),
                patch.object(service,'_spawn') as spawn):
            result=service.start(self.mission,night_lights=False,resource_profile='expanded')
        self.assertEqual(result['config']['adapter'],VERSION)
        self.assertEqual(result['config']['resource_profile']['requested'],'expanded')
        self.assertEqual(result['config']['resource_profile']['selected'],'expanded')
        self.assertEqual(result['config']['cpu_threads'],min(4,os.cpu_count() or 1))
        self.assertEqual(result['config']['resource_profile']['workers'],1)
        spawn.assert_called_once_with(self.mission,result['run'])
        with self.lib.connect(self.mission) as conn:
            saved=json.loads(conn.execute('SELECT config FROM inference_runs WHERE id=?',(result['run'],)).fetchone()['config'])
        self.assertEqual(saved['resource_profile'],result['config']['resource_profile'])

    def test_service_restart_marks_missing_worker_failed_with_progress_preserved(self):
        run=create_run(self.lib,self.mission,{'adapter':'fixture'})
        with self.lib.connect(self.mission) as conn:
            conn.execute("UPDATE inference_runs SET status='running',stage='detecting',done=1,worker_pid=2147483000 WHERE id=?",(run,))
        service=InferenceService(self.lib,self.root/'model-pack')
        job=next(row for row in service.status(self.mission) if row['id']==run)
        self.assertEqual(job['status'],'failed')
        self.assertEqual(job['done'],1)
        self.assertIn('Resume this run',job['error'])

    def test_resume_keeps_config_and_saved_predictions_and_requeues_only_remaining(self):
        from road_traffic.detector import VERSION
        from unittest.mock import Mock
        config={'adapter':VERSION,'model_sha256':'fixture-sha','threshold':.18,'tiled':True,'night_lights':False}
        run=create_run(self.lib,self.mission,config)
        with self.lib.connect(self.mission) as conn:
            first=conn.execute('SELECT image_id FROM inference_inputs WHERE inference_id=? ORDER BY image_id LIMIT 1',(run,)).fetchone()[0]
            saved_config=json.loads(conn.execute('SELECT config FROM inference_runs WHERE id=?',(run,)).fetchone()[0])
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(run,first,'[]','{}',.1,None))
            conn.execute("UPDATE inference_runs SET status='failed',stage='failed',done=1,error='interrupted' WHERE id=?",(run,))
        service=InferenceService(self.lib,self.root/'model-pack')
        process=Mock();process.poll.return_value=None
        with (patch('road_traffic.detector.model_pack',return_value=('fixture','fixture-sha')),
                patch('road_traffic.inference.subprocess.Popen',return_value=process)):
            result=service.resume(self.mission,run)
        self.assertEqual(result,{'run':run,'status':'queued','remaining':1})
        with self.lib.connect(self.mission) as conn:
            job=conn.execute('SELECT status,done,config FROM inference_runs WHERE id=?',(run,)).fetchone()
            count=conn.execute('SELECT COUNT(*) FROM predictions WHERE inference_id=?',(run,)).fetchone()[0]
        self.assertEqual((job['status'],job['done'],count),('queued',1,1))
        self.assertEqual(json.loads(job['config']),saved_config)
