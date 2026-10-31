from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from road_traffic.inference import InferenceService
from road_traffic.store import Library
from road_traffic.video_inference import VehicleProposalAdapter, VideoInferenceService
from road_traffic.video_store import create_run, register_video, transition


class VideoInferenceServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / 'input'
        self.source_root.mkdir()
        (self.source_root / 'clip.mp4').write_bytes(b'video-source')
        self.library = Library(self.root / 'workspace')
        self.mission = self.library.create('Video API mission', 'Israel', 'Asia/Jerusalem', {
            'input_type': 'videos', 'analysis_mode': 'video_traffic',
            'source_root': str(self.source_root)})
        self.pack = Path(__file__).resolve().parents[1] / 'model-packs' / 'yolox-s'
        self.inference = InferenceService(self.library, self.pack)
        self.worker = self.root / 'sleep-worker.py'
        self.worker.write_text('import time\ntime.sleep(30)\n')
        self.service = VideoInferenceService(self.library, self.pack, self.inference,
            worker_script=self.worker)

    def tearDown(self):
        for process in list(self.inference.processes.values()):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        self.temp.cleanup()

    @staticmethod
    def tracker_policy():
        return {'max_gap_seconds': [1, 2], 'minimum_iou': .2,
            'max_center_distance': .15, 'association_margin': .05,
            'compatible_label_groups': []}

    def source(self):
        return register_video(self.library, self.mission, 'clip.mp4')

    def test_worker_launcher_is_safe_when_reloaded_as_a_spawn_main_module(self):
        import runpy
        launcher=Path(__file__).resolve().parents[1]/'scripts'/'run-video-analysis.py'
        with patch('sys.argv',['run-video-analysis.py']):
            namespace=runpy.run_path(str(launcher),run_name='__mp_main__')
        self.assertIn('main',namespace)

    def test_video_adapter_persists_provider_fallback_only_when_state_changes(self):
        from road_traffic.video_inference import VehicleProposalAdapter
        class Detector:
            cpu_threads=2
            def __init__(self):
                self.session=SimpleNamespace(provider='DmlExecutionProvider',fallback_reason=None)
            def infer(self,frame):
                return [{'label':'car','box':[0,0,1,1],'score':.5}]
        detector=Detector();reported=[]
        adapter=VehicleProposalAdapter(detector,['car'],reported.append)
        self.assertEqual(len(reported),1)
        adapter.infer(None)
        self.assertEqual(len(reported),1)
        detector.session.provider='CPUExecutionProvider'
        detector.session.fallback_reason='DirectML execution failed'
        self.assertEqual(adapter.infer(None)[0]['label'],'car')
        self.assertEqual(reported[-1]['execution_provider'],'CPUExecutionProvider')
        self.assertEqual(reported[-1]['fallback_reason'],'DirectML execution failed')

    def test_video_adapter_attaches_review_only_road_geometry(self):
        class Detector:
            cpu_threads=2
            session=SimpleNamespace(provider='CPUExecutionProvider',fallback_reason=None)
            def infer(self,frame):
                return [{'label':'car','box':[.4,.3,.1,.3],'score':.8}]
        road=[{'kind':'include','role':'road_surface','polygon':[[0,.55],[1,.55],[1,1],[0,1]]}]
        result=VehicleProposalAdapter(Detector(),['car'],road_regions=road).infer(None)
        self.assertEqual(result[0]['road_relation']['label'],'on_road')
        self.assertEqual(result[0]['label'],'car')

    def test_supervisor_starts_cancels_recovers_and_resumes_same_run(self):
        source = self.source()
        result = self.service.start(self.mission, source['id'], self.tracker_policy(),
            threshold=.18, tiled=False,target_labels=['truck','bus'],resource_profile='conservative')
        self.assertEqual(result['config']['detector']['target_labels'],['bus','truck'])
        self.assertEqual(result['config']['detector']['resource_profile']['requested'],'conservative')
        self.assertEqual(result['config']['detector']['cpu_threads'],1)
        run_id = result['run']
        process = self.inference.processes[(self.mission, run_id)]
        self.assertIsNone(process.poll())
        with patch('road_traffic.hardware._process_sample',return_value=(0.0,12345)):
            live=self.service.status(self.mission)[0]
        self.assertEqual(live['status'], 'queued')
        self.assertEqual(live['resource_usage']['state'],'measured')
        self.assertGreater(live['resource_usage']['resident_memory_bytes'],0)
        self.service.cancel(self.mission, run_id)
        self.assertTrue(self.service.status(self.mission)[0]['cancel_requested'])
        process.terminate()
        process.wait(timeout=5)
        self.assertEqual(self.service.status(self.mission)[0]['status'], 'failed')

        resumed = self.service.resume(self.mission, run_id)
        self.assertEqual(resumed['run'], run_id)
        self.assertEqual(resumed['status'], 'queued')
        process = self.inference.processes[(self.mission, run_id)]
        process.terminate()
        process.wait(timeout=5)
        self.assertEqual(self.service.status(self.mission)[0]['status'], 'failed')

    def test_start_requires_explicit_valid_tracker_and_detector_policy(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, 'tracker_policy'):
            self.service.start(self.mission, source['id'], {}, threshold=.18, tiled=False)
        with self.assertRaisesRegex(ValueError, 'threshold'):
            self.service.start(self.mission, source['id'], self.tracker_policy(),
                threshold=.001, tiled=False)
        with self.assertRaisesRegex(ValueError,'vehicle classes'):
            self.service.start(self.mission,source['id'],self.tracker_policy(),
                threshold=.18,tiled=False,target_labels=[])
        with self.assertRaisesRegex(ValueError,'vehicle classes'):
            self.service.start(self.mission,source['id'],self.tracker_policy(),
                threshold=.18,tiled=False,target_labels=['semi-trailer'])
        self.assertEqual(self.service.status(self.mission), [])

    def test_manual_video_analysis_is_available_in_object_search_missions(self):
        mission=self.library.create('Video object search','Israel','Asia/Jerusalem',{
            'input_type':'videos','analysis_mode':'object_search','source_root':str(self.source_root)})
        source=register_video(self.library,mission,'clip.mp4')
        with patch('road_traffic.detector.model_pack',return_value=(self.pack,'fixture-sha256')) as model_pack, \
                patch.object(self.service,'_spawn') as spawn:
            result=self.service.start(mission,source['id'],self.tracker_policy(),
                threshold=.18,tiled=False,target_labels=['truck'])
        self.assertEqual(result['status'],'queued')
        self.assertEqual(result['config']['detector']['target_labels'],['truck'])
        model_pack.assert_called_once_with(self.pack)
        spawn.assert_called_once_with(mission,result['run'],self.pack)

    def test_video_run_uses_selected_candidate_and_resume_keeps_its_frozen_model(self):
        from road_traffic.model_registry import import_candidate,select_model
        source=self.source()
        candidate=self.root/'trained-candidate'
        candidate.mkdir()
        content=b'test candidate model'
        (candidate/'model.onnx').write_bytes(content)
        (candidate/'manifest.json').write_text(json.dumps({'name':'Reviewed candidate',
            'classes':['car','truck'],'files':{'model.onnx':{'sha256':hashlib.sha256(content).hexdigest(),
            'bytes':len(content)}}}),encoding='utf-8')
        imported=import_candidate(self.library,candidate)
        select_model(self.library,self.pack,self.mission,imported['id'],'Operator','Use reviewed video weights')
        selected_pack=self.library.root/'models'/'yolox-candidates'/imported['id']
        with patch.object(self.service,'_spawn') as spawn:
            result=self.service.start(self.mission,source['id'],self.tracker_policy(),
                threshold=.18,tiled=False,target_labels=['car'])
            self.assertEqual(result['config']['detector']['model_id'],imported['id'])
            self.assertEqual(result['config']['detector']['sha256'],imported['sha256'])
            spawn.assert_called_once_with(self.mission,result['run'],selected_pack.resolve())
            transition(self.library,self.mission,result['run'],'failed',error='synthetic restart')
            select_model(self.library,self.pack,self.mission,'base','Operator','Restore baseline for new runs')
            resumed=self.service.resume(self.mission,result['run'])
            self.assertEqual(resumed['run'],result['run'])
            self.assertEqual(spawn.call_args.args,(self.mission,result['run'],selected_pack.resolve()))

    def test_video_proposal_adapter_filters_to_the_saved_target_classes(self):
        class Detector:
            def infer(self,rgb):
                return [{'label':'truck','box':[0,0,.2,.2],'score':.8},
                    {'label':'car','box':[.3,0,.2,.2],'score':.7}]
        result=VehicleProposalAdapter(Detector(),['truck']).infer(object())
        self.assertEqual([item['label'] for item in result],['truck'])

    def test_plate_video_run_uses_only_the_plate_pack_and_existing_worker_slot(self):
        (self.root/'plate-output').mkdir()
        mission=self.library.create('Video plate mission','Israel','Asia/Jerusalem',{
            'input_type':'videos','analysis_mode':'license_plate','source_root':str(self.source_root),
            'output_root':str(self.root/'plate-output')})
        source=register_video(self.library,mission,'clip.mp4')
        result=self.service.start_plate_review(mission,source['id'],threshold=.18,tiled=False)
        self.assertEqual(result['status'],'queued')
        self.assertEqual(result['config']['task'],'license_plate')
        self.assertEqual(result['config']['plate_detector']['device'],'auto')
        self.assertEqual(result['config']['plate_detector']['threshold'],.18)
        self.assertFalse(result['config']['plate_detector']['tiled'])
        self.assertNotIn('recognition',result['config'])
        self.assertIn((mission,result['run']),self.inference.processes)

    def test_startup_marks_orphaned_runs_failed_with_checkpoint_preserved(self):
        source = self.source()
        run = create_run(self.library, self.mission, source['id'], {
            'detector': {'version': 'fixture', 'sha256': 'fixture'},
            'tracker_policy': self.tracker_policy()})
        transition(self.library, self.mission, run['id'], 'running')
        with self.library.connect(self.mission) as conn:
            conn.execute('UPDATE video_runs SET frames_done=7,worker_pid=? WHERE id=?',
                (2**30, run['id']))
        restarted = VideoInferenceService(self.library, self.pack, self.inference,
            worker_script=self.worker)
        job = restarted.status(self.mission)[0]
        self.assertEqual(job['status'], 'failed')
        self.assertIn('7 frames are saved', job['error'])
        self.assertEqual(job['frames_done'], 7)


if __name__ == '__main__':
    unittest.main()
