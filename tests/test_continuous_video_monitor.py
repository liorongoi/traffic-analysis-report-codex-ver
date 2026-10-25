import tempfile
import unittest
from pathlib import Path

from road_traffic.monitor import SourceMonitor
from road_traffic.store import Library
from road_traffic.inference import InferenceService
from road_traffic.video_inference import VideoInferenceService


class ContinuousVideoMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source_root=self.root/'incoming';self.source_root.mkdir()
        (self.source_root/'cam-a.mp4').write_bytes(b'video bytes v1')
        self.library=Library(self.root/'workspace')
        self.mission=self.library.create('Continuous video','Israel','Asia/Jerusalem',{
            'input_type':'videos','analysis_mode':'video_traffic','processing_mode':'continuous',
            'source_root':str(self.source_root)})
        self.policy={'threshold':.18,'tiled':False,'target_labels':['car'],
            'tracker_policy':{'max_gap_seconds':[1,2],'minimum_iou':.2,
                'max_center_distance':.15,'association_margin':.05,
                'compatible_label_groups':[]}}
        self.library.set_continuous_video_policy(self.mission,self.policy)
        self.started=[]
        self.monitor=SourceMonitor(self.library,stability_seconds=0,
            on_video=lambda mission,source,policy:self.started.append((source,policy)) or {'run':'fixture-run'})

    def tearDown(self):
        self.monitor.stop()
        self.temp.cleanup()

    def test_stable_video_is_versioned_once_then_replacement_is_preserved(self):
        self.monitor.scan(self.mission)
        self.monitor.scan(self.mission)
        with self.library.connect(self.mission) as conn:
            first=conn.execute('SELECT COUNT(*) FROM video_sources').fetchone()[0]
            batch=conn.execute('SELECT status,policy_json FROM video_monitor_batches').fetchone()
        self.assertEqual(first,1)
        self.assertEqual(batch['status'],'queued')
        self.assertEqual(__import__('json').loads(batch['policy_json']),self.policy)
        self.assertEqual(len(self.started),1)

        (self.source_root/'cam-a.mp4').write_bytes(b'video bytes v2')
        self.monitor.scan(self.mission)
        with self.library.connect(self.mission) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM video_sources').fetchone()[0],2)
        self.assertEqual(len(self.started),2)

    def test_video_policy_requires_valid_explicit_detector_and_tracker_values(self):
        pack=Path(__file__).resolve().parents[1]/'model-packs'/'yolox-s'
        service=VideoInferenceService(self.library,pack,InferenceService(self.library,pack),
            worker_script=self.root/'unused-worker.py')
        with self.assertRaisesRegex(ValueError,'threshold'):
            service.save_continuous_policy(self.mission,tracker_policy=self.policy['tracker_policy'],
                threshold=0,tiled=False,target_labels=['car'])
        with self.assertRaisesRegex(ValueError,'tracker_policy'):
            service.save_continuous_policy(self.mission,tracker_policy={},
                threshold=.18,tiled=False,target_labels=['car'])

    def test_busy_worker_keeps_durable_source_job_pending_for_next_scan(self):
        class BusyError(ValueError):retryable=True
        attempts=[]
        def start(mission,source,policy):
            attempts.append(source)
            if len(attempts)==1:raise BusyError('worker busy')
            return {'run':'retried-run'}
        self.monitor.on_video=start
        self.monitor.scan(self.mission)
        with self.library.connect(self.mission) as conn:
            self.assertEqual(conn.execute('SELECT status FROM video_monitor_batches').fetchone()['status'],'pending')
        self.monitor.scan(self.mission)
        with self.library.connect(self.mission) as conn:
            row=conn.execute('SELECT status,video_run_id FROM video_monitor_batches').fetchone()
        self.assertEqual((row['status'],row['video_run_id']),('queued','retried-run'))
        self.assertEqual(len(attempts),2)


if __name__=='__main__':
    unittest.main()
