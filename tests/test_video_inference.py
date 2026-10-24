from pathlib import Path
import tempfile
import unittest

from road_traffic.inference import InferenceService
from road_traffic.store import Library
from road_traffic.video_inference import VideoInferenceService
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

    def test_supervisor_starts_cancels_recovers_and_resumes_same_run(self):
        source = self.source()
        result = self.service.start(self.mission, source['id'], self.tracker_policy(),
            threshold=.18, tiled=False)
        run_id = result['run']
        process = self.inference.processes[(self.mission, run_id)]
        self.assertIsNone(process.poll())
        self.assertEqual(self.service.status(self.mission)[0]['status'], 'queued')
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
        self.assertEqual(self.service.status(self.mission), [])

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
