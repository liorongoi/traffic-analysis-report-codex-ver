from pathlib import Path
import tempfile
import unittest
from PIL import Image

from road_traffic.inference import InferenceBusy
from road_traffic.inference_store import create_run
from road_traffic.monitor import SourceMonitor
from road_traffic.store import Library


class ContinuousBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.library = Library(root / 'workspace')
        self.source = root / 'source'
        self.source.mkdir()
        self.mission = self.library.create('Continuous', 'Israel', 'Asia/Jerusalem', {
            'source_root': str(self.source), 'processing_mode': 'continuous'})
        self.image = self.source / '17052026-1605-camera.jpg'
        Image.new('RGB', (8, 8), (30, 60, 90)).save(self.image, 'JPEG')

    def tearDown(self):
        self.temp.cleanup()

    def test_monitor_dispatches_batch_once_and_tracks_inference_completion(self):
        created = []

        def dispatch(mission, source_run):
            run = create_run(self.library, mission, {'adapter': 'test'}, source_run)
            created.append((source_run, run))
            return {'run': run}

        monitor = SourceMonitor(self.library, stability_seconds=0, on_batch=dispatch)
        state = monitor.scan(self.mission)
        source_run = state['last_batch']['run']
        self.assertEqual(state['batches'], {'queued': 1})
        self.assertEqual(len(created), 1)
        with self.library.connect(self.mission) as conn:
            inference_run = conn.execute('SELECT id FROM inference_runs').fetchone()['id']
            inputs = conn.execute('SELECT COUNT(*) FROM inference_inputs WHERE inference_id=?', (inference_run,)).fetchone()[0]
            self.assertEqual(inputs, 1)
            conn.execute("UPDATE inference_runs SET status='completed',stage='finished' WHERE id=?", (inference_run,))
        state = monitor.status(self.mission)
        self.assertEqual(state['batches'], {'completed': 1})
        monitor.scan(self.mission)
        self.assertEqual(len(created), 1)
        self.assertNotEqual(source_run, inference_run)

    def test_busy_worker_leaves_batch_pending_for_retry(self):
        calls = []

        def dispatch(mission, source_run):
            calls.append(source_run)
            if len(calls) == 1:
                raise InferenceBusy('worker busy')
            return {'run': create_run(self.library, mission, {'adapter': 'test'}, source_run)}

        monitor = SourceMonitor(self.library, stability_seconds=0, on_batch=dispatch)
        first = monitor.scan(self.mission)
        self.assertEqual(first['batches'], {'pending': 1})
        monitor.scan(self.mission)
        self.assertEqual(len(calls), 2)
        self.assertEqual(monitor.status(self.mission)['batches'], {'queued': 1})


if __name__ == '__main__':
    unittest.main()
