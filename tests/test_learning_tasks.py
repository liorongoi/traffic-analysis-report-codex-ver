import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from road_traffic.learning import (
    cancel_task, queue_task, recover_interrupted_tasks, run_task, task_status,
)
from road_traffic.store import Library


class LearningTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.library = Library(Path(self.temp.name) / 'workspace')
        self.mission = self.library.create('Learning tasks', 'Israel', 'Asia/Jerusalem')

    def tearDown(self):
        self.temp.cleanup()

    def test_training_task_is_persisted_and_finishes_with_needs_labels(self):
        task = queue_task(self.library, self.mission, 'train')
        self.assertEqual(task_status(self.library, self.mission)[0]['status'], 'queued')

        result = run_task(self.library, self.mission, task['id'])

        self.assertEqual(result['status'], 'needs_labels')
        saved = task_status(self.library, self.mission)[0]
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['result_status'], 'needs_labels')
        self.assertIsNotNone(saved['started'])
        self.assertIsNotNone(saved['finished'])

    def test_only_one_learning_task_can_be_queued_at_a_time(self):
        queue_task(self.library, self.mission, 'train')
        with self.assertRaisesRegex(ValueError, 'already queued or running'):
            queue_task(self.library, self.mission, 'evaluate_test', {
                'candidate_id': 'candidate', 'operator': 'Reviewer'})

    def test_worker_cannot_claim_an_already_running_task_twice(self):
        task = queue_task(self.library, self.mission, 'train')
        with self.library.connect(self.mission) as connection:
            connection.execute("UPDATE learning_tasks SET status='running' WHERE id=?", (task['id'],))

        with self.assertRaisesRegex(ValueError, 'not queued'):
            run_task(self.library, self.mission, task['id'])

    def test_restart_requeues_tasks_and_fails_unmatched_candidate_rows(self):
        task = queue_task(self.library, self.mission, 'train')
        with self.library.connect(self.mission) as connection:
            connection.execute("UPDATE learning_tasks SET status='running' WHERE id=?", (task['id'],))
            connection.execute('''INSERT INTO night_training(id,status,progress,created,evidence_hash,evidence)
                VALUES ('interrupted-candidate','running',40,'2026-10-28T00:00:00Z','hash','{}')''')

        recovered = recover_interrupted_tasks(self.library)

        self.assertEqual(recovered, 1)
        saved = task_status(self.library, self.mission)[0]
        self.assertEqual(saved['status'], 'queued')
        with self.library.connect(self.mission) as connection:
            candidate = connection.execute(
                "SELECT status,error FROM night_training WHERE id='interrupted-candidate'").fetchone()
        self.assertEqual(candidate['status'], 'failed')
        self.assertIn('resumable training task', candidate['error'])

    def test_restart_preserves_checkpoint_and_requeues_matching_training(self):
        task = queue_task(self.library, self.mission, 'train')
        checkpoint = {'evidence_hash': 'same-evidence', 'stage': 'fit', 'next_step': 280}
        with self.library.connect(self.mission) as connection:
            connection.execute("UPDATE learning_tasks SET status='running',checkpoint=? WHERE id=?",
                (json.dumps(checkpoint), task['id']))
            connection.execute('''INSERT INTO night_training(id,status,progress,created,evidence_hash,evidence)
                VALUES ('candidate','running',40,'2026-10-28T00:00:00Z','same-evidence','{}')''')

        recover_interrupted_tasks(self.library)

        saved = task_status(self.library, self.mission)[0]
        self.assertEqual(saved['status'], 'queued')
        self.assertNotIn('checkpoint', saved)
        with self.library.connect(self.mission) as connection:
            row = connection.execute('SELECT checkpoint FROM learning_tasks WHERE id=?',(task['id'],)).fetchone()
            candidate = connection.execute("SELECT status FROM night_training WHERE id='candidate'").fetchone()
        self.assertEqual(json.loads(row['checkpoint']), checkpoint)
        self.assertEqual(candidate['status'], 'running')

    def test_requeued_task_receives_its_saved_checkpoint(self):
        from road_traffic import learning
        task = queue_task(self.library, self.mission, 'train')
        checkpoint = {'evidence_hash': 'same-evidence', 'stage': 'fit', 'next_step': 240}

        def interrupted_train(library, mission, *, yield_checkpoint, resume_checkpoint=None, save_checkpoint=None):
            save_checkpoint(checkpoint)
            raise RuntimeError('simulated abrupt failure')

        with patch.object(learning, 'train', side_effect=interrupted_train):
            with self.assertRaisesRegex(RuntimeError, 'simulated abrupt failure'):
                run_task(self.library, self.mission, task['id'])
        with self.library.connect(self.mission) as connection:
            connection.execute("UPDATE learning_tasks SET status='running' WHERE id=?", (task['id'],))

        recover_interrupted_tasks(self.library)
        resumed_checkpoints = []

        def resumed_train(library, mission, *, yield_checkpoint, resume_checkpoint=None, save_checkpoint=None):
            resumed_checkpoints.append(resume_checkpoint)
            return {'id': 'resumed-candidate', 'status': 'completed'}

        with patch.object(learning, 'train', side_effect=resumed_train):
            result = run_task(self.library, self.mission, task['id'])

        self.assertEqual(result['id'], 'resumed-candidate')
        self.assertEqual(resumed_checkpoints, [checkpoint])

    def test_queued_learning_task_can_be_cancelled_without_starting(self):
        task = queue_task(self.library, self.mission, 'train')
        cancel_task(self.library, self.mission, task['id'])
        saved = task_status(self.library, self.mission)[0]
        self.assertEqual(saved['result_status'], 'cancelled')
        with self.assertRaisesRegex(ValueError, 'not queued'):
            run_task(self.library, self.mission, task['id'])

    def test_failed_task_keeps_its_error_in_the_task_history(self):
        task = queue_task(self.library, self.mission, 'evaluate_test', {
            'candidate_id': 'missing-candidate', 'operator': 'Reviewer'})

        with self.assertRaises(ValueError):
            run_task(self.library, self.mission, task['id'])

        saved = task_status(self.library, self.mission)[0]
        self.assertEqual(saved['status'], 'failed')
        self.assertIn('No eligible independent test labels', saved['error'])

    def test_operational_worker_can_pause_and_resume_training_at_checkpoint(self):
        from road_traffic.inference import InferenceService
        service = InferenceService(self.library, Path(self.temp.name) / 'model-pack')
        service._has_active_worker = lambda: False
        service.training_active = True
        reached_checkpoint = threading.Event()
        resumed = threading.Event()

        def training():
            reached_checkpoint.set()
            service.training_pause_requested.wait(2)
            service.training_checkpoint()
            resumed.set()
            service.training_active = False

        worker = threading.Thread(target=training)
        worker.start()
        self.assertTrue(reached_checkpoint.wait(2))
        with service.lock:
            yielded = service._claim_worker_slot()
        self.assertTrue(yielded)
        self.assertTrue(service.training_paused.is_set())
        self.assertFalse(resumed.is_set())
        with service.lock:
            service._resume_training_after_worker()
        worker.join(2)
        self.assertTrue(resumed.is_set())
        self.assertFalse(service.training_paused.is_set())


if __name__ == '__main__':
    unittest.main()
