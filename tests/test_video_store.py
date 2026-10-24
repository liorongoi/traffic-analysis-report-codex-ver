import hashlib
from datetime import datetime, timezone, timedelta
from fractions import Fraction
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest

from road_traffic.store import Library
from road_traffic.tracking import TrackerPolicy, VideoTrackManager
from road_traffic.video_store import (checkpoint, create_run, events, register_video,
    execute_video_run, review_event, run_detail, source_path, transition)
from road_traffic.video_timeline import VideoTimestamp


class VideoStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / 'input-videos'
        self.source_root.mkdir()
        self.video_file = self.source_root / 'clip.mkv'
        self.original = b'video source fixture\x00\x01'
        self.video_file.write_bytes(self.original)
        self.library = Library(self.root / 'workspace')
        self.mission = self.library.create('Video mission', 'Israel', 'Asia/Jerusalem', {
            'input_type': 'videos', 'analysis_mode': 'video_traffic',
            'source_root': str(self.source_root), 'output_root': '',
        })
        self.origin = datetime(2026, 5, 17, 19, 4, tzinfo=timezone(timedelta(hours=3)))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def config():
        return {'detector': {'version': 'fixture-model', 'sha256': 'abc'},
            'tracker_policy': {'max_gap_seconds': [1, 2], 'minimum_iou': .2,
                'max_center_distance': .15, 'association_margin': .05,
                'compatible_label_groups': []}}

    class FakeDetector:
        def __init__(self):
            self.calls = 0

        def infer(self, rgb):
            self.calls += 1
            return [{'label': 'car', 'box': [.2 + self.calls * .002, .2, .1, .1]}]

    def test_source_snapshot_run_checkpoint_events_and_utc_are_durable(self):
        source = register_video(self.library, self.mission, 'clip.mkv',
            recording_start_utc=self.origin, origin_provenance='operator confirmed from recorder log')
        stored = source_path(self.library, self.mission, source['id'])
        self.assertEqual(stored.read_bytes(), self.original)
        self.assertEqual(source['sha256'], hashlib.sha256(self.original).hexdigest())
        duplicate_source = register_video(self.library, self.mission, 'clip.mkv',
            recording_start_utc=self.origin, origin_provenance='operator confirmed from recorder log')
        self.assertNotEqual(duplicate_source['id'], source['id'])
        self.assertEqual(duplicate_source['sha256'], source['sha256'])
        self.video_file.write_bytes(b'replaced after import')
        self.assertEqual(source_path(self.library, self.mission, source['id']).read_bytes(), self.original)

        run = create_run(self.library, self.mission, source['id'], self.config(), total_frames=2)
        self.assertEqual(run['status'], 'queued')
        self.assertEqual(run['config']['source_snapshot']['sha256'], source['sha256'])
        self.assertEqual(run['config']['source_snapshot']['origin_provenance'],
            'operator confirmed from recorder log')
        transition(self.library, self.mission, run['id'], 'running')

        tracker = VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1, 2),
            minimum_iou=.2, max_center_distance=.15, association_margin=.05), run_id=run['id'])
        tracker.update(VideoTimestamp(0, 1, 1000), [
            {'label': 'car', 'box': [.2, .2, .1, .1]}])
        mismatched_state = {**tracker.state(), 'run_id': 'another-run'}
        with self.assertRaisesRegex(ValueError, 'does not match this run'):
            checkpoint(self.library, self.mission, run['id'], frame_index=0,
                timestamp=VideoTimestamp(0, 1, 1000), tracker_state=mismatched_state)
        first = checkpoint(self.library, self.mission, run['id'], frame_index=0,
            timestamp=VideoTimestamp(0, 1, 1000), tracker_state=tracker.state())
        self.assertEqual(first['frames_done'], 1)

        tracker.update(VideoTimestamp(100, 1, 1000), [
            {'label': 'car', 'box': [.21, .2, .1, .1]}])
        final_tracks = tracker.finish()
        second = checkpoint(self.library, self.mission, run['id'], frame_index=1,
            timestamp=VideoTimestamp(100, 1, 1000), tracker_state=tracker.state(),
            closed_tracks=final_tracks)
        self.assertEqual(second['events_inserted'], 1)
        retry = checkpoint(self.library, self.mission, run['id'], frame_index=1,
            timestamp=VideoTimestamp(100, 1, 1000), tracker_state=tracker.state(),
            closed_tracks=final_tracks)
        self.assertTrue(retry['duplicate'])
        conflicting_state = {**tracker.state(), 'last_offset_seconds': [101, 1000]}
        with self.assertRaisesRegex(ValueError, 'conflicts with saved state'):
            checkpoint(self.library, self.mission, run['id'], frame_index=1,
                timestamp=VideoTimestamp(100, 1, 1000), tracker_state=conflicting_state,
                closed_tracks=final_tracks)
        with self.assertRaisesRegex(ValueError, 'exceeds the declared frame count'):
            checkpoint(self.library, self.mission, run['id'], frame_index=2,
                timestamp=VideoTimestamp(200, 1, 1000), tracker_state=tracker.state())
        completion = transition(self.library, self.mission, run['id'], 'completed', final_tracks=final_tracks)
        self.assertEqual(completion['events_inserted'], 0)

        saved_run = run_detail(self.library, self.mission, run['id'])
        self.assertEqual(saved_run['frames_done'], 2)
        self.assertEqual(saved_run['status'], 'completed')
        saved_events = events(self.library, self.mission, run['id'])
        self.assertEqual(len(saved_events), 1)
        self.assertEqual(len(events(self.library, self.mission, run['id'], limit=1, offset=0)), 1)
        with self.assertRaisesRegex(ValueError, 'between 1 and 100'):
            events(self.library, self.mission, run['id'], limit=101)
        item = saved_events[0]
        expected_origin = datetime(2026, 5, 17, 16, 4, tzinfo=timezone.utc)
        self.assertEqual(item['first_utc'], expected_origin.isoformat())
        self.assertEqual(item['last_utc'], (expected_origin + timedelta(milliseconds=100)).isoformat())
        self.assertEqual((item['first_offset_num'], item['first_offset_den']), (0, 1))
        self.assertEqual((item['last_offset_num'], item['last_offset_den']), (1, 10))
        self.assertEqual(item['review_status'], 'pending')

        reviewed = review_event(self.library, self.mission, item['id'], 0, 'Operator', 'accept')
        self.assertEqual(reviewed['revision'], 1)
        self.assertEqual(events(self.library, self.mission, run['id'])[0]['review_status'], 'accepted')
        with self.assertRaisesRegex(ValueError, 'changed'):
            review_event(self.library, self.mission, item['id'], 0, 'Operator', 'reject')
        with self.library.connect(self.mission) as conn:
            with self.assertRaisesRegex(sqlite3.IntegrityError, 'video events are immutable'):
                conn.execute("UPDATE video_events SET labels='[]' WHERE id=?", (item['id'],))

    def test_origin_requires_provenance_and_utc_is_never_guessed(self):
        with self.assertRaisesRegex(ValueError, 'provenance is required'):
            register_video(self.library, self.mission, 'clip.mkv', recording_start_utc=self.origin)
        source = register_video(self.library, self.mission, 'clip.mkv')
        self.assertIsNone(source['recording_start_utc'])
        self.assertIsNone(source['origin_provenance'])
        run = create_run(self.library, self.mission, source['id'], self.config())
        transition(self.library, self.mission, run['id'], 'running')
        tracker = VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1, 2),
            minimum_iou=.2, max_center_distance=.15, association_margin=.05), run_id=run['id'])
        tracker.update(VideoTimestamp(20, 1, 1000), [{'label': 'bus', 'box': [.1, .1, .2, .2]}])
        closed = tracker.finish()
        checkpoint(self.library, self.mission, run['id'], frame_index=0,
            timestamp=VideoTimestamp(20, 1, 1000), tracker_state=tracker.state(), closed_tracks=closed)
        saved = events(self.library, self.mission, run['id'])[0]
        self.assertIsNone(saved['first_utc'])
        self.assertEqual((saved['first_offset_num'], saved['first_offset_den']), (1, 50))

    def test_mission_isolation_and_run_state_transitions(self):
        other_root = self.root / 'other-videos'
        other_root.mkdir()
        other = self.library.create('Other', 'Latvia', 'Europe/Riga', {
            'input_type': 'videos', 'analysis_mode': 'video_traffic', 'source_root': str(other_root)})
        source = register_video(self.library, self.mission, 'clip.mkv')
        with self.assertRaisesRegex(ValueError, 'not found in this mission'):
            create_run(self.library, other, source['id'], self.config())
        run = create_run(self.library, self.mission, source['id'], self.config())
        transition(self.library, self.mission, run['id'], 'running')
        transition(self.library, self.mission, run['id'], 'paused')
        transition(self.library, self.mission, run['id'], 'running')
        transition(self.library, self.mission, run['id'], 'cancelled', error='operator stopped')
        transition(self.library, self.mission, run['id'], 'queued')
        with self.assertRaisesRegex(ValueError, 'Invalid video run transition'):
            transition(self.library, self.mission, run['id'], 'completed')

    def test_source_path_restrictions_and_event_correction_validation(self):
        with self.assertRaisesRegex(ValueError, 'relative to the configured source'):
            register_video(self.library, self.mission, str(self.video_file))
        source = register_video(self.library, self.mission, 'clip.mkv')
        run = create_run(self.library, self.mission, source['id'], self.config())
        transition(self.library, self.mission, run['id'], 'running')
        tracker = VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1, 2),
            minimum_iou=.2, max_center_distance=.15, association_margin=.05), run_id=run['id'])
        tracker.update(VideoTimestamp(0, 1, 30), [{'label': 'car', 'box': [.2, .2, .1, .1]}])
        closed = tracker.finish()
        checkpoint(self.library, self.mission, run['id'], frame_index=0,
            timestamp=VideoTimestamp(0, 1, 30), tracker_state=tracker.state(), closed_tracks=closed)
        event_id = events(self.library, self.mission, run['id'])[0]['id']
        with self.assertRaisesRegex(ValueError, 'non-empty label'):
            review_event(self.library, self.mission, event_id, 0, 'Operator', 'correct_class', {'label': ' '})
        corrected = review_event(self.library, self.mission, event_id, 0, 'Operator',
            'correct_class', {'label': 'heavy goods vehicle'})
        self.assertEqual(corrected['revision'], 1)
        self.assertEqual(events(self.library, self.mission, run['id'])[0]['review_status'], 'corrected')

    def test_source_registration_honors_flat_directory_setting(self):
        flat_mission = self.library.create('Flat video mission', 'Israel', 'Asia/Jerusalem', {
            'input_type': 'videos', 'analysis_mode': 'video_traffic',
            'source_root': str(self.source_root), 'include_subdirectories': False})
        nested = self.source_root / 'camera-1'
        nested.mkdir()
        (nested / 'clip.mkv').write_bytes(self.original)
        with self.assertRaisesRegex(ValueError, 'not configured to include video subdirectories'):
            register_video(self.library, flat_mission, 'camera-1/clip.mkv')

    def test_persisted_video_run_cancels_and_resumes_without_duplicate_frames_or_events(self):
        fixture = Path(__file__).with_name('create_vfr_fixture.py')
        subprocess.run([sys.executable, str(fixture), str(self.video_file)], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        source = register_video(self.library, self.mission, 'clip.mkv',
            recording_start_utc=self.origin, origin_provenance='operator confirmed')
        run = create_run(self.library, self.mission, source['id'], self.config(), total_frames=4)
        first_worker = self.FakeDetector()
        cancelled = execute_video_run(self.library, self.mission, run['id'], first_worker,
            should_cancel=lambda: first_worker.calls >= 1)
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertEqual(cancelled['frames_done'], 1)
        self.assertEqual(first_worker.calls, 1)

        second_worker = self.FakeDetector()
        completed = execute_video_run(self.library, self.mission, run['id'], second_worker)
        self.assertEqual(completed['status'], 'completed')
        self.assertEqual(completed['frames_done'], 4)
        self.assertEqual(second_worker.calls, 3)
        saved_events = events(self.library, self.mission, run['id'])
        self.assertEqual(len(saved_events), 1)
        self.assertEqual(saved_events[0]['observation_count'], 4)


if __name__ == '__main__':
    unittest.main()
