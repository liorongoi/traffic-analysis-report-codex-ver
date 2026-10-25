import hashlib
from datetime import datetime, timezone, timedelta
from fractions import Fraction
from pathlib import Path
import json
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from fractions import Fraction

import av
import numpy as np
from PIL import Image

from road_traffic.store import Library
from road_traffic.tracking import TrackerPolicy, VideoTrackManager
from road_traffic.video_store import (checkpoint, create_run, events, register_video,
    execute_video_run, event_thumbnail, frame_overlay, review_event, run_detail, source_path, transition)
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

    def test_stored_video_tampering_is_detected_even_after_digest_cache(self):
        source = register_video(self.library, self.mission, 'clip.mkv')
        self.assertEqual(source['metadata']['media_probe']['status'], 'unavailable')
        stored = source_path(self.library, self.mission, source['id'])
        stored.write_bytes(b'changed source bytes')
        with self.assertRaisesRegex(ValueError, 'SHA-256 check'):
            source_path(self.library, self.mission, source['id'])

    def test_video_registration_snapshots_probe_metadata_without_claiming_utc(self):
        fixture = Path(__file__).with_name('create_vfr_fixture.py')
        subprocess.run([sys.executable, str(fixture), str(self.video_file)], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        source = register_video(self.library, self.mission, 'clip.mkv')
        probe = source['metadata']['media_probe']
        self.assertEqual(probe['status'], 'available')
        self.assertEqual(probe['metadata']['codec_name'], 'mpeg4')
        self.assertEqual((probe['metadata']['width'], probe['metadata']['height']), (64, 48))
        self.assertIsNone(source['recording_start_utc'])
        self.assertIsNone(source['origin_provenance'])

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

    def test_event_preview_uses_verified_frame_and_marks_saved_vehicle_box(self):
        with av.open(str(self.video_file), mode='w', format='matroska') as output:
            stream = output.add_stream('mpeg4', rate=25)
            stream.width, stream.height = 64, 48
            stream.pix_fmt = 'yuv420p'
            stream.time_base = Fraction(1, 1000)
            for pts, value in ((0, 20), (40, 80), (160, 140)):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), value, dtype=np.uint8), format='rgb24')
                frame.pts = pts
                frame.time_base = Fraction(1, 1000)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        source = register_video(self.library, self.mission, 'clip.mkv')
        run = create_run(self.library, self.mission, source['id'], self.config())
        transition(self.library, self.mission, run['id'], 'running')
        tracker = VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1, 2),
            minimum_iou=.2, max_center_distance=.15, association_margin=.05), run_id=run['id'])
        tracker.update(VideoTimestamp(160, 1, 1000), [
            {'label': 'car', 'box': [.2, .25, .3, .4]}])
        transition(self.library, self.mission, run['id'], 'completed', final_tracks=tracker.finish())
        saved = events(self.library, self.mission, run['id'])
        self.assertEqual(saved[0]['representative_box'], [.2, .25, .3, .4])
        preview = event_thumbnail(self.library, self.mission, saved[0]['id'])
        self.assertTrue(preview.startswith(b'\xff\xd8'))
        from io import BytesIO
        with Image.open(BytesIO(preview)) as image:
            self.assertEqual(image.size, (64, 48))

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

    def test_frame_overlay_returns_nearest_saved_detection_without_extrapolating(self):
        source = register_video(self.library, self.mission, 'clip.mkv')
        run = create_run(self.library, self.mission, source['id'], self.config(), total_frames=2)
        transition(self.library, self.mission, run['id'], 'running')
        tracker = VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1, 2),
            minimum_iou=.2, max_center_distance=.15, association_margin=.05), run_id=run['id'])
        first_time = VideoTimestamp(0, 1, 1000)
        first_detection = {'label':'car','box':[.2,.2,.1,.1],'score':.83}
        first = tracker.update(first_time, [first_detection])
        checkpoint(self.library,self.mission,run['id'],frame_index=0,timestamp=first_time,
            tracker_state=tracker.state(),detections=[first_detection],assignments=first.assignments)
        second_time = VideoTimestamp(1000, 1, 1000)
        tracker.update(second_time, [])
        checkpoint(self.library,self.mission,run['id'],frame_index=1,timestamp=second_time,
            tracker_state=tracker.state(),detections=[])
        near_first=frame_overlay(self.library,self.mission,run['id'],.4)
        self.assertEqual(near_first['frame']['pts'],0)
        self.assertEqual(near_first['detections'][0]['label'],'car')
        self.assertEqual(near_first['detections'][0]['score'],.83)
        self.assertEqual(near_first['detections'][0]['track_id'],first.assignments[0].track.track_id)
        empty_frame=frame_overlay(self.library,self.mission,run['id'],1)
        self.assertEqual(empty_frame['frame']['index'],1)
        self.assertEqual(empty_frame['detections'],[])
        outside=frame_overlay(self.library,self.mission,run['id'],1.1)
        self.assertIsNone(outside['frame'])
        self.assertEqual(outside['detections'],[])
        with self.assertRaisesRegex(ValueError,'non-negative'):
            frame_overlay(self.library,self.mission,run['id'],-.1)

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

    def test_video_event_timing_correction_snaps_to_saved_track_frames(self):
        source = register_video(self.library, self.mission, 'clip.mkv',
            recording_start_utc=self.origin, origin_provenance='confirmed recorder log')
        run = create_run(self.library, self.mission, source['id'], self.config())
        event_id = 'timing-correction-event'
        track_id = f"{run['id']}:track-0001"
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (event_id,self.mission,run['id'],source['id'],track_id,0,1,2,3,1,2,0,1,3,2,
                 self.origin.isoformat(),(self.origin+timedelta(seconds=1.5)).isoformat(),
                 '["car"]',4,0,'[0.2,0.2,0.1,0.1]','[]','2026-10-25T00:00:00+00:00'))
            for frame_index in range(4):
                conn.execute('''INSERT INTO video_frame_detections(mission_id,run_id,source_id,frame_index,
                    pts,time_base_num,time_base_den,offset_seconds,detections,created)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (self.mission,run['id'],source['id'],frame_index,frame_index,1,2,
                     frame_index/2,json.dumps([{'track_id':track_id,'label':'car','box':[.2,.2,.1,.1]}]),
                     '2026-10-25T00:00:00+00:00'))
        with self.assertRaisesRegex(ValueError, 'within the original event interval'):
            review_event(self.library,self.mission,event_id,0,'Operator','correct_timing',
                {'first_offset_seconds':-0.1,'last_offset_seconds':1.5})
        with self.assertRaisesRegex(ValueError, 'start must not follow'):
            review_event(self.library,self.mission,event_id,0,'Operator','correct_timing',
                {'first_offset_seconds':1.2,'last_offset_seconds':0.6})
        saved = review_event(self.library,self.mission,event_id,0,'Operator','correct_timing',
            {'first_offset_seconds':'0.6','last_offset_seconds':'1.4'})
        self.assertEqual(saved['revision'],1)
        corrected = events(self.library,self.mission,run['id'])[0]
        self.assertEqual(corrected['review_status'],'corrected')
        self.assertEqual((corrected['first_pts'],corrected['last_pts']),(1,3))
        self.assertEqual((corrected['first_offset_num'],corrected['first_offset_den']),(1,2))
        self.assertEqual((corrected['last_offset_num'],corrected['last_offset_den']),(3,2))
        self.assertEqual(corrected['first_utc'],
            (self.origin+timedelta(seconds=.5)).astimezone(timezone.utc).isoformat())
        self.assertEqual((corrected['original_first_offset_seconds'],
            corrected['original_last_offset_seconds']),(0.0,1.5))
        self.assertEqual(corrected['payload']['first_frame_index'],1)
        self.assertEqual(corrected['payload']['last_frame_index'],3)
        reviewed = review_event(self.library,self.mission,event_id,1,'Operator','accept')
        self.assertEqual(reviewed['revision'],2)
        accepted = events(self.library,self.mission,run['id'])[0]
        self.assertEqual(accepted['review_status'],'accepted')
        self.assertEqual((accepted['first_pts'],accepted['last_pts']),(1,3))
        self.assertEqual(accepted['timing_payload']['first_frame_index'],1)

    def test_video_track_merge_projects_one_logical_event_and_can_be_cleared(self):
        source=register_video(self.library,self.mission,'clip.mkv')
        run=create_run(self.library,self.mission,source['id'],self.config())
        def add_event(event_id,track,start,end,label):
            with self.library.connect(self.mission) as conn:
                conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                    first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                    first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                    observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (event_id,self.mission,run['id'],source['id'],track,start,1,1,end,1,1,start,1,end,1,
                     None,None,json.dumps([label]),2,0,'[0.1,0.1,0.2,0.2]','[]','2026-10-25T00:00:00Z'))
        add_event('merge-source','track-a',0,2,'car')
        add_event('merge-target','track-b',2,4,'bus')
        corrected=review_event(self.library,self.mission,'merge-source',0,'Operator','merge_tracks',
            {'target_event_id':'merge-target'})
        self.assertEqual(corrected['action'],'merge_tracks')
        projected=events(self.library,self.mission,run['id'])
        self.assertEqual(len(projected),1)
        self.assertEqual(projected[0]['id'],'merge-target')
        self.assertEqual(projected[0]['merged_event_ids'],['merge-source','merge-target'])
        self.assertEqual(projected[0]['labels'],['bus','car'])
        self.assertEqual((projected[0]['first_offset_num'],projected[0]['last_offset_num']),(0,4))
        review_event(self.library,self.mission,'merge-source',1,'Operator','clear_identity_correction')
        self.assertEqual({item['id'] for item in events(self.library,self.mission,run['id'])},
            {'merge-source','merge-target'})

    def test_video_track_split_creates_two_corrected_children_with_saved_observations(self):
        source=register_video(self.library,self.mission,'clip.mkv')
        run=create_run(self.library,self.mission,source['id'],self.config())
        event_id='split-parent';track_id='track-split'
        with self.library.connect(self.mission) as conn:
            conn.execute('''INSERT INTO video_events(id,mission_id,run_id,source_id,track_id,
                first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
                first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
                observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (event_id,self.mission,run['id'],source['id'],track_id,0,1,1,3,1,1,0,1,3,1,
                 None,None,'["car"]',4,0,'[0.1,0.1,0.2,0.2]','[]','2026-10-25T00:00:00Z'))
            for frame_index in range(4):
                conn.execute('''INSERT INTO video_frame_detections(mission_id,run_id,source_id,frame_index,
                    pts,time_base_num,time_base_den,offset_seconds,detections,created)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (self.mission,run['id'],source['id'],frame_index,frame_index,1,1,float(frame_index),
                     json.dumps([{'track_id':track_id,'label':'car','box':[.1,.1,.2,.2]}]),'2026-10-25T00:00:00Z'))
        with self.assertRaisesRegex(ValueError,'saved observation'):
            review_event(self.library,self.mission,event_id,0,'Operator','split_track',{'frame_index':8})
        review_event(self.library,self.mission,event_id,0,'Operator','split_track',{'frame_index':2})
        children=events(self.library,self.mission,run['id'])
        self.assertEqual(len(children),2)
        self.assertEqual([item['first_offset_num'] for item in children],[0,2])
        self.assertEqual([item['last_offset_num'] for item in children],[1,3])
        self.assertEqual([item['observation_count'] for item in children],[2,2])
        self.assertEqual(children[0]['source_event_id'],event_id)
        self.assertNotEqual(children[0]['id'],children[1]['id'])

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
