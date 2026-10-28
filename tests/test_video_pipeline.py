import tempfile
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from road_traffic.tracking import TrackerPolicy, VideoTrackManager
from road_traffic.video_decoder import IsolatedVideoDecoder, VideoDecodeError
from road_traffic.video_pipeline import (VideoAnalysisCancelled, VideoAnalysisPipeline,
    VideoFrameAnalysis, VideoTrackClosure, SceneCutDetector)
from road_traffic.video_timeline import VideoTimestamp
from road_traffic.video_decoder import DecodedVideoFrame, VideoMetadata


def make_vfr_video(path):
    fixture = Path(__file__).with_name('create_vfr_fixture.py')
    subprocess.run([sys.executable, str(fixture), str(path)], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def infer(self, rgb):
        self.calls += 1
        self.assert_image_shape = rgb.shape == (48, 64, 3)
        return [{'kind': 'candidate', 'label': 'car', 'box': [.2 + self.calls * .002, .2, .1, .1]}]


def pipeline(detector=None):
    detector = detector or FakeDetector()
    tracker = VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1, 2),
        minimum_iou=.2, max_center_distance=.15, association_margin=.05), run_id='video-run')
    return VideoAnalysisPipeline(detector, tracker), detector


class VideoAnalysisPipelineTests(unittest.TestCase):
    def test_scene_cut_detector_uses_color_as_well_as_brightness(self):
        import numpy as np
        detector=SceneCutDetector(.65)
        red=np.zeros((12,12,3),dtype=np.uint8);red[:,:,0]=255
        blue=np.zeros((12,12,3),dtype=np.uint8);blue[:,:,2]=255
        self.assertFalse(detector.observe(red))
        self.assertTrue(detector.observe(blue))

    def test_detected_scene_cut_closes_tracks_and_never_associates_across_it(self):
        import numpy as np
        class Decoder:
            metadata=VideoMetadata('fixture','rawvideo',16,16,0,1,1000,25,1,4,120)
            def __init__(self,path):
                self.images=[np.zeros((16,16,3),dtype=np.uint8) for _ in range(2)]+[
                    np.full((16,16,3),255,dtype=np.uint8) for _ in range(2)]
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def frames(self):
                for index,image in enumerate(self.images):
                    yield DecodedVideoFrame(index,VideoTimestamp(index*40,1,1000),image)
        class StableDetector:
            def infer(self,image):return [{'label':'car','box':[.2,.2,.1,.1]}]
        tracker=VideoTrackManager(TrackerPolicy(max_gap_seconds=Fraction(1,2),minimum_iou=.2,
            max_center_distance=.15,association_margin=.05,scene_cut_threshold=.65),run_id='cut-run')
        records=list(VideoAnalysisPipeline(StableDetector(),tracker,decoder_factory=Decoder).analyze('fixture'))
        frames=[item for item in records if isinstance(item,VideoFrameAnalysis)]
        self.assertEqual([frame.scene_cut for frame in frames],[False,False,True,False])
        self.assertEqual(frames[0].assignments[0].track.track_id,frames[1].assignments[0].track.track_id)
        self.assertNotEqual(frames[1].assignments[0].track.track_id,frames[2].assignments[0].track.track_id)
        self.assertEqual(frames[2].closed_tracks[0].track_id,frames[1].assignments[0].track.track_id)
        self.assertIn('scene_cut_boundary',frames[2].closed_tracks[0].uncertainty_reasons)
        self.assertTrue(frames[3].assignments[0].matched_existing)

    def test_resumed_scene_cut_uses_saved_preceding_frame_for_detection(self):
        import numpy as np
        class Decoder:
            metadata=VideoMetadata('fixture','rawvideo',16,16,0,1,1000,25,1,2,80)
            def __init__(self,path):
                self.images=[np.zeros((16,16,3),dtype=np.uint8),np.full((16,16,3),255,dtype=np.uint8)]
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def frames(self):
                for index,image in enumerate(self.images):
                    yield DecodedVideoFrame(index,VideoTimestamp(index*40,1,1000),image)
        class StableDetector:
            def infer(self,image):return [{'label':'car','box':[.2,.2,.1,.1]}]
        configured=TrackerPolicy(max_gap_seconds=Fraction(1,2),minimum_iou=.2,
            max_center_distance=.15,association_margin=.05,scene_cut_threshold=.65)
        original=VideoTrackManager(configured,run_id='resumed-cut')
        prior=original.update(VideoTimestamp(0,1,1000),[{'label':'car','box':[.2,.2,.1,.1]}]).assignments[0]
        resumed=VideoTrackManager(configured,run_id='resumed-cut',state=original.state())
        records=list(VideoAnalysisPipeline(StableDetector(),resumed,decoder_factory=Decoder).analyze(
            'fixture',start_frame_index=1,expected_previous_timestamp=VideoTimestamp(0,1,1000)))
        frame=next(item for item in records if isinstance(item,VideoFrameAnalysis))
        self.assertTrue(frame.scene_cut)
        self.assertEqual(frame.closed_tracks[0].track_id,prior.track.track_id)
        self.assertNotEqual(frame.assignments[0].track.track_id,prior.track.track_id)

    def test_isolated_decode_runs_detector_tracker_and_progress_incrementally(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'variable-rate.mkv'
            make_vfr_video(path)
            processor, detector = pipeline()
            origin = datetime(2026, 5, 17, 16, 4, tzinfo=timezone.utc)
            progress = []
            decoded_frames=[]
            records = list(processor.analyze(path, recording_start_utc=origin,
                progress_callback=progress.append,frame_callback=decoded_frames.append))
            frames = [record for record in records if isinstance(record, VideoFrameAnalysis)]
            closure = records[-1]
            self.assertIsInstance(closure, VideoTrackClosure)
            self.assertEqual(detector.calls, 4)
            self.assertEqual([frame.index for frame in decoded_frames],[0,1,2,3])
            self.assertTrue(all(frame.image.shape==(48,64,3) for frame in decoded_frames))
            self.assertEqual([item.timestamp.pts for item in frames], [0, 40, 160, 200])
            self.assertEqual([item.utc_time for item in frames], [
                origin, datetime(2026, 5, 17, 16, 4, 0, 40000, tzinfo=timezone.utc),
                datetime(2026, 5, 17, 16, 4, 0, 160000, tzinfo=timezone.utc),
                datetime(2026, 5, 17, 16, 4, 0, 200000, tzinfo=timezone.utc)])
            self.assertEqual(len({item.assignments[0].track.track_id for item in frames}), 1)
            self.assertEqual([item.frames_processed for item in progress], [1, 2, 3, 4])
            self.assertEqual(len(closure.tracks), 1)
            self.assertEqual(closure.tracks[0].observation_count, 4)

    def test_unknown_origin_leaves_utc_unset_and_cancel_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'video.mkv'
            make_vfr_video(path)
            processor, detector = pipeline()
            analysis = processor.analyze(path)
            frames = (record for record in analysis if isinstance(record, VideoFrameAnalysis))
            first = next(frames)
            self.assertIsNone(first.utc_time)
            self.assertTrue(detector.assert_image_shape)
            analysis.close()
            processor, detector = pipeline()
            def cancel_after_first():
                return detector.calls >= 1
            with self.assertRaises(VideoAnalysisCancelled):
                list(processor.analyze(path, should_cancel=cancel_after_first))
            self.assertEqual(detector.calls, 1)
            checkpoint = processor.tracker.state()
            self.assertEqual(len(checkpoint['active']), 1)
            self.assertEqual(checkpoint['active'][0]['observation_count'], 1)

    def test_isolated_decoder_reports_decode_error_and_rejects_single_use(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'not-video.bin'
            path.write_bytes(b'no container')
            with self.assertRaisesRegex(VideoDecodeError, 'VideoDecodeError'):
                with IsolatedVideoDecoder(path):
                    pass
            path = Path(folder) / 'video.mkv'
            make_vfr_video(path)
            with IsolatedVideoDecoder(path) as decoder:
                list(decoder.frames())
                with self.assertRaisesRegex(RuntimeError, 'only be consumed once'):
                    list(decoder.frames())

    def test_resume_skips_saved_frames_and_checks_the_source_timeline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'variable-rate.mkv'
            make_vfr_video(path)
            detector = FakeDetector()
            processor, _ = pipeline(detector)
            first = processor.tracker.update(VideoTimestamp(0, 1, 1000),
                [{'label': 'car', 'box': [.202, .2, .1, .1]}]).assignments[0]
            expected = VideoTimestamp(0, 1, 1000)
            records = list(processor.analyze(path, start_frame_index=1,
                expected_previous_timestamp=expected))
            frames = [record for record in records if isinstance(record, VideoFrameAnalysis)]
            self.assertEqual([item.frame_index for item in frames], [1, 2, 3])
            self.assertEqual(detector.calls, 3)
            self.assertTrue(all(item.assignments[0].track.track_id == first.track.track_id for item in frames))

            detector = FakeDetector()
            invalid, _ = pipeline(detector)
            with self.assertRaisesRegex(ValueError, 'timeline changed'):
                list(invalid.analyze(path, start_frame_index=1,
                    expected_previous_timestamp=expected.__class__(1, 1, 1000)))
            self.assertEqual(detector.calls, 0)


if __name__ == '__main__':
    unittest.main()
