import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from road_traffic.video_inference import VehicleProposalAdapter
from road_traffic.video_pipeline import VideoAnalysisPipeline, VideoFrameAnalysis
from road_traffic.video_decoder import DecodedVideoFrame, VideoMetadata
from road_traffic.video_timeline import VideoTimestamp
from road_traffic.tracking import TrackerPolicy, VideoTrackManager
from road_traffic.video_store import _frame_detections


LIGHT = {'kind':'light','label':'possible paired vehicle lights','box':[.35,.4,.12,.06],
    'light_subtype':'uncertain_light','pairing':'hypothesis',
    'reasons':['geometry-only pairing; subtype uncertain']}


class FakeDetector:
    def __init__(self):
        self.night_top_calls=0
        self.infer_calls=0

    def infer(self, rgb):
        self.infer_calls+=1
        return [{'kind':'candidate','label':'car','box':[.2,.2,.2,.2],'score':.8}]

    def infer_night_top(self, rgb):
        self.night_top_calls+=1
        return [{'kind':'candidate','label':'car','box':[.1,.1,.1,.1],'score':.6}]


class VideoNightAdapterTests(unittest.TestCase):
    def test_night_frame_keeps_light_hypotheses_separate_from_vehicle_detections(self):
        detector=FakeDetector()
        adapter=VehicleProposalAdapter(detector,target_labels=['car'],night_lights=True)
        with patch('road_traffic.detector.illumination',return_value={
                'low_light':True,'scene_period':'night_like'}), \
             patch('road_traffic.detector.suppress',side_effect=lambda items,threshold=.45:items), \
             patch('road_traffic.night.propose',return_value=([LIGHT],{})) as propose:
            vehicles=adapter.infer(np.zeros((48,64,3),dtype=np.uint8))
            lights=adapter.take_review_hypotheses()

        self.assertEqual(detector.night_top_calls,1)
        self.assertEqual(len(vehicles),2)
        self.assertTrue(all('kind' not in item for item in vehicles))
        self.assertEqual(lights,[LIGHT])
        propose.assert_called_once()
        self.assertEqual(propose.call_args.args[1],None)

    def test_day_frame_has_no_night_pass_and_clears_previous_hypotheses(self):
        detector=FakeDetector()
        adapter=VehicleProposalAdapter(detector,target_labels=['car'],night_lights=True)
        with patch('road_traffic.detector.illumination',return_value={
                'low_light':False,'scene_period':'day_like'}), \
             patch('road_traffic.detector.suppress',side_effect=lambda items,threshold=.45:items), \
             patch('road_traffic.night.propose') as propose:
            adapter.infer(np.zeros((48,64,3),dtype=np.uint8))
            self.assertEqual(adapter.take_review_hypotheses(),[])
        self.assertEqual(detector.night_top_calls,0)
        propose.assert_not_called()

    def test_legacy_video_run_can_disable_new_night_adapter(self):
        detector=FakeDetector()
        adapter=VehicleProposalAdapter(detector,target_labels=['car'],night_lights=False)
        with patch('road_traffic.detector.illumination') as illumination:
            adapter.infer(np.zeros((48,64,3),dtype=np.uint8))
        illumination.assert_not_called()
        self.assertEqual(adapter.take_review_hypotheses(),[])

    def test_video_pipeline_keeps_review_lights_out_of_tracker(self):
        class Decoder:
            def __init__(self,path):pass
            metadata=VideoMetadata('fixture','rawvideo',16,16,0,1,1000,25,1,1,40)
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def frames(self):
                yield DecodedVideoFrame(0,VideoTimestamp(0,1,1000),
                    np.zeros((16,16,3),dtype=np.uint8))
        class DetectorWithReview:
            def infer(self, rgb):
                return [{'label':'car','box':[.2,.2,.2,.2]}]
            def take_review_hypotheses(self):
                return [dict(LIGHT)]
        tracker=VideoTrackManager(TrackerPolicy(max_gap_seconds=.5,minimum_iou=.2,
            max_center_distance=.15,association_margin=.05),run_id='video-night')
        records=list(VideoAnalysisPipeline(DetectorWithReview(),tracker,decoder_factory=Decoder).analyze('fixture'))
        frame=next(item for item in records if isinstance(item,VideoFrameAnalysis))
        self.assertEqual(len(frame.detections),1)
        self.assertEqual(len(frame.assignments),1)
        self.assertEqual(len(frame.review_hypotheses),1)
        self.assertEqual(frame.review_hypotheses[0]['kind'],'light')
        self.assertEqual(frame.progress.detections_in_frame,1)

    def test_frame_serialization_keeps_lights_untracked(self):
        vehicle={'label':'car','box':[.2,.2,.2,.2]}
        assignment=SimpleNamespace(detection_index=0,track=SimpleNamespace(track_id='run:car-1'))
        saved=_frame_detections([vehicle],[assignment],[dict(LIGHT)])
        self.assertEqual(saved[0]['track_id'],'run:car-1')
        self.assertEqual(saved[1]['kind'],'light')
        self.assertIsNone(saved[1]['track_id'])
        self.assertEqual(saved[1]['light_subtype'],'uncertain_light')
        self.assertEqual(len(saved),2)


if __name__=='__main__':
    unittest.main()
