import json
import unittest
from fractions import Fraction

from road_traffic.tracking import TrackerPolicy, VideoTrackManager
from road_traffic.video_timeline import VideoTimestamp


def timestamp(milliseconds):
    return VideoTimestamp(milliseconds, 1, 1000)


def detection(x, label='car', *, y=.2, width=.1, height=.1):
    return {'label': label, 'box': [x, y, width, height]}


def policy(**changes):
    values = {'max_gap_seconds': Fraction(1, 2), 'minimum_iou': .2,
        'max_center_distance': .12, 'association_margin': .05}
    values.update(changes)
    return TrackerPolicy(**values)


class VideoTrackManagerTests(unittest.TestCase):
    def test_same_object_is_associated_and_gap_closes_track(self):
        manager = VideoTrackManager(policy(), run_id='run-a')
        first = manager.update(timestamp(0), [detection(.20)]).assignments[0]
        second = manager.update(timestamp(40), [detection(.22)]).assignments[0]
        self.assertFalse(first.matched_existing)
        self.assertTrue(second.matched_existing)
        self.assertEqual(first.track.track_id, second.track.track_id)
        self.assertEqual(second.track.observation_count, 2)

        result = manager.update(timestamp(600), [detection(.80)])
        self.assertEqual([item.track_id for item in result.closed_tracks], [first.track.track_id])
        self.assertNotEqual(result.assignments[0].track.track_id, first.track.track_id)

    def test_chunk_checkpoint_preserves_identity_and_requires_same_policy(self):
        configured = policy()
        original = VideoTrackManager(configured, run_id='recording-1')
        first = original.update(timestamp(0), [detection(.3)]).assignments[0]
        checkpoint = json.loads(json.dumps(original.state()))
        restored = VideoTrackManager(configured, run_id='recording-1', state=checkpoint)
        next_result = restored.update(timestamp(40), [detection(.31)]).assignments[0]
        self.assertTrue(next_result.matched_existing)
        self.assertEqual(next_result.track.track_id, first.track.track_id)
        self.assertEqual(next_result.track.observation_count, 2)
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            VideoTrackManager(policy(max_gap_seconds=Fraction(1, 1)), run_id='recording-1', state=checkpoint)
        checkpoint['next_id'] = 1
        with self.assertRaisesRegex(ValueError, 'reuse'):
            VideoTrackManager(configured, run_id='recording-1', state=checkpoint)

    def test_ambiguous_competing_tracks_remain_uncertain(self):
        manager = VideoTrackManager(policy(max_center_distance=.25), run_id='ambiguous')
        manager.update(timestamp(0), [detection(.2), detection(.6)])
        result = manager.update(timestamp(40), [detection(.4)])
        assignment = result.assignments[0]
        self.assertFalse(assignment.matched_existing)
        self.assertTrue(assignment.association_uncertain)
        self.assertIn('competing_tracks', assignment.track.uncertainty_reasons)

    def test_two_detections_do_not_claim_the_same_track(self):
        manager = VideoTrackManager(policy(), run_id='collision')
        original = manager.update(timestamp(0), [detection(.3)]).assignments[0]
        result = manager.update(timestamp(40), [detection(.29, width=.08), detection(.31, width=.08)])
        self.assertEqual(len({item.track.track_id for item in result.assignments}), 2)
        self.assertTrue(all(item.association_uncertain for item in result.assignments))
        self.assertTrue(all('competing_detections' in item.track.uncertainty_reasons
                            for item in result.assignments))
        self.assertEqual(original.track.observation_count, 1)

    def test_configured_label_transition_keeps_one_track(self):
        manager = VideoTrackManager(policy(compatible_label_groups=(frozenset({'unresolved light', 'truck'}),)),
            run_id='transition')
        light = manager.update(timestamp(0), [detection(.4, 'unresolved light')]).assignments[0]
        vehicle = manager.update(timestamp(40), [detection(.41, 'truck')]).assignments[0]
        self.assertTrue(vehicle.matched_existing)
        self.assertEqual(vehicle.track.track_id, light.track.track_id)
        self.assertEqual(vehicle.track.labels, ('truck', 'unresolved light'))

    def test_input_time_and_boxes_are_validated(self):
        manager = VideoTrackManager(policy(), run_id='validation')
        manager.update(timestamp(50), [detection(.2)])
        with self.assertRaisesRegex(ValueError, 'monotonic'):
            manager.update(timestamp(40), [])
        with self.assertRaisesRegex(ValueError, 'inside normalized'):
            manager.update(timestamp(60), [{'label': 'car', 'box': [.95, .1, .1, .1]}])
        manager.update(timestamp(55), [detection(.21)])

    def test_thresholds_must_be_explicit(self):
        with self.assertRaises(TypeError):
            VideoTrackManager(run_id='missing-policy')
        with self.assertRaisesRegex(ValueError, 'positive'):
            policy(max_gap_seconds=Fraction(0))


if __name__ == '__main__':
    unittest.main()
