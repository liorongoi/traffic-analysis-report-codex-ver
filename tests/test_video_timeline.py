import unittest
from datetime import datetime, timedelta, timezone
from fractions import Fraction

from road_traffic.video_timeline import VideoTimestamp, utc_event_time


class VideoTimelineTests(unittest.TestCase):
    def test_variable_pts_gaps_are_preserved_instead_of_using_frame_rate(self):
        origin = datetime(2026, 5, 17, 16, 4, tzinfo=timezone.utc)
        frames = [VideoTimestamp(pts, 1, 90_000) for pts in (0, 3_600, 10_800)]
        self.assertEqual([f.offset_seconds for f in frames], [Fraction(0), Fraction(1, 25), Fraction(3, 25)])
        times = [f.utc_at(origin) for f in frames]
        self.assertEqual(times[1] - times[0], timedelta(milliseconds=40))
        self.assertEqual(times[2] - times[1], timedelta(milliseconds=80))

    def test_submicrosecond_offsets_keep_raw_pts_and_floor_datetime(self):
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        frame = VideoTimestamp(1, 1, 3_000_000)
        self.assertEqual(frame.offset_seconds, Fraction(1, 3_000_000))
        self.assertEqual(frame.utc_at(origin), origin)
        self.assertEqual(frame.pts, 1)

    def test_non_utc_aware_origin_is_normalized_to_utc(self):
        origin = datetime(2026, 1, 1, 3, tzinfo=timezone(timedelta(hours=3)))
        self.assertEqual(utc_event_time(origin, 90, 1, 90), datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc))

    def test_rejects_missing_origin_and_invalid_timebase_or_pts(self):
        with self.assertRaisesRegex(ValueError, 'timezone-aware'):
            VideoTimestamp(0, 1, 30).utc_at(datetime(2026, 1, 1))
        with self.assertRaisesRegex(ValueError, 'positive rational'):
            VideoTimestamp(1, 0, 30)
        with self.assertRaisesRegex(ValueError, 'integer tick'):
            VideoTimestamp(True, 1, 30)
