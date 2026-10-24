"""Exact media presentation-time arithmetic for video event timelines.

This module deliberately does not infer a recording's UTC origin. Callers must
supply an explicit, timezone-aware origin from approved media metadata or an
operator-confirmed value, and must retain the original PTS/time-base pair.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class VideoTimestamp:
    """A decoded frame's presentation timestamp in its stream time base."""

    pts: int
    time_base_num: int
    time_base_den: int

    def __post_init__(self):
        if isinstance(self.pts, bool) or not isinstance(self.pts, int):
            raise ValueError('PTS must be an integer tick count')
        if (isinstance(self.time_base_num, bool) or not isinstance(self.time_base_num, int)
                or isinstance(self.time_base_den, bool) or not isinstance(self.time_base_den, int)
                or self.time_base_num <= 0 or self.time_base_den <= 0):
            raise ValueError('Stream time base must be a positive rational number')

    @property
    def offset_seconds(self) -> Fraction:
        return Fraction(self.pts * self.time_base_num, self.time_base_den)

    def utc_at(self, recording_start_utc: datetime) -> datetime:
        """Return the UTC event time, flooring only at datetime's microsecond precision."""
        if not isinstance(recording_start_utc, datetime):
            raise ValueError('Recording start must be a timezone-aware datetime')
        if recording_start_utc.tzinfo is None or recording_start_utc.utcoffset() is None:
            raise ValueError('Recording start must be a timezone-aware datetime')
        origin = recording_start_utc.astimezone(timezone.utc)
        micros = self.offset_seconds * 1_000_000
        delta_microseconds = micros.numerator // micros.denominator
        return origin + timedelta(microseconds=delta_microseconds)


def utc_event_time(recording_start_utc: datetime, pts: int,
                   time_base_num: int, time_base_den: int) -> datetime:
    """Convert stream PTS to UTC without assuming a constant frame rate."""
    return VideoTimestamp(pts, time_base_num, time_base_den).utc_at(recording_start_utc)
