"""Incremental detector-to-tracker pipeline for one video recording."""
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
import time

from .video_decoder import IsolatedVideoDecoder
from .video_timeline import VideoTimestamp


class VideoAnalysisCancelled(RuntimeError):
    """Raised after the caller requests cancellation between decoded frames."""


@dataclass(frozen=True, slots=True)
class VideoProgress:
    frames_processed: int
    total_frames: int | None
    video_position_seconds: Fraction
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    detections_in_frame: int
    active_tracks: int


@dataclass(frozen=True, slots=True)
class VideoFrameAnalysis:
    frame_index: int
    timestamp: VideoTimestamp
    utc_time: datetime | None
    detections: tuple[dict, ...]
    assignments: tuple
    closed_tracks: tuple
    progress: VideoProgress


@dataclass(frozen=True, slots=True)
class VideoTrackClosure:
    tracks: tuple
    reason: str = 'recording_end'


class VideoAnalysisPipeline:
    """Stream decoded frames through a detector and explicit-policy tracker.

    This is an in-process processing component. It never buffers an entire
    recording: each yielded result contains detections and track metadata, not
    a retained image. The consumer should persist closed tracks as they arrive.
    """

    def __init__(self, detector, tracker, *, decoder_factory=IsolatedVideoDecoder):
        if not callable(getattr(detector, 'infer', None)):
            raise TypeError('detector must provide infer(rgb_frame)')
        if not callable(getattr(tracker, 'update', None)) or not callable(getattr(tracker, 'finish', None)):
            raise TypeError('tracker must provide update(timestamp, detections) and finish()')
        self.detector = detector
        self.tracker = tracker
        self.decoder_factory = decoder_factory
        self.metadata = None

    def analyze(self, path, *, recording_start_utc=None, should_cancel=None,
                progress_callback=None, frame_callback=None, start_frame_index=0,
                expected_previous_timestamp=None):
        """Yield per-frame analyses and a final track-closure record.

        `recording_start_utc` must be a timezone-aware origin confirmed from
        source metadata or by the operator. A missing value leaves UTC unset.
        Cancellation is checked between frames; tracker state remains available
        from `self.tracker.state()` for a future durable worker/checkpoint.
        """
        if recording_start_utc is not None:
            if (not isinstance(recording_start_utc, datetime)
                    or recording_start_utc.tzinfo is None
                    or recording_start_utc.utcoffset() is None):
                raise ValueError('Recording origin must be a timezone-aware datetime')
        if should_cancel is not None and not callable(should_cancel):
            raise TypeError('should_cancel must be callable')
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError('progress_callback must be callable')
        if frame_callback is not None and not callable(frame_callback):
            raise TypeError('frame_callback must be callable')
        if (isinstance(start_frame_index, bool) or not isinstance(start_frame_index, int)
                or start_frame_index < 0):
            raise ValueError('start_frame_index must be a non-negative integer')
        if start_frame_index == 0 and expected_previous_timestamp is not None:
            raise ValueError('A fresh video run cannot include a previous-frame checkpoint')
        if start_frame_index > 0 and not isinstance(expected_previous_timestamp, VideoTimestamp):
            raise ValueError('Resuming video analysis requires its saved previous-frame timestamp')

        started = time.monotonic()
        with self.decoder_factory(path) as decoder:
            metadata = decoder.metadata
            self.metadata = metadata
            frame_count = metadata.frame_count if metadata.frame_count > 0 else None
            verified_previous = start_frame_index == 0
            for decoded in decoder.frames():
                if decoded.index < start_frame_index:
                    if decoded.index == start_frame_index - 1:
                        if decoded.timestamp != expected_previous_timestamp:
                            raise ValueError('Video source timeline changed since the saved checkpoint')
                        verified_previous = True
                    continue
                if should_cancel is not None and should_cancel():
                    raise VideoAnalysisCancelled('Video analysis cancelled between frames')
                if frame_callback is not None:
                    frame_callback(decoded)
                detections = self.detector.infer(decoded.image)
                if not isinstance(detections, (list, tuple)):
                    raise ValueError('Detector must return a list of detection mappings')
                tracking = self.tracker.update(decoded.timestamp, detections)
                elapsed = max(0.0, time.monotonic() - started)
                processed = decoded.index + 1
                processed_this_session = processed - start_frame_index
                eta = None
                if (frame_count is not None and processed < frame_count
                        and processed_this_session > 0):
                    eta = elapsed / processed_this_session * (frame_count - processed)
                progress = VideoProgress(
                    frames_processed=processed,
                    total_frames=frame_count,
                    video_position_seconds=decoded.timestamp.offset_seconds,
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=eta,
                    detections_in_frame=len(detections),
                    active_tracks=self.tracker.active_track_count,
                )
                if progress_callback is not None:
                    progress_callback(progress)
                utc_time = decoded.utc_at(recording_start_utc)
                yield VideoFrameAnalysis(
                    frame_index=decoded.index,
                    timestamp=decoded.timestamp,
                    utc_time=utc_time,
                    detections=tuple(dict(item) for item in detections),
                    assignments=tracking.assignments,
                    closed_tracks=tracking.closed_tracks,
                    progress=progress,
                )
            if not verified_previous:
                raise ValueError('Video is shorter than the saved checkpoint; refusing an unsafe resume')
        yield VideoTrackClosure(self.tracker.finish())
