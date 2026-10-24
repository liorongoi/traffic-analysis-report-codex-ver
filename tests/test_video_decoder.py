import tempfile
import unittest
import wave
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from road_traffic.video_decoder import VideoDecodeError, VideoDecoder


def make_vfr_video(path, *, size=(64, 48), pixel_times=(0, 40, 160, 200)):
    width, height = size
    with av.open(str(path), mode='w', format='matroska') as output:
        stream = output.add_stream('mpeg4', rate=25)
        stream.width, stream.height = width, height
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 1000)
        for index, pts in enumerate(pixel_times):
            image = np.full((height, width, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


class VideoDecoderTests(unittest.TestCase):
    def test_real_vfr_file_retains_integer_pts_and_rational_timebase(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'variable-rate.mkv'
            make_vfr_video(path)
            with VideoDecoder(path) as decoder:
                metadata = decoder.metadata
                frames = list(decoder.frames())
            self.assertEqual(metadata.format_name, 'matroska,webm')
            self.assertEqual(metadata.codec_name, 'mpeg4')
            self.assertEqual(metadata.stream_time_base, Fraction(1, 1000))
            self.assertEqual([frame.timestamp.pts for frame in frames], [0, 40, 160, 200])
            self.assertEqual([frame.timestamp.offset_seconds for frame in frames], [
                Fraction(0), Fraction(1, 25), Fraction(4, 25), Fraction(1, 5)])
            self.assertEqual([right.timestamp.offset_seconds-left.timestamp.offset_seconds
                              for left, right in zip(frames, frames[1:])], [
                Fraction(1, 25), Fraction(3, 25), Fraction(1, 25)])
            self.assertEqual(frames[0].image.shape, (48, 64, 3))

    def test_absolute_utc_requires_explicit_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clip.mkv'
            make_vfr_video(path)
            origin = datetime(2026, 5, 17, 16, 4, tzinfo=timezone.utc)
            with VideoDecoder(path) as decoder:
                frames = list(decoder.frames())
            self.assertIsNone(frames[2].utc_at(None))
            self.assertEqual(frames[2].utc_at(origin), datetime(2026, 5, 17, 16, 4, 0, 160000, tzinfo=timezone.utc))

    def test_corrupt_input_and_pixel_limit_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.mkv'
            path.write_bytes(b'not a video container')
            with self.assertRaisesRegex(VideoDecodeError, 'corrupt, unsupported'):
                with VideoDecoder(path): pass
            path = Path(folder) / 'large.mkv'
            make_vfr_video(path)
            with self.assertRaisesRegex(VideoDecodeError, 'pixel decode limit'):
                with VideoDecoder(path, max_frame_pixels=10): pass

    def test_nonvideo_file_and_empty_stream_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio-only.wav'
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(b'\0\0' * 80)
            with self.assertRaisesRegex(VideoDecodeError, 'no video stream'):
                with VideoDecoder(path): pass

    def test_frame_stream_is_single_use(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'clip.mkv'
            make_vfr_video(path)
            with VideoDecoder(path) as decoder:
                list(decoder.frames())
                with self.assertRaisesRegex(RuntimeError, 'only be consumed once'):
                    list(decoder.frames())
