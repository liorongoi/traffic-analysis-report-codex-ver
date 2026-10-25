"""Bounded video decoding that retains FFmpeg frame PTS and time bases."""
from dataclasses import asdict, dataclass
from fractions import Fraction
import multiprocessing
from pathlib import Path

from .video_timeline import VideoTimestamp


class VideoDecodeError(ValueError):
    """An input cannot be decoded with a trustworthy frame timeline."""


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    index: int
    timestamp: VideoTimestamp
    image: object

    def utc_at(self, recording_start_utc):
        """Map to absolute UTC only when a caller supplies an explicit origin."""
        if recording_start_utc is None:
            return None
        return self.timestamp.utc_at(recording_start_utc)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    format_name: str
    codec_name: str
    width: int
    height: int
    stream_index: int
    stream_time_base_num: int | None
    stream_time_base_den: int | None
    average_rate_num: int | None
    average_rate_den: int | None
    frame_count: int
    duration_pts: int | None

    @property
    def stream_time_base(self):
        if self.stream_time_base_num is None or self.stream_time_base_den is None:
            return None
        return Fraction(self.stream_time_base_num, self.stream_time_base_den)

    @property
    def average_rate(self):
        if self.average_rate_num is None or self.average_rate_den is None:
            return None
        return Fraction(self.average_rate_num, self.average_rate_den)


class _PyAvVideoDecoder:
    """Decode one video incrementally inside the isolated FFmpeg process.

    PyAV/FFmpeg's Frame.pts and Frame.time_base are retained independently for
    every frame. VFR is therefore represented by differing PTS deltas; no frame
    number/FPS calculation is used. At most one RGB image is yielded at a time.
    """

    def __init__(self, path, *, max_frame_pixels=24_000_000, max_threads=2):
        self.path = Path(path)
        self.max_frame_pixels = max_frame_pixels
        self.max_threads = max_threads
        self.container = None
        self.stream = None
        self.metadata = None
        self._iterated = False

    def __enter__(self):
        import av
        if not self.path.is_file():
            raise VideoDecodeError(f'Video source does not exist or is not a file: {self.path}')
        if isinstance(self.max_frame_pixels, bool) or not isinstance(self.max_frame_pixels, int) or self.max_frame_pixels < 1:
            raise ValueError('max_frame_pixels must be a positive integer')
        if isinstance(self.max_threads, bool) or not isinstance(self.max_threads, int) or self.max_threads < 1:
            raise ValueError('max_threads must be a positive integer')
        try:
            self.container = av.open(str(self.path), mode='r')
            streams = self.container.streams.video
            if not streams:
                raise VideoDecodeError('Input contains no video stream')
            self.stream = streams[0]
            self.stream.codec_context.thread_count = self.max_threads
            width = int(self.stream.codec_context.width)
            height = int(self.stream.codec_context.height)
            if width < 1 or height < 1:
                raise VideoDecodeError('Video stream reports invalid frame dimensions')
            if width * height > self.max_frame_pixels:
                raise VideoDecodeError(f'Video frame exceeds the {self.max_frame_pixels:,}-pixel decode limit')
            codec = self.stream.codec_context.codec
            stream_tb = self.stream.time_base
            average_rate = self.stream.average_rate
            duration_pts = self.stream.duration
            fmt = self.container.format.name if self.container.format else 'unknown'
            self.metadata = VideoMetadata(
                format_name=fmt,
                codec_name=codec.name if codec else 'unknown',
                width=width,
                height=height,
                stream_index=int(self.stream.index),
                stream_time_base_num=int(stream_tb.numerator) if stream_tb else None,
                stream_time_base_den=int(stream_tb.denominator) if stream_tb else None,
                average_rate_num=int(average_rate.numerator) if average_rate else None,
                average_rate_den=int(average_rate.denominator) if average_rate else None,
                frame_count=max(0, int(self.stream.frames or 0)),
                duration_pts=int(duration_pts) if duration_pts is not None else None,
            )
            return self
        except VideoDecodeError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise VideoDecodeError(f'Video is corrupt, unsupported, or cannot be opened: {self.path.name}: {exc}') from exc

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None
            self.stream = None

    def frames(self):
        if self.container is None or self.stream is None:
            raise RuntimeError('Open VideoDecoder with a with block before reading frames')
        if self._iterated:
            raise RuntimeError('VideoDecoder frame stream can only be consumed once')
        self._iterated = True
        previous_seconds = None
        index = 0
        try:
            for frame in self.container.decode(self.stream):
                if frame.is_corrupt:
                    raise VideoDecodeError(f'Corrupt decoded frame at index {index}')
                if frame.pts is None:
                    raise VideoDecodeError(f'Missing presentation timestamp at decoded frame {index}')
                time_base = frame.time_base or self.stream.time_base
                if time_base is None:
                    raise VideoDecodeError(f'Missing stream time base at decoded frame {index}')
                time_base = Fraction(time_base)
                if time_base.numerator <= 0 or time_base.denominator <= 0:
                    raise VideoDecodeError(f'Invalid time base at decoded frame {index}')
                pts = int(frame.pts)
                seconds = Fraction(pts) * time_base
                if previous_seconds is not None and seconds < previous_seconds:
                    raise VideoDecodeError(f'Presentation timestamps moved backwards at decoded frame {index}')
                width, height = int(frame.width), int(frame.height)
                if width < 1 or height < 1 or width * height > self.max_frame_pixels:
                    raise VideoDecodeError(f'Decoded frame {index} exceeds the pixel limit or has invalid dimensions')
                if width != self.metadata.width or height != self.metadata.height:
                    raise VideoDecodeError(f'Decoded frame {index} dimensions changed unexpectedly')
                try:
                    rgb = frame.to_ndarray(format='rgb24')
                except Exception as exc:
                    raise VideoDecodeError(f'Could not convert decoded frame {index} to RGB: {exc}') from exc
                timestamp = VideoTimestamp(pts, time_base.numerator, time_base.denominator)
                yield DecodedVideoFrame(index, timestamp, rgb)
                previous_seconds = seconds
                index += 1
        except VideoDecodeError:
            raise
        except Exception as exc:
            raise VideoDecodeError(f'Video decoding failed at frame {index}: {exc}') from exc
        if index == 0:
            raise VideoDecodeError('Video contains no decodable frames')


def _decode_child(path, max_frame_pixels, max_threads, connection):
    """Spawn target: keep PyAV/FFmpeg native libraries out of detector process."""
    try:
        with _PyAvVideoDecoder(path, max_frame_pixels=max_frame_pixels, max_threads=max_threads) as decoder:
            connection.send(('ready', asdict(decoder.metadata)))
            for frame in decoder.frames():
                # Pipe backpressure limits the stream to the current frame; no
                # multiprocessing queue can accumulate unbounded RGB arrays.
                connection.send(('frame', frame.index, frame.timestamp.pts,
                    frame.timestamp.time_base_num, frame.timestamp.time_base_den, frame.image))
        connection.send(('done',))
    except BaseException as exc:
        try:
            connection.send(('error', f'{type(exc).__name__}: {exc}'))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class IsolatedVideoDecoder:
    """Bounded pipe-backed decoder process for use beside OpenCV inference.

    PyAV and OpenCV wheels bundle different FFmpeg dylibs on macOS. Isolating
    their native runtimes avoids loading those copies into the same process.
    The one-way pipe also provides frame-by-frame backpressure.
    """

    def __init__(self, path, *, max_frame_pixels=24_000_000, max_threads=2):
        self.path = Path(path)
        self.max_frame_pixels = max_frame_pixels
        self.max_threads = max_threads
        self.metadata = None
        self._process = None
        self._connection = None
        self._iterated = False
        self._finished = False

    def __enter__(self):
        context = multiprocessing.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        self._connection = receiver
        self._process = context.Process(target=_decode_child,
            args=(str(self.path), self.max_frame_pixels, self.max_threads, sender),
            name='traffic-video-decoder')
        try:
            self._process.start()
            sender.close()
            packet = receiver.recv()
            if packet[0] == 'error':
                self.close()
                raise VideoDecodeError(packet[1])
            if packet[0] != 'ready':
                self.close()
                raise VideoDecodeError('Video decoder process returned an invalid startup message')
            self.metadata = VideoMetadata(**packet[1])
            return self
        except VideoDecodeError:
            raise
        except (EOFError, OSError, RuntimeError) as exc:
            sender.close()
            self.close()
            raise VideoDecodeError(f'Video decoder process could not start: {exc}') from exc

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        if self._connection is not None:
            try:
                self._connection.close()
            except OSError:
                pass
            self._connection = None
        if self._process is not None:
            if self._process.pid is not None:
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join()
            self._process.close()
            self._process = None

    def frames(self):
        if self._iterated:
            raise RuntimeError('VideoDecoder frame stream can only be consumed once')
        if self._connection is None or self._process is None:
            raise RuntimeError('Open IsolatedVideoDecoder with a with block before reading frames')
        self._iterated = True
        try:
            while True:
                try:
                    packet = self._connection.recv()
                except EOFError as exc:
                    raise VideoDecodeError('Video decoder process exited before completing its frame stream') from exc
                if packet[0] == 'done':
                    self._finished = True
                    self._process.join()
                    if self._process.exitcode != 0:
                        raise VideoDecodeError(f'Video decoder process exited with status {self._process.exitcode}')
                    return
                if packet[0] == 'error':
                    self._process.join()
                    raise VideoDecodeError(packet[1])
                if packet[0] != 'frame':
                    raise VideoDecodeError('Video decoder process returned an invalid frame message')
                _, index, pts, numerator, denominator, image = packet
                yield DecodedVideoFrame(index,
                    VideoTimestamp(pts, numerator, denominator), image)
        finally:
            if self._finished and self._process is not None:
                self._process.close()
                self._process = None


# Public decoding always isolates PyAV's bundled FFmpeg libraries from OpenCV.
VideoDecoder = IsolatedVideoDecoder


def probe_video(path):
    """Read container/stream metadata without decoding or consuming frames."""
    with VideoDecoder(path) as decoder:
        return asdict(decoder.metadata)
