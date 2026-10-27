"""Isolated PyAV H.264 clip encoder, kept separate from OpenCV's FFmpeg build."""
from fractions import Fraction
import json
import math
import shutil
import sys
import traceback
from pathlib import Path


MIN_FREE_BYTES = 64 * 1024 * 1024


def ensure_output_space(destination):
    """Keep a small recovery reserve on the volume receiving a clip."""
    try:
        free = shutil.disk_usage(Path(destination).parent).free
    except OSError as exc:
        raise ValueError(f'Cannot check free space for clip output: {exc}') from exc
    if free < MIN_FREE_BYTES:
        raise ValueError(
            f'Clip export needs at least {MIN_FREE_BYTES} bytes free on the output drive '
            f'to preserve a safety reserve; only {free} bytes are available')


def encode(source, destination, start_offset, end_offset):
    import av
    if start_offset >= end_offset:
        raise ValueError('Clip interval must have positive duration')
    ensure_output_space(destination)
    with av.open(str(source), mode='r') as incoming:
        video = next((stream for stream in incoming.streams if stream.type == 'video'), None)
        if video is None or video.time_base is None:
            raise ValueError('Video source has no timestamped video stream')
        time_base = Fraction(video.time_base)
        rate = video.average_rate or video.base_rate or Fraction(30, 1)
        if time_base <= 0 or video.width <= 0 or video.height <= 0:
            raise ValueError('Video stream dimensions or time base are invalid')
        if video.width % 2 or video.height % 2:
            raise ValueError('H.264 clip export requires even video dimensions')
        incoming.seek(max(int(start_offset / time_base), video.start_time or 0),
            stream=video, backward=True, any_frame=False)
        with av.open(str(destination), mode='w', format='mp4', options={'movflags':'faststart'}) as outgoing:
            output = outgoing.add_stream('libx264', rate=rate)
            output.width, output.height, output.pix_fmt = video.width, video.height, 'yuv420p'
            output.time_base = time_base
            output.codec_context.time_base = time_base
            output.options = {'preset':'veryfast','crf':'20'}
            first_pts = previous_pts = None
            frame_count = 0
            space_check_frames = max(1, math.ceil(float(rate)))
            for frame in incoming.decode(video):
                if frame.pts is None:
                    continue
                pts = int(frame.pts)
                offset = Fraction(pts) * time_base
                if offset < start_offset:
                    continue
                if offset > end_offset:
                    break
                if previous_pts is not None and pts <= previous_pts:
                    raise ValueError('Video presentation timestamps are not increasing in the requested clip')
                if first_pts is None:
                    first_pts = pts
                previous_pts = pts
                frame.pts = pts - first_pts
                frame.time_base = time_base
                for packet in output.encode(frame):
                    outgoing.mux(packet)
                frame_count += 1
                if frame_count % space_check_frames == 0:
                    ensure_output_space(destination)
            for packet in output.encode(None):
                outgoing.mux(packet)
        ensure_output_space(destination)
        if not frame_count:
            raise ValueError('The requested clip interval contains no decodable video frames')
    return frame_count


def main():
    request = json.load(sys.stdin)
    try:
        count = encode(Path(request['source']), Path(request['destination']),
            Fraction(request['start_num'],request['start_den']),
            Fraction(request['end_num'],request['end_den']))
        print(json.dumps({'frame_count':count}))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
