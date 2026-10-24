"""Generate a tiny real VFR Matroska file for isolated decoder tests."""
from fractions import Fraction
from pathlib import Path
import sys

import av
import numpy as np


def main(path):
    width, height = 64, 48
    with av.open(str(Path(path)), mode='w', format='matroska') as output:
        stream = output.add_stream('mpeg4', rate=25)
        stream.width, stream.height = width, height
        stream.pix_fmt = 'yuv420p'
        stream.time_base = Fraction(1, 1000)
        for index, pts in enumerate((0, 40, 160, 200)):
            image = np.full((height, width, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


if __name__ == '__main__':
    main(sys.argv[1])
