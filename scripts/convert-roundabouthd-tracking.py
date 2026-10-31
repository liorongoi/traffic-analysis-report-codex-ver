#!/usr/bin/env python3
"""Convert RoundaboutHD SCT labels into a RoadLoom tracking ground-truth JSON."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.roundabouthd import convert_sct_labels, write_manifest


def _label_argument(value):
    sequence_id, separator, path = value.partition('=')
    if not separator or not sequence_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError('use SEQUENCE_ID=PATH')
    return sequence_id.strip(), path.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', action='append', type=_label_argument, required=True,
                        metavar='SEQUENCE_ID=PATH',
                        help='SCT annotation file; repeat once per camera sequence')
    parser.add_argument('--frame-count', required=True, type=int,
                        help='number of source video frames represented by each annotation file')
    parser.add_argument('--width', required=True, type=int, help='source video width in pixels')
    parser.add_argument('--height', required=True, type=int, help='source video height in pixels')
    parser.add_argument('--frame-offset', type=int, default=0,
                        help='source frame ID corresponding to output frame 0 (default: 0; observed RoundaboutHD SCT labels start at 1)')
    parser.add_argument('--preserve-duplicate-track-rows', action='store_true',
                        help='preserve repeated same-track boxes within a frame and report their count; never synthesize new identities')
    parser.add_argument('--country', default='United States',
                        help='dataset country metadata (default: United States)')
    parser.add_argument('--output', required=True, help='output ground-truth JSON path')
    args = parser.parse_args(argv)
    try:
        manifest = convert_sct_labels(args.labels, frame_count=args.frame_count,
            width=args.width, height=args.height, frame_offset=args.frame_offset,
            country=args.country, allow_duplicate_track_rows=args.preserve_duplicate_track_rows)
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'output': args.output,
        'sequences': [{'sequence_id': sequence['sequence_id'],
            'frames': len(sequence['frames']),
            'annotated_frames': sum(bool(frame['objects']) for frame in sequence['frames']),
            'objects': sum(len(frame['objects']) for frame in sequence['frames']),
            'duplicate_track_id_frame_rows': sequence['metadata']['duplicate_track_id_frame_rows']}
            for sequence in manifest['sequences']]}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
