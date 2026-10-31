#!/usr/bin/env python3
"""Export saved frame predictions from one completed RoadLoom video run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.tracking_benchmark import export_video_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, help='mission.sqlite3 from the target mission')
    parser.add_argument('--run-id', required=True, help='completed saved video run ID')
    parser.add_argument('--output', required=True, help='prediction-manifest JSON path')
    parser.add_argument('--sequence-id', help='optional sequence ID matching the ground-truth manifest')
    args = parser.parse_args(argv)
    try:
        result = export_video_run(args.database, args.run_id, args.output,
                                  sequence_id=args.sequence_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    sequence = result['sequences'][0]
    print(json.dumps({'output': args.output, 'sequence_id': sequence['sequence_id'],
                      'frames': len(sequence['frames']), 'model_id': sequence['metadata']['model_id']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
