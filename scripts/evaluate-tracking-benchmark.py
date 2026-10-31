#!/usr/bin/env python3
"""Score paired offline tracking annotations and predictions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.tracking_benchmark import evaluate_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', required=True,
                        help='UTF-8 JSON with sequences, frames, and human-labeled objects')
    parser.add_argument('--predictions', required=True,
                        help='UTF-8 JSON with matching sequence/frame IDs and predicted objects')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='minimum IoU for frame-level class-aware matching (default: 0.5)')
    parser.add_argument('--report', required=True, help='output JSON report path')
    args = parser.parse_args(argv)
    try:
        report = evaluate_files(args.ground_truth, args.predictions, args.report,
                                iou_threshold=args.iou_threshold)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'report': args.report, 'overall': report['overall']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
