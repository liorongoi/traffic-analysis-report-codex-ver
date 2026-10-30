"""Evaluate saved COCO detections against a frozen independently reviewed split."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.benchmark import evaluate_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', required=True, type=Path,
        help='COCO annotation JSON or RoadLoom dataset ZIP containing <split>/annotations.json')
    parser.add_argument('--predictions', required=True, type=Path,
        help='COCO result array, optionally wrapped as {predictions: [...], metadata: {...}}')
    parser.add_argument('--split', default='test', choices=('training', 'validation', 'test'))
    parser.add_argument('--operating-iou', default=.5, type=float,
        help='IoU threshold used for the displayed precision/recall/F1 (default: 0.5)')
    parser.add_argument('--report', required=True, type=Path, help='Output JSON report path')
    args = parser.parse_args()
    report = evaluate_files(args.ground_truth, args.predictions, args.report,
        split=args.split, operating_iou=args.operating_iou)
    overall = report['overall']
    print(f"Images: {report['images']}; report: {args.report}")
    print('Micro P/R/F1 at IoU {iou:.2f}: {p} / {r} / {f}'.format(
        iou=report['metrics']['operating_iou'],
        p=overall['micro'].get('precision'), r=overall['micro'].get('recall'),
        f=overall['micro'].get('f1')))
    print('Macro AP50 / mAP50:95: {} / {}'.format(
        overall['micro'].get('ap50_macro'), overall['micro'].get('map50_95_macro')))
    print('Per-image count MAE: {}'.format(
        overall['count_error'].get('mae_objects_per_image')))


if __name__ == '__main__':
    main()
