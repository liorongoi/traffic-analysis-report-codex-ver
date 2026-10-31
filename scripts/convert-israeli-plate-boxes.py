"""Convert local Israeli plate boxes to a diagnostic-only COCO ZIP."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.israeli_plate_boxes import convert_israeli_plate_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path,
        help='Local dataset directory with annotations.csv and img/')
    parser.add_argument('--output', required=True, type=Path,
        help='New COCO ZIP path; existing files are never overwritten')
    parser.add_argument('--exclude-list', type=Path,
        help='Optional JSON list of explicit imagefilename/reason exclusions')
    args = parser.parse_args()
    result = convert_israeli_plate_boxes(args.dataset_dir, args.output,
        exclusions_file=args.exclude_list)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Diagnostic only: this unpartitioned dataset cannot be used for training or acceptance.')


if __name__ == '__main__':
    main()
