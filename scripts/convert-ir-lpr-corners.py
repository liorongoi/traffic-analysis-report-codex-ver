"""Convert local IR-LPR-corners files to a diagnostic-only COCO ZIP."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.ir_lpr_corners import convert_ir_lpr_corners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path,
        help='Local IR-LPR-corners directory with annotations.csv and images/')
    parser.add_argument('--output', required=True, type=Path,
        help='New COCO ZIP path; existing files are never overwritten')
    args = parser.parse_args()
    result = convert_ir_lpr_corners(args.dataset_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Diagnostic only: this single unpartitioned dataset cannot be used for training or acceptance.')


if __name__ == '__main__':
    main()
