"""Package selected nighttime vehicle boxes into a COCO dataset ZIP."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.nighttime_vehicle_dataset import build_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True, type=Path,
        help='Dataset gt10913.txt annotation file')
    parser.add_argument('--images-dir', required=True, type=Path,
        help='Directory containing img_<id>.jpg images')
    parser.add_argument('--split-range', action='append', required=True,
        metavar='ROLE=FIRST-LAST',
        help='Explicit evidence role and inclusive image-ID range; repeat for disjoint ranges or roles')
    parser.add_argument('--exclude-image-id', action='append', type=int, default=[],
        help='Exclude a complete selected image when its source annotations fail validation')
    parser.add_argument('--output', required=True, type=Path, help='New dataset ZIP output path')
    args = parser.parse_args(argv)
    try:
        report = build_dataset(args.labels, args.images_dir, args.split_range, args.output,
            exclude_image_ids=args.exclude_image_id)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
