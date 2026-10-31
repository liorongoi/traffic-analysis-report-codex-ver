"""Run a local vehicle or plate detector over one frozen COCO split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from road_traffic.benchmark import predict_coco_zip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-zip', required=True, type=Path)
    parser.add_argument('--model-pack', required=True, type=Path)
    parser.add_argument('--model-kind', choices=('vehicle', 'plate'), default='vehicle')
    parser.add_argument('--split', choices=('training', 'validation', 'test', 'diagnostic'), default='test')
    parser.add_argument('--threshold', type=float,
        help='Confidence threshold; defaults to 0.18 for vehicles or 0.30 for plate regions')
    parser.add_argument('--no-tiles', action='store_true',
        help='Disable overlapping native-resolution tiling for the selected detector')
    parser.add_argument('--image-metadata', type=Path,
        help='Optional JSON object mapping COCO image IDs to country/condition/view/hardware labels')
    parser.add_argument('--hardware-profile',
        help='Run-level hardware profile label for stratified reporting')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    image_metadata = json.loads(args.image_metadata.read_text(encoding='utf-8')) if args.image_metadata else None
    strata = {'hardware_profile': args.hardware_profile} if args.hardware_profile else None
    bundle = predict_coco_zip(args.dataset_zip, args.model_pack, split=args.split,
        model_kind=args.model_kind, threshold=args.threshold, tiled=not args.no_tiles,
        image_metadata=image_metadata, strata=strata)
    bundle['metadata']['storage_bytes']['prediction_json'] = 0
    encoded = ''
    for _ in range(5):
        encoded = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        size = len(encoded.encode('utf-8'))
        if bundle['metadata']['storage_bytes']['prediction_json'] == size:
            break
        bundle['metadata']['storage_bytes']['prediction_json'] = size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name('.' + args.output.name + '.tmp')
    try:
        temporary.write_text(encoded, encoding='utf-8')
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {len(bundle['predictions'])} proposals for the {args.split} split to {args.output}")
    print(f"Model SHA-256: {bundle['metadata']['model_sha256']}; scope: {bundle['metadata']['pipeline_scope']}")


if __name__ == '__main__':
    main()
