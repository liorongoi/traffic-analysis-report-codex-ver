#!/usr/bin/env python3
"""Fine-tune a local RoadLoom COCO export with the staged YOLOX-S checkpoint."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from road_traffic.yolox_training import finetune_yolox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-zip', required=True, help='ZIP downloaded from the app learning panel')
    parser.add_argument('--output', required=True, help='New folder for training data, logs, checkpoints and ONNX pack')
    parser.add_argument('--yolox-root', help='Optional local checkout of Megvii YOLOX 0.3.0')
    parser.add_argument('--baseline', help='Override the bundled COCO checkpoint path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    result = finetune_yolox(args.dataset_zip, args.output, yolox_root=args.yolox_root,
        baseline=args.baseline, epochs=args.epochs, batch_size=args.batch_size, workers=args.workers)
    print(f"Trained model pack to import in RoadLoom: {result['model_pack']}")
    print(f"Training artifacts: {result['output']}")
    print(f"Classes: {', '.join(result['classes'])}")
    print(f"ONNX SHA-256: {result['model_sha256']}")
    print('The held-out test split was not used. Candidate selection and model promotion are not automatic.')


if __name__ == '__main__':
    main()
