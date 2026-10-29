#!/usr/bin/env python3
"""Download the optional public SAM 3.1 BF16 model without using an HF account."""
import argparse
import os
from pathlib import Path
import sys

REPOSITORY = 'unsloth/sam3.1'
FILES = [
    'LICENSE', 'config.json', 'merges.txt', 'model.safetensors',
    'processor_config.json', 'special_tokens_map.json', 'tokenizer.json',
    'tokenizer_config.json', 'vocab.json',
]
MINIMUM_WEIGHT_BYTES = 1_000_000_000


def default_destination():
    local_app_data = os.environ.get('LOCALAPPDATA')
    if local_app_data:
        return Path(local_app_data) / 'RoadTrafficAnalysis' / 'Models' / 'sam3.1-unsloth'
    return Path(__file__).resolve().parents[1] / 'model-packs' / 'sam3.1-unsloth'


def download(destination):
    # Explicitly opt out of any cached credential and hub telemetry. This script
    # performs one setup-time public download; the application loads local files.
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install the optional downloader with: python -m pip install huggingface_hub') from exc

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPOSITORY, local_dir=str(destination), allow_patterns=FILES,
        ignore_patterns=['*INT8*', '*FP8*'], token=False, max_workers=4)
    missing = [name for name in FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError('The public model snapshot is incomplete; missing: ' + ', '.join(missing))
    weight_size = (destination / 'model.safetensors').stat().st_size
    if weight_size < MINIMUM_WEIGHT_BYTES:
        raise RuntimeError('model.safetensors is unexpectedly small; refusing a pointer or partial download')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=str(default_destination()),
        help='Local folder for the optional ~1.68 GB BF16 weights')
    args = parser.parse_args()
    try:
        destination = download(args.destination)
    except Exception as exc:
        print(f'SAM 3.1 download failed: {exc}', file=sys.stderr)
        return 1
    print(f'SAM 3.1 BF16 files downloaded anonymously to: {destination}')
    print('No login/token was used. Keep this folder local or copy it to the offline workstation.')
    print('The core installer and repository do not include these large weight files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
