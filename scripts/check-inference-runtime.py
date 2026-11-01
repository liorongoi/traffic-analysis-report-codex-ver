"""Offline installed-runtime and optional still-image compatibility check."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch


parser = argparse.ArgumentParser(description='Measure the bundled models without network access.')
parser.add_argument('root', type=Path, help='Installed application directory')
parser.add_argument('--images', type=Path, help='Optional folder of still images to measure')
parser.add_argument('--max-images', type=int, default=8, help='Maximum still images to measure (default: 8)')
parser.add_argument('--output', type=Path, help='Optional path for a JSON report')
args = parser.parse_args()
if not 1 <= args.max_images <= 1000:
    parser.error('--max-images must be between 1 and 1000')
root = args.root.resolve()
if args.images and not args.images.is_dir():
    parser.error('--images must be an existing still-image folder')
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(root / 'src'))
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from road_traffic.detector import VehicleDetector
from road_traffic.plate_review import PlateDetector, _model_pack
from road_traffic.hardware import inventory


def summarize_ms(values):
    if not values:
        return {'count': 0, 'median_ms': None, 'p95_ms': None}
    ordered = sorted(values)
    return {'count': len(values), 'median_ms': round(ordered[len(ordered) // 2], 2),
        'p95_ms': round(ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)], 2)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


report = {'status': 'passed',
    'offline_check': {'python_socket_connect_blocked': True,
        'native_os_network_isolation_tested': False},
    'hardware': None, 'runtime': None, 'vehicle_model': None,
    'plate_region_model': None, 'representative_still_images': None,
    'interpretation': 'Synthetic blank-frame checks verify runtime only. Optional still-image timings measure compatibility and speed, not accuracy.'}

with (patch.object(socket.socket, 'connect',
        side_effect=RuntimeError('External connections blocked during offline smoke')),
        patch.object(socket.socket, 'connect_ex',
        side_effect=RuntimeError('External connections blocked during offline smoke'))):
    report['hardware'] = inventory(root)
    import_started = time.perf_counter()
    vehicle_model = VehicleDetector(root / 'model-packs' / 'yolox-s', tiled=False)
    vehicle_init_ms = round((time.perf_counter() - import_started) * 1000, 2)
    plate_pack = root / 'model-packs' / 'license-plate-yolox'
    _, plate_sha = _model_pack(plate_pack)
    import_started = time.perf_counter()
    plate_model = PlateDetector(plate_pack, tiled=False)
    plate_init_ms = round((time.perf_counter() - import_started) * 1000, 2)

    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    started = time.perf_counter()
    blank_vehicles = vehicle_model.infer(blank)
    blank_vehicle_ms = round((time.perf_counter() - started) * 1000, 2)
    if blank_vehicles:
        raise SystemExit('Vehicle model produced proposals for a blank smoke frame')
    started = time.perf_counter()
    blank_plates = plate_model.detect(blank)
    blank_plate_ms = round((time.perf_counter() - started) * 1000, 2)
    if blank_plates:
        raise SystemExit('Plate-region model produced proposals for a blank smoke frame')

    image_results = []
    vehicle_times, plate_times = [], []
    skipped = []
    if args.images:
        paths = sorted(path for path in args.images.rglob('*')
            if path.is_file() and path.suffix.casefold() in {'.jpg', '.jpeg', '.png'})
        for path in paths:
            if len(image_results) >= args.max_images:
                break
            try:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB'))
            except (OSError, ValueError) as exc:
                skipped.append({'file': str(path.relative_to(args.images)),
                    'reason': f'{type(exc).__name__}: {exc}'[:160]})
                continue
            height, width = rgb.shape[:2]
            if width * height > 24_000_000:
                skipped.append({'file': str(path.relative_to(args.images)),
                    'reason': 'exceeds the 24-megapixel inference limit'})
                continue
            started = time.perf_counter()
            vehicles = vehicle_model.infer(rgb)
            vehicle_ms = round((time.perf_counter() - started) * 1000, 2)
            started = time.perf_counter()
            plates = plate_model.detect(rgb)
            plate_ms = round((time.perf_counter() - started) * 1000, 2)
            vehicle_times.append(vehicle_ms)
            plate_times.append(plate_ms)
            image_results.append({'file': str(path.relative_to(args.images)),
                'sha256': sha256(path), 'width': width, 'height': height,
                'vehicle_proposals': len(vehicles), 'plate_region_proposals': len(plates),
                'vehicle_ms': vehicle_ms, 'plate_region_ms': plate_ms})
        if not image_results:
            raise SystemExit('The selected folder contains no readable images within the 24-megapixel limit')

report['runtime'] = {'numpy': np.__version__, 'opencv': cv2.__version__,
    'onnxruntime': ort.__version__, 'available_onnx_providers': ort.get_available_providers()}
report['vehicle_model'] = {'sha256': vehicle_model.sha,
    'selected_provider': getattr(vehicle_model.session, 'provider', 'unknown'),
    'session_providers': vehicle_model.session.get_providers(),
    'fallback_reason': getattr(vehicle_model.session, 'fallback_reason', None),
    'initialization_ms': vehicle_init_ms, 'blank_frame_proposals': len(blank_vehicles),
    'blank_frame_inference_ms': blank_vehicle_ms}
report['plate_region_model'] = {'sha256': plate_sha,
    'selected_provider': getattr(plate_model.session, 'provider', 'unknown'),
    'session_providers': plate_model.session.get_providers(),
    'fallback_reason': getattr(plate_model.session, 'fallback_reason', None),
    'initialization_ms': plate_init_ms, 'blank_frame_proposals': len(blank_plates),
    'blank_frame_inference_ms': blank_plate_ms, 'plate_text_recognition': 'excluded'}
if args.images:
    report['representative_still_images'] = {'folder': str(args.images.resolve()),
        'requested_limit': args.max_images, 'scored_images': len(image_results),
        'skipped_images': skipped, 'vehicle_timings': summarize_ms(vehicle_times),
        'plate_region_timings': summarize_ms(plate_times), 'images': image_results}
serialized = json.dumps(report, indent=2)
if args.output:
    args.output.write_text(serialized + '\n', encoding='utf-8')
else:
    print(serialized)
