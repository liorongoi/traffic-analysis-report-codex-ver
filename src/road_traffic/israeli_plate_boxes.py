"""Convert the public Israeli plate-box dataset into a diagnostic COCO ZIP.

The source contains one unpartitioned collection, so its output is intentionally
ineligible for training and product acceptance. This adapter handles plate
location boxes only and never reads or emits plate text.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from PIL import Image


SOURCE_NAME = 'Israeli license-plate object-detection dataset'
SOURCE_URL = 'https://huggingface.co/datasets/Norod78/il-license-plates'


def _write_member(archive: zipfile.ZipFile, name: str, *, source=None,
        content: bytes | None = None) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with archive.open(info, 'w') as target:
        if source is not None:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, target, length=1024 * 1024)
        else:
            target.write(content or b'')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _image_path(root: Path, filename: str) -> tuple[Path, str]:
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('Each annotation must name an image')
    relative = PurePosixPath(filename.strip().replace('\\', '/'))
    if relative.is_absolute() or any(part in ('', '.', '..') for part in relative.parts):
        raise ValueError(f'Unsafe image path in annotation: {filename!r}')
    if not relative.parts or relative.parts[0] != 'img':
        raise ValueError(f'Image path must remain inside the dataset img folder: {filename!r}')
    path = root.joinpath(*relative.parts)
    if path.is_symlink():
        raise ValueError(f'Symlinked dataset images are not accepted: {filename!r}')
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Image is missing or outside the dataset folder: {filename!r}') from exc
    if not resolved.is_file():
        raise ValueError(f'Annotation image is not a file: {filename!r}')
    return resolved, relative.as_posix()


def _boxes(entries: list[dict], width: int, height: int, filename: str) -> list[list[float]]:
    if not entries:
        raise ValueError(f'Image {filename!r} must have at least one plate box')
    boxes = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('label') != 'plate':
            raise ValueError(f'Unsupported annotation in {filename!r}; expected a plate box')
        try:
            x_min, x_max, y_min, y_max = (float(entry[key])
                for key in ('xMin', 'xMax', 'yMin', 'yMax'))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid plate box coordinates for {filename!r}') from exc
        if any(not math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
            raise ValueError(f'Plate box coordinates for {filename!r} must be finite')
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min \
                or x_max > width or y_max > height:
            raise ValueError(f'Plate box for {filename!r} falls outside its image')
        boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
    return boxes


def convert_israeli_plate_boxes(dataset_dir, output_zip, *, exclusions_file=None) -> dict:
    """Build a deterministic diagnostic COCO ZIP from a local dataset copy.

    The directory must contain ``annotations.csv`` and an ``img/`` folder.
    No network access, OCR, or plate-text handling occurs. Existing output
    files are never overwritten.
    """
    dataset_dir = Path(dataset_dir).resolve(strict=True)
    images_root = (dataset_dir / 'img').resolve(strict=True)
    if not images_root.is_dir():
        raise ValueError('Dataset must contain an img directory')
    annotations_path = dataset_dir / 'annotations.csv'
    if not annotations_path.is_file() or annotations_path.is_symlink():
        raise ValueError('Dataset must contain a regular annotations.csv file')
    output_zip = Path(output_zip).resolve()
    if output_zip.exists() or output_zip.is_symlink():
        raise ValueError('Output ZIP already exists; choose a new path')
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    grouped = {}
    try:
        with annotations_path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'image', 'label', 'xMin', 'xMax', 'yMin', 'yMax'}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError('annotations.csv must contain image,label,xMin,xMax,yMin,yMax columns')
            for row in reader:
                filename = row.get('image')
                if filename not in grouped:
                    grouped[filename] = []
                grouped[filename].append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError('Could not read annotations.csv') from exc
    if not grouped:
        raise ValueError('annotations.csv must contain at least one image row')
    rows = list(grouped.items())

    exclusions_path = Path(exclusions_file) if exclusions_file else None
    exclusions = {}
    if exclusions_path:
        if not exclusions_path.is_file() or exclusions_path.is_symlink():
            raise ValueError('Exclusion list must be a regular JSON file')
        exclusions_path = exclusions_path.resolve(strict=True)
        try:
            entries = json.loads(exclusions_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError('Could not read exclusion-list JSON') from exc
        if not isinstance(entries, list):
            raise ValueError('Exclusion list must be a JSON list of imagefilename/reason records')
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get('reason'), str) \
                    or not entry['reason'].strip():
                raise ValueError('Each exclusion needs an imagefilename and a non-empty reason')
            name = entry.get('imagefilename')
            if not isinstance(name, str) or name in exclusions:
                raise ValueError(f'Invalid or duplicate excluded image: {name!r}')
            exclusions[name] = entry['reason'].strip()

    images, annotations, members, excluded_images = [], [], [], []
    seen = set()
    annotation_id = 1
    for image_id, (filename, boxes_rows) in enumerate(rows, start=1):
        if filename in seen:
            raise ValueError(f'Duplicate image annotation: {filename!r}')
        seen.add(filename)
        if filename in exclusions:
            _image_path(dataset_dir, filename)
            excluded_images.append({'imagefilename': filename,
                'reason': exclusions.pop(filename)})
            continue
        source, relative = _image_path(dataset_dir, filename)
        try:
            with Image.open(source) as image:
                image.load()
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise ValueError(f'Could not decode dataset image {filename!r}') from exc
        if width <= 0 or height <= 0 or width * height > 24_000_000:
            raise ValueError(f'Dataset image {filename!r} is outside the 24-million-pixel adapter limit')
        boxes = _boxes(boxes_rows, width, height, filename)
        images.append({'id': image_id, 'file_name': relative,
            'width': width, 'height': height, 'country': 'Israel'})
        for box in boxes:
            annotations.append({'id': annotation_id, 'image_id': image_id,
                'category_id': 1, 'bbox': box, 'area': box[2] * box[3],
                'iscrowd': 0, 'segmentation': []})
            annotation_id += 1
        members.append((source, f'diagnostic/{relative}'))

    if exclusions:
        raise ValueError(f'Exclusion list refers to unknown images: {sorted(exclusions)}')
    if not images:
        raise ValueError('No images remain after applying the explicit exclusion list')

    coco = {'images': images, 'annotations': annotations,
        'categories': [{'id': 1, 'name': 'plate:region', 'kind': 'plate', 'label': 'region'}]}
    manifest = {'format': 'RoadLoom plate localization diagnostic v1',
        'source_name': SOURCE_NAME, 'source_url': SOURCE_URL,
        'source_annotations_sha256': _sha256(annotations_path),
        'country': 'Israel', 'task': 'plate-region box localization only; no text recognition',
        'image_count': len(images), 'box_count': len(annotations), 'split': 'diagnostic',
        'split_policy': 'The source has no protected train/validation/held-out partition; retained rows remain in one diagnostic set.',
        'excluded_image_count': len(excluded_images), 'excluded_images': excluded_images,
        'exclusion_list_sha256': _sha256(exclusions_path) if exclusions_path else None,
        'eligible_for_training': False, 'eligible_for_acceptance': False}
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{output_zip.name}.', suffix='.tmp',
        dir=output_zip.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            _write_member(archive, 'diagnostic/annotations.json', content=
                json.dumps(coco, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
            _write_member(archive, 'diagnostic-manifest.json', content=
                (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
            for source, member in members:
                _write_member(archive, member, source=source)
        os.replace(temporary, output_zip)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output_zip), **manifest, 'archive_bytes': output_zip.stat().st_size}
