"""Convert the public IR-LPR-corners imagefolder into a diagnostic COCO ZIP.

The source supplies one plate quadrilateral per image, but no train/validation/
held-out partition. The output is deliberately diagnostic-only and cannot be
mistaken for protected training data or acceptance evidence.
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


SOURCE_NAME = 'IR-LPR-corners'
SOURCE_URL = 'https://huggingface.co/datasets/mahdilotfi/IR-LPR-corners'
CORNER_COLUMNS = tuple(f'{axis}{index}' for index in range(1, 5) for axis in ('x', 'y'))


def _write_member(archive: zipfile.ZipFile, name: str, source=None, content: bytes | None = None) -> None:
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
    path = root.joinpath(*relative.parts)
    if path.is_symlink():
        raise ValueError(f'Symlinked dataset images are not accepted: {filename!r}')
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Image is missing or outside the images folder: {filename!r}') from exc
    if not resolved.is_file():
        raise ValueError(f'Annotation image is not a file: {filename!r}')
    return resolved, relative.as_posix()


def _bbox(values: list[str], width: int, height: int, filename: str) -> list[float]:
    try:
        coordinates = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid plate corner coordinates for {filename!r}') from exc
    if len(coordinates) != 8 or any(not math.isfinite(value) or not 0 <= value <= 1
                                    for value in coordinates):
        raise ValueError(f'Plate corners for {filename!r} must be eight normalized finite values')
    xs, ys = coordinates[::2], coordinates[1::2]
    left, top = min(xs) * width, min(ys) * height
    right, bottom = max(xs) * width, max(ys) * height
    if right <= left or bottom <= top:
        raise ValueError(f'Plate quadrilateral for {filename!r} has zero area')
    return [left, top, right - left, bottom - top]


def convert_ir_lpr_corners(dataset_dir, output_zip) -> dict:
    """Build a reproducible, one-split diagnostic COCO ZIP from local files.

    The source directory must contain ``annotations.csv`` and ``images/``.
    No network access, model training, OCR, or plate-text handling occurs.
    Existing output files are never overwritten.
    """
    dataset_dir = Path(dataset_dir).resolve(strict=True)
    images_root = (dataset_dir / 'images').resolve(strict=True)
    if not images_root.is_dir():
        raise ValueError('Dataset must contain an images directory')
    annotations_path = dataset_dir / 'annotations.csv'
    if not annotations_path.is_file() or annotations_path.is_symlink():
        raise ValueError('Dataset must contain a regular annotations.csv file')
    output_zip = Path(output_zip).resolve()
    if output_zip.exists() or output_zip.is_symlink():
        raise ValueError('Output ZIP already exists; choose a new path')
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    images, annotations, members = [], [], []
    seen = set()
    with annotations_path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image', *CORNER_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('annotations.csv must contain image and x1,y1,...,x4,y4 columns')
        for image_id, row in enumerate(reader, start=1):
            filename = row.get('image', '')
            if filename in seen:
                raise ValueError(f'Duplicate image annotation: {filename!r}')
            seen.add(filename)
            source, relative = _image_path(images_root, filename)
            try:
                with Image.open(source) as image:
                    image.load()
                    width, height = image.size
            except (OSError, ValueError) as exc:
                raise ValueError(f'Could not decode dataset image {filename!r}') from exc
            if width <= 0 or height <= 0 or width * height > 24_000_000:
                raise ValueError(f'Dataset image {filename!r} is outside the 24-million-pixel adapter limit')
            box = _bbox([row.get(column, '') for column in CORNER_COLUMNS],
                width, height, filename)
            coco_image = {'id': image_id, 'file_name': relative,
                'width': width, 'height': height, 'country': 'Iran'}
            basename = PurePosixPath(relative).name.lower()
            if basename.startswith('day_'):
                coco_image['day_night'] = 'day'
            elif basename.startswith('night_'):
                coco_image['day_night'] = 'night'
            images.append(coco_image)
            annotations.append({'id': image_id, 'image_id': image_id,
                'category_id': 1, 'bbox': box, 'area': box[2] * box[3],
                'iscrowd': 0, 'segmentation': []})
            members.append((source, f'diagnostic/{relative}'))
    if not images:
        raise ValueError('annotations.csv contains no image rows')

    coco = {'images': images, 'annotations': annotations,
        'categories': [{'id': 1, 'name': 'plate:region', 'kind': 'plate', 'label': 'region'}]}
    manifest = {'format': 'RoadLoom plate localization diagnostic v1',
        'source_name': SOURCE_NAME, 'source_url': SOURCE_URL,
        'source_annotations_sha256': _sha256(annotations_path),
        'country': 'Iran', 'task': 'plate-region box localization only; no text recognition',
        'annotation_conversion': 'axis-aligned box enclosing the four supplied normalized plate corners',
        'image_count': len(images), 'split': 'diagnostic',
        'split_policy': 'The source provides no train/validation/held-out partition; all rows remain in one diagnostic set.',
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
    return {'path': str(output_zip), **manifest,
        'archive_bytes': output_zip.stat().st_size}
