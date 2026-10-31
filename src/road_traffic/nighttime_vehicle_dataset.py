"""Package explicitly selected Nighttime Vehicle Detection images as COCO ZIP."""
import hashlib
import json
import re
import tempfile
from pathlib import Path
import zipfile

ROLES = ('training', 'validation', 'test')


def parse_split_range(value):
    """Parse ``role=first-last`` without assigning dataset roles implicitly."""
    if not isinstance(value, str) or '=' not in value:
        raise ValueError('split ranges must use role=first-last syntax')
    role, bounds = value.split('=', 1)
    match = re.fullmatch(r'(\d+)-(\d+)', bounds)
    if role not in ROLES or match is None:
        raise ValueError('split ranges must use training, validation, or test followed by first-last')
    first, last = map(int, match.groups())
    if first > last:
        raise ValueError('split range start must not exceed its end')
    return role, first, last


def _read_labels(path):
    rows = {}
    try:
        text = Path(path).expanduser().read_text(encoding='utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('annotation file must be UTF-8 text') from exc
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f'{path}:{line_number}: expected image ID and vehicle count')
        try:
            image_id, count = int(fields[0]), int(fields[1])
            values = [int(value) for value in fields[2:]]
        except ValueError as exc:
            raise ValueError(f'{path}:{line_number}: IDs, counts, and boxes must be integers') from exc
        if image_id < 0 or count < 0 or len(values) != count * 4:
            raise ValueError(f'{path}:{line_number}: expected {count} boxes as x y width height groups')
        if image_id in rows:
            raise ValueError(f'{path}:{line_number}: duplicate image ID {image_id}')
        rows[image_id] = [values[index:index + 4] for index in range(0, len(values), 4)]
    if not rows:
        raise ValueError('annotation file contains no image rows')
    return rows, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _image_path(directory, image_id):
    directory = Path(directory).expanduser().resolve(strict=True)
    names = (f'img_{image_id:05d}.jpg', f'img_{image_id}.jpg')
    candidates = [directory / name for name in dict.fromkeys(names) if (directory / name).is_file()]
    if len(candidates) != 1:
        if not candidates:
            raise ValueError(f'missing image for ID {image_id} (expected {names[0]} or {names[-1]})')
        raise ValueError(f'ambiguous image names for ID {image_id}: {", ".join(path.name for path in candidates)}')
    return candidates[0]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(labels_path, images_dir, split_ranges, destination, *, exclude_image_ids=()):
    """Build a COCO ZIP from operator-assigned ID ranges and source labels.

    Images are assigned only to roles explicitly provided in ``split_ranges``.
    An out-of-bounds annotation rejects the entire image; callers may exclude
    that image explicitly, but may not drop only the offending box.
    """
    parsed = [parse_split_range(value) if isinstance(value, str) else tuple(value)
              for value in split_ranges]
    if not parsed:
        raise ValueError('at least one explicit split range is required')
    assignments = {}
    role_ranges = {role: [] for role in ROLES}
    for role, first, last in parsed:
        if role not in ROLES or isinstance(first, bool) or isinstance(last, bool):
            raise ValueError('split ranges must use a supported role and integer bounds')
        if not isinstance(first, int) or not isinstance(last, int) or first < 0 or first > last:
            raise ValueError('split range bounds must be nonnegative integers in ascending order')
        role_ranges[role].append([first, last])
        for image_id in range(first, last + 1):
            previous = assignments.setdefault(image_id, role)
            if previous != role:
                raise ValueError(f'image ID {image_id} is assigned to multiple evidence roles')

    excluded = set(exclude_image_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in excluded):
        raise ValueError('excluded image IDs must be nonnegative integers')
    unknown_exclusions = excluded - assignments.keys()
    if unknown_exclusions:
        raise ValueError(f'excluded image IDs are not in a selected range: {sorted(unknown_exclusions)}')
    selected = {image_id: role for image_id, role in assignments.items() if image_id not in excluded}
    if not selected:
        raise ValueError('all selected image IDs were excluded')
    labels, label_sha256 = _read_labels(labels_path)
    missing = sorted(set(selected) - labels.keys())
    if missing:
        raise ValueError(f'annotation file is missing selected image IDs: {missing[:10]}')

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Pillow is required to package nighttime vehicle images') from exc

    split_rows = {role: {'images': [], 'annotations': []} for role in ROLES}
    image_files = {}
    annotation_id = 1
    for image_id, role in sorted(selected.items()):
        path = _image_path(images_dir, image_id)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if width <= 0 or height <= 0 or width * height > 24_000_000:
            raise ValueError(f'image {image_id} has unsupported dimensions {width}x{height}')
        file_name = f'images/{path.name}'
        image_sha256 = _sha256_file(path)
        image_files[(role, image_id)] = (file_name, path, image_sha256)
        split_rows[role]['images'].append({
            'id': image_id,
            'file_name': file_name,
            'width': width,
            'height': height,
            'condition': 'night',
            'source_dataset': 'Nighttime Vehicle Detection Dataset',
            'source_image_id': image_id,
            'source_sha256': image_sha256,
        })
        for x, y, box_width, box_height in labels[image_id]:
            if (x < 0 or y < 0 or box_width <= 0 or box_height <= 0
                    or x + box_width > width or y + box_height > height):
                raise ValueError(f'image {image_id} has an out-of-bounds box {(x, y, box_width, box_height)}; '
                                 'exclude the complete image ID explicitly or correct the source labels')
            split_rows[role]['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'bbox': [x, y, box_width, box_height],
                'area': box_width * box_height,
                'iscrowd': 0,
            })
            annotation_id += 1

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'dataset output already exists: {destination}')
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{destination.name}.', suffix='.tmp',
                dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(temporary_name, 'w', compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            for role in ROLES:
                if not role_ranges[role]:
                    continue
                rows = split_rows[role]
                coco = {
                    'info': {
                        'description': 'Explicitly selected nighttime vehicle images; no UTC or temporal track labels',
                        'source_dataset': 'University of Nevada, Reno Nighttime Vehicle Detection Dataset',
                        'source_annotation_sha256': label_sha256,
                        'selected_ranges': role_ranges[role],
                        'excluded_image_ids': sorted(excluded & set(assignments)),
                    },
                    'images': rows['images'],
                    'annotations': rows['annotations'],
                    'categories': [{'id': 1, 'name': 'vehicle', 'kind': 'vehicle', 'label': 'vehicle'}],
                }
                archive.writestr(f'{role}/annotations.json',
                    json.dumps(coco, ensure_ascii=False, separators=(',', ':')))
                archive.writestr(f'{role}/images.json',
                    json.dumps(rows['images'], ensure_ascii=False, separators=(',', ':')))
                for image_id in sorted(selected):
                    if selected[image_id] == role:
                        member, path, expected_sha256 = image_files[(role, image_id)]
                        if _sha256_file(path) != expected_sha256:
                            raise ValueError(f'image source changed while packaging: {path}')
                        archive.write(path, f'{role}/{member}')
        Path(temporary_name).replace(destination)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()
    return {
        'output': str(destination),
        'annotation_sha256': label_sha256,
        'images': sum(len(rows['images']) for rows in split_rows.values()),
        'annotations': sum(len(rows['annotations']) for rows in split_rows.values()),
        'excluded_image_ids': sorted(excluded),
        'splits': {role: {'images': len(rows['images']), 'annotations': len(rows['annotations'])}
                   for role, rows in split_rows.items() if role_ranges[role]},
    }
