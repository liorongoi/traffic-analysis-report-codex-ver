"""Convert RoundaboutHD single-camera tracking labels to RoadLoom manifests."""
import hashlib
import json
import math
from pathlib import Path


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _number(token, name):
    try:
        value = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def convert_sct_labels(label_sources, *, frame_count, width, height, frame_offset=0,
                       country='United States', allow_duplicate_track_rows=False):
    """Return a paired-tracking ground-truth manifest.

    ``label_sources`` is an iterable of ``(sequence_id, path)`` pairs. SCT
    rows may use the documented six-column layout (frame, object ID, xmin,
    ymin, xmax, ymax) or the observed seven-column variant with a trailing
    source label. That label is preserved without assigning it semantics.
    ``frame_offset`` is explicit because frame numbering must never be guessed.
    Every frame in the requested clip is emitted, including empty frames.
    """
    frame_count = _positive_integer(frame_count, 'frame_count')
    width = _positive_integer(width, 'width')
    height = _positive_integer(height, 'height')
    if isinstance(frame_offset, bool) or not isinstance(frame_offset, int):
        raise ValueError('frame_offset must be an integer')
    if country is not None and (not isinstance(country, str) or not country.strip()):
        raise ValueError('country must be a non-empty string or null')
    if not isinstance(allow_duplicate_track_rows, bool):
        raise ValueError('allow_duplicate_track_rows must be a boolean')

    sequences = []
    seen_ids = set()
    for sequence_id, source_path in label_sources:
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            raise ValueError('sequence IDs must be non-empty strings')
        sequence_id = sequence_id.strip()
        if sequence_id in seen_ids:
            raise ValueError(f'duplicate sequence ID: {sequence_id}')
        seen_ids.add(sequence_id)

        path = Path(source_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'annotation source is not a file: {path}')
        data = path.read_bytes()
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError(f'{path} must be UTF-8 text') from exc

        frames = [[] for _ in range(frame_count)]
        occupied = set()
        duplicate_track_id_frame_rows = 0
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            if len(fields) not in (6, 7):
                raise ValueError(f'{path}:{line_number}: expected 6 or 7 columns, got {len(fields)}')
            try:
                source_frame = int(fields[0])
                object_id = int(fields[1])
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: frame and object IDs must be integers') from exc
            frame_id = source_frame - frame_offset
            if frame_id < 0 or frame_id >= frame_count:
                raise ValueError(f'{path}:{line_number}: frame ID {source_frame} is outside the requested clip')
            x1, y1, x2, y2 = (_number(value, 'bounding-box coordinate') for value in fields[2:6])
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
                raise ValueError(f'{path}:{line_number}: bounding box is invalid or outside {width}x{height}')
            identity = (frame_id, object_id)
            if identity in occupied:
                if not allow_duplicate_track_rows:
                    raise ValueError(f'{path}:{line_number}: duplicate object {object_id} in frame {source_frame}')
                duplicate_track_id_frame_rows += 1
            occupied.add(identity)
            annotation = {
                'track_id': f'{sequence_id}:{object_id}',
                'category_id': 'vehicle',
                'bbox': [x1, y1, x2 - x1, y2 - y1],
            }
            if len(fields) == 7:
                if not fields[6]:
                    raise ValueError(f'{path}:{line_number}: trailing source label must not be empty')
                annotation['source_label'] = fields[6]
            frames[frame_id].append(annotation)

        for objects in frames:
            objects.sort(key=lambda item: item['track_id'])
        metadata = {
            'dataset': 'RoundaboutHD',
            'camera_view': sequence_id,
            'frame_width': width,
            'frame_height': height,
            'frame_count': frame_count,
            'frame_offset': frame_offset,
            'duplicate_track_id_frame_rows': duplicate_track_id_frame_rows,
            'source_annotation': path.name,
            'source_annotation_sha256': hashlib.sha256(data).hexdigest(),
        }
        if country:
            metadata['country'] = country.strip()
        sequences.append({
            'sequence_id': sequence_id,
            'metadata': metadata,
            'frames': [{'frame_id': index, 'objects': objects}
                       for index, objects in enumerate(frames)],
        })

    if not sequences:
        raise ValueError('at least one annotation source is required')
    sequences.sort(key=lambda item: item['sequence_id'])
    return {'sequences': sequences}


def write_manifest(manifest, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
