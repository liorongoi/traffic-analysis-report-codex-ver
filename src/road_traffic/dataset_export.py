"""Package explicitly reviewed, split-protected images as a COCO dataset."""
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone


ROLES = ('training', 'validation', 'test')


def _reviewed_regions(raw):
    result = []
    for index, item in enumerate(json.loads(raw or '[]')):
        if (item.get('origin') != 'human' or item.get('reviewed') is not True
                or item.get('rejected') is True):
            continue
        label = item.get('label') or item.get('night_class')
        kind = item.get('kind')
        box = item.get('box')
        if not isinstance(kind, str) or not kind or not isinstance(label, str) or not label:
            continue
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
            continue
        x, y, width, height = map(float, box)
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
            continue
        result.append({'index': index, 'kind': kind, 'label': label, 'box': [x, y, width, height]})
    return result


def _image_numeric_id(image_id):
    return int(hashlib.sha256(image_id.encode('utf-8')).hexdigest()[:15], 16)


def build(library, mission):
    """Return a temporary directory and COCO ZIP path for a stable review snapshot.

    Only fully reviewed images with a saved training/validation/test role are
    packaged. Partially reviewed images are excluded because omitted regions
    cannot safely be treated as background. Known cross-role byte-identical
    images and camera/hour overlap stop export until the split is corrected.
    """
    temporary = tempfile.TemporaryDirectory(prefix='roadloom-learning-dataset-', dir=library.root)
    root = Path(temporary.name)
    index_db = root / 'snapshot.sqlite3'
    output = root / 'roadloom-learning-dataset.zip'
    conn = sqlite3.connect(index_db)
    conn.execute('''CREATE TABLE records (
        role TEXT NOT NULL, image_id TEXT NOT NULL, sha256 TEXT NOT NULL,
        name TEXT NOT NULL, capture_utc TEXT NOT NULL, revision INTEGER NOT NULL,
        annotations TEXT NOT NULL, view_hour TEXT, view_id TEXT, PRIMARY KEY(role, image_id))''')
    conn.execute('CREATE INDEX record_hash ON records(sha256)')
    snapshot_rowid = 0
    partial_count = 0
    try:
        mission_settings = library.settings(mission)
        target_country = mission_settings.get('country')
        if mission_settings['input_type'] != 'images':
            raise ValueError('Reviewed image datasets can only be exported from still-image missions')
        with library.connect(mission) as db:
            snapshot_rowid = db.execute('SELECT COALESCE(MAX(rowid),0) FROM images').fetchone()[0]
        categories = set()
        for row in library.iter_images(mission, batch_size=200, snapshot_rowid=snapshot_rowid):
            role = row.get('evidence_role')
            if role not in ROLES:
                continue
            if row.get('review_status') != 'reviewed':
                if row.get('review_status') == 'partial':
                    partial_count += 1
                continue
            regions = _reviewed_regions(row.get('annotations'))
            categories.update((item['kind'], item['label']) for item in regions)
            view = row.get('assigned_view_id') or ''
            if not view and json.loads(row.get('provenance') or '{}').get('preset') == 'fintraffic':
                view = Path(row['name']).stem.split('_')[0]
            view_hour = f"{view}\0{row['capture_utc'][:13]}" if view else None
            conn.execute('INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?,?)',
                (role, row['id'], row['sha256'], row['name'], row['capture_utc'],
                 row['revision'], json.dumps(regions, ensure_ascii=False, separators=(',', ':')),
                 view_hour, view or None))
        conn.commit()
        if not categories:
            raise ValueError('No explicit reviewed human labels are available for dataset export')

        # Detect protected split leaks only among complete reviews eligible for this package.
        hash_roles = {}
        group_roles = {}
        for role, sha, view_hour in conn.execute('SELECT role,sha256,view_hour FROM records'):
            hash_roles.setdefault(sha, set()).add(role)
            if view_hour:
                group_roles.setdefault(view_hour, set()).add(role)
        if any(len(roles) > 1 for roles in hash_roles.values()):
            raise ValueError('Dataset export stopped: identical image bytes occur across review splits. Reassign the duplicate to one split first.')
        if any(len(roles) > 1 for roles in group_roles.values()):
            raise ValueError('Dataset export stopped: a known camera/hour group occurs across review splits. Correct the split assignments first.')

        category_rows = [
            {'id': i + 1, 'name': f'{kind}:{label}', 'kind': kind, 'label': label}
            for i, (kind, label) in enumerate(sorted(categories))
        ]
        category_ids = {(row['kind'], row['label']): row['id'] for row in category_rows}
        counts = {role: {'images': 0, 'annotations': 0} for role in ROLES}
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for role in ROLES:
                images_member = f'{role}/images.json'
                annotations_member = f'{role}/annotations.json'
                for image_id, in conn.execute(
                        'SELECT image_id FROM records WHERE role=? ORDER BY capture_utc,name,image_id', (role,)):
                    source = library.image_path(mission, image_id)
                    archive.write(source, f'{role}/images/{image_id}{source.suffix.lower()}')
                with archive.open(images_member, 'w') as stream:
                    stream.write(b'[')
                    first = True
                    for image_id, sha, name, capture_utc, revision, view_id in conn.execute(
                            'SELECT image_id,sha256,name,capture_utc,revision,view_id FROM records WHERE role=? ORDER BY capture_utc,name,image_id', (role,)):
                        source = library.image_path(mission, image_id)
                        from PIL import Image
                        with Image.open(source) as image:
                            width, height = image.size
                        item = {'id': _image_numeric_id(image_id), 'file_name': f'images/{image_id}{source.suffix.lower()}',
                            'width': width, 'height': height, 'capture_utc': capture_utc,
                            'source_id': image_id, 'source_sha256': sha,
                            'review_revision': revision, 'review_role': role,
                            'country': target_country, 'mission_country_context': target_country,
                            'camera_view': view_id, 'view_id': view_id,
                            'country_context_basis': 'mission_configuration'}
                        if not first:
                            stream.write(b',')
                        stream.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                        first = False
                        counts[role]['images'] += 1
                    stream.write(b']')

                with archive.open(annotations_member, 'w') as stream:
                    stream.write(b'{"images":')
                    stream.write(b'[')
                    first_image = True
                    for image_id, sha, capture_utc, revision, view_id in conn.execute(
                            'SELECT image_id,sha256,capture_utc,revision,view_id FROM records WHERE role=? ORDER BY capture_utc,name,image_id', (role,)):
                        source = library.image_path(mission, image_id)
                        from PIL import Image
                        with Image.open(source) as image:
                            width, height = image.size
                        item = {'id': _image_numeric_id(image_id),
                            'file_name': f'images/{image_id}{source.suffix.lower()}',
                            'width': width, 'height': height, 'capture_utc': capture_utc,
                            'source_id': image_id, 'source_sha256': sha,
                            'review_revision': revision, 'review_role': role,
                            'country': target_country, 'mission_country_context': target_country,
                            'camera_view': view_id, 'view_id': view_id,
                            'country_context_basis': 'mission_configuration'}
                        if not first_image:
                            stream.write(b',')
                        stream.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                        first_image = False
                    stream.write(b']')
                    stream.write(b',"annotations":[')
                    first = True
                    for image_id, sha, raw in conn.execute(
                            'SELECT image_id,sha256,annotations FROM records WHERE role=? ORDER BY capture_utc,name,image_id', (role,)):
                        image_path = library.image_path(mission, image_id)
                        from PIL import Image
                        with Image.open(image_path) as image:
                            width, height = image.size
                        numeric_image_id = _image_numeric_id(image_id)
                        for region in json.loads(raw):
                            x, y, box_width, box_height = region['box']
                            annotation = {'id': _image_numeric_id(f"{image_id}:{region['index']}"),
                                'image_id': numeric_image_id, 'category_id': category_ids[(region['kind'], region['label'])],
                                'bbox': [x * width, y * height, box_width * width, box_height * height],
                                'area': box_width * width * box_height * height, 'iscrowd': 0,
                                'source_annotation_index': region['index']}
                            if not first:
                                stream.write(b',')
                            stream.write(json.dumps(annotation, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                            first = False
                            counts[role]['annotations'] += 1
                    stream.write(b'],"categories":')
                    stream.write(json.dumps(category_rows, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                    stream.write(b'}')

            evidence_hash = hashlib.sha256()
            for record in conn.execute('''SELECT role,image_id,sha256,revision,annotations
                    FROM records ORDER BY role,image_id'''):
                evidence_hash.update(json.dumps(record, ensure_ascii=False,
                    separators=(',', ':')).encode('utf-8'))
                evidence_hash.update(b'\n')
            manifest = {'format': 'RoadLoom COCO dataset v1', 'mission_id': mission,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'source_snapshot_max_rowid': snapshot_rowid,
                'evidence_version_sha256': evidence_hash.hexdigest(),
                'label_policy': 'Latest full image reviews only; explicit human reviewed non-rejected boxes only. Model proposals and bulk approvals are excluded. Partial reviews are excluded, not treated as negatives.',
                'country_context_basis': 'mission_configuration; not inferred from image contents',
                'split_policy': 'Training, validation and test are preserved. Export stops on cross-split duplicate source hashes or known camera/hour overlap.',
                'partial_reviews_excluded': partial_count, 'roles': counts, 'categories': category_rows,
                'vehicle_identity_inference': False, 'source_images_are_copies': True}
            archive.writestr('dataset-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        return temporary, output
    except Exception:
        conn.close()
        temporary.cleanup()
        raise
    finally:
        if conn:
            conn.close()
