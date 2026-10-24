"""One database per mission, immutable source snapshots and revisioned human labels."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from .domain import capture_time, named_zone


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(root, relative):
    root = Path(root).resolve(strict=True)
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Source path escapes configured folder')
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink() or (hasattr(cursor, 'is_junction') and cursor.is_junction()):
            raise ValueError('Linked source paths are not allowed')
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError('Source must be a regular file within the authorized root')
    return resolved


class Library:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def location(self, mission):
        if str(uuid.UUID(mission)) != mission:
            raise ValueError('Invalid mission ID')
        return self.root / mission

    def connect(self, mission):
        path = self.location(mission) / 'mission.sqlite3'
        if not path.is_file():
            raise ValueError('Mission not found')
        conn = sqlite3.connect(path, timeout=20)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    def create(self, name, country, zone):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
            raise ValueError('Mission name must contain 1–120 characters')
        # Finland is a sample-only option; operational target countries remain supported.
        if country not in ('Iran', 'Lebanon', 'Israel', 'Latvia', 'Finland'):
            raise ValueError('Unsupported country')
        named_zone(zone)
        mission = str(uuid.uuid4())
        folder = self.location(mission)
        folder.mkdir()
        (folder / 'sources').mkdir()
        conn = sqlite3.connect(folder / 'mission.sqlite3')
        try:
            with conn:
                conn.executescript('''
                CREATE TABLE mission(id TEXT PRIMARY KEY,name TEXT,country TEXT,zone TEXT,created TEXT);
                CREATE TABLE runs(id TEXT PRIMARY KEY,preset TEXT,source_root TEXT,status TEXT,created TEXT,error TEXT);
                CREATE TABLE images(id TEXT PRIMARY KEY,run_id TEXT REFERENCES runs(id),name TEXT,capture_utc TEXT,
                    precision TEXT,sha256 TEXT,snapshot TEXT,provenance TEXT,UNIQUE(run_id,name));
                CREATE TABLE reviews(id INTEGER PRIMARY KEY,image_id TEXT REFERENCES images(id),revision INTEGER,
                    operator TEXT,role TEXT,status TEXT,annotations TEXT,created TEXT,UNIQUE(image_id,revision));
                CREATE TRIGGER immutable_images_update BEFORE UPDATE ON images BEGIN SELECT RAISE(ABORT,'immutable source'); END;
                CREATE TRIGGER immutable_images_delete BEFORE DELETE ON images BEGIN SELECT RAISE(ABORT,'immutable source'); END;
                CREATE TRIGGER immutable_reviews_update BEFORE UPDATE ON reviews BEGIN SELECT RAISE(ABORT,'append-only review'); END;
                CREATE TRIGGER immutable_reviews_delete BEFORE DELETE ON reviews BEGIN SELECT RAISE(ABORT,'append-only review'); END;
                ''')
                conn.execute('INSERT INTO mission VALUES (?,?,?,?,?)', (mission, name.strip(), country, zone, now()))
        finally:
            conn.close()
        return mission

    def missions(self):
        result = []
        for path in self.root.glob('*/mission.sqlite3'):
            with self.connect(path.parent.name) as conn:
                result.append(dict(conn.execute('SELECT * FROM mission').fetchone()))
        return sorted(result, key=lambda row: row['created'], reverse=True)

    def import_folder(self, mission, root, preset):
        root = Path(root).resolve(strict=True)
        if not root.is_dir() or self.root.is_relative_to(root) or root.is_relative_to(self.root):
            raise ValueError('Choose a source outside application-managed storage')
        if preset not in ('fintraffic', 'camera'):
            raise ValueError('Select an explicit import preset')
        entries = []
        if preset == 'fintraffic':
            manifest = safe_file(root, 'manifest.json')
            data = json.loads(manifest.read_text(encoding='utf-8'))
            manifest_hash = digest(manifest)
            for row in data['downloads']:
                if 'file' not in row or 'sha256' not in row:
                    raise ValueError('Manifest entry lacks file/checksum')
                entries.append((row['file'], row['sha256'], {'preset': preset, 'manifest_sha256': manifest_hash,
                    'attribution': data.get('attribution'), 'original': row}))
        else:
            for base, directories, files in os.walk(root, followlinks=False):
                directories[:] = sorted(d for d in directories if not (Path(base) / d).is_symlink())
                for name in sorted(files):
                    if Path(name).suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        entries.append((str((Path(base) / name).relative_to(root)), None, {'preset': preset}))
        if not entries:
            raise ValueError('No images found')
        run = str(uuid.uuid4())
        with self.connect(mission) as conn:
            conn.execute('INSERT INTO runs VALUES (?,?,?,?,?,?)', (run, preset, str(root), 'importing', now(), None))
        folder = self.location(mission) / 'sources' / run
        folder.mkdir()
        try:
            with self.connect(mission) as conn:
                seen = set()
                for relative, expected, provenance in entries:
                    if relative in seen:
                        raise ValueError('Duplicate source entry')
                    seen.add(relative)
                    source = safe_file(root, relative)
                    timestamp, precision = capture_time(source.name, preset)
                    before = source.stat()
                    image = str(uuid.uuid4())
                    target = folder / (image + source.suffix.lower())
                    shutil.copyfile(source, target)
                    actual = digest(target)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('Source changed during import; retry after transfer finishes')
                    if actual != digest(source) or (expected and actual != expected):
                        raise ValueError('Source checksum mismatch: ' + relative)
                    conn.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?)',
                        (image, run, relative, timestamp.isoformat(), precision, actual,
                         str(target.relative_to(self.location(mission))), json.dumps(provenance, ensure_ascii=False)))
                conn.execute("UPDATE runs SET status='ready_for_review' WHERE id=?", (run,))
            return {'run': run, 'images': len(entries), 'status': 'ready_for_review',
                    'inference': 'not_run', 'message': 'Source import only; no automatic detections'}
        except Exception as exc:
            shutil.rmtree(folder)
            with self.connect(mission) as conn:
                conn.execute("UPDATE runs SET status='failed',error=? WHERE id=?", (str(exc), run))
            raise

    def images(self, mission, prediction_run=None):
        with self.connect(mission) as conn:
            rows = conn.execute('''SELECT images.*, COALESCE(reviews.revision,0) revision,
                COALESCE(reviews.status,'unreviewed') review_status, COALESCE(reviews.annotations,'[]') annotations, COALESCE(reviews.role,'none') evidence_role, COALESCE(reviews.operator,'') operator
                FROM images LEFT JOIN reviews ON reviews.image_id=images.id AND reviews.revision=
                (SELECT MAX(r.revision) FROM reviews r WHERE r.image_id=images.id)
                ORDER BY capture_utc,name''').fetchall()
        from .inference_store import enrich
        return enrich(self,mission,[dict(row) for row in rows],prediction_run)

    def image_path(self, mission, image):
        with self.connect(mission) as conn:
            row = conn.execute('SELECT * FROM images WHERE id=?', (image,)).fetchone()
        if not row:
            raise ValueError('Image not found in mission')
        path = safe_file(self.location(mission), row['snapshot'])
        if digest(path) != row['sha256']:
            raise ValueError('Saved image checksum mismatch')
        return path

    def review(self, mission, image, expected_revision, annotations, operator, role='none', status='reviewed', prediction_run=None):
        from .inference_store import prepare,link_review
        prepare(self,mission)
        if role not in ('none', 'training', 'validation', 'test') or status not in ('reviewed', 'partial'):
            raise ValueError('Invalid review role/status')
        if not isinstance(operator, str) or not 1 <= len(operator.strip()) <= 120:
            raise ValueError('Operator name required')
        if not isinstance(annotations, list) or len(annotations) > 1000:
            raise ValueError('Invalid annotation list')
        for item in annotations:
            if not isinstance(item,dict):raise ValueError('Invalid annotation object')
            if item.get('kind') not in ('vehicle', 'light', 'candidate', 'background'):
                raise ValueError('Invalid annotation kind')
            if not isinstance(item.get('label'), str) or not 1 <= len(item['label']) <= 120:
                raise ValueError('Annotation label required')
            if item.get('night_class') is not None and item['night_class'] not in ('headlight','streetlight','reflection','other'):
                raise ValueError('Invalid explicit night class')
            if 'review_confidence' in item and (type(item['review_confidence']) is not int or not 1<=item['review_confidence']<=5):
                raise ValueError('Review confidence must be an integer from 1 to 5')
            box = item.get('box')
            if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in box):
                raise ValueError('Rectangle must use normalized image coordinates')
            x, y, w, h = box
            if w <= 0 or h <= 0 or x+w > 1.000001 or y+h > 1.000001:
                raise ValueError('Rectangle outside image')
        with self.connect(mission) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not conn.execute('SELECT 1 FROM images WHERE id=?', (image,)).fetchone():
                raise ValueError('Image not found in this mission')
            previous = conn.execute('SELECT * FROM reviews WHERE image_id=? ORDER BY revision DESC LIMIT 1', (image,)).fetchone()
            revision = previous['revision'] if previous else 0
            if revision != expected_revision:
                raise ValueError('Review changed. Reload before saving.')
            # Holdout roles are sticky across revisions; protect independent evaluation.
            protected = conn.execute("SELECT role FROM reviews WHERE image_id=? AND role IN ('validation','test') LIMIT 1", (image,)).fetchone()
            if protected and role != protected['role']:
                raise ValueError('Held-out image cannot change evidence role')
            cursor=conn.execute('INSERT INTO reviews(image_id,revision,operator,role,status,annotations,created) VALUES (?,?,?,?,?,?,?)',
                (image, revision+1, operator.strip(), role, status, json.dumps(annotations, ensure_ascii=False), now()))
            baseline=conn.execute('SELECT annotations FROM predictions WHERE inference_id=? AND image_id=?',(prediction_run,image)).fetchone() if prediction_run else None
            before=json.loads(previous['annotations']) if previous else json.loads(baseline['annotations']) if baseline else []
            link_review(conn,cursor.lastrowid,image,prediction_run,before,annotations)
        return {'revision': revision+1}
