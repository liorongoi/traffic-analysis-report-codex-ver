"""One database per mission, immutable source snapshots and revisioned human labels."""
import csv
import hashlib
import io
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
        # Add configuration storage to databases created before the mission
        # wizard. This migration is additive and preserves all existing work.
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mission_settings'").fetchone():
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS mission_settings(
                    mission_id TEXT PRIMARY KEY REFERENCES mission(id),
                    input_type TEXT NOT NULL DEFAULT 'images',
                    analysis_mode TEXT NOT NULL DEFAULT 'general_traffic',
                    location TEXT NOT NULL DEFAULT '',
                    source_root TEXT NOT NULL DEFAULT '',
                    include_subdirectories INTEGER NOT NULL DEFAULT 1,
                    processing_mode TEXT NOT NULL DEFAULT 'once',
                    output_root TEXT NOT NULL DEFAULT '',
                    export_formats TEXT NOT NULL DEFAULT '["csv"]'
                );
                INSERT OR IGNORE INTO mission_settings(mission_id)
                SELECT id FROM mission;
            ''')
        conn.execute('''CREATE TABLE IF NOT EXISTS exports(
            id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
            inference_run TEXT, format TEXT NOT NULL, output_path TEXT NOT NULL,
            snapshot_sha256 TEXT, row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL, created TEXT NOT NULL, finished TEXT, error TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS monitored_files(
            relative_path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            stable_since TEXT NOT NULL, last_seen TEXT NOT NULL, imported_sha256 TEXT,
            status TEXT NOT NULL, error TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS monitor_state(
            id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL,
            last_scan TEXT, last_import TEXT, error TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS scan_windows(
            id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
            start_date TEXT, end_date TEXT, weekdays TEXT NOT NULL,
            start_time TEXT NOT NULL, end_time TEXT NOT NULL, created TEXT NOT NULL
        )''')
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS views(
                id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
                name TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
                UNIQUE(mission_id,name));
            CREATE TABLE IF NOT EXISTS image_view_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT NOT NULL REFERENCES mission(id),
                image_id TEXT NOT NULL REFERENCES images(id), view_id TEXT REFERENCES views(id),
                revision INTEGER NOT NULL, operator TEXT NOT NULL, created TEXT NOT NULL,
                UNIQUE(image_id,revision));
            CREATE TABLE IF NOT EXISTS view_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT NOT NULL REFERENCES mission(id),
                view_id TEXT REFERENCES views(id), action TEXT NOT NULL, details TEXT NOT NULL,
                operator TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS view_region_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT NOT NULL REFERENCES mission(id),
                region_key TEXT NOT NULL, view_id TEXT NOT NULL REFERENCES views(id), revision INTEGER NOT NULL,
                action TEXT NOT NULL, kind TEXT NOT NULL, polygon TEXT NOT NULL,
                operator TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(region_key,revision));
            CREATE INDEX IF NOT EXISTS view_region_history_latest_idx
                ON view_region_history(mission_id,view_id,region_key,revision);
            CREATE INDEX IF NOT EXISTS image_view_history_mission_idx
                ON image_view_history(mission_id,image_id,revision);
            CREATE TRIGGER IF NOT EXISTS immutable_image_view_history_u BEFORE UPDATE ON image_view_history
                BEGIN SELECT RAISE(ABORT,'view assignment history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_image_view_history_d BEFORE DELETE ON image_view_history
                BEGIN SELECT RAISE(ABORT,'view assignment history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_view_events_u BEFORE UPDATE ON view_events
                BEGIN SELECT RAISE(ABORT,'view events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_view_events_d BEFORE DELETE ON view_events
                BEGIN SELECT RAISE(ABORT,'view events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_view_region_history_u BEFORE UPDATE ON view_region_history
                BEGIN SELECT RAISE(ABORT,'view region history is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_view_region_history_d BEFORE DELETE ON view_region_history
                BEGIN SELECT RAISE(ABORT,'view region history is immutable'); END;
        ''')
        return conn

    def create(self, name, country, zone, settings=None):
        settings = settings or {}
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
            raise ValueError('Mission name must contain 1–120 characters')
        # Finland is a sample-only option; operational target countries remain supported.
        if country not in ('Iran', 'Lebanon', 'Israel', 'Latvia', 'Finland'):
            raise ValueError('Unsupported country')
        named_zone(zone)
        input_type = settings.get('input_type', 'images')
        analysis_mode = settings.get('analysis_mode', 'general_traffic')
        processing_mode = settings.get('processing_mode', 'once')
        export_formats = settings.get('export_formats', ['csv'])
        if isinstance(export_formats, str):
            export_formats = {'csv': ['csv'], 'xlsx': ['xlsx'], 'both': ['csv', 'xlsx']}.get(export_formats, [])
        if input_type not in ('images', 'videos'):
            raise ValueError('Input type must be images or videos')
        supported_modes = ('general_traffic', 'object_search', 'license_plate') if input_type == 'images' else ('video_traffic', 'object_search', 'license_plate')
        if analysis_mode not in supported_modes:
            raise ValueError('Analysis mode is not valid for this input type')
        if processing_mode not in ('once', 'continuous'):
            raise ValueError('Processing mode must be once or continuous')
        if type(settings.get('include_subdirectories', True)) is not bool:
            raise ValueError('include_subdirectories must be a boolean')
        if not isinstance(export_formats, list) or not export_formats or any(not isinstance(value, str) or value not in ('csv', 'xlsx') for value in export_formats) or len(set(export_formats)) != len(export_formats):
            raise ValueError('Choose CSV, XLSX, or both export formats')
        location = settings.get('location', '')
        if not isinstance(location, str) or len(location.strip()) > 200:
            raise ValueError('Location must be a text value up to 200 characters')
        source_root = self._configured_directory(settings.get('source_root', ''), 'source', writable=False)
        output_root = self._configured_directory(settings.get('output_root', ''), 'output', writable=True)
        if source_root and (self.root.is_relative_to(Path(source_root)) or Path(source_root).is_relative_to(self.root)):
            raise ValueError('Choose a source outside application-managed storage')
        if output_root and (self.root.is_relative_to(Path(output_root)) or Path(output_root).is_relative_to(self.root)):
            raise ValueError('Choose an output outside application-managed storage')
        if source_root and output_root and (
            Path(source_root).is_relative_to(Path(output_root)) or
            Path(output_root).is_relative_to(Path(source_root))
        ):
            raise ValueError('Source and output directories must not overlap')
        mission = str(uuid.uuid4())
        folder = self.location(mission)
        folder.mkdir()
        (folder / 'sources').mkdir()
        conn = sqlite3.connect(folder / 'mission.sqlite3')
        try:
            with conn:
                conn.executescript('''
                CREATE TABLE mission(id TEXT PRIMARY KEY,name TEXT,country TEXT,zone TEXT,created TEXT);
                CREATE TABLE mission_settings(
                    mission_id TEXT PRIMARY KEY REFERENCES mission(id),
                    input_type TEXT NOT NULL,
                    analysis_mode TEXT NOT NULL,
                    location TEXT NOT NULL,
                    source_root TEXT NOT NULL,
                    include_subdirectories INTEGER NOT NULL,
                    processing_mode TEXT NOT NULL,
                    output_root TEXT NOT NULL,
                    export_formats TEXT NOT NULL
                );
                CREATE TABLE exports(
                    id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
                    inference_run TEXT, format TEXT NOT NULL, output_path TEXT NOT NULL,
                    snapshot_sha256 TEXT, row_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL, created TEXT NOT NULL, finished TEXT, error TEXT
                );
                CREATE TABLE monitored_files(
                    relative_path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                    stable_since TEXT NOT NULL, last_seen TEXT NOT NULL, imported_sha256 TEXT,
                    status TEXT NOT NULL, error TEXT
                );
                CREATE TABLE monitor_state(
                    id INTEGER PRIMARY KEY CHECK(id=1), status TEXT NOT NULL,
                    last_scan TEXT, last_import TEXT, error TEXT
                );
                INSERT INTO monitor_state(id,status) VALUES (1,'stopped');
                CREATE TABLE scan_windows(
                    id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
                    start_date TEXT, end_date TEXT, weekdays TEXT NOT NULL,
                    start_time TEXT NOT NULL, end_time TEXT NOT NULL, created TEXT NOT NULL
                );
                CREATE TABLE views(
                    id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
                    name TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
                    UNIQUE(mission_id,name));
                CREATE TABLE image_view_history(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT NOT NULL REFERENCES mission(id),
                    image_id TEXT NOT NULL REFERENCES images(id), view_id TEXT REFERENCES views(id),
                    revision INTEGER NOT NULL, operator TEXT NOT NULL, created TEXT NOT NULL,
                    UNIQUE(image_id,revision));
                CREATE TABLE view_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT NOT NULL REFERENCES mission(id),
                    view_id TEXT REFERENCES views(id), action TEXT NOT NULL, details TEXT NOT NULL,
                    operator TEXT NOT NULL, created TEXT NOT NULL);
                CREATE TABLE view_region_history(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, mission_id TEXT NOT NULL REFERENCES mission(id),
                    region_key TEXT NOT NULL, view_id TEXT NOT NULL REFERENCES views(id), revision INTEGER NOT NULL,
                    action TEXT NOT NULL, kind TEXT NOT NULL, polygon TEXT NOT NULL,
                    operator TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(region_key,revision));
                CREATE INDEX view_region_history_latest_idx ON view_region_history(mission_id,view_id,region_key,revision);
                CREATE INDEX image_view_history_mission_idx ON image_view_history(mission_id,image_id,revision);
                CREATE TRIGGER immutable_image_view_history_u BEFORE UPDATE ON image_view_history
                    BEGIN SELECT RAISE(ABORT,'view assignment history is immutable'); END;
                CREATE TRIGGER immutable_image_view_history_d BEFORE DELETE ON image_view_history
                    BEGIN SELECT RAISE(ABORT,'view assignment history is immutable'); END;
                CREATE TRIGGER immutable_view_events_u BEFORE UPDATE ON view_events
                    BEGIN SELECT RAISE(ABORT,'view events are immutable'); END;
                CREATE TRIGGER immutable_view_events_d BEFORE DELETE ON view_events
                    BEGIN SELECT RAISE(ABORT,'view events are immutable'); END;
                CREATE TRIGGER immutable_view_region_history_u BEFORE UPDATE ON view_region_history
                    BEGIN SELECT RAISE(ABORT,'view region history is immutable'); END;
                CREATE TRIGGER immutable_view_region_history_d BEFORE DELETE ON view_region_history
                    BEGIN SELECT RAISE(ABORT,'view region history is immutable'); END;
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
                conn.execute('INSERT INTO mission_settings VALUES (?,?,?,?,?,?,?,?,?)',
                    (mission, input_type, analysis_mode, location.strip(), source_root,
                     int(settings.get('include_subdirectories', True)), processing_mode,
                     output_root, json.dumps(export_formats)))
        finally:
            conn.close()
        return mission

    @staticmethod
    def _configured_directory(value, label, writable):
        if not value:
            return ''
        if not isinstance(value, str):
            raise ValueError(f'{label.title()} directory must be a path string')
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f'{label.title()} directory must be an absolute path on the backend workstation')
        try:
            path = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f'{label.title()} directory is unavailable') from exc
        if not path.is_dir():
            raise ValueError(f'{label.title()} path must be a directory')
        if not writable and not os.access(path, os.R_OK | os.X_OK):
            raise ValueError(f'{label.title()} directory is not readable')
        if writable and not os.access(path, os.W_OK):
            raise ValueError(f'{label.title()} directory is not writable')
        return str(path)

    def settings(self, mission):
        with self.connect(mission) as conn:
            row = conn.execute('SELECT * FROM mission_settings WHERE mission_id=?', (mission,)).fetchone()
        result = dict(row)
        result['include_subdirectories'] = bool(result['include_subdirectories'])
        result['export_formats'] = json.loads(result['export_formats'])
        return result

    def exports(self, mission):
        with self.connect(mission) as conn:
            return [dict(row) for row in conn.execute(
                'SELECT * FROM exports WHERE mission_id=? ORDER BY created DESC', (mission,)
            )]

    def scan_windows(self, mission):
        with self.connect(mission) as conn:
            return [{**dict(row), 'weekdays': json.loads(row['weekdays'])} for row in
                conn.execute('SELECT * FROM scan_windows WHERE mission_id=? ORDER BY created,id', (mission,))]

    def add_scan_window(self, mission, window):
        from .schedules import validate_window
        normalized = validate_window(window)
        identifier = str(uuid.uuid4())
        with self.connect(mission) as conn:
            conn.execute('INSERT INTO scan_windows VALUES (?,?,?,?,?,?,?,?)',
                (identifier, mission, normalized['start_date'], normalized['end_date'],
                 json.dumps(normalized['weekdays']), normalized['start_time'], normalized['end_time'], now()))
        return next(window for window in self.scan_windows(mission) if window['id'] == identifier)

    def delete_scan_window(self, mission, identifier):
        with self.connect(mission) as conn:
            cursor = conn.execute('DELETE FROM scan_windows WHERE id=? AND mission_id=?', (identifier, mission))
            if not cursor.rowcount:
                raise ValueError('Schedule window not found')
        return {'id': identifier, 'deleted': True}

    def views(self, mission):
        with self.connect(mission) as conn:
            rows=conn.execute('''WITH latest AS (
                    SELECT image_id,MAX(revision) revision FROM image_view_history WHERE mission_id=? GROUP BY image_id),
                counts AS (
                    SELECT h.view_id,COUNT(*) image_count,MIN(i.name) representative_name,MIN(i.id) representative_image_id
                    FROM latest l JOIN image_view_history h ON h.image_id=l.image_id AND h.revision=l.revision
                    JOIN images i ON i.id=h.image_id WHERE h.view_id IS NOT NULL GROUP BY h.view_id)
                SELECT v.id,v.name,v.created,v.updated,COALESCE(c.image_count,0) image_count,
                    c.representative_name,c.representative_image_id
                FROM views v LEFT JOIN counts c ON c.view_id=v.id
                WHERE v.mission_id=? ORDER BY v.name COLLATE NOCASE''',(mission,mission)).fetchall()
            return [dict(row) for row in rows]

    def view_history(self, mission, image=None):
        with self.connect(mission) as conn:
            if image and not conn.execute('SELECT 1 FROM images WHERE id=?',(image,)).fetchone():
                raise ValueError('Image not found in this mission')
            where='AND h.image_id=?' if image else ''
            params=(mission,image) if image else (mission,)
            return [dict(row) for row in conn.execute(f'''SELECT h.*,v.name view_name FROM image_view_history h
                LEFT JOIN views v ON v.id=h.view_id WHERE h.mission_id=? {where} ORDER BY h.created,h.id''',params)]

    def create_view(self, mission, name, operator):
        name=name.strip() if isinstance(name,str) else ''
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(name) <= 80:raise ValueError('View name must contain 1–80 characters')
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        identifier=str(uuid.uuid4());created=now()
        with self.connect(mission) as conn:
            if conn.execute('SELECT 1 FROM views WHERE mission_id=? AND name=? COLLATE NOCASE',(mission,name)).fetchone():
                raise ValueError('A camera view with this name already exists')
            conn.execute('INSERT INTO views VALUES (?,?,?,?,?)',(identifier,mission,name,created,created))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,identifier,'created',json.dumps({'name':name},ensure_ascii=False),operator,created))
        return {'id':identifier,'name':name,'created':created,'updated':created}

    def rename_view(self, mission, identifier, name, operator):
        name=name.strip() if isinstance(name,str) else ''
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(name) <= 80:raise ValueError('View name must contain 1–80 characters')
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        with self.connect(mission) as conn:
            current=conn.execute('SELECT name FROM views WHERE id=? AND mission_id=?',(identifier,mission)).fetchone()
            if not current:raise ValueError('Camera view not found')
            if current['name']==name:return {'id':identifier,'name':name,'unchanged':True}
            if conn.execute('SELECT 1 FROM views WHERE mission_id=? AND id<>? AND name=? COLLATE NOCASE',
                (mission,identifier,name)).fetchone():
                raise ValueError('A camera view with this name already exists')
            conn.execute('UPDATE views SET name=?,updated=? WHERE id=? AND mission_id=?',(name,now(),identifier,mission))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,identifier,'renamed',json.dumps({'before':current['name'],'after':name},ensure_ascii=False),operator,now()))
        return {'id':identifier,'name':name}

    def assign_view(self, mission, image, view_id, operator):
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        with self.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM images WHERE id=?',(image,)).fetchone():
                raise ValueError('Image not found in this mission')
            if view_id and not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=?',(view_id,mission)).fetchone():
                raise ValueError('Camera view not found in this mission')
            previous=conn.execute('SELECT view_id FROM image_view_history WHERE image_id=? ORDER BY revision DESC LIMIT 1',(image,)).fetchone()
            if previous and previous['view_id']==view_id:
                return {'image_id':image,'view_id':view_id,'unchanged':True}
            revision=conn.execute('SELECT COALESCE(MAX(revision),0)+1 FROM image_view_history WHERE image_id=?',(image,)).fetchone()[0]
            created=now()
            conn.execute('INSERT INTO image_view_history(mission_id,image_id,view_id,revision,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,image,view_id,revision,operator,created))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,view_id,'image_assigned' if view_id else 'image_unassigned',
                 json.dumps({'image_id':image,'revision':revision},ensure_ascii=False),operator,created))
            return {'image_id':image,'view_id':view_id,'revision':revision,'created':created}

    def view_regions(self, mission, view_id):
        with self.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=?',(view_id,mission)).fetchone():
                raise ValueError('Camera view not found in this mission')
            return [{**dict(row),'polygon':json.loads(row['polygon'])} for row in conn.execute('''SELECT * FROM view_region_history h
                WHERE h.mission_id=? AND h.view_id=?
                AND h.revision=(SELECT MAX(x.revision) FROM view_region_history x WHERE x.region_key=h.region_key)
                AND h.action='upsert' ORDER BY h.created,h.id''',(mission,view_id))]

    def view_region_history(self, mission, view_id):
        with self.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=?',(view_id,mission)).fetchone():
                raise ValueError('Camera view not found in this mission')
            return [{**dict(row),'polygon':json.loads(row['polygon'])} for row in conn.execute(
                'SELECT * FROM view_region_history WHERE mission_id=? AND view_id=? ORDER BY region_key,revision',
                (mission,view_id))]

    def add_view_region(self, mission, view_id, kind, polygon, operator):
        from .regions import validate_polygon
        polygon=validate_polygon(polygon)
        operator=operator.strip() if isinstance(operator,str) else ''
        if kind not in ('include','exclude'):raise ValueError('Region kind must be include or exclude')
        if not 1<=len(operator)<=120:raise ValueError('Operator is required for region history')
        key=str(uuid.uuid4());created=now()
        with self.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=?',(view_id,mission)).fetchone():
                raise ValueError('Camera view not found in this mission')
            conn.execute('''INSERT INTO view_region_history
                (mission_id,region_key,view_id,revision,action,kind,polygon,operator,created) VALUES (?,?,?,?,?,?,?,?,?)''',
                (mission,key,view_id,1,'upsert',kind,json.dumps(polygon),operator,created))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,view_id,'region_created',json.dumps({'region_id':key,'kind':kind},ensure_ascii=False),operator,created))
        return next(region for region in self.view_regions(mission,view_id) if region['region_key']==key)

    def update_view_region(self, mission, region_key, kind, polygon, operator):
        from .regions import validate_polygon
        polygon=validate_polygon(polygon)
        operator=operator.strip() if isinstance(operator,str) else ''
        if kind not in ('include','exclude'):raise ValueError('Region kind must be include or exclude')
        if not 1<=len(operator)<=120:raise ValueError('Operator is required for region history')
        with self.connect(mission) as conn:
            previous=conn.execute('SELECT * FROM view_region_history WHERE mission_id=? AND region_key=? ORDER BY revision DESC LIMIT 1',
                (mission,region_key)).fetchone()
            if not previous or previous['action']=='delete':raise ValueError('View region not found')
            revision=previous['revision']+1;created=now()
            conn.execute('''INSERT INTO view_region_history
                (mission_id,region_key,view_id,revision,action,kind,polygon,operator,created) VALUES (?,?,?,?,?,?,?,?,?)''',
                (mission,region_key,previous['view_id'],revision,'upsert',kind,json.dumps(polygon),operator,created))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,previous['view_id'],'region_updated',json.dumps({'region_id':region_key,'revision':revision},ensure_ascii=False),operator,created))
        return next(region for region in self.view_regions(mission,previous['view_id']) if region['region_key']==region_key)

    def delete_view_region(self, mission, region_key, operator):
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1<=len(operator)<=120:raise ValueError('Operator is required for region history')
        with self.connect(mission) as conn:
            previous=conn.execute('SELECT * FROM view_region_history WHERE mission_id=? AND region_key=? ORDER BY revision DESC LIMIT 1',
                (mission,region_key)).fetchone()
            if not previous or previous['action']=='delete':raise ValueError('View region not found')
            revision=previous['revision']+1;created=now()
            conn.execute('''INSERT INTO view_region_history
                (mission_id,region_key,view_id,revision,action,kind,polygon,operator,created) VALUES (?,?,?,?,?,?,?,?,?)''',
                (mission,region_key,previous['view_id'],revision,'delete',previous['kind'],previous['polygon'],operator,created))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,previous['view_id'],'region_deleted',json.dumps({'region_id':region_key,'revision':revision},ensure_ascii=False),operator,created))
        return {'id':region_key,'deleted':True,'revision':revision}

    def reconcile_exports(self):
        """Mark exports left in `writing` by a stopped process as incomplete."""
        reconciled = 0
        for mission_path in self.root.glob('*/mission.sqlite3'):
            mission = mission_path.parent.name
            with self.connect(mission) as conn:
                rows = [dict(row) for row in conn.execute(
                    "SELECT id,output_path FROM exports WHERE status='writing' AND mission_id=?", (mission,)
                )]
                for row in rows:
                    status = 'partial' if Path(row['output_path']).is_file() else 'failed'
                    conn.execute('UPDATE exports SET status=?,finished=?,error=? WHERE id=?',
                        (status, now(), 'Export process stopped before completion', row['id']))
                    reconciled += 1
        return reconciled

    def export_csv(self, mission, inference_run=None):
        settings = self.settings(mission)
        if 'xlsx' in settings['export_formats']:
            raise ValueError('This mission requests XLSX output, which is not implemented yet; no export was generated')
        if 'csv' not in settings['export_formats']:
            raise ValueError('CSV is not configured for this mission')
        if not settings['output_root']:
            raise ValueError('Set an output directory in the mission settings before exporting')
        output_root = self._configured_directory(settings['output_root'], 'output', writable=True)
        if output_root != settings['output_root']:
            raise ValueError('Configured output directory changed; update the mission settings before exporting')
        if self.root.is_relative_to(Path(output_root)) or Path(output_root).is_relative_to(self.root):
            raise ValueError('Choose an output outside application-managed storage')

        rows = self.images(mission, inference_run)
        export_id = str(uuid.uuid4())
        target = Path(output_root) / f'traffic-analysis-{mission[:8]}-{export_id}.csv'
        created = now()
        with self.connect(mission) as conn:
            conn.execute('INSERT INTO exports(id,mission_id,inference_run,format,output_path,row_count,status,created) VALUES (?,?,?,?,?,?,?,?)',
                (export_id, mission, inference_run, 'csv', str(target), 0, 'writing', created))
        row_count = 0
        try:
            with target.open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['mission_id','inference_run','source_id','source_run_id','source_sha256',
                    'filename','capture_utc','timestamp_precision','review_status','revision',
                    'confirmed_vehicle_count','candidate_count','light_count','annotations',
                    'prediction_run','inference_status','original_predictions','review_prediction_run',
                    'evidence_role','operator'])
                for row in rows:
                    annotations = json.loads(row['annotations'])
                    active = [annotation for annotation in annotations if not annotation.get('rejected')]
                    filename = row['name']
                    operator = row.get('operator', '')
                    if filename.lstrip().startswith(('=', '+', '-', '@')):
                        filename = "'" + filename
                    if operator.lstrip().startswith(('=', '+', '-', '@')):
                        operator = "'" + operator
                    writer.writerow([mission, inference_run or '', row['id'], row['run_id'], row['sha256'],
                        filename, row['capture_utc'], row['precision'], row['review_status'], row['revision'],
                        sum(annotation['kind']=='vehicle' for annotation in active),
                        sum(annotation['kind']=='candidate' for annotation in active),
                        sum(annotation['kind']=='light' for annotation in active),
                        json.dumps(annotations, ensure_ascii=False), row.get('prediction_run'),
                        row.get('inference_status'), json.dumps(row.get('predictions', []), ensure_ascii=False),
                        row.get('review_prediction_run'), row.get('evidence_role'), operator])
                    row_count += 1
                stream.flush()
                os.fsync(stream.fileno())
            snapshot_hash = digest(target)
            with self.connect(mission) as conn:
                conn.execute("UPDATE exports SET snapshot_sha256=?,row_count=?,status='completed',finished=? WHERE id=?",
                    (snapshot_hash, row_count, now(), export_id))
            return {'id': export_id, 'path': str(target), 'format': 'csv',
                'snapshot_sha256': snapshot_hash, 'row_count': row_count, 'status': 'completed'}
        except Exception as exc:
            status = 'partial' if target.exists() else 'failed'
            error = str(exc)[:1000]
            with self.connect(mission) as conn:
                conn.execute('UPDATE exports SET row_count=?,status=?,finished=?,error=? WHERE id=?',
                    (row_count, status, now(), error, export_id))
            raise ValueError(f'Export {export_id} {status}: {error}') from exc

    def missions(self):
        result = []
        for path in self.root.glob('*/mission.sqlite3'):
            with self.connect(path.parent.name) as conn:
                row = dict(conn.execute('SELECT * FROM mission').fetchone())
                settings = dict(conn.execute('SELECT * FROM mission_settings WHERE mission_id=?', (row['id'],)).fetchone())
                settings['include_subdirectories'] = bool(settings['include_subdirectories'])
                settings['export_formats'] = json.loads(settings['export_formats'])
                result.append({**row, **settings})
        return sorted(result, key=lambda row: row['created'], reverse=True)

    def import_folder(self, mission, root, preset, selected_paths=None):
        root = Path(root).resolve(strict=True)
        if not root.is_dir() or self.root.is_relative_to(root) or root.is_relative_to(self.root):
            raise ValueError('Choose a source outside application-managed storage')
        settings = self.settings(mission)
        if settings['input_type'] != 'images':
            raise ValueError('This mission is configured for video input; image import is not available')
        if settings['source_root'] and root != Path(settings['source_root']):
            raise ValueError('Choose the source directory configured for this mission')
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
            if not settings['include_subdirectories'] and any(len(Path(relative).parts) > 1 for relative, _, _ in entries):
                raise ValueError('Manifest includes nested files but this mission excludes subdirectories')
        else:
            if settings['include_subdirectories']:
                for base, directories, files in os.walk(root, followlinks=False):
                    directories[:] = sorted(d for d in directories if not (Path(base) / d).is_symlink())
                    for name in sorted(files):
                        if Path(name).suffix.lower() in ('.jpg', '.jpeg', '.png'):
                            entries.append((str((Path(base) / name).relative_to(root)), None, {'preset': preset}))
            else:
                for name in sorted(os.listdir(root)):
                    source = root / name
                    if source.is_file() and not source.is_symlink() and source.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        entries.append((name, None, {'preset': preset}))
        if selected_paths is not None:
            selected = {Path(value).as_posix() for value in selected_paths}
            entries = [entry for entry in entries if Path(entry[0]).as_posix() in selected]
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
