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
import zipfile
from datetime import datetime, timezone
from .domain import capture_time, named_zone
from xml.sax.saxutils import escape


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_image_file(path, relative_name):
    """Fully decode supported still-image formats before accepting a source."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.format not in ('JPEG','PNG'):
                raise ValueError(f'Unsupported image format {image.format!r}')
            if image.width*image.height>24_000_000:
                raise ValueError('Image exceeds the current 24-megapixel processing limit')
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        raise ValueError(f'Corrupt or unsupported image {relative_name}: {exc}') from exc


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
                    export_formats TEXT NOT NULL DEFAULT '["csv"]',
                    continuous_video_policy TEXT NOT NULL DEFAULT '',
                    resource_profile TEXT NOT NULL DEFAULT 'auto'
                );
                INSERT OR IGNORE INTO mission_settings(mission_id)
                SELECT id FROM mission;
            ''')
        settings_columns={row['name'] for row in conn.execute('PRAGMA table_info(mission_settings)')}
        if 'continuous_video_policy' not in settings_columns:
            conn.execute("ALTER TABLE mission_settings ADD COLUMN continuous_video_policy TEXT NOT NULL DEFAULT ''")
        if 'resource_profile' not in settings_columns:
            conn.execute("ALTER TABLE mission_settings ADD COLUMN resource_profile TEXT NOT NULL DEFAULT 'auto'")
        conn.execute('''CREATE TABLE IF NOT EXISTS exports(
            id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES mission(id),
            inference_run TEXT, format TEXT NOT NULL, output_path TEXT NOT NULL,
            snapshot_sha256 TEXT, row_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL, created TEXT NOT NULL, finished TEXT, error TEXT,
            bulk_approval_id TEXT
        )''')
        export_columns={row['name'] for row in conn.execute('PRAGMA table_info(exports)')}
        if 'bulk_approval_id' not in export_columns:
            conn.execute('ALTER TABLE exports ADD COLUMN bulk_approval_id TEXT')
        conn.execute('''CREATE TABLE IF NOT EXISTS monitored_files(
            relative_path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            stable_since TEXT NOT NULL, last_seen TEXT NOT NULL, imported_sha256 TEXT,
            status TEXT NOT NULL, error TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS monitor_batches(
            source_run_id TEXT PRIMARY KEY REFERENCES runs(id), status TEXT NOT NULL,
            inference_run_id TEXT, created TEXT NOT NULL, error TEXT
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS video_monitor_batches(
            source_id TEXT PRIMARY KEY, status TEXT NOT NULL, video_run_id TEXT,
            created TEXT NOT NULL, error TEXT, policy_json TEXT NOT NULL DEFAULT ''
        )''')
        video_batch_columns={row['name'] for row in conn.execute('PRAGMA table_info(video_monitor_batches)')}
        if 'policy_json' not in video_batch_columns:
            conn.execute("ALTER TABLE video_monitor_batches ADD COLUMN policy_json TEXT NOT NULL DEFAULT ''")
        conn.execute('''CREATE TABLE IF NOT EXISTS monitor_control(
            id INTEGER PRIMARY KEY CHECK(id=1), enabled INTEGER NOT NULL CHECK(enabled IN (0,1))
        )''')
        conn.execute('INSERT OR IGNORE INTO monitor_control(id,enabled) VALUES (1,1)')
        conn.execute('''CREATE TABLE IF NOT EXISTS monitor_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT, source_run_id TEXT, kind TEXT NOT NULL,
            details TEXT NOT NULL, created TEXT NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0,
            UNIQUE(source_run_id,kind)
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
                merged_into TEXT REFERENCES views(id),
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
        view_columns={row['name'] for row in conn.execute('PRAGMA table_info(views)')}
        if 'merged_into' not in view_columns:
            conn.execute('ALTER TABLE views ADD COLUMN merged_into TEXT REFERENCES views(id)')
        conn.execute('''CREATE TRIGGER IF NOT EXISTS one_way_view_merge BEFORE UPDATE OF merged_into ON views
            WHEN OLD.merged_into IS NOT NULL OR NEW.merged_into IS NULL
            BEGIN SELECT RAISE(ABORT,'camera view merge history cannot be reversed'); END''')
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
        resource_profile = settings.get('resource_profile', 'auto')
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
        if resource_profile not in ('auto', 'conservative', 'expanded'):
            raise ValueError('Resource profile must be auto, conservative, or expanded')
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
                    export_formats TEXT NOT NULL,
                    continuous_video_policy TEXT NOT NULL DEFAULT '',
                    resource_profile TEXT NOT NULL DEFAULT 'auto'
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
                conn.execute('INSERT INTO mission_settings(mission_id,input_type,analysis_mode,location,source_root,include_subdirectories,processing_mode,output_root,export_formats,continuous_video_policy,resource_profile) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (mission, input_type, analysis_mode, location.strip(), source_root,
                     int(settings.get('include_subdirectories', True)), processing_mode,
                     output_root, json.dumps(export_formats), '', resource_profile))
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
        result['continuous_video_policy'] = json.loads(result['continuous_video_policy']) if result.get('continuous_video_policy') else None
        return result

    def set_continuous_video_policy(self, mission, policy):
        settings=self.settings(mission)
        if settings['processing_mode']!='continuous' or settings['input_type']!='videos' or settings['analysis_mode']!='video_traffic':
            raise ValueError('A saved automatic-run policy requires a continuous Video Traffic mission')
        encoded=json.dumps(policy,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        with self.connect(mission) as conn:
            conn.execute('UPDATE mission_settings SET continuous_video_policy=? WHERE mission_id=?',(encoded,mission))
        return {'saved':True,'policy':policy}

    def monitoring_enabled(self, mission):
        with self.connect(mission) as conn:
            row=conn.execute('SELECT enabled FROM monitor_control WHERE id=1').fetchone()
        return bool(row['enabled']) if row else True

    def set_monitoring(self, mission, enabled):
        if type(enabled) is not bool:
            raise ValueError('Monitoring state must be enabled or paused')
        settings=self.settings(mission)
        if settings['processing_mode']!='continuous' or settings['input_type'] not in ('images','videos'):
            raise ValueError('Continuous monitoring controls require a continuous image or video mission')
        with self.connect(mission) as conn:
            conn.execute('UPDATE monitor_control SET enabled=? WHERE id=1',(int(enabled),))
            conn.execute('''INSERT INTO monitor_state(id,status,last_scan,last_import,error) VALUES (1,?,?,NULL,NULL)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,last_scan=excluded.last_scan,error=NULL''',
                ('active' if enabled else 'paused',now()))
        return {'enabled':enabled}

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
                SELECT v.id,v.name,v.created,v.updated,v.merged_into,target.name merged_into_name,
                    COALESCE(c.image_count,0) image_count,c.representative_name,c.representative_image_id
                FROM views v LEFT JOIN counts c ON c.view_id=v.id
                LEFT JOIN views target ON target.id=v.merged_into AND target.mission_id=v.mission_id
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
            conn.execute('INSERT INTO views(id,mission_id,name,created,updated) VALUES (?,?,?,?,?)',
                (identifier,mission,name,created,created))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,identifier,'created',json.dumps({'name':name},ensure_ascii=False),operator,created))
        return {'id':identifier,'name':name,'created':created,'updated':created}

    def rename_view(self, mission, identifier, name, operator):
        name=name.strip() if isinstance(name,str) else ''
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(name) <= 80:raise ValueError('View name must contain 1–80 characters')
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        with self.connect(mission) as conn:
            current=conn.execute('SELECT name,merged_into FROM views WHERE id=? AND mission_id=?',(identifier,mission)).fetchone()
            if not current:raise ValueError('Camera view not found')
            if current['merged_into']:raise ValueError('A merged camera view is retained for history and cannot be renamed')
            if current['name']==name:return {'id':identifier,'name':name,'unchanged':True}
            if conn.execute('SELECT 1 FROM views WHERE mission_id=? AND id<>? AND name=? COLLATE NOCASE',
                (mission,identifier,name)).fetchone():
                raise ValueError('A camera view with this name already exists')
            conn.execute('UPDATE views SET name=?,updated=? WHERE id=? AND mission_id=?',(name,now(),identifier,mission))
            conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                (mission,identifier,'renamed',json.dumps({'before':current['name'],'after':name},ensure_ascii=False),operator,now()))
        return {'id':identifier,'name':name}

    def merge_views(self, mission, source_id, target_id, operator):
        """Move current source-view assignments to target and retain a merge audit."""
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        if source_id==target_id:raise ValueError('Choose two different camera views')
        with self.connect(mission) as conn:
            conn.execute('BEGIN IMMEDIATE')
            source=conn.execute('SELECT * FROM views WHERE id=? AND mission_id=?',(source_id,mission)).fetchone()
            target=conn.execute('SELECT * FROM views WHERE id=? AND mission_id=?',(target_id,mission)).fetchone()
            if not source or not target:raise ValueError('Both camera views must belong to this mission')
            if source['merged_into']:raise ValueError('This source camera view was already merged')
            if target['merged_into']:raise ValueError('Choose an active target camera view')
            image_ids=[row['image_id'] for row in conn.execute('''SELECT h.image_id
                FROM image_view_history h JOIN (
                  SELECT image_id,MAX(revision) revision FROM image_view_history WHERE mission_id=? GROUP BY image_id
                ) latest ON latest.image_id=h.image_id AND latest.revision=h.revision
                WHERE h.mission_id=? AND h.view_id=?''',(mission,mission,source_id))]
            created=now()
            for image_id in image_ids:
                revision=conn.execute('SELECT COALESCE(MAX(revision),0)+1 FROM image_view_history WHERE mission_id=? AND image_id=?',
                    (mission,image_id)).fetchone()[0]
                conn.execute('''INSERT INTO image_view_history(mission_id,image_id,view_id,revision,operator,created)
                    VALUES (?,?,?,?,?,?)''',(mission,image_id,target_id,revision,operator,created))
                conn.execute('''INSERT INTO view_events(mission_id,view_id,action,details,operator,created)
                    VALUES (?,?,?,?,?,?)''',(mission,target_id,'image_assigned',json.dumps({
                        'image_id':image_id,'revision':revision,'reason':'view_merge','source_view_id':source_id},
                        ensure_ascii=False,sort_keys=True),operator,created))
            details=json.dumps({'source_view_id':source_id,'source_name':source['name'],
                'target_view_id':target_id,'target_name':target['name'],'moved_images':len(image_ids),
                'source_regions_retained':True,'target_regions_apply_to_reassigned_images':True},
                ensure_ascii=False,sort_keys=True)
            conn.execute('UPDATE views SET merged_into=?,updated=? WHERE id=? AND mission_id=?',
                (target_id,created,source_id,mission))
            conn.execute('''INSERT INTO view_events(mission_id,view_id,action,details,operator,created)
                VALUES (?,?,?,?,?,?)''',(mission,source_id,'merged_into',details,operator,created))
            conn.execute('''INSERT INTO view_events(mission_id,view_id,action,details,operator,created)
                VALUES (?,?,?,?,?,?)''',(mission,target_id,'merged_from',details,operator,created))
        return {'source_view_id':source_id,'target_view_id':target_id,'moved_images':len(image_ids),
            'source_regions_retained':True,'target_regions_apply_to_reassigned_images':True}

    def assign_view(self, mission, image, view_id, operator):
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        with self.connect(mission) as conn:
            if not conn.execute('SELECT 1 FROM images WHERE id=?',(image,)).fetchone():
                raise ValueError('Image not found in this mission')
            if view_id and not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=? AND merged_into IS NULL',(view_id,mission)).fetchone():
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

    def assign_views(self, mission, images, view_id, operator):
        """Atomically reassign a bounded image selection, preserving per-image history."""
        operator=operator.strip() if isinstance(operator,str) else ''
        if not 1 <= len(operator) <= 120:raise ValueError('Operator is required for view history')
        if not isinstance(images,list) or not images or len(images)>1000:
            raise ValueError('Choose between 1 and 1000 images')
        if any(not isinstance(image,str) or not image for image in images):
            raise ValueError('Image IDs must be non-empty strings')
        images=list(dict.fromkeys(images))
        with self.connect(mission) as conn:
            if view_id and not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=? AND merged_into IS NULL',(view_id,mission)).fetchone():
                raise ValueError('Camera view not found in this mission')
            if not conn.execute('SELECT 1 FROM mission WHERE id=?',(mission,)).fetchone():
                raise ValueError('Mission not found')
            placeholders=','.join('?' for _ in images)
            found={row['id'] for row in conn.execute(
                f'SELECT id FROM images WHERE id IN ({placeholders})',images)}
            if found!=set(images):
                raise ValueError('One or more images were not found in this mission')
            changed=[]
            for image in images:
                previous=conn.execute('SELECT view_id FROM image_view_history WHERE mission_id=? AND image_id=? ORDER BY revision DESC LIMIT 1',(mission,image)).fetchone()
                if previous and previous['view_id']==view_id:continue
                revision=conn.execute('SELECT COALESCE(MAX(revision),0)+1 FROM image_view_history WHERE mission_id=? AND image_id=?',(mission,image)).fetchone()[0]
                created=now()
                conn.execute('INSERT INTO image_view_history(mission_id,image_id,view_id,revision,operator,created) VALUES (?,?,?,?,?,?)',
                    (mission,image,view_id,revision,operator,created))
                conn.execute('INSERT INTO view_events(mission_id,view_id,action,details,operator,created) VALUES (?,?,?,?,?,?)',
                    (mission,view_id,'images_assigned' if view_id else 'images_unassigned',
                     json.dumps({'image_id':image,'revision':revision,'selection_size':len(images)},ensure_ascii=False),operator,created))
                changed.append({'image_id':image,'view_id':view_id,'revision':revision})
            return {'requested':len(images),'changed':len(changed),'unchanged':len(images)-len(changed),'assignments':changed}

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
            if not conn.execute('SELECT 1 FROM views WHERE id=? AND mission_id=? AND merged_into IS NULL',(view_id,mission)).fetchone():
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
            previous=conn.execute('''SELECT h.* FROM view_region_history h JOIN views v ON v.id=h.view_id
                WHERE h.mission_id=? AND h.region_key=? AND v.merged_into IS NULL ORDER BY h.revision DESC LIMIT 1''',
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
            previous=conn.execute('''SELECT h.* FROM view_region_history h JOIN views v ON v.id=h.view_id
                WHERE h.mission_id=? AND h.region_key=? AND v.merged_into IS NULL ORDER BY h.revision DESC LIMIT 1''',
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

    def export_csv(self, mission, inference_run=None, snapshot_rowid=None, bulk_approval_id=None):
        settings = self.settings(mission)
        if 'csv' not in settings['export_formats']:
            raise ValueError('CSV is not configured for this mission')
        if not settings['output_root']:
            raise ValueError('Set an output directory in the mission settings before exporting')
        output_root = self._configured_directory(settings['output_root'], 'output', writable=True)
        if output_root != settings['output_root']:
            raise ValueError('Configured output directory changed; update the mission settings before exporting')
        if self.root.is_relative_to(Path(output_root)) or Path(output_root).is_relative_to(self.root):
            raise ValueError('Choose an output outside application-managed storage')

        rows = self.iter_images(mission, inference_run,snapshot_rowid=snapshot_rowid)
        export_id = str(uuid.uuid4())
        target = Path(output_root) / f'traffic-analysis-{mission[:8]}-{export_id}.csv'
        created = now()
        with self.connect(mission) as conn:
            conn.execute('INSERT INTO exports(id,mission_id,inference_run,format,output_path,row_count,status,created,bulk_approval_id) VALUES (?,?,?,?,?,?,?,?,?)',
                (export_id, mission, inference_run, 'csv', str(target), 0, 'writing', created,bulk_approval_id))
        row_count = 0
        try:
            with target.open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['mission_id','inference_run','source_id','source_run_id','source_sha256',
                    'filename','capture_utc','timestamp_precision','review_status','revision',
                    'confirmed_vehicle_count','candidate_count','light_count','annotations',
                    'prediction_run','inference_status','original_predictions','review_prediction_run',
                    'evidence_role','operator','bulk_approval_id','bulk_approval_type'])
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
                        row.get('review_prediction_run'), row.get('evidence_role'), operator,
                        bulk_approval_id or '', 'bulk_unreviewed' if bulk_approval_id else ''])
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

    def export_xlsx(self, mission, inference_run=None, snapshot_rowid=None, bulk_approval_id=None):
        """Write a dependency-free, Unicode-safe XLSX snapshot without formula cells."""
        settings=self.settings(mission)
        if 'xlsx' not in settings['export_formats']:
            raise ValueError('XLSX is not configured for this mission')
        if not settings['output_root']:
            raise ValueError('Set an output directory in the mission settings before exporting')
        output_root=self._configured_directory(settings['output_root'],'output',writable=True)
        if output_root!=settings['output_root']:
            raise ValueError('Configured output directory changed; update the mission settings before exporting')
        if self.root.is_relative_to(Path(output_root)) or Path(output_root).is_relative_to(self.root):
            raise ValueError('Choose an output outside application-managed storage')

        with self.connect(mission) as conn:
            if snapshot_rowid is None:
                snapshot_rowid=conn.execute('SELECT COALESCE(MAX(rowid),0) FROM images').fetchone()[0]
            count=conn.execute('SELECT COUNT(*) FROM images WHERE rowid<=?',(snapshot_rowid,)).fetchone()[0]
        rows=self.iter_images(mission,inference_run,snapshot_rowid=snapshot_rowid)
        headers=['mission_id','inference_run','source_id','source_run_id','source_sha256','filename','capture_utc',
            'timestamp_precision','review_status','revision','confirmed_vehicle_count','candidate_count','light_count',
            'annotations','prediction_run','inference_status','original_predictions','review_prediction_run','evidence_role','operator',
            'bulk_approval_id','bulk_approval_type']
        max_data_rows=1_048_575
        sheet_count=max(1,(count+max_data_rows-1)//max_data_rows)
        sheet_names=['Traffic' if sheet_count==1 else f'Traffic {number+1}' for number in range(sheet_count)]
        export_id=str(uuid.uuid4());target=Path(output_root)/f'traffic-analysis-{mission[:8]}-{export_id}.xlsx';created=now()
        with self.connect(mission) as conn:
            conn.execute('INSERT INTO exports(id,mission_id,inference_run,format,output_path,row_count,status,created,bulk_approval_id) VALUES (?,?,?,?,?,?,?,?,?)',
                (export_id,mission,inference_run,'xlsx',str(target),0,'writing',created,bulk_approval_id))
        row_count=0
        try:
            with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
                archive.writestr('[Content_Types].xml','''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'''+''.join(
                    f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(sheet_count))+'</Types>')
                archive.writestr('_rels/.rels','''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''')
                archive.writestr('xl/workbook.xml','''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'''+''.join(
                    f'<sheet name="{name}" sheetId="{i+1}" r:id="rId{i+1}"/>' for i,name in enumerate(sheet_names))+'</sheets></workbook>')
                archive.writestr('xl/_rels/workbook.xml.rels','''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'''+''.join(
                    f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i+1}.xml"/>' for i in range(sheet_count))+'</Relationships>')

                def cell_xml(column,row_number,value):
                    reference=f'{column}{row_number}'
                    if isinstance(value,int):return f'<c r="{reference}"><v>{value}</v></c>'
                    value='' if value is None else str(value)
                    if len(value)>32767:raise ValueError(f'Cell {reference} exceeds the XLSX 32,767-character limit')
                    if any((ord(char)<32 and char not in '\t\n\r') or ord(char) in (0xfffe,0xffff) for char in value):
                        raise ValueError(f'Cell {reference} contains a character Excel cannot store')
                    return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'

                def column_name(number):
                    name=''
                    while number:
                        number,remainder=divmod(number-1,26);name=chr(65+remainder)+name
                    return name

                for sheet_index in range(sheet_count):
                    sheet_rows=min(max_data_rows,max(0,count-sheet_index*max_data_rows))
                    with archive.open(f'xl/worksheets/sheet{sheet_index+1}.xml','w') as sheet:
                        sheet.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>')
                        header_xml=''.join(cell_xml(column_name(i+1),1,value) for i,value in enumerate(headers))
                        sheet.write(f'<row r="1">{header_xml}</row>'.encode('utf-8'))
                        for row_number in range(2,sheet_rows+2):
                            row=next(rows)
                            annotations=json.loads(row['annotations']);active=[item for item in annotations if not item.get('rejected')]
                            values=[mission,inference_run or '',row['id'],row['run_id'],row['sha256'],row['name'],row['capture_utc'],
                                row['precision'],row['review_status'],row['revision'],sum(item['kind']=='vehicle' for item in active),
                                sum(item['kind']=='candidate' for item in active),sum(item['kind']=='light' for item in active),
                                json.dumps(annotations,ensure_ascii=False),row.get('prediction_run'),row.get('inference_status'),
                                json.dumps(row.get('predictions',[]),ensure_ascii=False),row.get('review_prediction_run'),
                                row.get('evidence_role'),row.get('operator'),bulk_approval_id or '',
                                'bulk_unreviewed' if bulk_approval_id else '']
                            content=''.join(cell_xml(column_name(i+1),row_number,value) for i,value in enumerate(values))
                            sheet.write(f'<row r="{row_number}">{content}</row>'.encode('utf-8'));row_count+=1
                        sheet.write(b'</sheetData></worksheet>')
            with target.open('rb') as stream:os.fsync(stream.fileno())
            snapshot_hash=digest(target)
            with self.connect(mission) as conn:
                conn.execute("UPDATE exports SET snapshot_sha256=?,row_count=?,status='completed',finished=? WHERE id=?",
                    (snapshot_hash,row_count,now(),export_id))
            return {'id':export_id,'path':str(target),'format':'xlsx','snapshot_sha256':snapshot_hash,
                'row_count':row_count,'status':'completed','sheets':sheet_count}
        except Exception as exc:
            status='partial' if target.exists() else 'failed';error=str(exc)[:1000]
            with self.connect(mission) as conn:
                conn.execute('UPDATE exports SET row_count=?,status=?,finished=?,error=? WHERE id=?',
                    (row_count,status,now(),error,export_id))
            raise ValueError(f'Export {export_id} {status}: {error}') from exc

    def export_configured(self,mission,inference_run=None,snapshot_rowid=None,bulk_approval_id=None):
        formats=self.settings(mission)['export_formats'];files=[];errors=[]
        if snapshot_rowid is None:
            with self.connect(mission) as conn:
                snapshot_rowid=conn.execute('SELECT COALESCE(MAX(rowid),0) FROM images').fetchone()[0]
        for format_name in formats:
            try:files.append(self.export_csv(mission,inference_run,snapshot_rowid,bulk_approval_id) if format_name=='csv' else self.export_xlsx(mission,inference_run,snapshot_rowid,bulk_approval_id))
            except ValueError as exc:errors.append(str(exc))
        if errors:raise ValueError('Some configured exports failed: '+'; '.join(errors))
        result={'files':files,'status':'completed'}
        if len(files)==1:result.update(files[0])
        return result

    def missions(self):
        result = []
        for path in self.root.glob('*/mission.sqlite3'):
            with self.connect(path.parent.name) as conn:
                row = dict(conn.execute('SELECT * FROM mission').fetchone())
                settings = dict(conn.execute('SELECT * FROM mission_settings WHERE mission_id=?', (row['id'],)).fetchone())
                settings['include_subdirectories'] = bool(settings['include_subdirectories'])
                settings['export_formats'] = json.loads(settings['export_formats'])
                settings['continuous_video_policy'] = json.loads(settings['continuous_video_policy']) if settings.get('continuous_video_policy') else None
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
                    validate_image_file(target,relative)
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

    def iter_images(self,mission,prediction_run=None,batch_size=250,snapshot_rowid=None):
        """Yield a bounded image snapshot; later continuous imports are excluded."""
        if type(batch_size) is not int or not 1<=batch_size<=2000:
            raise ValueError('Image export batch size must be between 1 and 2000')
        if snapshot_rowid is None:
            with self.connect(mission) as conn:
                snapshot_rowid=conn.execute('SELECT COALESCE(MAX(rowid),0) FROM images').fetchone()[0]
        last=None
        while True:
            with self.connect(mission) as conn:
                query='''SELECT images.*, images.rowid source_rowid, COALESCE(reviews.revision,0) revision,
                    COALESCE(reviews.status,'unreviewed') review_status, COALESCE(reviews.annotations,'[]') annotations,
                    COALESCE(reviews.role,'none') evidence_role, COALESCE(reviews.operator,'') operator
                    FROM images LEFT JOIN reviews ON reviews.image_id=images.id AND reviews.revision=
                    (SELECT MAX(r.revision) FROM reviews r WHERE r.image_id=images.id)
                    WHERE images.rowid<=?'''
                if last is not None:
                    query+=' AND (capture_utc,name,images.rowid)>(?,?,?)'
                    parameters=(snapshot_rowid,*last,batch_size)
                else:parameters=(snapshot_rowid,batch_size)
                rows=conn.execute(query+' ORDER BY capture_utc,name,images.rowid LIMIT ?',parameters).fetchall()
            if not rows:break
            from .inference_store import enrich
            batch=enrich(self,mission,[dict(row) for row in rows],prediction_run)
            yield from batch
            last=(rows[-1]['capture_utc'],rows[-1]['name'],rows[-1]['source_rowid'])

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
