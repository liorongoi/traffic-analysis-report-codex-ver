"""Persistent, backend-owned still-image source monitoring."""
from datetime import datetime, timezone
import os
import threading
from pathlib import Path
from .store import digest, now, safe_file, validate_image_file

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


class SourceMonitor:
    def __init__(self, library, interval=2.0, stability_seconds=5.0,on_batch=None):
        self.library = library
        self.interval = interval
        self.stability_seconds = stability_seconds
        self.on_batch=on_batch
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name='source-monitor', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval + 1))

    def set_enabled(self, mission, enabled):
        # Serialize the control change with scans: once pause returns, an
        # in-flight scan has finished and no later scan can import new files.
        with self.lock:
            result=self.library.set_monitoring(mission,enabled)
            return {**self.status(mission),**result}

    def _run(self):
        while not self.stop_event.is_set():
            for mission in self.library.missions():
                if self.stop_event.is_set():
                    break
                if mission['processing_mode'] == 'continuous' and mission['input_type'] == 'images':
                    try:
                        self.scan(mission['id'])
                    except Exception as exc:
                        self._state(mission['id'], 'error', str(exc))
            self.stop_event.wait(self.interval)

    def _state(self, mission, status, error=None, imported=False):
        with self.library.connect(mission) as conn:
            previous = conn.execute('SELECT last_import FROM monitor_state WHERE id=1').fetchone()
            last_import = now() if imported else (previous['last_import'] if previous else None)
            conn.execute('''INSERT INTO monitor_state(id,status,last_scan,last_import,error) VALUES (1,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status,last_scan=excluded.last_scan,
                last_import=excluded.last_import,error=excluded.error''',
                (status, now(), last_import, error))

    def status(self, mission):
        self._sync_batches(mission)
        enabled=self.library.monitoring_enabled(mission)
        with self.library.connect(mission) as conn:
            row = conn.execute('SELECT * FROM monitor_state WHERE id=1').fetchone()
            state = dict(row) if row else {'id': 1, 'status': 'stopped', 'last_scan': None,
                'last_import': None, 'error': None}
            state['enabled']=enabled
            state['files'] = {row['status']: row['count'] for row in conn.execute(
                'SELECT status,COUNT(*) count FROM monitored_files GROUP BY status')}
            state['recent_errors'] = [dict(row) for row in conn.execute(
                "SELECT relative_path,error FROM monitored_files WHERE error IS NOT NULL ORDER BY last_seen DESC LIMIT 20")]
            state['batches']={row['status']:row['count'] for row in conn.execute(
                'SELECT status,COUNT(*) count FROM monitor_batches GROUP BY status')}
            state['recent_errors'].extend({'relative_path':'source batch '+row['source_run_id'],'error':row['error']}
                for row in conn.execute("SELECT source_run_id,error FROM monitor_batches WHERE status IN ('failed','completed_with_errors') AND error IS NOT NULL ORDER BY created DESC LIMIT 20"))
            return state

    def _sync_batches(self,mission):
        with self.library.connect(mission) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='inference_runs'").fetchone():return
            pending=conn.execute('''SELECT b.source_run_id,b.inference_run_id,r.status,r.error FROM monitor_batches b
                LEFT JOIN inference_runs r ON r.id=b.inference_run_id
                WHERE b.status IN ('queued','failed','cancelled') AND b.inference_run_id IS NOT NULL''').fetchall()
            for row in pending:
                if row['status'] in ('completed','completed_with_errors','failed','cancelled'):
                    error=(row['error'] or row['status']) if row['status']!='completed' else None
                    batch_status='completed' if row['status']=='completed' else row['status']
                    conn.execute('UPDATE monitor_batches SET status=?,error=? WHERE source_run_id=?',
                        (batch_status,error,row['source_run_id']))

    def _dispatch_batches(self,mission):
        if not self.on_batch:return
        with self.library.connect(mission) as conn:
            pending=conn.execute("SELECT source_run_id FROM monitor_batches WHERE status='pending' ORDER BY created,source_run_id").fetchall()
        for row in pending:
            source_run=row['source_run_id']
            try:
                result=self.on_batch(mission,source_run)
            except Exception as exc:
                if getattr(exc,'retryable',False):return
                with self.library.connect(mission) as conn:
                    conn.execute("UPDATE monitor_batches SET status='failed',error=? WHERE source_run_id=? AND status='pending'",
                        (str(exc)[:1000],source_run))
                continue
            with self.library.connect(mission) as conn:
                conn.execute("UPDATE monitor_batches SET status='queued',inference_run_id=?,error=NULL WHERE source_run_id=? AND status='pending'",
                    (result['run'],source_run))
        self._sync_batches(mission)

    def scan(self, mission):
        with self.lock:
            settings = self.library.settings(mission)
            if settings['processing_mode'] != 'continuous':
                self._state(mission, 'stopped')
                return self.status(mission)
            if not self.library.monitoring_enabled(mission):
                self._state(mission, 'paused')
                return self.status(mission)
            if settings['input_type'] != 'images':
                self._state(mission, 'unsupported', 'Continuous monitoring currently supports still images only')
                return self.status(mission)
            if not settings['source_root']:
                self._state(mission, 'unconfigured', 'Set a source directory in the mission settings')
                return self.status(mission)
            root = Path(settings['source_root'])
            try:
                root = root.resolve(strict=True)
                if not root.is_dir() or str(root) != settings['source_root']:
                    raise OSError('Configured source directory is unavailable')
            except OSError as exc:
                self._state(mission, 'unavailable', str(exc))
                return self.status(mission)

            scan_time = datetime.now(timezone.utc)
            scan_text = scan_time.isoformat()
            discovered = []
            if settings['include_subdirectories']:
                for base, directories, files in os.walk(root, followlinks=False):
                    directories[:] = sorted(name for name in directories if not (Path(base) / name).is_symlink())
                    for name in files:
                        path = Path(base) / name
                        if path.suffix.lower() in IMAGE_SUFFIXES and not path.is_symlink():
                            discovered.append(path)
            else:
                discovered = [path for path in root.iterdir() if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_SUFFIXES]

            ready = []
            with self.library.connect(mission) as conn:
                conn.execute("UPDATE monitor_state SET status='scanning',last_scan=?,error=NULL WHERE id=1", (scan_text,))
                for path in sorted(discovered):
                    relative = path.relative_to(root).as_posix()
                    try:
                        safe_file(root, relative)
                        stat = path.stat()
                    except (OSError, ValueError):
                        continue
                    row = conn.execute('SELECT * FROM monitored_files WHERE relative_path=?', (relative,)).fetchone()
                    unchanged = row and row['size'] == stat.st_size and row['mtime_ns'] == stat.st_mtime_ns
                    stable_since = row['stable_since'] if unchanged else scan_text
                    status = row['status'] if unchanged else 'stabilizing'
                    conn.execute('''INSERT INTO monitored_files(relative_path,size,mtime_ns,stable_since,last_seen,status,error)
                        VALUES (?,?,?,?,?,?,NULL) ON CONFLICT(relative_path) DO UPDATE SET
                        size=excluded.size,mtime_ns=excluded.mtime_ns,stable_since=excluded.stable_since,
                        last_seen=excluded.last_seen,status=excluded.status,error=NULL''',
                        (relative, stat.st_size, stat.st_mtime_ns, stable_since, scan_text, status))
                    elapsed = (scan_time - datetime.fromisoformat(stable_since)).total_seconds()
                    if elapsed >= self.stability_seconds and status != 'imported':
                        try:
                            source_file=safe_file(root, relative)
                            validate_image_file(source_file,relative)
                            checksum = digest(source_file)
                            existing = conn.execute('''SELECT 1 FROM images WHERE name=? AND sha256=? LIMIT 1''',
                                (relative, checksum)).fetchone()
                            if existing:
                                conn.execute("UPDATE monitored_files SET imported_sha256=?,status='imported' WHERE relative_path=?",
                                    (checksum, relative))
                            else:
                                ready.append((relative, checksum))
                                conn.execute("UPDATE monitored_files SET status='ready' WHERE relative_path=?", (relative,))
                        except (OSError, ValueError) as exc:
                            conn.execute("UPDATE monitored_files SET status='error',error=? WHERE relative_path=?", (str(exc)[:1000], relative))

            if ready:
                try:
                    result = self.library.import_folder(mission, root, 'camera', [relative for relative, _ in ready])
                    with self.library.connect(mission) as conn:
                        for relative, checksum in ready:
                            conn.execute("UPDATE monitored_files SET imported_sha256=?,status='imported',error=NULL WHERE relative_path=?",
                                (checksum, relative))
                        if self.on_batch:
                            conn.execute("INSERT OR IGNORE INTO monitor_batches(source_run_id,status,created) VALUES (?,'pending',?)",
                                (result['run'],now()))
                    self._state(mission, 'active', imported=True)
                    self._dispatch_batches(mission)
                    return {**self.status(mission), 'last_batch': result}
                except Exception as exc:
                    with self.library.connect(mission) as conn:
                        for relative, _ in ready:
                            conn.execute("UPDATE monitored_files SET status='error',error=? WHERE relative_path=?",
                                (str(exc)[:1000], relative))
                    self._state(mission, 'error', str(exc))
                    return self.status(mission)
            self._dispatch_batches(mission)
            self._state(mission, 'active')
            return self.status(mission)
