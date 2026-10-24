"""Per-mission immutable video sources, durable runs, events, and review history."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .store import now, safe_file
from .tracking import TrackerPolicy
from .video_timeline import VideoTimestamp


def schema(conn):
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS video_sources(
      id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
      source_relative_path TEXT NOT NULL,stored_relative_path TEXT NOT NULL,
      original_name TEXT NOT NULL,sha256 TEXT NOT NULL,size INTEGER NOT NULL,
      metadata TEXT NOT NULL,recording_start_utc TEXT,origin_provenance TEXT,
      created TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS video_sources_mission_idx ON video_sources(mission_id,created,id);
    CREATE TABLE IF NOT EXISTS video_runs(
      id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
      source_id TEXT NOT NULL REFERENCES video_sources(id),status TEXT NOT NULL,
      stage TEXT NOT NULL,total_frames INTEGER,frames_done INTEGER NOT NULL DEFAULT 0,
      current_pts INTEGER,current_time_base_num INTEGER,current_time_base_den INTEGER,
      tracker_checkpoint TEXT NOT NULL,last_frame_fingerprint TEXT,config TEXT NOT NULL,created TEXT NOT NULL,
      updated TEXT NOT NULL,finished TEXT,error TEXT,worker_pid INTEGER,cancel_requested INTEGER NOT NULL DEFAULT 0);
    CREATE INDEX IF NOT EXISTS video_runs_mission_idx ON video_runs(mission_id,created,id);
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_video_run ON video_runs(mission_id)
      WHERE status IN ('queued','running','paused');
    CREATE TABLE IF NOT EXISTS video_events(
      id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
      run_id TEXT NOT NULL REFERENCES video_runs(id),source_id TEXT NOT NULL REFERENCES video_sources(id),
      track_id TEXT NOT NULL,first_pts INTEGER NOT NULL,first_time_base_num INTEGER NOT NULL,
      first_time_base_den INTEGER NOT NULL,last_pts INTEGER NOT NULL,
      last_time_base_num INTEGER NOT NULL,last_time_base_den INTEGER NOT NULL,
      first_offset_num INTEGER NOT NULL,first_offset_den INTEGER NOT NULL,
      last_offset_num INTEGER NOT NULL,last_offset_den INTEGER NOT NULL,
      first_utc TEXT,last_utc TEXT,labels TEXT NOT NULL,observation_count INTEGER NOT NULL,
      association_uncertain INTEGER NOT NULL CHECK(association_uncertain IN (0,1)),
      uncertainty_reasons TEXT NOT NULL,created TEXT NOT NULL,UNIQUE(run_id,track_id));
    CREATE INDEX IF NOT EXISTS video_events_run_time_idx ON video_events(run_id,first_offset_num,first_offset_den,id);
    CREATE TABLE IF NOT EXISTS video_event_reviews(
      id INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT NOT NULL REFERENCES video_events(id),
      revision INTEGER NOT NULL,operator TEXT NOT NULL,action TEXT NOT NULL,
      payload TEXT NOT NULL,created TEXT NOT NULL,UNIQUE(event_id,revision));
    CREATE TRIGGER IF NOT EXISTS immutable_video_sources_u BEFORE UPDATE ON video_sources
      BEGIN SELECT RAISE(ABORT,'video source versions are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_sources_d BEFORE DELETE ON video_sources
      BEGIN SELECT RAISE(ABORT,'video source versions are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_events_u BEFORE UPDATE ON video_events
      BEGIN SELECT RAISE(ABORT,'video events are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_events_d BEFORE DELETE ON video_events
      BEGIN SELECT RAISE(ABORT,'video events are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_reviews_u BEFORE UPDATE ON video_event_reviews
      BEGIN SELECT RAISE(ABORT,'video event reviews are append-only'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_reviews_d BEFORE DELETE ON video_event_reviews
      BEGIN SELECT RAISE(ABORT,'video event reviews are append-only'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_run_config BEFORE UPDATE OF id,mission_id,source_id,config,created
      ON video_runs BEGIN SELECT RAISE(ABORT,'video run configuration is immutable'); END;
    ''')
    columns = {row['name'] for row in conn.execute('PRAGMA table_info(video_runs)')}
    if 'last_frame_fingerprint' not in columns:
        conn.execute('ALTER TABLE video_runs ADD COLUMN last_frame_fingerprint TEXT')
    if 'worker_pid' not in columns:
        conn.execute('ALTER TABLE video_runs ADD COLUMN worker_pid INTEGER')
    if 'cancel_requested' not in columns:
        conn.execute('ALTER TABLE video_runs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0')


def prepare(library, mission):
    with library.connect(mission) as conn:
        schema(conn)


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Value is not JSON serializable: {exc}') from exc


def _origin(recording_start_utc, provenance):
    if recording_start_utc is None:
        if provenance is not None:
            raise ValueError('Origin provenance requires an explicit recording UTC time')
        return None, None
    if isinstance(recording_start_utc, str):
        try:
            recording_start_utc = datetime.fromisoformat(recording_start_utc)
        except ValueError as exc:
            raise ValueError('Recording origin must be an ISO-8601 timezone-aware datetime') from exc
    if (not isinstance(recording_start_utc, datetime) or recording_start_utc.tzinfo is None
            or recording_start_utc.utcoffset() is None):
        raise ValueError('Recording origin must be a timezone-aware datetime')
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError('Recording origin provenance is required when UTC is supplied')
    return recording_start_utc.astimezone(timezone.utc).isoformat(), provenance.strip()


def _sha_and_copy(source, destination):
    before = source.stat()
    digest = hashlib.sha256()
    copied = 0
    fd, temporary = tempfile.mkstemp(prefix='.video-', suffix='.partial', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output, source.open('rb') as input_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b''):
                output.write(block)
                digest.update(block)
                copied += len(block)
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
                or copied != after.st_size):
            raise ValueError('Video source changed while it was being snapshotted; retry after it is stable')
        source_hash = hashlib.sha256()
        with source.open('rb') as input_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b''):
                source_hash.update(block)
        if source_hash.hexdigest() != digest.hexdigest():
            raise ValueError('Video source changed while it was being snapshotted; retry after it is stable')
        os.replace(temporary, destination)
        return digest.hexdigest(), copied, after.st_mtime_ns
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def register_video(library, mission, relative_path, *, metadata=None,
                   recording_start_utc=None, origin_provenance=None):
    """Create a versioned private copy of one authorized video input.

    Identical bytes are not deduplicated: each explicit import is a distinct
    source observation and retains its own optional origin provenance.
    """
    prepare(library, mission)
    settings = library.settings(mission)
    if settings['input_type'] != 'videos':
        raise ValueError('This mission is not configured for video input')
    root_text = settings['source_root']
    if not root_text:
        raise ValueError('Configure an authorized video source directory first')
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError('Video path must be relative to the configured source directory')
    selected_path = Path(relative_path)
    if selected_path.is_absolute() or '..' in selected_path.parts:
        raise ValueError('Video path must remain relative to the configured source directory')
    if not settings['include_subdirectories'] and len(selected_path.parts) != 1:
        raise ValueError('This mission is not configured to include video subdirectories')
    source = safe_file(root_text, relative_path)
    source_relative = Path(relative_path).as_posix()
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError('Video metadata must be a mapping')
    metadata_json = _json(metadata)
    origin, provenance = _origin(recording_start_utc, origin_provenance)
    source_id = str(uuid.uuid4())
    mission_root = library.location(mission)
    original_name = Path(source_relative).name
    destination_relative = Path('sources') / 'videos' / source_id / original_name
    destination = mission_root / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=False)
    try:
        sha256, size, mtime_ns = _sha_and_copy(source, destination)
        source_info = {**metadata, 'source_mtime_ns_at_import': mtime_ns}
        metadata_json = _json(source_info)
        with library.connect(mission) as conn:
            conn.execute('INSERT INTO video_sources VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (source_id, mission, source_relative, destination_relative.as_posix(),
                 original_name, sha256, size, metadata_json, origin, provenance, now()))
    except BaseException:
        try:
            destination.unlink()
            destination.parent.rmdir()
        except FileNotFoundError:
            pass
        raise
    return source_detail(library, mission, source_id)


def source_detail(library, mission, source_id):
    prepare(library, mission)
    with library.connect(mission) as conn:
        row = conn.execute('SELECT * FROM video_sources WHERE mission_id=? AND id=?',
            (mission, source_id)).fetchone()
    if not row:
        raise ValueError('Video source not found in this mission')
    result = dict(row)
    result['metadata'] = json.loads(result['metadata'])
    return result


def source_path(library, mission, source_id):
    source = source_detail(library, mission, source_id)
    try:
        path = safe_file(library.location(mission), source['stored_relative_path'])
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError('Stored video evidence is missing or unsafe') from exc
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != source['sha256']:
        raise ValueError('Stored video evidence failed its SHA-256 check')
    return path


def create_run(library, mission, source_id, config, *, total_frames=None):
    prepare(library, mission)
    if not isinstance(config, dict) or not isinstance(config.get('tracker_policy'), dict):
        raise ValueError('Video run config must snapshot an explicit tracker_policy')
    config = dict(config)
    config['tracker_policy'] = _canonical_tracker_policy(config['tracker_policy'])
    if total_frames is not None and (isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 1):
        raise ValueError('total_frames must be a positive integer or unknown')
    run_id = str(uuid.uuid4())
    created = now()
    # Verify the private source snapshot before any run is allowed to consume it.
    source_path(library, mission, source_id)
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        settings = conn.execute('SELECT input_type,analysis_mode FROM mission_settings WHERE mission_id=?',
            (mission,)).fetchone()
        if not settings or settings['input_type'] != 'videos':
            raise ValueError('Video runs require a video mission')
        if settings['analysis_mode'] != 'video_traffic':
            raise ValueError('Automatic analysis currently supports Video Traffic missions only')
        source = conn.execute('SELECT * FROM video_sources WHERE mission_id=? AND id=?',
            (mission, source_id)).fetchone()
        if not source:
            raise ValueError('Video source not found in this mission')
        if conn.execute("SELECT 1 FROM video_runs WHERE mission_id=? AND status IN ('queued','running','paused')",
                (mission,)).fetchone():
            raise ValueError('This mission already has an active video run')
        recorded_config = {**config, 'video_config_schema_version': 1,
            'mission_snapshot': dict(conn.execute('''SELECT m.id,m.name,m.country,m.zone,s.analysis_mode,
                s.source_root,s.output_root,s.export_formats FROM mission m JOIN mission_settings s
                ON s.mission_id=m.id WHERE m.id=?''', (mission,)).fetchone()),
            'source_snapshot': {'id': source['id'], 'relative_path': source['source_relative_path'],
                'stored_relative_path': source['stored_relative_path'], 'sha256': source['sha256'],
                'size': source['size'], 'metadata': json.loads(source['metadata']),
                'recording_start_utc': source['recording_start_utc'],
                'origin_provenance': source['origin_provenance']}}
        conn.execute('''INSERT INTO video_runs(id,mission_id,source_id,status,stage,total_frames,
            frames_done,tracker_checkpoint,config,created,updated) VALUES (?,?,?,'queued','queued',?,0,'{}',?,?,?)''',
            (run_id, mission, source_id, total_frames, _json(recorded_config), created, created))
    return run_detail(library, mission, run_id)


def _track_event_values(track, recording_start_utc):
    first = VideoTimestamp(track.first_pts, track.first_time_base_num, track.first_time_base_den)
    last = VideoTimestamp(track.last_pts, track.last_time_base_num, track.last_time_base_den)
    first_offset, last_offset = first.offset_seconds, last.offset_seconds
    first_utc = first.utc_at(recording_start_utc).isoformat() if recording_start_utc else None
    last_utc = last.utc_at(recording_start_utc).isoformat() if recording_start_utc else None
    return first, last, first_offset, last_offset, first_utc, last_utc


def _track_payload(track):
    return {'track_id': track.track_id, 'first_pts': track.first_pts,
        'first_time_base_num': track.first_time_base_num,
        'first_time_base_den': track.first_time_base_den, 'last_pts': track.last_pts,
        'last_time_base_num': track.last_time_base_num,
        'last_time_base_den': track.last_time_base_den, 'labels': list(track.labels),
        'observation_count': track.observation_count,
        'association_uncertain': track.association_uncertain,
        'uncertainty_reasons': list(track.uncertainty_reasons)}


def _canonical_tracker_policy(value):
    required = {'max_gap_seconds', 'minimum_iou', 'max_center_distance', 'association_margin'}
    missing = required - set(value)
    if missing:
        raise ValueError('tracker_policy must explicitly set: ' + ', '.join(sorted(missing)))
    gap = value['max_gap_seconds']
    try:
        if isinstance(gap, bool) or (isinstance(gap, (list, tuple)) and any(isinstance(x, bool) for x in gap)):
            raise ValueError('max_gap_seconds cannot be boolean')
        gap = Fraction(*gap) if isinstance(gap, (list, tuple)) else Fraction(gap)
        groups = tuple(frozenset(group) for group in value.get('compatible_label_groups', ()))
        policy = TrackerPolicy(gap, value['minimum_iou'], value['max_center_distance'],
            value['association_margin'], groups)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'Invalid explicit tracker_policy: {exc}') from exc
    return {'max_gap_seconds': [policy.max_gap_seconds.numerator, policy.max_gap_seconds.denominator],
        'minimum_iou': policy.minimum_iou, 'max_center_distance': policy.max_center_distance,
        'association_margin': policy.association_margin,
        'compatible_label_groups': [sorted(group) for group in policy.compatible_label_groups]}


def _insert_closed_tracks(conn, mission, run_id, source, tracks, created):
    origin = datetime.fromisoformat(source['recording_start_utc']) if source['recording_start_utc'] else None
    inserted = 0
    for track in tracks:
        if not track.track_id.startswith(run_id + ':'):
            raise ValueError('Closed track belongs to a different video run')
        first, last, first_offset, last_offset, first_utc, last_utc = _track_event_values(track, origin)
        if first_offset > last_offset:
            raise ValueError('Closed track has an invalid presentation-time interval')
        event_id = str(uuid.uuid5(uuid.UUID(run_id), track.track_id))
        cursor = conn.execute('''INSERT OR IGNORE INTO video_events(id,mission_id,run_id,source_id,track_id,
            first_pts,first_time_base_num,first_time_base_den,last_pts,last_time_base_num,last_time_base_den,
            first_offset_num,first_offset_den,last_offset_num,last_offset_den,first_utc,last_utc,labels,
            observation_count,association_uncertain,uncertainty_reasons,created)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (event_id, mission, run_id, source['id'], track.track_id, first.pts, first.time_base_num,
             first.time_base_den, last.pts, last.time_base_num, last.time_base_den,
             first_offset.numerator, first_offset.denominator, last_offset.numerator, last_offset.denominator,
             first_utc, last_utc, _json(list(track.labels)), track.observation_count,
             int(track.association_uncertain), _json(list(track.uncertainty_reasons)), created))
        inserted += cursor.rowcount
    return inserted


def checkpoint(library, mission, run_id, *, frame_index, timestamp, tracker_state, closed_tracks=()):
    prepare(library, mission)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError('frame_index must be a non-negative integer')
    if not isinstance(timestamp, VideoTimestamp):
        raise TypeError('timestamp must be a VideoTimestamp')
    state_json = _json(tracker_state)
    fingerprint_payload = {'frame_index': frame_index, 'timestamp': [timestamp.pts,
        timestamp.time_base_num, timestamp.time_base_den], 'tracker_state': tracker_state,
        'closed_tracks': [_track_payload(track) for track in closed_tracks]}
    fingerprint = hashlib.sha256(_json(fingerprint_payload).encode('utf-8')).hexdigest()
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = conn.execute('SELECT * FROM video_runs WHERE mission_id=? AND id=?',
            (mission, run_id)).fetchone()
        if not run:
            raise ValueError('Video run not found')
        if run['status'] != 'running':
            raise ValueError('Video run must be running before checkpointing')
        try:
            run_config = json.loads(run['config'])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError('Video run config is corrupt') from exc
        if (not isinstance(tracker_state, dict) or tracker_state.get('run_id') != run_id
                or tracker_state.get('policy') != run_config.get('tracker_policy')):
            raise ValueError('Tracker checkpoint does not match this run and its frozen policy')
        if frame_index < run['frames_done']:
            if (frame_index == run['frames_done'] - 1 and run['current_pts'] == timestamp.pts
                    and run['current_time_base_num'] == timestamp.time_base_num
                    and run['current_time_base_den'] == timestamp.time_base_den
                    and run['tracker_checkpoint'] == state_json
                    and run['last_frame_fingerprint'] == fingerprint):
                return {'duplicate': True, 'frames_done': run['frames_done'], 'events_inserted': 0}
            raise ValueError('Video frame checkpoint moved backwards or conflicts with saved state')
        if frame_index != run['frames_done']:
            raise ValueError('Video frame checkpoints must be contiguous')
        if run['total_frames'] is not None and frame_index >= run['total_frames']:
            raise ValueError('Video frame checkpoint exceeds the declared frame count')
        source = conn.execute('SELECT * FROM video_sources WHERE id=? AND mission_id=?',
            (run['source_id'], mission)).fetchone()
        offset = timestamp.offset_seconds
        if run['current_pts'] is not None:
            previous = Fraction(run['current_pts'] * run['current_time_base_num'], run['current_time_base_den'])
            if offset < previous:
                raise ValueError('Video frame checkpoint timestamp moved backwards')
        events_inserted = _insert_closed_tracks(conn, mission, run_id, source, closed_tracks, now())
        conn.execute('''UPDATE video_runs SET stage='analyzing',frames_done=?,current_pts=?,
            current_time_base_num=?,current_time_base_den=?,tracker_checkpoint=?,
            last_frame_fingerprint=?,updated=? WHERE id=?''',
            (frame_index + 1, timestamp.pts, timestamp.time_base_num, timestamp.time_base_den,
             state_json, fingerprint, now(), run_id))
    return {'duplicate': False, 'frames_done': frame_index + 1, 'events_inserted': events_inserted}


def set_total_frames(library, mission, run_id, total_frames):
    prepare(library, mission)
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 1:
        raise ValueError('total_frames must be a positive integer')
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = conn.execute('SELECT status,total_frames FROM video_runs WHERE mission_id=? AND id=?',
            (mission, run_id)).fetchone()
        if not run:
            raise ValueError('Video run not found')
        if run['status'] != 'running':
            raise ValueError('Video run must be running before setting its frame total')
        if run['total_frames'] is not None and run['total_frames'] != total_frames:
            raise ValueError('Video frame total changed after processing started')
        conn.execute('UPDATE video_runs SET total_frames=?,updated=? WHERE id=?',
            (total_frames, now(), run_id))


def execute_video_run(library, mission, run_id, detector, *, should_cancel=None,
                      progress_callback=None):
    """Execute or resume one persisted run synchronously in a worker process."""
    from .tracking import TrackerPolicy, VideoTrackManager
    from .video_pipeline import (VideoAnalysisCancelled, VideoAnalysisPipeline,
        VideoFrameAnalysis, VideoTrackClosure)

    run = run_detail(library, mission, run_id)
    if run['status'] in ('failed', 'cancelled'):
        transition(library, mission, run_id, 'queued')
        run = run_detail(library, mission, run_id)
    if run['status'] not in ('queued', 'paused'):
        raise ValueError(f'Video run cannot start from {run["status"]}')
    transition(library, mission, run_id, 'running')
    run = run_detail(library, mission, run_id)
    try:
        source = run['config']['source_snapshot']
        path = source_path(library, mission, run['source_id'])
        policy = TrackerPolicy.from_config(run['config']['tracker_policy'])
        state = run['tracker_checkpoint'] if run['frames_done'] else None
        tracker = VideoTrackManager(policy, run_id=run_id, state=state)
        pipeline = VideoAnalysisPipeline(detector, tracker)
        previous = None
        if run['frames_done']:
            previous = VideoTimestamp(run['current_pts'], run['current_time_base_num'],
                run['current_time_base_den'])

        def progress(item):
            if item.total_frames is not None and run['total_frames'] is None:
                set_total_frames(library, mission, run_id, item.total_frames)
                run['total_frames'] = item.total_frames
            if progress_callback is not None:
                progress_callback(item)

        for record in pipeline.analyze(path,
                recording_start_utc=(datetime.fromisoformat(source['recording_start_utc'])
                    if source['recording_start_utc'] else None),
                should_cancel=should_cancel, progress_callback=progress,
                start_frame_index=run['frames_done'], expected_previous_timestamp=previous):
            if isinstance(record, VideoFrameAnalysis):
                checkpoint(library, mission, run_id, frame_index=record.frame_index,
                    timestamp=record.timestamp, tracker_state=tracker.state(),
                    closed_tracks=record.closed_tracks)
            elif isinstance(record, VideoTrackClosure):
                transition(library, mission, run_id, 'completed', final_tracks=record.tracks)
        return run_detail(library, mission, run_id)
    except VideoAnalysisCancelled as exc:
        current = run_detail(library, mission, run_id)
        if current['status'] == 'running':
            transition(library, mission, run_id, 'cancelled', error=str(exc))
        return run_detail(library, mission, run_id)
    except Exception as exc:
        current = run_detail(library, mission, run_id)
        if current['status'] == 'running':
            transition(library, mission, run_id, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


def transition(library, mission, run_id, target, *, error=None, final_tracks=()):
    allowed = {'queued': {'running', 'cancelled', 'failed'},
        'running': {'paused', 'completed', 'cancelled', 'failed'},
        'paused': {'running', 'cancelled', 'failed'}, 'completed': set(),
        'cancelled': {'queued'}, 'failed': {'queued'}}
    if target not in allowed:
        raise ValueError('Unsupported video run state')
    prepare(library, mission)
    if target == 'failed' and (not isinstance(error, str) or not error.strip()):
        raise ValueError('A failure reason is required')
    if target == 'queued':
        existing = run_detail(library, mission, run_id)
        source_path(library, mission, existing['source_id'])
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        run = conn.execute('SELECT * FROM video_runs WHERE mission_id=? AND id=?',
            (mission, run_id)).fetchone()
        if not run:
            raise ValueError('Video run not found')
        if target not in allowed.get(run['status'], set()):
            raise ValueError(f'Invalid video run transition {run["status"]} → {target}')
        if target == 'running' and conn.execute("SELECT 1 FROM video_runs WHERE mission_id=? AND id!=? AND status IN ('queued','running','paused')",
                (mission, run_id)).fetchone():
            raise ValueError('Another video run is active in this mission')
        source = conn.execute('SELECT * FROM video_sources WHERE id=? AND mission_id=?',
            (run['source_id'], mission)).fetchone()
        event_count = 0
        if target == 'completed':
            event_count = _insert_closed_tracks(conn, mission, run_id, source, final_tracks, now())
        if target in ('failed', 'cancelled') and error is not None:
            if not isinstance(error, str) or not error.strip():
                raise ValueError('Run error must be non-empty text')
            error = error.strip()[:2000]
        elif error is not None:
            raise ValueError('Only failed or cancelled runs can include an error')
        finished = now() if target in ('completed', 'failed', 'cancelled') else None
        worker_pid = None if target in ('queued', 'completed', 'failed', 'cancelled') else run['worker_pid']
        cancel_requested = 0 if target == 'queued' else run['cancel_requested']
        conn.execute('UPDATE video_runs SET status=?,stage=?,updated=?,finished=?,error=?,worker_pid=?,cancel_requested=? WHERE id=?',
            (target, target, now(), finished, error, worker_pid, cancel_requested, run_id))
    return {'run': run_id, 'status': target, 'events_inserted': event_count}


def run_detail(library, mission, run_id):
    prepare(library, mission)
    with library.connect(mission) as conn:
        row = conn.execute('SELECT * FROM video_runs WHERE mission_id=? AND id=?',
            (mission, run_id)).fetchone()
    if not row:
        raise ValueError('Video run not found')
    result = dict(row)
    result['tracker_checkpoint'] = json.loads(result['tracker_checkpoint'])
    result['config'] = json.loads(result['config'])
    return result


def sources(library, mission):
    prepare(library, mission)
    with library.connect(mission) as conn:
        rows = conn.execute('SELECT * FROM video_sources WHERE mission_id=? ORDER BY created DESC,id',
            (mission,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item['metadata'] = json.loads(item['metadata'])
        result.append(item)
    return result


def runs(library, mission):
    prepare(library, mission)
    with library.connect(mission) as conn:
        rows = conn.execute('SELECT id,mission_id,source_id,status,stage,total_frames,frames_done,created,updated,finished,error,worker_pid,cancel_requested FROM video_runs WHERE mission_id=? ORDER BY created DESC,id',
            (mission,)).fetchall()
    return [dict(row) for row in rows]


def set_worker_pid(library, mission, run_id, pid):
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError('worker pid must be a positive integer')
    prepare(library, mission)
    with library.connect(mission) as conn:
        conn.execute("UPDATE video_runs SET worker_pid=?,updated=? WHERE mission_id=? AND id=? AND status IN ('queued','running')",
            (pid, now(), mission, run_id))


def request_cancel(library, mission, run_id):
    prepare(library, mission)
    with library.connect(mission) as conn:
        cursor = conn.execute("UPDATE video_runs SET cancel_requested=1,updated=? WHERE mission_id=? AND id=? AND status IN ('queued','running')",
            (now(), mission, run_id))
        if cursor.rowcount == 0:
            row = conn.execute('SELECT status FROM video_runs WHERE mission_id=? AND id=?',
                (mission, run_id)).fetchone()
            if not row:
                raise ValueError('Video run not found')
            raise ValueError(f'Video run cannot be cancelled from {row["status"]}')
    return {'run': run_id, 'cancel_requested': True}


def fail_worker(library, mission, run_id, error):
    prepare(library, mission)
    reason = str(error).strip()[:2000] or 'Video worker stopped unexpectedly'
    with library.connect(mission) as conn:
        conn.execute("UPDATE video_runs SET status='failed',stage='failed',updated=?,finished=?,error=?,worker_pid=NULL WHERE mission_id=? AND id=? AND status IN ('queued','running')",
            (now(), now(), reason, mission, run_id))


def events(library, mission, run_id):
    prepare(library, mission)
    with library.connect(mission) as conn:
        if not conn.execute('SELECT 1 FROM video_runs WHERE id=? AND mission_id=?',
                (run_id, mission)).fetchone():
            raise ValueError('Video run not found')
        rows = conn.execute('''SELECT e.*,r.revision,r.operator,r.action,r.payload,r.created reviewed_at
            FROM video_events e LEFT JOIN video_event_reviews r ON r.event_id=e.id
            AND r.revision=(SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id)
            WHERE e.mission_id=? AND e.run_id=? ORDER BY e.id''',
            (mission, run_id)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        for key in ('labels', 'uncertainty_reasons', 'payload'):
            item[key] = json.loads(item[key]) if item[key] is not None else None
        item['review_status'] = {'accept': 'accepted', 'reject': 'rejected',
            'reopen': 'pending', 'correct_class': 'corrected', 'mark_duplicate': 'duplicate'}.get(item['action'], 'pending')
        result.append(item)
    return sorted(result, key=lambda item: (Fraction(item['first_offset_num'], item['first_offset_den']), item['id']))


def review_event(library, mission, event_id, expected_revision, operator, action, payload=None):
    prepare(library, mission)
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise ValueError('expected_revision must be a non-negative integer')
    operator = operator.strip() if isinstance(operator, str) else ''
    if not 1 <= len(operator) <= 120:
        raise ValueError('Operator is required for video event review')
    if action not in ('accept', 'reject', 'reopen', 'correct_class', 'mark_duplicate'):
        raise ValueError('Unsupported video event correction; merge, split, and timing edits await detailed design')
    payload = {} if payload is None else payload
    if not isinstance(payload, dict):
        raise ValueError('Video event review payload must be a mapping')
    if action == 'correct_class':
        label = payload.get('label')
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Class correction requires a non-empty label')
        payload = {'label': label.strip()}
    elif action == 'mark_duplicate':
        duplicate_of = payload.get('duplicate_of')
        if not isinstance(duplicate_of, str) or duplicate_of == event_id:
            raise ValueError('Duplicate correction requires a different event ID')
        payload = {'duplicate_of': duplicate_of}
    elif payload:
        raise ValueError(f'{action} does not accept a payload')
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        event = conn.execute('SELECT id,run_id FROM video_events WHERE id=? AND mission_id=?',
            (event_id, mission)).fetchone()
        if not event:
            raise ValueError('Video event not found in this mission')
        if action == 'mark_duplicate' and not conn.execute(
                'SELECT 1 FROM video_events WHERE id=? AND run_id=?',
                (payload['duplicate_of'], event['run_id'])).fetchone():
            raise ValueError('Duplicate target must be another event in the same run')
        current = conn.execute('SELECT COALESCE(MAX(revision),0) FROM video_event_reviews WHERE event_id=?',
            (event_id,)).fetchone()[0]
        if current != expected_revision:
            raise ValueError('Video event review changed; refresh and retry')
        revision = current + 1
        conn.execute('INSERT INTO video_event_reviews(event_id,revision,operator,action,payload,created) VALUES (?,?,?,?,?,?)',
            (event_id, revision, operator, action, _json(payload), now()))
    return {'event_id': event_id, 'revision': revision, 'action': action}
