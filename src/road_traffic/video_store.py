"""Per-mission immutable video sources, durable runs, events, and review history."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from functools import lru_cache
from contextlib import nullcontext

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
      updated TEXT NOT NULL,finished TEXT,error TEXT,worker_pid INTEGER,cancel_requested INTEGER NOT NULL DEFAULT 0,
      progress_elapsed_seconds REAL,estimated_remaining_seconds REAL,
      runtime_details TEXT NOT NULL DEFAULT '{}');
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
      representative_box TEXT,
      uncertainty_reasons TEXT NOT NULL,created TEXT NOT NULL,UNIQUE(run_id,track_id));
    CREATE INDEX IF NOT EXISTS video_events_run_time_idx ON video_events(run_id,first_offset_num,first_offset_den,id);
    CREATE TABLE IF NOT EXISTS video_frame_detections(
      mission_id TEXT NOT NULL REFERENCES mission(id),run_id TEXT NOT NULL REFERENCES video_runs(id),
      source_id TEXT NOT NULL REFERENCES video_sources(id),frame_index INTEGER NOT NULL,
      pts INTEGER NOT NULL,time_base_num INTEGER NOT NULL,time_base_den INTEGER NOT NULL,
      offset_seconds REAL NOT NULL,detections TEXT NOT NULL,created TEXT NOT NULL,
      PRIMARY KEY(run_id,frame_index));
    CREATE INDEX IF NOT EXISTS video_frame_detections_time_idx
      ON video_frame_detections(run_id,offset_seconds,frame_index);
    CREATE TABLE IF NOT EXISTS video_event_reviews(
      id INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT NOT NULL REFERENCES video_events(id),
      revision INTEGER NOT NULL,operator TEXT NOT NULL,action TEXT NOT NULL,
      payload TEXT NOT NULL,created TEXT NOT NULL,UNIQUE(event_id,revision));
    CREATE TABLE IF NOT EXISTS video_clip_exports(
      id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
      run_id TEXT NOT NULL REFERENCES video_runs(id),event_id TEXT NOT NULL REFERENCES video_events(id),
      source_id TEXT NOT NULL REFERENCES video_sources(id),operator TEXT NOT NULL,
      pre_seconds REAL NOT NULL,post_seconds REAL NOT NULL,
      requested_start_num INTEGER NOT NULL,requested_start_den INTEGER NOT NULL,
      requested_end_num INTEGER NOT NULL,requested_end_den INTEGER NOT NULL,
      output_path TEXT NOT NULL,source_sha256 TEXT NOT NULL,sha256 TEXT,
      frame_count INTEGER NOT NULL DEFAULT 0,status TEXT NOT NULL,error TEXT,
      created TEXT NOT NULL,finished TEXT);
    CREATE INDEX IF NOT EXISTS video_clip_exports_mission_idx ON video_clip_exports(mission_id,created,id);
    CREATE TRIGGER IF NOT EXISTS immutable_video_sources_u BEFORE UPDATE ON video_sources
      BEGIN SELECT RAISE(ABORT,'video source versions are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_sources_d BEFORE DELETE ON video_sources
      BEGIN SELECT RAISE(ABORT,'video source versions are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_events_u BEFORE UPDATE ON video_events
      BEGIN SELECT RAISE(ABORT,'video events are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_events_d BEFORE DELETE ON video_events
      BEGIN SELECT RAISE(ABORT,'video events are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_frame_detections_u BEFORE UPDATE ON video_frame_detections
      BEGIN SELECT RAISE(ABORT,'video frame detections are immutable'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_video_frame_detections_d BEFORE DELETE ON video_frame_detections
      BEGIN SELECT RAISE(ABORT,'video frame detections are immutable'); END;
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
    if 'progress_elapsed_seconds' not in columns:
        conn.execute('ALTER TABLE video_runs ADD COLUMN progress_elapsed_seconds REAL')
    if 'estimated_remaining_seconds' not in columns:
        conn.execute('ALTER TABLE video_runs ADD COLUMN estimated_remaining_seconds REAL')
    if 'runtime_details' not in columns:
        conn.execute("ALTER TABLE video_runs ADD COLUMN runtime_details TEXT NOT NULL DEFAULT '{}'")
    event_columns = {row['name'] for row in conn.execute('PRAGMA table_info(video_events)')}
    if 'representative_box' not in event_columns:
        conn.execute('ALTER TABLE video_events ADD COLUMN representative_box TEXT')


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
    required_bytes = source.stat().st_size
    available_bytes = shutil.disk_usage(library.location(mission)).free
    if required_bytes > available_bytes:
        raise ValueError(f'Video import needs {required_bytes:,} bytes but only {available_bytes:,} bytes are free in mission storage')
    source_id = str(uuid.uuid4())
    mission_root = library.location(mission)
    original_name = Path(source_relative).name
    destination_relative = Path('sources') / 'videos' / source_id / original_name
    destination = mission_root / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=False)
    try:
        sha256, size, mtime_ns = _sha_and_copy(source, destination)
        try:
            from .video_decoder import probe_video
            media_probe = {'status': 'available', 'metadata': probe_video(destination)}
        except (ValueError, OSError, RuntimeError) as exc:
            # Keep the immutable evidence snapshot even when the local decoder
            # cannot inspect it; the UI makes this explicit before a run starts.
            media_probe = {'status': 'unavailable', 'error': str(exc)[:500]}
        source_info = {**metadata, 'source_mtime_ns_at_import': mtime_ns,
            'media_probe': media_probe}
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


@lru_cache(maxsize=256)
def _verify_source_digest(path_text, expected_sha256, signature):
    path = Path(path_text)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    current = path.stat()
    current_signature = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if current_signature != signature:
        raise ValueError('Stored video evidence changed during its SHA-256 check')
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Stored video evidence failed its SHA-256 check')
    return True


def source_path(library, mission, source_id):
    source = source_detail(library, mission, source_id)
    try:
        path = safe_file(library.location(mission), source['stored_relative_path'])
        stat = path.stat()
    except (ValueError, FileNotFoundError) as exc:
        raise ValueError('Stored video evidence is missing or unsafe') from exc
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    _verify_source_digest(str(path), source['sha256'], signature)
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
        task = config.get('task', 'video_traffic')
        if settings['analysis_mode'] != task:
            raise ValueError('Video run task must match the mission analysis mode')
        if task not in ('video_traffic', 'object_search', 'license_plate'):
            raise ValueError('Unsupported video analysis task')
        if task == 'license_plate' and not isinstance(config.get('plate_detector'), dict):
            raise ValueError('Plate crop review runs must snapshot plate_detector settings')
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
        'last_box': list(track.last_box),
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
            observation_count,association_uncertain,representative_box,uncertainty_reasons,created)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (event_id, mission, run_id, source['id'], track.track_id, first.pts, first.time_base_num,
             first.time_base_den, last.pts, last.time_base_num, last.time_base_den,
             first_offset.numerator, first_offset.denominator, last_offset.numerator, last_offset.denominator,
             first_utc, last_utc, _json(list(track.labels)), track.observation_count,
             int(track.association_uncertain), _json(list(track.last_box)),
             _json(list(track.uncertainty_reasons)), created))
        inserted += cursor.rowcount
    return inserted


def _frame_detections(detections, assignments):
    if not isinstance(detections, (list, tuple)) or not isinstance(assignments, (list, tuple)):
        raise ValueError('Frame detections and assignments must be sequences')
    tracks = {}
    for assignment in assignments:
        index = getattr(assignment, 'detection_index', None)
        track = getattr(assignment, 'track', None)
        track_id = getattr(track, 'track_id', None)
        if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                or index >= len(detections) or not isinstance(track_id, str) or not track_id):
            raise ValueError('Frame assignment does not match its detection')
        if index in tracks:
            raise ValueError('A frame detection cannot have multiple track assignments')
        tracks[index] = track_id
    result = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError('Frame detection must be a mapping')
        box = detection.get('box')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError('Frame detection box must contain normalized x, y, width, height')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in box):
            raise ValueError('Frame detection box must contain finite numbers')
        x, y, width, height = map(float, box)
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 1 or y + height > 1:
            raise ValueError('Frame detection box must be inside normalized image bounds')
        label = detection.get('label')
        if not isinstance(label, str) or not label:
            raise ValueError('Frame detection label is required')
        item = {'box': [x, y, width, height], 'label': label,
            'track_id': tracks.get(index)}
        score = detection.get('score', detection.get('confidence'))
        if score is not None:
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Frame detection score must be between zero and one')
            item['score'] = float(score)
        result.append(item)
    return result


def checkpoint(library, mission, run_id, *, frame_index, timestamp, tracker_state, closed_tracks=(),
               detections=(), assignments=(), progress_elapsed_seconds=None,
               estimated_remaining_seconds=None):
    prepare(library, mission)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError('frame_index must be a non-negative integer')
    if not isinstance(timestamp, VideoTimestamp):
        raise TypeError('timestamp must be a VideoTimestamp')
    state_json = _json(tracker_state)
    frame_detections = _frame_detections(detections, assignments)
    fingerprint_payload = {'frame_index': frame_index, 'timestamp': [timestamp.pts,
        timestamp.time_base_num, timestamp.time_base_den], 'tracker_state': tracker_state,
        'closed_tracks': [_track_payload(track) for track in closed_tracks],
        'detections': frame_detections}
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
        offset = timestamp.offset_seconds
        conn.execute('''INSERT INTO video_frame_detections(mission_id,run_id,source_id,frame_index,
            pts,time_base_num,time_base_den,offset_seconds,detections,created) VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (mission,run_id,source['id'],frame_index,timestamp.pts,timestamp.time_base_num,
             timestamp.time_base_den,float(offset),_json(frame_detections),now()))
        conn.execute('''UPDATE video_runs SET stage='analyzing',frames_done=?,current_pts=?,
            current_time_base_num=?,current_time_base_den=?,tracker_checkpoint=?,
            last_frame_fingerprint=?,progress_elapsed_seconds=?,estimated_remaining_seconds=?,updated=? WHERE id=?''',
            (frame_index + 1, timestamp.pts, timestamp.time_base_num, timestamp.time_base_den,
             state_json, fingerprint, progress_elapsed_seconds, estimated_remaining_seconds, now(), run_id))
    return {'duplicate': False, 'frames_done': frame_index + 1, 'events_inserted': events_inserted}


def frame_overlay(library, mission, run_id, seconds):
    """Return saved detections from the nearest analyzed frame covering a playback time."""
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError('Playback time must be a finite non-negative number')
    prepare(library, mission)
    with library.connect(mission) as conn:
        run = conn.execute('''SELECT r.id,r.source_id,r.config,r.frames_done
            FROM video_runs r WHERE r.mission_id=? AND r.id=?''', (mission,run_id)).fetchone()
        if not run:
            raise ValueError('Video run not found in this mission')
        config = json.loads(run['config'])
        if config.get('task', 'video_traffic') not in ('video_traffic', 'object_search'):
            return {'run_id': run_id, 'source_id': run['source_id'], 'frame': None, 'detections': []}
        before = conn.execute('''SELECT * FROM video_frame_detections WHERE mission_id=? AND run_id=?
            AND offset_seconds<=? ORDER BY offset_seconds DESC,frame_index DESC LIMIT 1''',
            (mission,run_id,float(seconds))).fetchone()
        after = conn.execute('''SELECT * FROM video_frame_detections WHERE mission_id=? AND run_id=?
            AND offset_seconds>=? ORDER BY offset_seconds,frame_index LIMIT 1''',
            (mission,run_id,float(seconds))).fetchone()
    # Don't extrapolate before the first or after the last analyzed frame.
    if before is None or after is None:
        return {'run_id':run_id,'source_id':run['source_id'],'frame':None,'detections':[]}
    nearest = before if seconds - before['offset_seconds'] <= after['offset_seconds'] - seconds else after
    try:
        boxes = json.loads(nearest['detections'])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Saved video frame detections are corrupt') from exc
    return {'run_id':run_id,'source_id':run['source_id'],
        'frame':{'index':nearest['frame_index'],'pts':nearest['pts'],
            'time_base_num':nearest['time_base_num'],'time_base_den':nearest['time_base_den'],
            'offset_seconds':nearest['offset_seconds']},'detections':boxes}


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
                      progress_callback=None, frame_callback=None):
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

        latest_progress = {}
        def progress(item):
            if item.total_frames is not None and run['total_frames'] is None:
                set_total_frames(library, mission, run_id, item.total_frames)
                run['total_frames'] = item.total_frames
            latest_progress['elapsed_seconds'] = item.elapsed_seconds
            latest_progress['estimated_remaining_seconds'] = item.estimated_remaining_seconds
            if progress_callback is not None:
                progress_callback(item)

        for record in pipeline.analyze(path,
                recording_start_utc=(datetime.fromisoformat(source['recording_start_utc'])
                    if source['recording_start_utc'] else None),
                should_cancel=should_cancel, progress_callback=progress,
                frame_callback=frame_callback,
                start_frame_index=run['frames_done'], expected_previous_timestamp=previous):
            if isinstance(record, VideoFrameAnalysis):
                checkpoint(library, mission, run_id, frame_index=record.frame_index,
                    timestamp=record.timestamp, tracker_state=tracker.state(),
                    closed_tracks=record.closed_tracks, detections=record.detections,
                    assignments=record.assignments,
                    progress_elapsed_seconds=latest_progress.get('elapsed_seconds'),
                    estimated_remaining_seconds=latest_progress.get('estimated_remaining_seconds'))
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
        conn.execute('''UPDATE video_runs SET status=?,stage=?,updated=?,finished=?,error=?,worker_pid=?,cancel_requested=?,
            progress_elapsed_seconds=CASE WHEN ?='running' THEN NULL ELSE progress_elapsed_seconds END,
            estimated_remaining_seconds=CASE WHEN ?='running' THEN NULL ELSE estimated_remaining_seconds END
            WHERE id=?''',
            (target, target, now(), finished, error, worker_pid, cancel_requested,target,target,run_id))
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
    result['runtime_details'] = json.loads(result.get('runtime_details') or '{}')
    return result


def update_runtime_details(library, mission, run_id, details):
    """Persist actual execution-provider telemetry separately from immutable run config."""
    if not isinstance(details, dict):
        raise ValueError('Runtime details must be a JSON object')
    encoded = _json(details)
    if len(encoded.encode('utf-8')) > 4096:
        raise ValueError('Runtime details exceed the 4 KiB limit')
    prepare(library, mission)
    with library.connect(mission) as conn:
        cursor = conn.execute('''UPDATE video_runs SET runtime_details=?,updated=?
            WHERE mission_id=? AND id=?''', (encoded, now(), mission, run_id))
        if cursor.rowcount != 1:
            raise ValueError('Video run not found')
    return details


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
        rows = conn.execute('SELECT id,mission_id,source_id,status,stage,total_frames,frames_done,progress_elapsed_seconds,estimated_remaining_seconds,runtime_details,created,updated,finished,error,worker_pid,cancel_requested FROM video_runs WHERE mission_id=? ORDER BY created DESC,id',
            (mission,)).fetchall()
    result = [dict(row) for row in rows]
    for row in result:
        row['runtime_details'] = json.loads(row['runtime_details'] or '{}')
    return result


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


def events(library, mission, run_id, *, limit=100, offset=0, prepared=False, connection=None):
    if not prepared:
        prepare(library, mission)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError('event page limit must be between 1 and 100')
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('event page offset must be a non-negative integer')
    with (nullcontext(connection) if connection is not None else library.connect(mission)) as conn:
        if not conn.execute('SELECT 1 FROM video_runs WHERE id=? AND mission_id=?',
                (run_id, mission)).fetchone():
            raise ValueError('Video run not found')
        has_active_identity_edits=bool(conn.execute('''SELECT 1 FROM video_event_reviews r JOIN (
            SELECT event_id,MAX(revision) revision FROM video_event_reviews
            WHERE action IN ('merge_tracks','split_track','clear_identity_correction') GROUP BY event_id
          ) x ON x.event_id=r.event_id AND x.revision=r.revision
          JOIN video_events e ON e.id=r.event_id WHERE e.mission_id=? AND e.run_id=?
            AND r.action IN ('merge_tracks','split_track') LIMIT 1''',(mission,run_id)).fetchone())
        already_paged=not has_active_identity_edits
        rows = conn.execute('''SELECT e.*,r.revision,r.operator,r.action,r.payload,r.created reviewed_at,
            s.recording_start_utc,t.payload timing_payload,c.payload class_payload
            FROM video_events e LEFT JOIN video_event_reviews r ON r.event_id=e.id
            AND r.revision=(SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id)
            LEFT JOIN video_event_reviews t ON t.event_id=e.id AND t.revision=(
              SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id AND x.action='correct_timing')
            LEFT JOIN video_event_reviews c ON c.event_id=e.id AND c.revision=(
              SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id AND x.action='correct_class')
            JOIN video_sources s ON s.id=e.source_id AND s.mission_id=e.mission_id
            WHERE e.mission_id=? AND e.run_id=? ORDER BY (1.0*e.first_offset_num/e.first_offset_den),e.id'''+
            (' LIMIT ? OFFSET ?' if already_paged else ''),
            (mission,run_id,limit,offset) if already_paged else (mission,run_id)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ('labels', 'uncertainty_reasons', 'payload', 'timing_payload', 'class_payload', 'representative_box'):
                item[key] = json.loads(item[key]) if item[key] is not None else None
            if item['timing_payload'] is not None:
                item = apply_event_timing(conn, item)
            if item['class_payload'] is not None:
                item['original_labels'] = item['labels']
                corrected_label = item['class_payload'].get('label')
                if corrected_label:
                    item['labels'] = [corrected_label]
            item['review_status'] = {'accept': 'accepted', 'reject': 'rejected',
                'reopen': 'pending', 'correct_class': 'corrected', 'correct_timing': 'corrected',
                'mark_duplicate': 'duplicate', 'merge_tracks': 'corrected',
                'split_track': 'corrected', 'clear_identity_correction': 'corrected'}.get(item['action'], 'pending')
            item['source_event_id'] = item['id']
            result.append(item)
        latest_identity = conn.execute('''SELECT r.event_id,r.action,r.payload,r.revision,r.operator,r.created
            FROM video_event_reviews r JOIN (
              SELECT event_id,MAX(revision) revision FROM video_event_reviews
              WHERE action IN ('merge_tracks','split_track','clear_identity_correction') GROUP BY event_id
            ) x ON x.event_id=r.event_id AND x.revision=r.revision
            JOIN video_events e ON e.id=r.event_id WHERE e.mission_id=? AND e.run_id=?''',
            (mission, run_id)).fetchall()
        edits = {row['event_id']: dict(row) for row in latest_identity}
        for edit in edits.values():
            edit['payload'] = json.loads(edit['payload']) if edit['payload'] else {}
        by_id = {item['id']: item for item in result}
        # Merge source events into their selected target and project one logical event.
        groups = {}
        for item in result:
            event_id = item['id']; seen = set()
            while event_id in edits and edits[event_id]['action'] == 'merge_tracks':
                if event_id in seen:
                    raise ValueError('Saved track merge history contains a cycle')
                seen.add(event_id); event_id = edits[event_id]['payload']['target_event_id']
            groups.setdefault(event_id, []).append(item)
        projected = []
        for root_id, members in groups.items():
            root = dict(by_id.get(root_id, members[0]))
            if len(members) > 1:
                first_member = min(members,key=lambda item:Fraction(item['first_offset_num'],item['first_offset_den']))
                last_member = max(members,key=lambda item:Fraction(item['last_offset_num'],item['last_offset_den']))
                root.update({key:first_member[key] for key in ('first_pts','first_time_base_num','first_time_base_den','first_offset_num','first_offset_den','first_utc')})
                root.update({key:last_member[key] for key in ('last_pts','last_time_base_num','last_time_base_den','last_offset_num','last_offset_den','last_utc')})
                root['labels'] = sorted({label for member in members for label in member['labels']})
                root['original_labels'] = sorted({label for member in members
                    for label in member.get('original_labels',member['labels'])})
                root['representative_box'] = last_member['representative_box']
                root['preview_event_id'] = last_member['id']
                root['observation_count'] = sum(member['observation_count'] for member in members)
                root['association_uncertain'] = int(any(member['association_uncertain'] for member in members))
                root['merged_event_ids'] = sorted(member['id'] for member in members)
                root['original_track_ids'] = sorted(member['track_id'] for member in members)
                merged_from=[member['id'] for member in members if member['id']!=root_id]
                root['identity_correction'] = {'action':'merge_tracks','event_ids':root['merged_event_ids'],
                    'merged_from_event_ids':sorted(merged_from)}
                if merged_from:
                    active_edit=edits[merged_from[0]]
                    root['identity_action_event_id']=merged_from[0]
                    root['identity_action_revision']=active_edit['revision']
                root['review_status'] = 'corrected'
            edit = edits.get(root_id)
            if edit and edit['action'] == 'split_track':
                projected.extend(_split_logical_event(library, mission, root, edit['payload'], connection=conn))
            else:
                projected.append(root)
        result = sorted(projected, key=lambda item: (Fraction(item['first_offset_num'], item['first_offset_den']), item['id']))
        return result if already_paged else result[offset:offset+limit]


def _split_logical_event(library, mission, event, payload, connection=None):
    split_at = payload.get('frame_index')
    if isinstance(split_at, bool) or not isinstance(split_at, int):
        raise ValueError('Saved track split frame is invalid')
    with (nullcontext(connection) if connection is not None else library.connect(mission)) as conn:
        rows = conn.execute('''SELECT frame_index,pts,time_base_num,time_base_den,detections
            FROM video_frame_detections WHERE mission_id=? AND run_id=? ORDER BY frame_index''',
            (mission,event['run_id'])).fetchall()
    track = event['track_id']
    timing=event.get('timing_payload') or {}
    min_frame=timing.get('first_frame_index',-1);max_frame=timing.get('last_frame_index',2**63-1)
    observations=[]
    for row in rows:
        if row['frame_index']<min_frame or row['frame_index']>max_frame:
            continue
        try: detections=json.loads(row['detections'])
        except (TypeError,json.JSONDecodeError) as exc: raise ValueError('Saved frame detections are corrupt') from exc
        match=next((d for d in detections if isinstance(d,dict) and d.get('track_id')==track),None)
        if match:
            stamp=VideoTimestamp(row['pts'],row['time_base_num'],row['time_base_den'])
            observations.append((row['frame_index'],stamp,match))
    left=[obs for obs in observations if obs[0]<split_at]
    right=[obs for obs in observations if obs[0]>=split_at]
    if not left or not right:
        raise ValueError('A split must leave at least one saved observation in each child event')
    origin=datetime.fromisoformat(event['recording_start_utc']) if event.get('recording_start_utc') else None
    children=[]
    for ordinal,part in enumerate((left,right),1):
        first=part[0][1];last=part[-1][1]
        child=dict(event)
        child['id']=event['id'] if ordinal==1 else str(uuid.uuid5(uuid.NAMESPACE_URL,f"traffic-event-split:{event['id']}:{split_at}"))
        child['source_event_id']=event['id'];child['track_id']=f"{track}:child-{ordinal}"
        child.update(first_pts=first.pts,first_time_base_num=first.time_base_num,first_time_base_den=first.time_base_den,
            first_offset_num=first.offset_seconds.numerator,first_offset_den=first.offset_seconds.denominator,
            last_pts=last.pts,last_time_base_num=last.time_base_num,last_time_base_den=last.time_base_den,
            last_offset_num=last.offset_seconds.numerator,last_offset_den=last.offset_seconds.denominator,
            first_utc=first.utc_at(origin).isoformat() if origin else None,last_utc=last.utc_at(origin).isoformat() if origin else None,
            observation_count=len(part),labels=sorted({str(obs[2].get('label')) for obs in part if obs[2].get('label')}),
            representative_box=part[-1][2].get('box'),review_status='corrected',
            identity_correction={'action':'split_track','frame_index':split_at,'child':ordinal,'children':2})
        child['first_frame_index']=part[0][0];child['last_frame_index']=part[-1][0]
        if (event.get('class_payload') or {}).get('label'):
            child['labels']=[event['class_payload']['label']]
        child['timing_payload']=None
        children.append(child)
    return children


def _frame_timestamp_for_track(conn, event, frame_index):
    row = conn.execute('''SELECT pts,time_base_num,time_base_den,detections
        FROM video_frame_detections WHERE mission_id=? AND run_id=? AND frame_index=?''',
        (event['mission_id'], event['run_id'], frame_index)).fetchone()
    if not row:
        raise ValueError('Timing correction must use an analyzed frame from this event')
    try:
        detections = json.loads(row['detections'])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Saved frame detections are corrupt') from exc
    if not any(isinstance(detection, dict) and detection.get('track_id') == event['track_id']
            for detection in detections):
        raise ValueError('Timing correction frame must contain this event track')
    return VideoTimestamp(row['pts'], row['time_base_num'], row['time_base_den'])


def apply_event_timing(conn, event):
    """Project an append-only timing correction onto a saved event row."""
    result = dict(event)
    payload = result.get('timing_payload', result.get('payload'))
    if not isinstance(payload, dict):
        raise ValueError('Saved event timing correction is invalid')
    first_index = payload.get('first_frame_index')
    last_index = payload.get('last_frame_index')
    if (isinstance(first_index, bool) or not isinstance(first_index, int)
            or isinstance(last_index, bool) or not isinstance(last_index, int)
            or first_index > last_index):
        raise ValueError('Saved event timing correction has invalid frame indexes')
    first = _frame_timestamp_for_track(conn, result, first_index)
    last = _frame_timestamp_for_track(conn, result, last_index)
    original_first = Fraction(result['first_offset_num'], result['first_offset_den'])
    original_last = Fraction(result['last_offset_num'], result['last_offset_den'])
    if (first.offset_seconds > last.offset_seconds or first.offset_seconds < original_first
            or last.offset_seconds > original_last):
        raise ValueError('Saved event timing correction has a reversed interval')
    result['original_first_offset_seconds'] = float(Fraction(result['first_offset_num'], result['first_offset_den']))
    result['original_last_offset_seconds'] = float(Fraction(result['last_offset_num'], result['last_offset_den']))
    result.update(first_pts=first.pts, first_time_base_num=first.time_base_num,
        first_time_base_den=first.time_base_den, first_offset_num=first.offset_seconds.numerator,
        first_offset_den=first.offset_seconds.denominator, last_pts=last.pts,
        last_time_base_num=last.time_base_num, last_time_base_den=last.time_base_den,
        last_offset_num=last.offset_seconds.numerator, last_offset_den=last.offset_seconds.denominator)
    origin_text = result.get('recording_start_utc')
    origin = datetime.fromisoformat(origin_text) if origin_text else None
    result['first_utc'] = first.utc_at(origin).isoformat() if origin else None
    result['last_utc'] = last.utc_at(origin).isoformat() if origin else None
    return result


def logical_event_by_id(library, mission, event_id):
    prepare(library,mission)
    with library.connect(mission) as conn:
        run_ids=[row['run_id'] for row in conn.execute('SELECT DISTINCT run_id FROM video_events WHERE mission_id=?',(mission,))]
    for run_id in run_ids:
        offset=0
        while True:
            page=events(library,mission,run_id,limit=100,offset=offset,prepared=True)
            logical=next((item for item in page if item['id']==event_id),None)
            if logical:return logical
            if len(page)<100:break
            offset+=len(page)
    raise ValueError('Video event not found in this mission')


def event_thumbnail(library, mission, event_id):
    """Render a bounded JPEG preview from the event's verified source frame."""
    logical=logical_event_by_id(library,mission,event_id)
    source_event_id=logical.get('preview_event_id',logical.get('source_event_id',event_id))
    with library.connect(mission) as conn:
        event = conn.execute('''SELECT e.*,s.recording_start_utc,t.payload timing_payload
            FROM video_events e JOIN video_sources s ON s.id=e.source_id AND s.mission_id=e.mission_id
            LEFT JOIN video_event_reviews t ON t.event_id=e.id AND t.revision=(
              SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id AND x.action='correct_timing')
            WHERE e.id=? AND e.mission_id=?''',
            (source_event_id, mission)).fetchone()
    if not event:
        raise ValueError('Video event not found in this mission')
    event = dict(event)
    event.update({key:logical[key] for key in ('first_pts','first_time_base_num','first_time_base_den',
        'last_pts','last_time_base_num','last_time_base_den','representative_box') if key in logical})
    event['timing_payload']=None
    if not event['representative_box']:
        raise ValueError('This saved event predates frame preview metadata')
    box = json.loads(event['representative_box']) if isinstance(event['representative_box'],str) else event['representative_box']
    if (not isinstance(box, list) or len(box) != 4 or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in box)):
        raise ValueError('Video event preview geometry is invalid')
    import io
    import av
    from PIL import Image, ImageDraw
    x, y, box_width, box_height = box
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0 or x + box_width > 1 or y + box_height > 1:
        raise ValueError('Video event preview geometry is outside normalized image bounds')

    path = source_path(library, mission, event['source_id'])
    target = Fraction(event['last_pts'] * event['last_time_base_num'], event['last_time_base_den'])
    with av.open(str(path), mode='r') as container:
        streams = container.streams.video
        if not streams:
            raise ValueError('Stored video source has no video stream')
        stream = streams[0]
        stream_time_base = Fraction(stream.time_base)
        seek_pts = target / stream_time_base
        container.seek(seek_pts.numerator // seek_pts.denominator, stream=stream, backward=True)
        selected = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = Fraction(frame.pts) * Fraction(frame.time_base or stream.time_base)
            if frame_time == target:
                selected = frame
                break
            if frame_time > target:
                break
    if selected is None:
        raise ValueError('Could not seek to the saved event frame')
    image = Image.fromarray(selected.to_ndarray(format='rgb24'), mode='RGB')
    width, height = image.size
    left = max(0, min(width - 1, round(x * width)))
    top = max(0, min(height - 1, round(y * height)))
    right = max(left + 1, min(width - 1, round((x + box_width) * width)))
    bottom = max(top + 1, min(height - 1, round((y + box_height) * height)))
    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, right, bottom), outline=(255, 48, 48), width=max(2, round(min(width, height) / 120)))
    image.thumbnail((800, 450), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=82, optimize=True)
    return output.getvalue()


def review_event(library, mission, event_id, expected_revision, operator, action, payload=None):
    prepare(library, mission)
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise ValueError('expected_revision must be a non-negative integer')
    operator = operator.strip() if isinstance(operator, str) else ''
    if not 1 <= len(operator) <= 120:
        raise ValueError('Operator is required for video event review')
    if action not in ('accept', 'reject', 'reopen', 'correct_class', 'correct_timing', 'mark_duplicate',
            'merge_tracks', 'split_track', 'clear_identity_correction'):
        raise ValueError('Unsupported video event correction')
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
    elif action == 'correct_timing':
        requested = []
        for name in ('first_offset_seconds', 'last_offset_seconds'):
            value = payload.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError('Timing correction requires start and end seconds')
            try:
                offset = Fraction(str(value))
            except (ValueError, ZeroDivisionError):
                raise ValueError('Timing correction seconds must be finite numbers')
            requested.append(offset)
        if requested[0] > requested[1]:
            raise ValueError('Timing correction start must not follow its end')
    elif action == 'merge_tracks':
        target_id=payload.get('target_event_id')
        if not isinstance(target_id,str) or not target_id.strip() or target_id==event_id:
            raise ValueError('Track merge requires a different target event ID')
        payload={'target_event_id':target_id.strip()}
    elif action == 'split_track':
        frame_index=payload.get('frame_index')
        if isinstance(frame_index,bool) or not isinstance(frame_index,int) or frame_index<0:
            raise ValueError('Track split requires a non-negative analyzed frame index')
        payload={'frame_index':frame_index}
    elif payload:
        raise ValueError(f'{action} does not accept a payload')
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        event = conn.execute('SELECT * FROM video_events WHERE id=? AND mission_id=?',
            (event_id, mission)).fetchone()
        if not event:
            raise ValueError('Video event not found in this mission')
        if action == 'mark_duplicate' and not conn.execute(
                'SELECT 1 FROM video_events WHERE id=? AND run_id=?',
                (payload['duplicate_of'], event['run_id'])).fetchone():
            raise ValueError('Duplicate target must be another event in the same run')
        if action in ('merge_tracks','split_track','clear_identity_correction'):
            latest=conn.execute('''SELECT r.event_id,r.action,r.payload FROM video_event_reviews r JOIN (
                SELECT event_id,MAX(revision) revision FROM video_event_reviews
                WHERE action IN ('merge_tracks','split_track','clear_identity_correction') GROUP BY event_id
              ) x ON x.event_id=r.event_id AND x.revision=r.revision
              JOIN video_events e ON e.id=r.event_id WHERE e.run_id=?''',(event['run_id'],)).fetchall()
            active={row['event_id']:row for row in latest if row['action']!='clear_identity_correction'}
            if action=='clear_identity_correction':
                if event_id not in active:
                    raise ValueError('This event has no active identity correction to clear')
            elif action=='merge_tracks':
                target=conn.execute('SELECT * FROM video_events WHERE id=? AND run_id=?',(payload['target_event_id'],event['run_id'])).fetchone()
                if not target: raise ValueError('Merge target must be another event in the same run')
                if event_id in active:
                    raise ValueError('Clear the existing identity correction before merging this event')
                if payload['target_event_id'] in active and active[payload['target_event_id']]['action']=='split_track':
                    raise ValueError('A split event cannot be a merge target')
                if payload['target_event_id'] in active and active[payload['target_event_id']]['action']=='merge_tracks':
                    raise ValueError('Choose the current root event as the merge target')
                cursor=payload['target_event_id'];seen={event_id}
                while cursor in active and active[cursor]['action']=='merge_tracks':
                    if cursor in seen: raise ValueError('This merge would create an identity cycle')
                    seen.add(cursor);cursor=json.loads(active[cursor]['payload'])['target_event_id']
                if cursor==event_id: raise ValueError('This merge would create an identity cycle')
            elif action=='split_track':
                if event_id in active:
                    raise ValueError('Clear the existing identity correction before splitting this event')
                incoming=conn.execute('''SELECT r.event_id,r.payload FROM video_event_reviews r JOIN (
                    SELECT event_id,MAX(revision) revision FROM video_event_reviews
                    WHERE action IN ('merge_tracks','split_track','clear_identity_correction') GROUP BY event_id
                  ) x ON x.event_id=r.event_id AND x.revision=r.revision
                  WHERE r.action='merge_tracks' AND r.payload LIKE ?''',(f'%"target_event_id":"{event_id}"%',)).fetchone()
                if incoming: raise ValueError('A merged event must be separated before it can be split')
                frame=conn.execute('''SELECT frame_index,detections FROM video_frame_detections
                    WHERE mission_id=? AND run_id=? ORDER BY frame_index''',(mission,event['run_id'])).fetchall()
                observed=[]
                for saved in frame:
                    try:detections=json.loads(saved['detections'])
                    except (TypeError,json.JSONDecodeError) as exc:raise ValueError('Saved frame detections are corrupt') from exc
                    if any(isinstance(d,dict) and d.get('track_id')==event['track_id'] for d in detections):observed.append(saved['frame_index'])
                if payload['frame_index'] not in observed or not any(i<payload['frame_index'] for i in observed) or not any(i>=payload['frame_index'] for i in observed):
                    raise ValueError('Split frame must be a saved observation with track observations on both sides')
        if action == 'correct_timing':
            first_original = Fraction(event['first_offset_num'], event['first_offset_den'])
            last_original = Fraction(event['last_offset_num'], event['last_offset_den'])
            if requested[0] < first_original or requested[1] > last_original:
                raise ValueError('Timing corrections may trim only within the original event interval')
            frames = conn.execute('''SELECT frame_index,pts,time_base_num,time_base_den,detections
                FROM video_frame_detections WHERE mission_id=? AND run_id=?
                AND offset_seconds>=? AND offset_seconds<=? ORDER BY frame_index''',
                (mission, event['run_id'], float(first_original)-1e-9,
                 float(last_original)+1e-9)).fetchall()
            candidates = []
            for frame in frames:
                timestamp = VideoTimestamp(frame['pts'], frame['time_base_num'], frame['time_base_den'])
                if not first_original <= timestamp.offset_seconds <= last_original:
                    continue
                try:
                    frame_detections = json.loads(frame['detections'])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError('Saved frame detections are corrupt') from exc
                if any(isinstance(detection, dict) and detection.get('track_id') == event['track_id']
                        for detection in frame_detections):
                    candidates.append((timestamp.offset_seconds, frame['frame_index']))
            if not candidates:
                raise ValueError('No saved analyzed observations are available for this event')
            first_observation = min(candidates, key=lambda pair: (abs(pair[0] - requested[0]), pair[1]))
            last_observation = min(candidates, key=lambda pair: (abs(pair[0] - requested[1]), pair[1]))
            if first_observation[1] > last_observation[1] or first_observation[0] > last_observation[0]:
                raise ValueError('Timing correction resolves to a reversed frame interval')
            payload = {'first_frame_index': first_observation[1], 'last_frame_index': last_observation[1],
                'requested_first_offset': [requested[0].numerator, requested[0].denominator],
                'requested_last_offset': [requested[1].numerator, requested[1].denominator]}
        current = conn.execute('SELECT COALESCE(MAX(revision),0) FROM video_event_reviews WHERE event_id=?',
            (event_id,)).fetchone()[0]
        if current != expected_revision:
            raise ValueError('Video event review changed; refresh and retry')
        revision = current + 1
        conn.execute('INSERT INTO video_event_reviews(event_id,revision,operator,action,payload,created) VALUES (?,?,?,?,?,?)',
            (event_id, revision, operator, action, _json(payload), now()))
    return {'event_id': event_id, 'revision': revision, 'action': action}
