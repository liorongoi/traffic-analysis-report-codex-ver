"""Audited user-contextualized video event clip exports."""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from . import video_store
from .store import now


def _seconds(value, label):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a number between 0 and 600 seconds')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f'{label} must be a number between 0 and 600 seconds')
    if not result.is_finite() or not Decimal(0) <= result <= Decimal(600):
        raise ValueError(f'{label} must be a number between 0 and 600 seconds')
    return result


def _encode_clip(source, destination, start_offset, end_offset):
    package_root = str(Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None,
        (package_root, env.get('PYTHONPATH', ''))))
    request = {'source':str(source),'destination':str(destination),
        'start_num':start_offset.numerator,'start_den':start_offset.denominator,
        'end_num':end_offset.numerator,'end_den':end_offset.denominator}
    result = subprocess.run([sys.executable,'-m','road_traffic.video_clip_worker'],
        input=json.dumps(request),text=True,capture_output=True,env=env,timeout=3600)
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'Clip worker failed'
        raise ValueError(detail[:500])
    try:
        return int(json.loads(result.stdout)['frame_count'])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Clip worker returned invalid completion data') from exc


def _export_detail(library, mission, export_id):
    video_store.prepare(library, mission)
    with library.connect(mission) as conn:
        row = conn.execute('SELECT * FROM video_clip_exports WHERE id=? AND mission_id=?',
            (export_id, mission)).fetchone()
    if not row:
        raise ValueError('Video clip export not found in this mission')
    item = dict(row)
    item['download_url'] = f'/api/video/clips/download?mission={mission}&export={export_id}'
    return item


def export_event_clip(library, mission, event_id, pre_seconds, post_seconds, operator):
    video_store.prepare(library, mission)
    logical=video_store.logical_event_by_id(library,mission,event_id)
    source_event_id=logical.get('source_event_id',event_id)
    pre = _seconds(pre_seconds, 'Pre-event context')
    post = _seconds(post_seconds, 'Post-event context')
    operator = operator.strip() if isinstance(operator, str) else ''
    if not 1 <= len(operator) <= 120:
        raise ValueError('Operator is required for clip export')
    with library.connect(mission) as conn:
        event = conn.execute('''SELECT e.*,r.source_id,s.sha256 source_sha256,
            s.recording_start_utc,review.action,review.payload,timing.payload timing_payload
            FROM video_events e JOIN video_runs r ON r.id=e.run_id AND r.mission_id=e.mission_id
            JOIN video_sources s ON s.id=e.source_id AND s.mission_id=e.mission_id
            LEFT JOIN video_event_reviews review ON review.event_id=e.id AND review.revision=(
              SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id)
            LEFT JOIN video_event_reviews timing ON timing.event_id=e.id AND timing.revision=(
              SELECT MAX(x.revision) FROM video_event_reviews x WHERE x.event_id=e.id AND x.action='correct_timing')
            WHERE e.id=? AND e.mission_id=?''', (source_event_id, mission)).fetchone()
    if not event:
        raise ValueError('Video event not found in this mission')
    event = dict(event)
    event.update({key:logical[key] for key in ('first_pts','first_time_base_num','first_time_base_den',
        'first_offset_num','first_offset_den','last_pts','last_time_base_num','last_time_base_den',
        'last_offset_num','last_offset_den','first_utc','last_utc') if key in logical})
    settings = library.settings(mission)
    output_text = library._configured_directory(settings['output_root'], 'output', writable=True)
    if not output_text:
        raise ValueError('Set an output directory in the mission settings before exporting a clip')
    output_root = Path(output_text)
    workspace = library.root.resolve()
    if output_root.is_relative_to(workspace) or workspace.is_relative_to(output_root):
        raise ValueError('Choose an output outside application-managed storage')
    source = video_store.source_path(library, mission, event['source_id'])
    start = Fraction(event['first_offset_num'], event['first_offset_den']) - Fraction(pre)
    end = Fraction(event['last_offset_num'], event['last_offset_den']) + Fraction(post)
    if start >= end:
        raise ValueError('Clip interval must have positive duration')
    export_id = str(uuid.uuid4())
    target = output_root / f'traffic-event-{event_id[:8]}-{export_id[:8]}.mp4'
    fd, temporary = tempfile.mkstemp(prefix='.traffic-event-', suffix='.mp4', dir=output_root)
    os.close(fd)
    os.unlink(temporary)
    created = now()
    with library.connect(mission) as conn:
        conn.execute('''INSERT INTO video_clip_exports(id,mission_id,run_id,event_id,source_id,operator,
            pre_seconds,post_seconds,requested_start_num,requested_start_den,requested_end_num,
            requested_end_den,output_path,source_sha256,status,created)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (export_id,mission,event['run_id'],source_event_id,event['source_id'],operator,float(pre),float(post),
             start.numerator,start.denominator,end.numerator,end.denominator,str(target),
             event['source_sha256'],'writing',created))
    try:
        frame_count = _encode_clip(source, temporary, start, end)
        digest = hashlib.sha256()
        with open(temporary, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if target.exists():
            raise FileExistsError('Unique clip output path already exists')
        os.replace(temporary, target)
        with library.connect(mission) as conn:
            conn.execute('''UPDATE video_clip_exports SET sha256=?,frame_count=?,status='completed',finished=?
                WHERE id=? AND mission_id=?''', (digest.hexdigest(),frame_count,now(),export_id,mission))
        return _export_detail(library, mission, export_id)
    except BaseException as exc:
        for path in (Path(temporary), target):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        with library.connect(mission) as conn:
            conn.execute('''UPDATE video_clip_exports SET status='failed',error=?,finished=?
                WHERE id=? AND mission_id=?''', (f'{type(exc).__name__}: {exc}',now(),export_id,mission))
        raise


def exports(library, mission):
    video_store.prepare(library, mission)
    with library.connect(mission) as conn:
        rows = conn.execute('SELECT id FROM video_clip_exports WHERE mission_id=? ORDER BY created DESC,id',
            (mission,)).fetchall()
    return [_export_detail(library, mission, row['id']) for row in rows]


def export_file(library, mission, export_id):
    item = _export_detail(library, mission, export_id)
    if item['status'] != 'completed' or not item['sha256']:
        raise ValueError('Video clip export is not complete')
    path = Path(item['output_path'])
    if not path.is_file():
        raise ValueError('Video clip export file is missing')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise ValueError('Video clip export failed its SHA-256 check')
    return path


def reconcile_exports(library):
    """Mark clip exports interrupted by service shutdown as incomplete."""
    reconciled = 0
    for mission_path in library.root.glob('*/mission.sqlite3'):
        mission = mission_path.parent.name
        video_store.prepare(library, mission)
        with library.connect(mission) as conn:
            rows = conn.execute("SELECT id,output_path FROM video_clip_exports WHERE status='writing' AND mission_id=?",
                (mission,)).fetchall()
            for row in rows:
                status = 'partial' if Path(row['output_path']).is_file() else 'failed'
                conn.execute('UPDATE video_clip_exports SET status=?,finished=?,error=? WHERE id=?',
                    (status,now(),'Clip export process stopped before completion',row['id']))
                reconciled += 1
    return reconciled
