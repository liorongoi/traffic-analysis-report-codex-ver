"""Traceable, configured CSV/XLSX snapshots of one saved video run's events."""
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile
from xml.sax.saxutils import escape

from . import video_store
from .store import now


HEADERS = ('mission_id', 'run_id', 'source_id', 'source_sha256', 'video_name',
    'source_relative_path', 'event_id', 'track_id', 'first_pts', 'first_time_base_num',
    'first_time_base_den', 'last_pts', 'last_time_base_num', 'last_time_base_den',
    'first_offset_seconds', 'last_offset_seconds', 'first_utc', 'last_utc', 'labels',
    'original_labels', 'timing_correction', 'class_correction',
    'original_first_offset_seconds', 'original_last_offset_seconds',
    'observation_count', 'association_uncertain', 'uncertainty_reasons', 'review_revision',
    'review_action', 'review_status', 'review_operator', 'reviewed_at', 'review_payload',
    'representative_box', 'clip_exports', 'original_event_ids', 'identity_correction')


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _cell(value):
    if value is None:
        return ''
    text = str(value)
    if text.lstrip().startswith(('=', '+', '-', '@')):
        text = "'" + text
    return text


def _directory(library, mission):
    settings = library.settings(mission)
    if not settings['output_root']:
        raise ValueError('Set an output directory in the mission settings before exporting event tables')
    output = library._configured_directory(settings['output_root'], 'output', writable=True)
    if output != settings['output_root']:
        raise ValueError('Configured output directory changed; update the mission settings before exporting')
    root = Path(output).resolve(strict=True)
    workspace = library.root.resolve()
    if root.is_relative_to(workspace) or workspace.is_relative_to(root):
        raise ValueError('Choose an output outside application-managed storage')
    return root


def _prepare(library, mission):
    video_store.prepare(library, mission)
    with library.connect(mission) as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS video_event_table_exports(
          id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
          run_id TEXT NOT NULL REFERENCES video_runs(id),operator TEXT NOT NULL,
          format TEXT NOT NULL,output_root TEXT NOT NULL DEFAULT '',
          output_path TEXT NOT NULL,temp_path TEXT NOT NULL,
          snapshot_sha256 TEXT,row_count INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL,error TEXT,created TEXT NOT NULL,finished TEXT);
        CREATE INDEX IF NOT EXISTS video_event_table_exports_mission_idx
          ON video_event_table_exports(mission_id,created,id);
        ''')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(video_event_table_exports)')}
        if 'output_root' not in columns:
            conn.execute("ALTER TABLE video_event_table_exports ADD COLUMN output_root TEXT NOT NULL DEFAULT ''")


def _rows(conn, mission, run_id):
    raw={row['id']:dict(row) for row in conn.execute('''SELECT e.id,e.mission_id,e.run_id,e.source_id,
        e.track_id,e.labels,e.observation_count,e.association_uncertain,e.uncertainty_reasons,
        e.representative_box,e.first_offset_num,e.first_offset_den,e.last_offset_num,e.last_offset_den,
        s.original_name,s.source_relative_path,s.sha256 source_sha256,
        s.recording_start_utc FROM video_events e JOIN video_sources s ON s.id=e.source_id
        WHERE e.mission_id=? AND e.run_id=?''',(mission,run_id))}
    offset=0
    while True:
        page=video_store.events(None,mission,run_id,limit=100,offset=offset,prepared=True,connection=conn)
        if not page:break
        for event in page:
            original=dict(raw[event['source_event_id']]);row=dict(original);row.update(event)
            row['original_labels']=original['labels'] if isinstance(original['labels'],list) else json.loads(original['labels'])
            row.setdefault('original_first_offset_seconds',original['first_offset_num']/original['first_offset_den'])
            row.setdefault('original_last_offset_seconds',original['last_offset_num']/original['last_offset_den'])
            if event.get('merged_event_ids'):
                row['original_event_ids']=event['merged_event_ids']
                row['original_track_ids']=event.get('original_track_ids',[])
            else:
                row['original_event_ids']=[event['source_event_id']]
                row['original_track_ids']=[raw[event['source_event_id']]['track_id']]
            yield row
        offset+=len(page)


def _values(conn, row, clips):
    row = dict(row)
    payload = row.get('payload')
    if isinstance(payload,str):payload=json.loads(payload) if payload else None
    timing_payload = row.get('timing_payload')
    if isinstance(timing_payload,str):timing_payload=json.loads(timing_payload) if timing_payload else None
    class_payload = row.get('class_payload')
    if isinstance(class_payload,str):class_payload=json.loads(class_payload) if class_payload else None
    original_labels = row.get('original_labels',row['labels'])
    if isinstance(original_labels,str):original_labels=json.loads(original_labels)
    if class_payload and class_payload.get('label'):
        row['labels'] = [class_payload['label']]
    if isinstance(row['labels'],str):row['labels']=json.loads(row['labels'])
    action = row.get('action',row.get('review_action'))
    status = {'accept': 'accepted', 'reject': 'rejected', 'reopen': 'pending',
        'correct_class': 'corrected', 'correct_timing': 'corrected',
        'mark_duplicate': 'duplicate','merge_tracks':'corrected','split_track':'corrected',
        'clear_identity_correction':'corrected'}.get(action, 'pending')
    first_offset = row['first_offset_num'] / row['first_offset_den']
    last_offset = row['last_offset_num'] / row['last_offset_den']
    original_first = row.get('original_first_offset_seconds', first_offset)
    original_last = row.get('original_last_offset_seconds', last_offset)
    return [row['mission_id'], row['run_id'], row['source_id'], row['source_sha256'],
        row['original_name'], row['source_relative_path'], row['id'], row['track_id'],
        row['first_pts'], row['first_time_base_num'], row['first_time_base_den'],
        row['last_pts'], row['last_time_base_num'], row['last_time_base_den'],
        first_offset, last_offset, row['first_utc'], row['last_utc'],
        json.dumps(row['labels'], ensure_ascii=False),
        json.dumps(original_labels, ensure_ascii=False),
        json.dumps(timing_payload, ensure_ascii=False) if timing_payload is not None else '',
        json.dumps(class_payload, ensure_ascii=False) if class_payload is not None else '',
        original_first, original_last,
        row['observation_count'],
        bool(row['association_uncertain']), json.dumps(row['uncertainty_reasons'] if isinstance(row['uncertainty_reasons'],list)
            else json.loads(row['uncertainty_reasons']), ensure_ascii=False),
        row.get('review_revision',row.get('revision')) or 0, action or '', status,
        row.get('review_operator',row.get('operator')) or '', row.get('reviewed_at') or '',
        json.dumps(payload, ensure_ascii=False) if payload is not None else '',
        json.dumps(row['representative_box'],ensure_ascii=False) if isinstance(row['representative_box'],list) else row['representative_box'] or '',
        json.dumps(clips, ensure_ascii=False),json.dumps(row.get('original_event_ids',[]),ensure_ascii=False),
        json.dumps(row.get('identity_correction'),ensure_ascii=False) if row.get('identity_correction') else '']


def _clips_for_event(conn, event_id):
    rows = conn.execute('''SELECT id,output_path,sha256,status,pre_seconds,post_seconds,frame_count
        FROM video_clip_exports WHERE event_id=? ORDER BY created,id''', (event_id,)).fetchall()
    return [dict(row) for row in rows]


def _clips_for_events(conn,event_ids):
    result=[]
    for event_id in event_ids:
        result.extend(_clips_for_event(conn,event_id))
    return result


def _column_name(number):
    name = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(column, row_number, value):
    reference = f'{column}{row_number}'
    value = _cell(value)
    if len(value) > 32767:
        raise ValueError(f'Cell {reference} exceeds the XLSX 32,767-character limit')
    if any((ord(char) < 32 and char not in '\t\n\r') or ord(char) in (0xfffe, 0xffff) for char in value):
        raise ValueError(f'Cell {reference} contains a character Excel cannot store')
    return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'


def _write_xlsx(conn, mission, run_id, temp_path, row_count):
    max_data_rows = 1_048_575
    sheet_count = max(1, (row_count + max_data_rows - 1) // max_data_rows)
    sheet_names = ['Events' if sheet_count == 1 else f'Events {i + 1}' for i in range(sheet_count)]
    with zipfile.ZipFile(temp_path, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'''+''.join(
            f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(sheet_count))+'</Types>')
        archive.writestr('_rels/.rels', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''')
        archive.writestr('xl/workbook.xml', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'''+''.join(
            f'<sheet name="{name}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, name in enumerate(sheet_names))+'</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'''+''.join(
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>' for i in range(sheet_count))+'</Relationships>')
        rows = _rows(conn, mission, run_id)
        for sheet_index in range(sheet_count):
            data_rows = min(max_data_rows, max(0, row_count - sheet_index * max_data_rows))
            with archive.open(f'xl/worksheets/sheet{sheet_index + 1}.xml', 'w') as sheet:
                sheet.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>')
                sheet.write(('<row r="1">'+''.join(_cell_xml(_column_name(i + 1), 1, name)
                    for i, name in enumerate(HEADERS))+'</row>').encode('utf-8'))
                for row_number in range(2, data_rows + 2):
                    row = next(rows)
                    values = _values(conn, row, _clips_for_events(conn,row.get('merged_event_ids',[row['source_event_id']])))
                    content = ''.join(_cell_xml(_column_name(i + 1), row_number, value)
                        for i, value in enumerate(values))
                    sheet.write(f'<row r="{row_number}">{content}</row>'.encode('utf-8'))
                sheet.write(b'</sheetData></worksheet>')


def _write_csv(conn, mission, run_id, temp_path):
    row_count = 0
    with Path(temp_path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        rows = _rows(conn, mission, run_id)
        for row in rows:
            writer.writerow([_cell(value) for value in _values(conn, row,
                _clips_for_events(conn,row.get('merged_event_ids',[row['source_event_id']])))])
            row_count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return row_count


def _row_count(conn, mission, run_id):
    return sum(1 for _ in _rows(conn,mission,run_id))


def _export_detail(library, mission, export_id):
    _prepare(library, mission)
    with library.connect(mission) as conn:
        row = conn.execute('SELECT * FROM video_event_table_exports WHERE id=? AND mission_id=?',
            (export_id, mission)).fetchone()
    if not row:
        raise ValueError('Video event-table export not found in this mission')
    item = dict(row)
    item['download_url'] = f'/api/video/event-exports/download?mission={mission}&export={export_id}'
    return item


def _export_format(library, mission, run_id, operator, format_name):
    settings = library.settings(mission)
    if format_name not in settings['export_formats']:
        raise ValueError(f'{format_name.upper()} is not configured for this mission')
    _prepare(library, mission)
    with library.connect(mission) as conn:
        if not conn.execute('SELECT 1 FROM video_runs WHERE id=? AND mission_id=?',
                (run_id, mission)).fetchone():
            raise ValueError('Video run not found in this mission')
    output = _directory(library, mission)
    export_id = str(uuid.uuid4())
    extension = 'xlsx' if format_name == 'xlsx' else 'csv'
    target = output / f'traffic-video-events-{mission[:8]}-{run_id[:8]}-{export_id[:8]}.{extension}'
    temp_name = str(output / f'.traffic-video-events-{export_id}.{extension}.partial')
    created = now()
    with library.connect(mission) as conn:
        conn.execute('''INSERT INTO video_event_table_exports(id,mission_id,run_id,operator,format,
            output_root,output_path,temp_path,row_count,status,created) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (export_id, mission, run_id, operator, format_name, str(output), str(target), temp_name, 0, 'writing', created))
    row_count = 0
    try:
        with library.connect(mission) as conn:
            conn.execute('BEGIN')
            count = _row_count(conn, mission, run_id)
            if format_name == 'csv':
                row_count = _write_csv(conn, mission, run_id, temp_name)
            else:
                _write_xlsx(conn, mission, run_id, temp_name, count)
                with Path(temp_name).open('r+b') as stream:
                    os.fsync(stream.fileno())
                row_count = count
            if row_count != count:
                raise ValueError(f'Event snapshot changed during export: expected {count}, wrote {row_count}')
            snapshot = _digest(temp_name)
        with library.connect(mission) as conn:
            conn.execute('''UPDATE video_event_table_exports SET snapshot_sha256=?,row_count=?,status='publishing'
                WHERE id=? AND mission_id=?''', (snapshot, row_count, export_id, mission))
        if target.exists():
            raise FileExistsError('Unique event-table output path already exists')
        os.replace(temp_name, target)
        try:
            directory_fd = os.open(output, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        with library.connect(mission) as conn:
            conn.execute('''UPDATE video_event_table_exports SET status='completed',finished=?
                WHERE id=? AND mission_id=?''', (now(), export_id, mission))
        return _export_detail(library, mission, export_id)
    except Exception as exc:
        try:
            with library.connect(mission) as conn:
                record = conn.execute('''SELECT status,snapshot_sha256 FROM video_event_table_exports
                    WHERE id=? AND mission_id=?''', (export_id, mission)).fetchone()
            if (record and record['status'] == 'publishing' and target.is_file()
                    and record['snapshot_sha256'] == _digest(target)):
                with library.connect(mission) as conn:
                    conn.execute('''UPDATE video_event_table_exports SET status='completed',finished=?
                        WHERE id=? AND mission_id=?''', (now(), export_id, mission))
                return _export_detail(library, mission, export_id)
        except Exception:
            # Leave publishing rows for startup reconciliation if SQLite is unavailable.
            raise ValueError(f'Event-table export {export_id} needs startup reconciliation: {exc}') from exc
        try:
            Path(temp_name).unlink(missing_ok=True)
        except OSError:
            pass
        with library.connect(mission) as conn:
            conn.execute('''UPDATE video_event_table_exports SET status='failed',row_count=?,error=?,finished=?
                WHERE id=? AND mission_id=?''', (row_count, f'{type(exc).__name__}: {exc}'[:1000], now(), export_id, mission))
        raise ValueError(f'Event-table export {export_id} failed: {exc}') from exc


def export_run(library, mission, run_id, operator):
    operator = operator.strip() if isinstance(operator, str) else ''
    if not 1 <= len(operator) <= 120:
        raise ValueError('Operator is required for event-table export')
    _prepare(library, mission)
    with library.connect(mission) as conn:
        if not conn.execute('SELECT 1 FROM video_runs WHERE id=? AND mission_id=?',
                (run_id, mission)).fetchone():
            raise ValueError('Video run not found in this mission')
    formats = library.settings(mission)['export_formats']
    results, errors = [], []
    for format_name in formats:
        try:
            results.append(_export_format(library, mission, run_id, operator, format_name))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        return {'run_id': run_id, 'files': exports(library, mission, run_id),
            'status': 'partial' if results else 'failed', 'errors': errors}
    return {'run_id': run_id, 'files': results, 'status': 'completed', 'errors': []}


def exports(library, mission, run_id=None):
    _prepare(library, mission)
    with library.connect(mission) as conn:
        if run_id is None:
            rows = conn.execute('''SELECT id FROM video_event_table_exports WHERE mission_id=?
                ORDER BY created DESC,id''', (mission,)).fetchall()
        else:
            rows = conn.execute('''SELECT id FROM video_event_table_exports WHERE mission_id=? AND run_id=?
                ORDER BY created DESC,id''', (mission, run_id)).fetchall()
    return [_export_detail(library, mission, row['id']) for row in rows]


def export_file(library, mission, export_id):
    item = _export_detail(library, mission, export_id)
    if item['status'] != 'completed' or not item['snapshot_sha256']:
        raise ValueError('Event-table export is not complete')
    if not item['output_root']:
        raise ValueError('Event-table export predates saved output-folder provenance')
    root = Path(item['output_root']).resolve(strict=True)
    workspace = library.root.resolve()
    if root.is_relative_to(workspace) or workspace.is_relative_to(root):
        raise ValueError('Event-table export root is inside application-managed storage')
    path = Path(item['output_path']).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Event-table export file is missing or outside the configured output folder')
    if _digest(path) != item['snapshot_sha256']:
        raise ValueError('Event-table export failed its SHA-256 check')
    return path


def reconcile_exports(library):
    repaired = 0
    for mission_path in library.root.glob('*/mission.sqlite3'):
        mission = mission_path.parent.name
        _prepare(library, mission)
        with library.connect(mission) as conn:
            rows = conn.execute('''SELECT * FROM video_event_table_exports WHERE status IN ('writing','publishing')
                AND mission_id=?''', (mission,)).fetchall()
        for row in rows:
            target, temp = Path(row['output_path']), Path(row['temp_path'])
            status, error = 'failed', 'Export process stopped before completion'
            try:
                if row['status'] == 'publishing' and row['snapshot_sha256']:
                    if target.is_file() and _digest(target) == row['snapshot_sha256']:
                        status, error = 'completed', None
                    elif temp.is_file() and _digest(temp) == row['snapshot_sha256'] and not target.exists():
                        os.replace(temp, target)
                        status, error = 'completed', None
                    else:
                        status = 'partial' if target.exists() else 'failed'
            except OSError as exc:
                status, error = ('partial' if target.exists() else 'failed'), str(exc)[:1000]
            try:
                temp.unlink(missing_ok=True)
            except OSError as exc:
                status, error = 'partial', f'{error}; temporary file cleanup failed: {exc}'[:1000]
            with library.connect(mission) as conn:
                conn.execute('UPDATE video_event_table_exports SET status=?,error=?,finished=? WHERE id=?',
                    (status, error, now(), row['id']))
            repaired += 1
    return repaired
