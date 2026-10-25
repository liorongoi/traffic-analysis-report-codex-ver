"""Explicit bulk sign-off records for a fixed still-image result snapshot.

Bulk sign-off never writes individual image reviews and is never ground truth.
"""
import json
import uuid

from .store import now


def _prepare(library,mission):
    with library.connect(mission) as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS bulk_approvals(
                id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
                preview_id TEXT NOT NULL UNIQUE,inference_run TEXT NOT NULL,snapshot_rowid INTEGER NOT NULL,
                image_count INTEGER NOT NULL,operator TEXT NOT NULL,status TEXT NOT NULL,
                exports_json TEXT NOT NULL DEFAULT '[]',created TEXT NOT NULL,
                finished TEXT,error TEXT);
            CREATE TABLE IF NOT EXISTS bulk_approval_previews(
                id TEXT PRIMARY KEY,mission_id TEXT NOT NULL REFERENCES mission(id),
                inference_run TEXT NOT NULL,snapshot_rowid INTEGER NOT NULL,
                image_count INTEGER NOT NULL,created TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_bulk_preview_update
                BEFORE UPDATE ON bulk_approval_previews BEGIN SELECT RAISE(ABORT,'bulk preview scope is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_bulk_preview_delete
                BEFORE DELETE ON bulk_approval_previews BEGIN SELECT RAISE(ABORT,'bulk previews are append-only'); END;
            CREATE INDEX IF NOT EXISTS bulk_approvals_created
                ON bulk_approvals(created DESC,id);
            CREATE TRIGGER IF NOT EXISTS immutable_bulk_approval_scope
                BEFORE UPDATE OF id,mission_id,preview_id,inference_run,snapshot_rowid,image_count,operator,created
                ON bulk_approvals BEGIN SELECT RAISE(ABORT,'bulk approval scope is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_bulk_approval_delete
                BEFORE DELETE ON bulk_approvals BEGIN SELECT RAISE(ABORT,'bulk approvals are append-only'); END;
        ''')


def _scope(library,mission,inference_run=None):
    settings=library.settings(mission)
    if settings['input_type']!='images' or settings['analysis_mode']!='general_traffic':
        raise ValueError('Bulk sign-off is available only for still-image General Traffic missions')
    if not settings['output_root']:
        raise ValueError('Set an output directory before bulk sign-off and export')
    from . import inference_store
    jobs=inference_store.runs(library,mission)
    selected=inference_run or inference_store.latest_run_id(library,mission)
    run=next((item for item in jobs if item['id']==selected),None)
    if not run:raise ValueError('Choose a saved inference run before bulk sign-off')
    if run['status']!='completed' or run['done']!=run['total']:
        raise ValueError('Bulk sign-off requires a completed inference run')
    roi=inference_store.roi_review_state(library,mission,selected)
    if roi['review_required']:
        raise ValueError('Review the changed camera regions before bulk sign-off')
    with library.connect(mission) as conn:
        row=conn.execute('SELECT COALESCE(MAX(rowid),0) snapshot_rowid FROM images').fetchone()
        snapshot_rowid=row['snapshot_rowid']
        count=conn.execute('SELECT COUNT(*) FROM images WHERE rowid<=?',(snapshot_rowid,)).fetchone()[0]
    if count<1:raise ValueError('There are no image results in this snapshot')
    return {'mission_id':mission,'inference_run':selected,'run_status':run['status'],
        'snapshot_rowid':snapshot_rowid,'image_count':count,
        'approval_type':'bulk_without_individual_review','independent_ground_truth':False,
        'includes_later_imports':False}


def preview(library,mission,inference_run=None):
    scope=_scope(library,mission,inference_run)
    _prepare(library,mission)
    preview_id=str(uuid.uuid4())
    with library.connect(mission) as conn:
        conn.execute('INSERT INTO bulk_approval_previews VALUES (?,?,?,?,?,?)',
            (preview_id,mission,scope['inference_run'],scope['snapshot_rowid'],scope['image_count'],now()))
    return {**scope,'preview_id':preview_id}


def approve_and_export(library,mission,*,preview_id,operator):
    if not isinstance(operator,str) or not 1<=len(operator.strip())<=120:
        raise ValueError('Operator name required for bulk sign-off')
    _prepare(library,mission)
    with library.connect(mission) as conn:
        scope=conn.execute('SELECT * FROM bulk_approval_previews WHERE mission_id=? AND id=?',
            (mission,preview_id)).fetchone()
    if not scope:raise ValueError('Bulk sign-off preview expired or does not belong to this mission')
    _scope(library,mission,scope['inference_run'])
    with library.connect(mission) as conn:
        actual=conn.execute('SELECT COUNT(*) FROM images WHERE rowid<=?',(scope['snapshot_rowid'],)).fetchone()[0]
    if actual!=scope['image_count']:
        raise ValueError('Image snapshot no longer matches the preview; refresh the scope before confirming')
    identifier=str(uuid.uuid4());created=now()
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('''INSERT INTO bulk_approvals(id,mission_id,preview_id,inference_run,snapshot_rowid,
            image_count,operator,status,created) VALUES (?,?,?,?,?,?,?,?,?)''',
            (identifier,mission,preview_id,scope['inference_run'],scope['snapshot_rowid'],
                scope['image_count'],operator.strip(),'exporting',created))
    try:
        result=library.export_configured(mission,scope['inference_run'],scope['snapshot_rowid'],identifier)
        with library.connect(mission) as conn:
            conn.execute('UPDATE bulk_approvals SET status=?,exports_json=?,finished=? WHERE id=?',
                ('completed',json.dumps(result['files'],ensure_ascii=False),now(),identifier))
    except Exception as exc:
        with library.connect(mission) as conn:
            conn.execute('UPDATE bulk_approvals SET status=?,error=?,finished=? WHERE id=?',
                ('failed',str(exc)[:2000],now(),identifier))
        raise
    return {'id':identifier,'mission_id':mission,'inference_run':scope['inference_run'],
        'preview_id':preview_id,'snapshot_rowid':scope['snapshot_rowid'],
        'image_count':scope['image_count'],'operator':operator.strip(),
        'status':'completed','approval_type':'bulk_without_individual_review',
        'individual_reviews_created':0,'independent_ground_truth':False,'files':result['files']}


def history(library,mission):
    _prepare(library,mission)
    with library.connect(mission) as conn:
        pending=conn.execute("SELECT id FROM bulk_approvals WHERE mission_id=? AND status='exporting'",
            (mission,)).fetchall()
        for approval in pending:
            exports=conn.execute('SELECT 1 FROM exports WHERE bulk_approval_id=? LIMIT 1',
                (approval['id'],)).fetchone()
            states=[row['status'] for row in conn.execute(
                'SELECT status FROM exports WHERE bulk_approval_id=?',(approval['id'],))]
            if not exports:
                conn.execute("UPDATE bulk_approvals SET status='failed',error=?,finished=? WHERE id=?",
                    ('Export stopped before an output record was created',now(),approval['id']))
            elif all(status=='completed' for status in states):
                details=[dict(row) for row in conn.execute('SELECT * FROM exports WHERE bulk_approval_id=?',(approval['id'],))]
                conn.execute("UPDATE bulk_approvals SET status='completed',exports_json=?,finished=? WHERE id=?",
                    (json.dumps(details,ensure_ascii=False),now(),approval['id']))
            elif any(status in ('failed','partial') for status in states):
                details=[dict(row) for row in conn.execute('SELECT * FROM exports WHERE bulk_approval_id=?',(approval['id'],))]
                conn.execute("UPDATE bulk_approvals SET status='failed',exports_json=?,error=?,finished=? WHERE id=?",
                    (json.dumps(details,ensure_ascii=False),'One or more snapshot exports stopped before completion',now(),approval['id']))
        rows=conn.execute('SELECT * FROM bulk_approvals WHERE mission_id=? ORDER BY created DESC,id',
            (mission,)).fetchall()
    return [{**dict(row),'exports':json.loads(row['exports_json']),
        'individual_reviews_created':0,'independent_ground_truth':False} for row in rows]
