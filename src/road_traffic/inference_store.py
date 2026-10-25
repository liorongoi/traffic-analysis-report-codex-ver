"""Durable inference ledger. No ML frameworks are imported by the web service."""
import json
import hashlib
import uuid
from .store import now


def schema(conn):
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS inference_runs(
      id TEXT PRIMARY KEY,status TEXT NOT NULL,stage TEXT NOT NULL,total INTEGER NOT NULL,
      done INTEGER NOT NULL DEFAULT 0,config TEXT NOT NULL,created TEXT NOT NULL,finished TEXT,
      stop_requested INTEGER NOT NULL DEFAULT 0,error TEXT,worker_pid INTEGER);
    CREATE TABLE IF NOT EXISTS inference_inputs(
      inference_id TEXT REFERENCES inference_runs(id),image_id TEXT REFERENCES images(id),sha256 TEXT NOT NULL,
      PRIMARY KEY(inference_id,image_id));
    CREATE TABLE IF NOT EXISTS inference_input_views(
      inference_id TEXT NOT NULL,image_id TEXT NOT NULL,view_id TEXT REFERENCES views(id),view_revision INTEGER NOT NULL,
      PRIMARY KEY(inference_id,image_id),
      FOREIGN KEY(inference_id,image_id) REFERENCES inference_inputs(inference_id,image_id));
    CREATE TABLE IF NOT EXISTS predictions(
      inference_id TEXT REFERENCES inference_runs(id),image_id TEXT REFERENCES images(id),
      annotations TEXT NOT NULL,details TEXT NOT NULL,elapsed REAL NOT NULL,error TEXT,
      PRIMARY KEY(inference_id,image_id),
      FOREIGN KEY(inference_id,image_id) REFERENCES inference_inputs(inference_id,image_id));
    CREATE TABLE IF NOT EXISTS review_inference(
      review_id INTEGER PRIMARY KEY REFERENCES reviews(id),inference_id TEXT REFERENCES inference_runs(id),
      changes TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS roi_rerun_acknowledgments(
      id INTEGER PRIMARY KEY,source_run TEXT NOT NULL REFERENCES inference_runs(id),
      rerun_run TEXT NOT NULL UNIQUE REFERENCES inference_runs(id),operator TEXT NOT NULL,
      drift_sha256 TEXT NOT NULL,changed_views TEXT NOT NULL,created TEXT NOT NULL,
      UNIQUE(source_run,drift_sha256));
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_inference ON inference_runs((1)) WHERE status IN ('queued','running');
    CREATE TRIGGER IF NOT EXISTS immutable_run_config BEFORE UPDATE OF id,config,total,created ON inference_runs BEGIN SELECT RAISE(ABORT,'immutable run configuration'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_predictions_u BEFORE UPDATE ON predictions BEGIN SELECT RAISE(ABORT,'immutable prediction'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_predictions_d BEFORE DELETE ON predictions BEGIN SELECT RAISE(ABORT,'immutable prediction'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_inputs_u BEFORE UPDATE ON inference_inputs BEGIN SELECT RAISE(ABORT,'immutable inputs'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_inputs_d BEFORE DELETE ON inference_inputs BEGIN SELECT RAISE(ABORT,'immutable inputs'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_input_views_u BEFORE UPDATE ON inference_input_views BEGIN SELECT RAISE(ABORT,'immutable input view snapshot'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_input_views_d BEFORE DELETE ON inference_input_views BEGIN SELECT RAISE(ABORT,'immutable input view snapshot'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_review_link_u BEFORE UPDATE ON review_inference BEGIN SELECT RAISE(ABORT,'immutable review lineage'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_review_link_d BEFORE DELETE ON review_inference BEGIN SELECT RAISE(ABORT,'immutable review lineage'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_roi_ack_u BEFORE UPDATE ON roi_rerun_acknowledgments BEGIN SELECT RAISE(ABORT,'immutable ROI acknowledgment'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_roi_ack_d BEFORE DELETE ON roi_rerun_acknowledgments BEGIN SELECT RAISE(ABORT,'immutable ROI acknowledgment'); END;
    ''')


def prepare(library,mission):
    with library.connect(mission) as conn:schema(conn)


def _mission_snapshot(conn,mission):
    row=conn.execute('''SELECT m.id,m.name,m.country,m.zone,m.created,
            s.input_type,s.analysis_mode,s.location,s.source_root,s.include_subdirectories,
            s.processing_mode,s.output_root,s.export_formats
        FROM mission m JOIN mission_settings s ON s.mission_id=m.id WHERE m.id=?''',(mission,)).fetchone()
    if not row:raise ValueError('Mission not found')
    snapshot=dict(row)
    snapshot['include_subdirectories']=bool(snapshot['include_subdirectories'])
    snapshot['export_formats']=json.loads(snapshot['export_formats'])
    return snapshot


def _source_run_snapshots(conn,image_ids):
    image_ids=sorted(set(image_ids))
    if not image_ids:return []
    placeholders=','.join('?' for _ in image_ids)
    run_ids=[row['run_id'] for row in conn.execute(
        f'SELECT DISTINCT run_id FROM images WHERE id IN ({placeholders}) AND run_id IS NOT NULL',image_ids)]
    if not run_ids:return []
    placeholders=','.join('?' for _ in run_ids)
    return [dict(row) for row in conn.execute(
        f'SELECT id,preset,source_root,status,created,error FROM runs WHERE id IN ({placeholders}) ORDER BY created,id',run_ids)]


def create_run(library,mission,config,source_run=None):
    prepare(library,mission)
    run=str(uuid.uuid4())
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        mission_snapshot=_mission_snapshot(conn,mission)
        windows=[{**dict(row),'weekdays':json.loads(row['weekdays'])} for row in conn.execute(
            'SELECT * FROM scan_windows WHERE mission_id=? ORDER BY created,id',(mission,))]
        if source_run and not conn.execute('SELECT 1 FROM runs WHERE id=?',(source_run,)).fetchone():
            raise ValueError('Source batch not found in this mission')
        rows=conn.execute('SELECT id,run_id,sha256,capture_utc FROM images WHERE (? IS NULL OR run_id=?) ORDER BY capture_utc,name',
            (source_run,source_run)).fetchall()
        if not rows:raise ValueError('Import images before running inference')
        from .schedules import compile_windows,eligible_compiled
        compiled=compile_windows(windows)
        eligible_rows=[row for row in rows if eligible_compiled(row['capture_utc'],mission_snapshot['zone'],compiled)]
        if not eligible_rows:
            raise ValueError('No imported images fall within the configured local-time scan windows')
        if conn.execute("SELECT 1 FROM inference_runs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('This mission already has an active inference run')
        view_rows={row['image_id']:(row['view_id'],row['revision']) for row in conn.execute('''SELECT image_id,view_id,revision
            FROM image_view_history WHERE mission_id=? AND revision=(SELECT MAX(h.revision) FROM image_view_history h WHERE h.image_id=image_view_history.image_id)''',(mission,))}
        regions={}
        for item in conn.execute('''SELECT h.region_key,h.view_id,h.revision,h.kind,h.polygon FROM view_region_history h
            WHERE h.mission_id=? AND h.action='upsert'
            AND h.revision=(SELECT MAX(x.revision) FROM view_region_history x WHERE x.region_key=h.region_key)''',(mission,)):
            regions.setdefault(item['view_id'],[]).append({'id':item['region_key'],'revision':item['revision'],
                'kind':item['kind'],'polygon':json.loads(item['polygon'])})
        recorded_config={**config,'run_config_schema_version':1,'mission_snapshot':mission_snapshot,
            'mission_snapshot_provenance':'captured_at_run_start','source_run_id':source_run,
            'source_runs_snapshot':_source_run_snapshots(conn,[row['id'] for row in eligible_rows]),
            'scan_schedule_snapshot':{'timezone':mission_snapshot['zone'],'windows':windows,
            'excluded_image_count':len(rows)-len(eligible_rows)},'view_regions_snapshot':regions}
        conn.execute('INSERT INTO inference_runs(id,status,stage,total,config,created) VALUES (?,?,?,?,?,?)',
            (run,'queued','queued',len(eligible_rows),json.dumps(recorded_config),now()))
        conn.executemany('INSERT INTO inference_inputs VALUES (?,?,?)',[(run,r['id'],r['sha256']) for r in eligible_rows])
        conn.executemany('INSERT INTO inference_input_views VALUES (?,?,?,?)',[
            (run,row['id'],*(view_rows.get(row['id']) or (None,0))) for row in eligible_rows])
    return run


def runs(library,mission):
    prepare(library,mission)
    with library.connect(mission) as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM inference_runs ORDER BY created DESC')]


def latest_run_id(library,mission):
    prepare(library,mission)
    with library.connect(mission) as conn:
        row=conn.execute('SELECT id FROM inference_runs ORDER BY created DESC,id DESC LIMIT 1').fetchone()
    return row['id'] if row else None


def search_scope_snapshot(library,mission,run):
    """Return the immutable schedule, view assignments and ROI used by a saved run."""
    if not run:return None
    prepare(library,mission)
    with library.connect(mission) as conn:
        row=conn.execute('SELECT config,created,total FROM inference_runs WHERE id=?',(run,)).fetchone()
        if not row:raise ValueError('Inference run is not part of this mission')
        config=json.loads(row['config'])
        view_rows=conn.execute('''SELECT view_id,view_revision,COUNT(*) image_count
            FROM inference_input_views WHERE inference_id=? GROUP BY view_id,view_revision
            ORDER BY view_id,view_revision''',(run,)).fetchall()
        grouped={}
        for item in view_rows:
            key=item['view_id']
            grouped.setdefault(key,{'view_id':key,'image_count':0,'assignment_revisions':[]})
            grouped[key]['image_count']+=item['image_count']
            grouped[key]['assignment_revisions'].append(item['view_revision'])
        view_assignments=[]
        for item in grouped.values():
            name=None
            if item['view_id']:
                event=conn.execute('''SELECT action,details FROM view_events WHERE view_id=?
                    AND action IN ('created','renamed') AND created<=?
                    ORDER BY created DESC,id DESC LIMIT 1''',(item['view_id'],row['created'])).fetchone()
                if event:
                    details=json.loads(event['details'])
                    name=details.get('name') if event['action']=='created' else details.get('after')
            view_assignments.append({**item,'view_name':name})
    used={item['view_id'] for item in view_assignments if item['view_id']}
    all_regions=config.get('view_regions_snapshot',{})
    regions={view_id:all_regions.get(view_id,[]) for view_id in sorted(used)}
    return {
        'run_id':run,
        'run_created':row['created'],
        'input_image_count':row['total'],
        'schedule':config.get('scan_schedule_snapshot'),
        'view_assignments':view_assignments,
        'regions_by_view':regions,
    }


def roi_drift(library,mission,run):
    """List views whose effective current ROI differs from this run's snapshot."""
    prepare(library,mission)
    with library.connect(mission) as conn:
        job=conn.execute('SELECT config FROM inference_runs WHERE id=?',(run,)).fetchone()
        if not job:raise ValueError('Inference run not found')
        used={row['view_id'] for row in conn.execute(
            'SELECT DISTINCT view_id FROM inference_input_views WHERE inference_id=? AND view_id IS NOT NULL',(run,))}
        names={row['id']:row['name'] for row in conn.execute('SELECT id,name FROM views WHERE mission_id=?',(mission,))}
        current={}
        for row in conn.execute('''SELECT h.region_key,h.view_id,h.kind,h.polygon FROM view_region_history h
            WHERE h.mission_id=? AND h.action='upsert'
            AND h.revision=(SELECT MAX(x.revision) FROM view_region_history x WHERE x.region_key=h.region_key)''',(mission,)):
            current.setdefault(row['view_id'],{})[row['region_key']]=(row['kind'],json.dumps(json.loads(row['polygon']),sort_keys=True,separators=(',',':')))
    snapshot=json.loads(job['config']).get('view_regions_snapshot',{})
    changed=[]
    for view_id in sorted(used):
        saved={item['id']:(item['kind'],json.dumps(item['polygon'],sort_keys=True,separators=(',',':')))
            for item in snapshot.get(view_id,[])}
        if saved!=current.get(view_id,{}):
            changed.append({'view_id':view_id,'view_name':names.get(view_id,view_id)})
    return changed


def _current_regions(conn,mission,view_ids):
    snapshot={view_id:[] for view_id in view_ids if view_id}
    if not snapshot:return snapshot
    for row in conn.execute('''SELECT h.region_key,h.view_id,h.revision,h.kind,h.polygon FROM view_region_history h
        WHERE h.mission_id=? AND h.action='upsert'
        AND h.revision=(SELECT MAX(x.revision) FROM view_region_history x WHERE x.region_key=h.region_key)
        ORDER BY h.region_key,h.created''',(mission,)):
        if row['view_id'] in snapshot:
            snapshot[row['view_id']].append({'id':row['region_key'],'revision':row['revision'],
                'kind':row['kind'],'polygon':json.loads(row['polygon'])})
    return snapshot


def _roi_review_state_conn(conn,mission,run):
    job=conn.execute('SELECT config FROM inference_runs WHERE id=?',(run,)).fetchone()
    if not job:raise ValueError('Inference run not found')
    used={row['view_id'] for row in conn.execute(
        'SELECT DISTINCT view_id FROM inference_input_views WHERE inference_id=? AND view_id IS NOT NULL',(run,))}
    names={row['id']:row['name'] for row in conn.execute('SELECT id,name FROM views WHERE mission_id=?',(mission,))}
    current=_current_regions(conn,mission,used);saved=json.loads(job['config']).get('view_regions_snapshot',{})
    changed=[]
    for view_id in sorted(used):
        before={item['id']:(item['kind'],json.dumps(item['polygon'],sort_keys=True,separators=(',',':')))
            for item in saved.get(view_id,[])}
        after={item['id']:(item['kind'],json.dumps(item['polygon'],sort_keys=True,separators=(',',':')))
            for item in current.get(view_id,[])}
        if before!=after:changed.append({'view_id':view_id,'view_name':names.get(view_id,view_id)})
    payload={'changed_views':[item['view_id'] for item in changed],
        'current_regions':{key:current.get(key,[]) for key in sorted(item['view_id'] for item in changed)}}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    ack=conn.execute('''SELECT a.*,r.status rerun_status FROM roi_rerun_acknowledgments a
        JOIN inference_runs r ON r.id=a.rerun_run WHERE a.source_run=? AND a.drift_sha256=?''',(run,digest)).fetchone()
    return {'changed_views':changed,'stale':bool(changed),'review_required':bool(changed and not ack),
        'acknowledgment':dict(ack) if ack else None,'drift_sha256':digest}


def roi_review_state(library,mission,run):
    """Return current drift and any acknowledgment for this exact effective ROI state."""
    prepare(library,mission)
    with library.connect(mission) as conn:return _roi_review_state_conn(conn,mission,run)


def create_roi_rerun(library,mission,source_run,operator):
    """Create a same-input rerun with current ROIs and durable operator acknowledgment."""
    operator=operator.strip() if isinstance(operator,str) else ''
    if not 1<=len(operator)<=120:raise ValueError('Operator is required for ROI acknowledgment')
    prepare(library,mission)
    run=str(uuid.uuid4());created=now()
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        state=_roi_review_state_conn(conn,mission,source_run)
        if not state['stale']:raise ValueError('This run has no ROI changes to review')
        if state['acknowledgment']:raise ValueError('This ROI change already has an acknowledged rerun')
        changed=state['changed_views']
        source=conn.execute('SELECT status,config FROM inference_runs WHERE id=?',(source_run,)).fetchone()
        if not source:raise ValueError('Inference run not found')
        if source['status'] in ('queued','running'):raise ValueError('Wait for the source inference run to finish before rerunning')
        inputs=conn.execute('SELECT image_id,sha256 FROM inference_inputs WHERE inference_id=? ORDER BY image_id',(source_run,)).fetchall()
        if not inputs:raise ValueError('Source inference run has no saved inputs')
        if conn.execute("SELECT 1 FROM inference_runs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('Wait for the active inference run to finish before rerunning')
        used={row['view_id'] for row in conn.execute(
            'SELECT DISTINCT view_id FROM inference_input_views WHERE inference_id=? AND view_id IS NOT NULL',(source_run,))}
        config=json.loads(source['config']);config['rerun_of']=source_run
        if 'mission_snapshot' not in config:
            config['run_config_schema_version']=1
            config['mission_snapshot']=_mission_snapshot(conn,mission)
            config['mission_snapshot_provenance']='captured_at_roi_rerun_for_legacy_source'
        config.setdefault('source_runs_snapshot',_source_run_snapshots(conn,[row['image_id'] for row in inputs]))
        config['view_regions_snapshot']=_current_regions(conn,mission,used)
        conn.execute('INSERT INTO inference_runs(id,status,stage,total,config,created) VALUES (?,?,?,?,?,?)',
            (run,'queued','queued',len(inputs),json.dumps(config,ensure_ascii=False),created))
        conn.executemany('INSERT INTO inference_inputs VALUES (?,?,?)',[(run,row['image_id'],row['sha256']) for row in inputs])
        conn.execute('''INSERT INTO inference_input_views(inference_id,image_id,view_id,view_revision)
            SELECT ?,image_id,view_id,view_revision FROM inference_input_views WHERE inference_id=?''',(run,source_run))
        conn.execute('''INSERT INTO roi_rerun_acknowledgments(source_run,rerun_run,operator,drift_sha256,changed_views,created)
            VALUES (?,?,?,?,?,?)''',(source_run,run,operator,state['drift_sha256'],
            json.dumps(changed,ensure_ascii=False),created))
    return {'run':run,'rerun_of':source_run,'status':'queued','total':len(inputs),'changed_views':changed,'operator':operator}


def enrich(library,mission,rows,selected=None):
    prepare(library,mission)
    with library.connect(mission) as conn:
        if selected and not conn.execute('SELECT 1 FROM inference_runs WHERE id=?',(selected,)).fetchone():
            raise ValueError('Inference run is not part of this mission')
        current_views={r['image_id']:dict(r) for r in conn.execute('''SELECT h.image_id,h.view_id,v.name view_name,
                h.revision view_revision,h.operator view_operator
            FROM image_view_history h LEFT JOIN views v ON v.id=h.view_id
            WHERE h.mission_id=? AND h.revision=(SELECT MAX(x.revision) FROM image_view_history x WHERE x.image_id=h.image_id)''',(mission,))}
        latest=selected or next((r['id'] for r in conn.execute('SELECT id FROM inference_runs ORDER BY created DESC LIMIT 1')),None)
        for row in rows:
            # A saved review stays tied to the exact predictions it corrected.
            review=conn.execute('SELECT id FROM reviews WHERE image_id=? ORDER BY revision DESC LIMIT 1',(row['id'],)).fetchone()
            link=conn.execute('SELECT inference_id FROM review_inference WHERE review_id=?',(review['id'],)).fetchone() if review else None
            prediction_run=latest
            row['review_prediction_run']=link['inference_id'] if link else None
            row['selected_prediction_run']=latest
            if review and link and not selected:prediction_run=link['inference_id']
            snapshot=conn.execute('''SELECT view_id,view_revision FROM inference_input_views
                WHERE inference_id=? AND image_id=?''',(prediction_run,row['id'])).fetchone() if prediction_run else None
            view=current_views.get(row['id'])
            if snapshot:
                view={'view_id':snapshot['view_id'],'view_revision':snapshot['view_revision'],'view_operator':None,'view_name':None}
                if snapshot['view_id']:
                    assignment=conn.execute('SELECT operator FROM image_view_history WHERE image_id=? AND revision=?',
                        (row['id'],snapshot['view_revision'])).fetchone()
                    event=conn.execute('''SELECT action,details FROM view_events WHERE view_id=?
                        AND action IN ('created','renamed')
                        AND created<=(SELECT created FROM inference_runs WHERE id=?) ORDER BY created DESC,id DESC LIMIT 1''',
                        (snapshot['view_id'],prediction_run)).fetchone()
                    if assignment:view['view_operator']=assignment['operator']
                    if event:
                        details=json.loads(event['details'])
                        view['view_name']=details.get('name') if event['action']=='created' else details.get('after')
            row['view_id']=view['view_id'] if view else None
            row['view_name']=view['view_name'] if view else None
            row['view_revision']=view['view_revision'] if view else 0
            row['view_operator']=view['view_operator'] if view else None
            pred=conn.execute('SELECT * FROM predictions WHERE inference_id=? AND image_id=?',(prediction_run,row['id'])).fetchone()
            row['prediction_run']=prediction_run
            row['predictions']=json.loads(pred['annotations']) if pred else []
            row['inference_details']=json.loads(pred['details']) if pred else None
            row['inference_error']=pred['error'] if pred else None
            included = bool(prediction_run and conn.execute(
                'SELECT 1 FROM inference_inputs WHERE inference_id=? AND image_id=?',
                (prediction_run, row['id'])).fetchone())
            row['inference_included']=included
            row['inference_status']='failed' if pred and pred['error'] else 'completed' if pred else 'not_run' if included or not prediction_run else 'excluded_by_schedule'
            # Never replace a saved human review with newer model outputs.
            if not review:row['annotations']=pred['annotations'] if pred else '[]'
    return rows


def link_review(conn,review_id,image,run,before,after):
    if run and not conn.execute('SELECT 1 FROM predictions WHERE inference_id=? AND image_id=? AND error IS NULL',(run,image)).fetchone():
        raise ValueError('Review source prediction is unavailable for this image')
    conn.execute('INSERT INTO review_inference VALUES (?,?,?)',(review_id,run,json.dumps({'before':before,'after':after})))
