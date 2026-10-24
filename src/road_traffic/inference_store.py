"""Durable inference ledger. No ML frameworks are imported by the web service."""
import json
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
    CREATE TABLE IF NOT EXISTS predictions(
      inference_id TEXT REFERENCES inference_runs(id),image_id TEXT REFERENCES images(id),
      annotations TEXT NOT NULL,details TEXT NOT NULL,elapsed REAL NOT NULL,error TEXT,
      PRIMARY KEY(inference_id,image_id),
      FOREIGN KEY(inference_id,image_id) REFERENCES inference_inputs(inference_id,image_id));
    CREATE TABLE IF NOT EXISTS review_inference(
      review_id INTEGER PRIMARY KEY REFERENCES reviews(id),inference_id TEXT REFERENCES inference_runs(id),
      changes TEXT NOT NULL);
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_inference ON inference_runs((1)) WHERE status IN ('queued','running');
    CREATE TRIGGER IF NOT EXISTS immutable_run_config BEFORE UPDATE OF id,config,total,created ON inference_runs BEGIN SELECT RAISE(ABORT,'immutable run configuration'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_predictions_u BEFORE UPDATE ON predictions BEGIN SELECT RAISE(ABORT,'immutable prediction'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_predictions_d BEFORE DELETE ON predictions BEGIN SELECT RAISE(ABORT,'immutable prediction'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_inputs_u BEFORE UPDATE ON inference_inputs BEGIN SELECT RAISE(ABORT,'immutable inputs'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_inputs_d BEFORE DELETE ON inference_inputs BEGIN SELECT RAISE(ABORT,'immutable inputs'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_review_link_u BEFORE UPDATE ON review_inference BEGIN SELECT RAISE(ABORT,'immutable review lineage'); END;
    CREATE TRIGGER IF NOT EXISTS immutable_review_link_d BEFORE DELETE ON review_inference BEGIN SELECT RAISE(ABORT,'immutable review lineage'); END;
    ''')


def prepare(library,mission):
    with library.connect(mission) as conn:schema(conn)


def create_run(library,mission,config):
    prepare(library,mission)
    run=str(uuid.uuid4())
    windows=library.scan_windows(mission)
    with library.connect(mission) as conn:
        zone=conn.execute('SELECT zone FROM mission WHERE id=?',(mission,)).fetchone()['zone']
    with library.connect(mission) as conn:
        conn.execute('BEGIN IMMEDIATE')
        rows=conn.execute('SELECT id,sha256,capture_utc FROM images ORDER BY capture_utc,name').fetchall()
        if not rows:raise ValueError('Import images before running inference')
        from .schedules import compile_windows,eligible_compiled
        compiled=compile_windows(windows)
        eligible_rows=[row for row in rows if eligible_compiled(row['capture_utc'],zone,compiled)]
        if not eligible_rows:
            raise ValueError('No imported images fall within the configured local-time scan windows')
        if conn.execute("SELECT 1 FROM inference_runs WHERE status IN ('queued','running')").fetchone():
            raise ValueError('This mission already has an active inference run')
        recorded_config={**config,'scan_schedule_snapshot':{'timezone':zone,'windows':windows,
            'excluded_image_count':len(rows)-len(eligible_rows)}}
        conn.execute('INSERT INTO inference_runs(id,status,stage,total,config,created) VALUES (?,?,?,?,?,?)',
            (run,'queued','queued',len(eligible_rows),json.dumps(recorded_config),now()))
        conn.executemany('INSERT INTO inference_inputs VALUES (?,?,?)',[(run,r['id'],r['sha256']) for r in eligible_rows])
    return run


def runs(library,mission):
    prepare(library,mission)
    with library.connect(mission) as conn:
        return [dict(r) for r in conn.execute('SELECT * FROM inference_runs ORDER BY created DESC')]


def enrich(library,mission,rows,selected=None):
    prepare(library,mission)
    with library.connect(mission) as conn:
        if selected and not conn.execute('SELECT 1 FROM inference_runs WHERE id=?',(selected,)).fetchone():
            raise ValueError('Inference run is not part of this mission')
        latest=selected or next((r['id'] for r in conn.execute('SELECT id FROM inference_runs ORDER BY created DESC LIMIT 1')),None)
        for row in rows:
            # A saved review stays tied to the exact predictions it corrected.
            review=conn.execute('SELECT id FROM reviews WHERE image_id=? ORDER BY revision DESC LIMIT 1',(row['id'],)).fetchone()
            link=conn.execute('SELECT inference_id FROM review_inference WHERE review_id=?',(review['id'],)).fetchone() if review else None
            prediction_run=latest
            row['review_prediction_run']=link['inference_id'] if link else None
            row['selected_prediction_run']=latest
            if review and link and not selected:prediction_run=link['inference_id']
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
