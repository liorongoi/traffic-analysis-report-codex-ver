"""Versioned, mission-local light correction head. Does not fine-tune YOLOX."""
import hashlib
import json
from pathlib import Path
import uuid
from datetime import datetime
from .store import now,digest
from .night import NIGHT_CLASSES,FEATURE_SCHEMA

BASE_SUBTYPE_CLASSES={
    'headlight_candidate':'headlight',
    'streetlight_suspect':'streetlight',
    'reflection_suspect':'reflection',
    'uncertain_light':'other',
}
NIGHT_TRAINING_PROFILE={
    'id':'night-head-cpu-bounded-v1',
    'device':'CPU',
    'workers':1,
    'concurrent_inference_or_training':'training yields cooperatively at bounded feature and fit-step checkpoints',
    'max_training_regions':5000,
    'max_validation_regions':5000,
    'max_image_pixels':24_000_000,
    'feature_extraction':'one source image loaded at a time; one 13-value feature vector per region',
    'host_memory_selection':'not applicable to the bounded downstream head; detector resource profiles are separate',
}


def prepare(library,mission):
    with library.connect(mission) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS night_training(
          id TEXT PRIMARY KEY,status TEXT,progress INTEGER,created TEXT,finished TEXT,evidence_hash TEXT UNIQUE,
          evidence TEXT,artifact TEXT,artifact_hash TEXT,report TEXT,error TEXT);
          CREATE TRIGGER IF NOT EXISTS immutable_night_artifact BEFORE UPDATE OF artifact,artifact_hash,evidence,evidence_hash ON night_training
          WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'immutable trained candidate'); END;
          CREATE TABLE IF NOT EXISTS night_evaluations(
            id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,candidate_id TEXT NOT NULL REFERENCES night_training(id),
            candidate_hash TEXT NOT NULL,test_fingerprint TEXT NOT NULL,operator TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
            report TEXT,error TEXT,created TEXT NOT NULL,finished TEXT);
          CREATE UNIQUE INDEX IF NOT EXISTS one_completed_night_test_manifest
            ON night_evaluations(mission_id,test_fingerprint) WHERE status='completed';
          CREATE TRIGGER IF NOT EXISTS immutable_night_evaluations_u BEFORE UPDATE ON night_evaluations
            WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'completed test evaluation is immutable'); END;
          CREATE TRIGGER IF NOT EXISTS immutable_night_evaluations_d BEFORE DELETE ON night_evaluations
            BEGIN SELECT RAISE(ABORT,'test evaluation history is append-only'); END;
          CREATE TABLE IF NOT EXISTS night_model_decisions(
            id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,candidate_id TEXT,candidate_hash TEXT,
            previous_candidate_id TEXT,action TEXT NOT NULL CHECK(action IN ('promote','rollback')),
            evaluation_id TEXT,operator TEXT NOT NULL,reason TEXT NOT NULL,created TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS night_model_decisions_latest_idx
            ON night_model_decisions(mission_id,created);
          CREATE TRIGGER IF NOT EXISTS immutable_night_model_decisions_u BEFORE UPDATE ON night_model_decisions
            BEGIN SELECT RAISE(ABORT,'model decision history is append-only'); END;
          CREATE TRIGGER IF NOT EXISTS immutable_night_model_decisions_d BEFORE DELETE ON night_model_decisions
            BEGIN SELECT RAISE(ABORT,'model decision history is append-only'); END;''')
        c.execute('''CREATE TABLE IF NOT EXISTS learning_tasks(
          id TEXT PRIMARY KEY,mission_id TEXT NOT NULL,kind TEXT NOT NULL CHECK(kind IN ('train','evaluate_test')),
          payload TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
          result_id TEXT,result_status TEXT,error TEXT,created TEXT NOT NULL,started TEXT,finished TEXT,
          cancel_requested INTEGER NOT NULL DEFAULT 0,checkpoint TEXT,checkpoint_features TEXT)''')
        columns={row['name'] for row in c.execute('PRAGMA table_info(learning_tasks)')}
        if 'cancel_requested' not in columns:
            c.execute('ALTER TABLE learning_tasks ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0')
        if 'checkpoint' not in columns:
            c.execute('ALTER TABLE learning_tasks ADD COLUMN checkpoint TEXT')
        if 'checkpoint_features' not in columns:
            c.execute('ALTER TABLE learning_tasks ADD COLUMN checkpoint_features TEXT')
        c.execute('CREATE INDEX IF NOT EXISTS learning_tasks_recent ON learning_tasks(created DESC)')


def queue_task(library,mission,kind,payload=None):
    """Persist a user-started learning job before dispatching its worker."""
    if kind not in ('train','evaluate_test'):
        raise ValueError('Unsupported learning task')
    prepare(library,mission)
    identifier=str(uuid.uuid4())
    with library.connect(mission) as c:
        c.execute('BEGIN IMMEDIATE')
        active=c.execute("SELECT id FROM learning_tasks WHERE status IN ('queued','running') LIMIT 1").fetchone()
        if active:
            raise ValueError('A learning task is already queued or running for this mission')
        c.execute('''INSERT INTO learning_tasks(id,mission_id,kind,payload,status,created)
            VALUES (?,?,?,?,?,?)''',(identifier,mission,kind,json.dumps(payload or {},sort_keys=True),'queued',now()))
    return {'id':identifier,'mission_id':mission,'kind':kind,'status':'queued'}


def run_task(library,mission,task_id,yield_checkpoint=None):
    """Run one persisted learning task; progress and the result ledger survive browser closure."""
    prepare(library,mission)
    with library.connect(mission) as c:
        c.execute("UPDATE learning_tasks SET status='running',started=?,cancel_requested=0 WHERE id=? AND mission_id=? AND status='queued'",
            (now(),task_id,mission))
        claimed=c.execute('SELECT changes()').fetchone()[0]
        row=c.execute('SELECT * FROM learning_tasks WHERE id=? AND mission_id=?',(task_id,mission)).fetchone()
    if claimed!=1 or not row or row['status']!='running':
        raise ValueError('Learning task is not queued')
    payload=json.loads(row['payload'])
    resume_checkpoint=json.loads(row['checkpoint']) if row['checkpoint'] else {}
    if row['checkpoint_features']:
        resume_checkpoint.update(json.loads(row['checkpoint_features']))
    def save_checkpoint(checkpoint):
        feature_keys=('training_vectors','validation_vectors','vectors')
        feature_state={key:checkpoint[key] for key in feature_keys if key in checkpoint}
        control_state={key:value for key,value in checkpoint.items() if key not in feature_keys}
        encoded=json.dumps(control_state,separators=(',',':'))
        with library.connect(mission) as c:
            if feature_state:
                encoded_features=json.dumps(feature_state,separators=(',',':'))
                current=c.execute('SELECT checkpoint_features FROM learning_tasks WHERE id=?',(task_id,)).fetchone()
                if current and current['checkpoint_features']!=encoded_features:
                    c.execute("UPDATE learning_tasks SET checkpoint=?,checkpoint_features=? WHERE id=? AND status='running'",
                        (encoded,encoded_features,task_id))
                else:
                    c.execute("UPDATE learning_tasks SET checkpoint=? WHERE id=? AND status='running'",
                        (encoded,task_id))
            else:
                c.execute("UPDATE learning_tasks SET checkpoint=? WHERE id=? AND status='running'",
                    (encoded,task_id))
    def checkpoint():
        if yield_checkpoint:yield_checkpoint()
        with library.connect(mission) as c:
            state=c.execute('SELECT cancel_requested FROM learning_tasks WHERE id=?',(task_id,)).fetchone()
        if state and state['cancel_requested']:
            raise LearningTaskCancelled('Cancelled by operator')
    try:
        if row['kind']=='train':
            result=train(library,mission,yield_checkpoint=checkpoint,
                resume_checkpoint=resume_checkpoint,save_checkpoint=save_checkpoint)
        elif row['kind']=='evaluate_test':
            result=evaluate_test(library,mission,payload.get('candidate_id'),payload.get('operator'),
                yield_checkpoint=checkpoint,resume_checkpoint=resume_checkpoint,
                save_checkpoint=save_checkpoint)
        else:
            raise ValueError('Unsupported learning task')
        with library.connect(mission) as c:
            c.execute("UPDATE learning_tasks SET status='completed',result_id=?,result_status=?,finished=?,checkpoint=NULL,checkpoint_features=NULL WHERE id=? AND status='running'",
                (result.get('id'),result.get('status','completed'),now(),task_id))
        return result
    except LearningTaskCancelled as exc:
        with library.connect(mission) as c:
            c.execute("UPDATE learning_tasks SET status='failed',result_status='cancelled',error=?,finished=? WHERE id=? AND status='running'",
                (str(exc)[:1000],now(),task_id))
        raise
    except Exception as exc:
        with library.connect(mission) as c:
            c.execute("UPDATE learning_tasks SET status='failed',error=?,finished=? WHERE id=? AND status='running'",
                (str(exc)[:1000],now(),task_id))
        raise


class LearningTaskCancelled(Exception):
    """Raised at a safe feature or optimizer checkpoint after operator cancellation."""


def cancel_task(library,mission,task_id):
    prepare(library,mission)
    with library.connect(mission) as c:
        row=c.execute("SELECT status FROM learning_tasks WHERE id=? AND mission_id=?",
            (task_id,mission)).fetchone()
        if not row or row['status'] not in ('queued','running'):
            raise ValueError('Learning task is no longer active')
        if row['status']=='queued':
            c.execute("UPDATE learning_tasks SET status='failed',result_status='cancelled',error='Cancelled by operator',finished=?,cancel_requested=1 WHERE id=?",
                (now(),task_id))
        else:
            c.execute('UPDATE learning_tasks SET cancel_requested=1 WHERE id=?',(task_id,))
    return {'id':task_id,'mission_id':mission,'cancel_requested':True}


def task_status(library,mission):
    prepare(library,mission)
    with library.connect(mission) as c:
        rows=[dict(row) for row in c.execute(
            'SELECT * FROM learning_tasks WHERE mission_id=? ORDER BY created DESC LIMIT 20',(mission,))]
    for row in rows:
        row.pop('checkpoint',None)
        row.pop('checkpoint_features',None)
    return rows


def recover_interrupted_tasks(library):
    """Requeue persisted jobs for checkpoint-aware recovery after process restart."""
    recovered=0
    for mission in library.missions():
        prepare(library,mission['id'])
        with library.connect(mission['id']) as c:
            stamp=now()
            resumable_hashes=set()
            resumable_evaluations=set()
            rows=[dict(row) for row in c.execute("SELECT * FROM learning_tasks WHERE status IN ('queued','running')")]
            for task in rows:
                if task['cancel_requested']:
                    c.execute("UPDATE learning_tasks SET status='failed',result_status='cancelled',error='Cancelled before application restart',finished=? WHERE id=?",
                        (stamp,task['id']))
                    continue
                payload=json.loads(task['payload'] or '{}')
                checkpoint=json.loads(task['checkpoint']) if task['checkpoint'] else {}
                if task['kind']=='train' and checkpoint.get('evidence_hash'):
                    resumable_hashes.add(checkpoint['evidence_hash'])
                if task['kind']=='evaluate_test' and payload.get('candidate_id'):
                    current_fingerprint=_test_fingerprint(inventory(library,mission['id'])['test'])
                    resumable_evaluations.add((payload['candidate_id'],current_fingerprint))
                c.execute("UPDATE learning_tasks SET status='queued',error=NULL,finished=NULL WHERE id=?",
                    (task['id'],))
                recovered+=1
            if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='night_training'").fetchone():
                candidates=[dict(row) for row in c.execute("SELECT id,evidence_hash FROM night_training WHERE status='running'")]
                for candidate in candidates:
                    if candidate['evidence_hash'] not in resumable_hashes:
                        c.execute("UPDATE night_training SET status='failed',finished=?,error=? WHERE id=?",
                            (stamp,'Application stopped before a resumable training task was recorded.',candidate['id']))
            if c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='night_evaluations'").fetchone():
                evaluations=[dict(row) for row in c.execute("SELECT id,candidate_id,test_fingerprint FROM night_evaluations WHERE status='running'")]
                for evaluation in evaluations:
                    if (evaluation['candidate_id'],evaluation['test_fingerprint']) not in resumable_evaluations:
                        c.execute("UPDATE night_evaluations SET status='failed',finished=?,error=? WHERE id=?",
                            (stamp,'Application stopped before a resumable held-out evaluation was recorded.',evaluation['id']))
    return recovered


def inventory(library,mission):
    """Only explicit human classes from latest revisions; protect hashes and adjacent hour groups."""
    with library.connect(mission) as c:
        rows=[dict(r) for r in c.execute('''SELECT reviews.*,images.sha256,images.name,images.capture_utc,images.provenance,
          COALESCE((SELECT h.view_id FROM image_view_history h WHERE h.mission_id=?
            AND h.image_id=images.id ORDER BY h.revision DESC LIMIT 1),'') assigned_view_id
          FROM reviews JOIN images ON images.id=reviews.image_id
          WHERE reviews.revision=(SELECT MAX(r.revision) FROM reviews r WHERE r.image_id=reviews.image_id)''',(mission,))]
        protected=[dict(r) for r in c.execute('''SELECT DISTINCT images.sha256,images.name,images.capture_utc,images.provenance,
          COALESCE((SELECT h.view_id FROM image_view_history h WHERE h.mission_id=?
            AND h.image_id=images.id ORDER BY h.revision DESC LIMIT 1),'') assigned_view_id
          FROM reviews JOIN images ON images.id=reviews.image_id WHERE role IN ('validation','test')''',(mission,))]
    def view_key(r):
        if r.get('assigned_view_id'):
            return 'manual:'+r['assigned_view_id']
        if json.loads(r['provenance']).get('preset')=='fintraffic':
            return Path(r['name']).stem.split('_')[0]
        return None
    def group(r):return (view_key(r) or 'unknown-view',r['capture_utc'][:13])
    hashes={r['sha256'] for r in protected};groups={group(r) for r in protected}
    result={'training':[],'validation':[],'test':[],'excluded':[],'review_count':len(rows),'source':'current mission only'}
    for r in rows:
        if r['role'] not in ('training','validation','test'):continue
        near_holdout=any(group(r)[0]==group(h)[0] and abs((datetime.fromisoformat(r['capture_utc'])-datetime.fromisoformat(h['capture_utc'])).total_seconds())<=1800 for h in protected)
        if r['role']=='training' and (r['sha256'] in hashes or group(r) in groups or near_holdout):
            result['excluded'].append({'review':r['id'],'reason':'held-out hash or same camera/hour group'});continue
        eligible=[]
        for i,a in enumerate(json.loads(r['annotations'])):
            label=a.get('night_class')
            if label not in NIGHT_CLASSES or not a.get('reviewed') or a.get('rejected') or a.get('review_confidence',0)<3:continue
            candidate={'review_id':r['id'],'revision':r['revision'],'image_id':r['image_id'],'sha256':r['sha256'],
                'view':view_key(r),
                'annotation_index':i,'class':label,'box':a['box'],'role':r['role'],'review_status':r['status'],
                'prediction_id':a.get('prediction_id'),
                'base_rule_class':BASE_SUBTYPE_CLASSES.get(a.get('light_subtype'))
                    if a.get('origin')=='night-light-heuristic' and a.get('kind')=='light' and a.get('prediction_id') else None}
            # Avoid overweighting duplicate overlapping rectangles; latest annotation owns overlap.
            from .detector import iou
            eligible=[old for old in eligible if iou(old['box'],a['box'])<=0]
            eligible.append(candidate)
        result[r['role']].extend(eligible)
    for role in ('training','validation','test'):
        unique={}
        for row in sorted(result[role],key=lambda r:r['review_id']):unique[(row['sha256'],tuple(round(v,6) for v in row['box']))]=row
        result[role]=list(unique.values())
    result['review_completion']={}
    for role in ('training','validation','test'):
        images={}
        for row in result[role]:
            images[row['sha256']]=row['review_status']
        result['review_completion'][role]={
            'images_with_eligible_labels':len(images),
            'fully_reviewed_images':sum(status=='reviewed' for status in images.values()),
            'partially_reviewed_images':sum(status=='partial' for status in images.values()),
            'eligible_regions':len(result[role]),
        }
    result['eligible_training_images']=len({r['sha256'] for r in result['training']})
    result['classes']=sorted({r['class'] for r in result['training']})
    return result


def reviewed_evidence_inventory(library, mission):
    """Summarize explicit human image labels by saved evidence role.

    This is an inspection report, not a training-data export or a declaration
    that a detector has a working training adapter. Model predictions, plate
    crop proposals, bulk approvals, and unmarked content in partial reviews
    are never counted as labels.
    """
    result = {
        'source': 'current mission only',
        'purpose': 'evidence inventory; not a training dataset or adapter claim',
        'model_predictions_used_as_labels': False,
        'bulk_approvals_used_as_labels': False,
        'partial_unmarked_regions_used_as_negatives': False,
        'roles': {role: {
            'images': 0, 'fully_reviewed_images': 0, 'partially_reviewed_images': 0,
            'images_with_explicit_human_labels': 0, 'explicit_human_regions': 0,
            'explicitly_rejected_model_regions': 0, 'regions_by_kind': {}, 'regions_by_label': {},
            'camera_hour_groups': 0,
        } for role in ('training', 'validation', 'test')},
        'cross_role_duplicate_hashes': 0,
        'cross_role_camera_hour_groups': 0,
    }
    by_hash_roles = {}
    by_view_time_roles = {}
    label_images = {role: set() for role in ('training', 'validation', 'test')}
    view_groups = {role: set() for role in ('training', 'validation', 'test')}
    with library.connect(mission) as conn:
        rows = conn.execute('''
            SELECT r.role,r.status,r.annotations,
                i.id AS image_id,i.sha256,i.capture_utc,i.name,i.provenance,
                COALESCE((SELECT h.view_id FROM image_view_history h
                    WHERE h.mission_id=? AND h.image_id=i.id
                    ORDER BY h.revision DESC LIMIT 1),'') AS assigned_view_id
            FROM reviews r JOIN images i ON i.id=r.image_id
            WHERE r.revision=(SELECT MAX(latest.revision) FROM reviews latest
                WHERE latest.image_id=r.image_id)
            AND r.role IN ('training','validation','test')
            ORDER BY r.id
        ''', (mission,))
        for raw_row in rows:
            row = dict(raw_row)
            role = row['role']
            counts = result['roles'][role]
            counts['images'] += 1
            counts['fully_reviewed_images'] += row['status'] == 'reviewed'
            counts['partially_reviewed_images'] += row['status'] == 'partial'
            by_hash_roles.setdefault(row['sha256'], set()).add(role)
            provenance = json.loads(row['provenance'] or '{}')
            view = ('manual:' + row['assigned_view_id'] if row['assigned_view_id'] else
                Path(row['name']).stem.split('_')[0] if provenance.get('preset') == 'fintraffic' else None)
            if view:
                hour_group = (view, row['capture_utc'][:13])
                by_view_time_roles.setdefault(hour_group, set()).add(role)
                view_groups[role].add(hour_group)
            for annotation in json.loads(row['annotations'] or '[]'):
                if (annotation.get('origin') == 'model' and annotation.get('reviewed') is True
                        and annotation.get('rejected') is True):
                    counts['explicitly_rejected_model_regions'] += 1
                    continue
                if (annotation.get('origin') != 'human' or annotation.get('reviewed') is not True
                        or annotation.get('rejected') is True):
                    continue
                counts['explicit_human_regions'] += 1
                label_images[role].add(row['image_id'])
                kind = annotation.get('kind') or 'unknown'
                label = annotation.get('label') or 'unlabeled'
                counts['regions_by_kind'][kind] = counts['regions_by_kind'].get(kind, 0) + 1
                counts['regions_by_label'][label] = counts['regions_by_label'].get(label, 0) + 1
    for role, counts in result['roles'].items():
        counts['images_with_explicit_human_labels'] = len(label_images[role])
        counts['regions_by_kind'] = dict(sorted(counts['regions_by_kind'].items()))
        counts['regions_by_label'] = dict(sorted(counts['regions_by_label'].items()))
        counts['camera_hour_groups'] = len(view_groups[role])
    result['cross_role_duplicate_hashes'] = sum(len(roles) > 1 for roles in by_hash_roles.values())
    result['cross_role_camera_hour_groups'] = sum(len(roles) > 1 for roles in by_view_time_roles.values())
    return result


class Head:
    def __init__(self,data,identifier):self.data,self.identifier=data,identifier
    def predict(self,features):
        import numpy as np
        d=self.data;x=(np.asarray(features)-np.asarray(d['mean']))/np.asarray(d['scale'])
        logits=x@np.asarray(d['weights'])+np.asarray(d['bias']);logits-=logits.max();p=np.exp(logits);p/=p.sum()
        return d['classes'][int(p.argmax())],{k:float(v) for k,v in zip(d['classes'],p)}


def candidates(library,mission):
    prepare(library,mission)
    with library.connect(mission) as c:return [dict(r) for r in c.execute('SELECT * FROM night_training ORDER BY created DESC')]


def load_head(library,mission,identifier,expected_hash=None):
    from .store import safe_file
    matches=[r for r in candidates(library,mission) if r['id']==identifier and r['status']=='completed']
    if not matches:raise ValueError('Correction model not found in this mission')
    row=matches[0];path=safe_file(library.location(mission),row['artifact'])
    if digest(path)!=row['artifact_hash'] or (expected_hash and row['artifact_hash']!=expected_hash):raise ValueError('Correction model checksum mismatch')
    data=json.loads(path.read_text())
    if data['feature_schema']!=FEATURE_SCHEMA:raise ValueError('Incompatible correction feature schema')
    return Head(data,identifier),row['artifact_hash']


def _test_fingerprint(records):
    manifest=[{'sha256':row['sha256'],'box':[round(float(v),6) for v in row['box']],
        'class':row['class']}
        for row in records]
    manifest.sort(key=lambda item:(item['sha256'],item['box'],item['class']))
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _legacy_test_fingerprint(records):
    """Recognize pre-label fingerprints so upgrades cannot silently reuse test data."""
    manifest=[{'sha256':row['sha256'],'box':[round(float(v),6) for v in row['box']]}
        for row in records]
    manifest.sort(key=lambda item:(item['sha256'],item['box']))
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _feature_vectors(library,mission,records,yield_checkpoint=None,*,
        resume_checkpoint=None,save_checkpoint=None,checkpoint_identity=None):
    import numpy as np
    from PIL import Image
    from .night import patch_features
    resume_checkpoint=resume_checkpoint or {}
    vectors=list(resume_checkpoint.get('vectors',[])) if resume_checkpoint.get('identity')==checkpoint_identity else []
    if len(vectors)>len(records):vectors=[]
    for index,row in enumerate(records[len(vectors):],start=len(vectors)):
        if yield_checkpoint:yield_checkpoint()
        with Image.open(library.image_path(mission,row['image_id'])) as image:
            if image.width*image.height>24_000_000:raise ValueError('Evaluation image exceeds adapter pixel budget')
            rgb=np.asarray(image.convert('RGB'))
        # Keep checkpoint JSON compatible with both the production list return
        # value and array-like feature extractors used by alternate backends.
        vectors.append([float(value) for value in patch_features(rgb,row['box'])])
        if save_checkpoint and (len(vectors)%64==0 or len(vectors)==len(records)):
            save_checkpoint({'identity':checkpoint_identity,'vectors':vectors})
    return np.asarray(vectors,dtype=np.float64)


def _classification_metrics(actual,predicted):
    labels=sorted(set(actual)|set(predicted))
    metrics={};confusion={label:{other:0 for other in labels} for label in labels}
    for truth,guess in zip(actual,predicted):confusion[truth][guess]+=1
    for label in labels:
        tp=sum(t==label and p==label for t,p in zip(actual,predicted))
        fp=sum(t!=label and p==label for t,p in zip(actual,predicted))
        fn=sum(t==label and p!=label for t,p in zip(actual,predicted))
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        metrics[label]={'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'support':actual.count(label)}
    return {'per_class':metrics,'confusion_matrix':confusion,
        'accuracy':sum(a==p for a,p in zip(actual,predicted))/len(actual),
        'macro_f1':sum(item['f1'] for item in metrics.values())/len(metrics)}


def evaluation_status(library,mission):
    prepare(library,mission)
    data=inventory(library,mission)
    fingerprint=_test_fingerprint(data['test'])
    legacy_fingerprint=_legacy_test_fingerprint(data['test'])
    with library.connect(mission) as c:
        rows=[dict(row) for row in c.execute('''SELECT * FROM night_evaluations
          WHERE mission_id=? ORDER BY created,id''',(mission,))]
    for row in rows:
        row['report']=json.loads(row['report']) if row['report'] else None
    return {'test_fingerprint':fingerprint,'test_regions':len(data['test']),
        'test_images':len({row['image_id'] for row in data['test']}),
        'already_evaluated':any(row['status']=='completed' and row['test_fingerprint'] in
            {fingerprint,legacy_fingerprint} for row in rows),
        'evaluations':rows}


def governance_status(library,mission):
    prepare(library,mission)
    with library.connect(mission) as c:
        rows=[dict(row) for row in c.execute('''SELECT * FROM night_model_decisions
            WHERE mission_id=? ORDER BY rowid''',(mission,))]
    latest=rows[-1] if rows else None
    return {'promoted_candidate_id':latest['candidate_id'] if latest else None,
        'promoted_candidate_hash':latest['candidate_hash'] if latest else None,
        'rollback_target_id':latest['previous_candidate_id'] if latest else None,
        'history':rows}


def record_model_decision(library,mission,action,candidate_id,operator,reason):
    if action not in ('promote','rollback'):
        raise ValueError('Choose promote or rollback')
    if not isinstance(operator,str) or not operator.strip() or len(operator.strip())>120:
        raise ValueError('Enter an operator name of up to 120 characters')
    if not isinstance(reason,str) or not reason.strip() or len(reason.strip())>1000:
        raise ValueError('Record a promotion or rollback rationale of up to 1000 characters')
    prepare(library,mission)
    current=governance_status(library,mission)
    previous=current['promoted_candidate_id']
    if candidate_id==previous:
        raise ValueError('The selected candidate is already active')
    candidate_hash=None
    evaluation_id=None
    if candidate_id is not None:
        _,candidate_hash=load_head(library,mission,candidate_id)
    if action=='promote':
        if candidate_id is None:
            raise ValueError('Promote a completed candidate; use rollback to return to the base model')
        data=inventory(library,mission)
        fingerprint=_test_fingerprint(data['test'])
        if not data['test']:
            raise ValueError('Promotion requires an independent held-out test evaluation')
        with library.connect(mission) as c:
            evaluation=c.execute('''SELECT id,candidate_hash FROM night_evaluations
                WHERE mission_id=? AND candidate_id=? AND test_fingerprint=? AND status='completed'
                ORDER BY finished DESC,id DESC LIMIT 1''',(mission,candidate_id,fingerprint)).fetchone()
        if not evaluation:
            raise ValueError('Run the current held-out test evaluation before promotion')
        if evaluation['candidate_hash']!=candidate_hash:
            raise ValueError('The evaluated candidate artifact does not match the current candidate')
        evaluation_id=evaluation['id']
    else:
        history_ids={row['candidate_id'] for row in current['history'] if row['candidate_id']}
        if candidate_id is not None and candidate_id not in history_ids:
            raise ValueError('Rollback target must be the base model or a previously promoted candidate')
    identifier=str(uuid.uuid4())
    with library.connect(mission) as c:
        c.execute('''INSERT INTO night_model_decisions(id,mission_id,candidate_id,candidate_hash,
            previous_candidate_id,action,evaluation_id,operator,reason,created)
            VALUES (?,?,?,?,?,?,?,?,?,?)''',(identifier,mission,candidate_id,candidate_hash,previous,
            action,evaluation_id,operator.strip(),reason.strip(),now()))
    return {'id':identifier,'mission_id':mission,'candidate_id':candidate_id,
        'candidate_hash':candidate_hash,'previous_candidate_id':previous,'action':action,
        'evaluation_id':evaluation_id,'operator':operator.strip(),'reason':reason.strip()}


def evaluate_test(library,mission,candidate_id,operator,yield_checkpoint=None,
        resume_checkpoint=None,save_checkpoint=None):
    if not isinstance(operator,str) or not operator.strip() or len(operator.strip())>120:
        raise ValueError('Enter an operator name of up to 120 characters')
    prepare(library,mission)
    data=inventory(library,mission)
    if not data['test']:
        raise ValueError('No eligible independent test labels are available')
    head,candidate_hash=load_head(library,mission,candidate_id)
    model=head.data
    test_views={row['view'] for row in data['test']}
    if test_views!={model.get('view')}:
        raise ValueError('Test examples must match the candidate’s verified camera view')
    fingerprint=_test_fingerprint(data['test']);legacy_fingerprint=_legacy_test_fingerprint(data['test'])
    checkpoint_identity=hashlib.sha256(f'{candidate_hash}:{fingerprint}'.encode()).hexdigest()
    evaluation_id=None
    with library.connect(mission) as c:
        c.execute('BEGIN IMMEDIATE')
        if c.execute('''SELECT 1 FROM night_evaluations WHERE mission_id=?
            AND test_fingerprint IN (?,?) AND status='completed' LIMIT 1''',
                     (mission,fingerprint,legacy_fingerprint)).fetchone():
            raise ValueError('This unchanged test set has already been used for an independent evaluation')
        existing=c.execute('''SELECT id FROM night_evaluations WHERE mission_id=? AND candidate_id=?
          AND candidate_hash=? AND test_fingerprint=? AND status IN ('running','failed')
          ORDER BY created DESC,id DESC LIMIT 1''',
          (mission,candidate_id,candidate_hash,fingerprint)).fetchone()
        if existing:
            evaluation_id=existing['id']
            c.execute("UPDATE night_evaluations SET status='running',error=NULL,finished=NULL,operator=? WHERE id=?",
                (operator.strip(),evaluation_id))
        else:
            evaluation_id=str(uuid.uuid4())
            c.execute('''INSERT INTO night_evaluations(id,mission_id,candidate_id,candidate_hash,
              test_fingerprint,operator,status,created) VALUES (?,?,?,?,?,?,?,?)''',
              (evaluation_id,mission,candidate_id,candidate_hash,fingerprint,operator.strip(),'running',now()))
    try:
        vectors=_feature_vectors(library,mission,data['test'],yield_checkpoint=yield_checkpoint,
            resume_checkpoint=resume_checkpoint,save_checkpoint=save_checkpoint,
            checkpoint_identity=checkpoint_identity)
        actual=[row['class'] for row in data['test']]
        predicted=[head.predict(vector)[0] for vector in vectors]
        candidate_metrics=_classification_metrics(actual,predicted)
        current=governance_status(library,mission)
        baseline_comparison={'status':'unavailable','reason':'No distinct promoted candidate is available for comparison.'}
        base_context_comparison={'status':'unavailable',
            'reason':'No held-out reviewed regions retain a linked contextual-light proposal.',
            'paired_regions':0,'test_regions':len(actual),'coverage':0.0}
        paired=[index for index,row in enumerate(data['test']) if row.get('base_rule_class') in NIGHT_CLASSES]
        if paired:
            paired_actual=[actual[index] for index in paired]
            base_predictions=[data['test'][index]['base_rule_class'] for index in paired]
            candidate_paired=[predicted[index] for index in paired]
            candidate_paired_metrics=_classification_metrics(paired_actual,candidate_paired)
            base_metrics=_classification_metrics(paired_actual,base_predictions)
            base_context_comparison={'status':'compared','baseline':'saved_contextual_light_proposals',
                'regions':len(paired),'paired_regions':len(paired),'test_regions':len(actual),
                'coverage':len(paired)/len(actual),'candidate':candidate_paired_metrics,
                'current':base_metrics,
                'delta':{'accuracy':candidate_paired_metrics['accuracy']-base_metrics['accuracy'],
                    'macro_f1':candidate_paired_metrics['macro_f1']-base_metrics['macro_f1']},
                'test_labels_used_for_fit_or_validation':False,
                'scope':'paired reviewed regions retaining their original production contextual-light proposal; does not rerun detector or background construction'}
        baseline_id=current.get('promoted_candidate_id')
        if baseline_id and baseline_id!=candidate_id:
            try:
                baseline_head,baseline_hash=load_head(library,mission,baseline_id)
                if baseline_head.data.get('view')!=model.get('view'):
                    baseline_comparison={'status':'unavailable','reason':'The current candidate uses a different camera view.'}
                else:
                    baseline_predictions=[baseline_head.predict(vector)[0] for vector in vectors]
                    baseline_metrics=_classification_metrics(actual,baseline_predictions)
                    comparison_candidate=_classification_metrics(actual,predicted)
                    baseline_comparison={'status':'compared','current_candidate_id':baseline_id,
                        'current_candidate_hash':baseline_hash,'regions':len(actual),
                        'candidate':comparison_candidate,'current':baseline_metrics,
                        'delta':{'accuracy':comparison_candidate['accuracy']-baseline_metrics['accuracy'],
                            'macro_f1':comparison_candidate['macro_f1']-baseline_metrics['macro_f1']},
                        'test_labels_used_for_fit_or_validation':False}
            except ValueError as exc:
                baseline_comparison={'status':'unavailable','reason':str(exc)}
        report={'regions':len(actual),'images':len({row['image_id'] for row in data['test']}),
            **candidate_metrics,'baseline_comparison':baseline_comparison,
            'base_context_comparison':base_context_comparison,
            'scope':'held-out labeled light patches for this frozen candidate; not whole-image vehicle or traffic-count accuracy',
            'test_labels_used_for_fit_or_validation':False,'automatic_promotion':False}
        with library.connect(mission) as c:
            c.execute("UPDATE night_evaluations SET status='completed',report=?,finished=? WHERE id=? AND status='running'",
                (json.dumps(report,sort_keys=True),now(),evaluation_id))
        return {'id':evaluation_id,'candidate_id':candidate_id,'candidate_hash':candidate_hash,
            'test_fingerprint':fingerprint,'operator':operator.strip(),'status':'completed','report':report}
    except Exception as exc:
        with library.connect(mission) as c:
            c.execute("UPDATE night_evaluations SET status='failed',error=?,finished=? WHERE id=? AND status='running'",
                (str(exc)[:1000],now(),evaluation_id))
        raise


def train(library,mission,yield_checkpoint=None,resume_checkpoint=None,save_checkpoint=None):
    prepare(library,mission);data=inventory(library,mission)
    # Technical fitting requirements, not an automatic retraining policy or quality guarantee.
    if data['eligible_training_images']<2 or len(data['classes'])<2 or len(data['training'])<4:
        return {'status':'needs_labels','message':'Need explicit training labels (confidence >=3) for at least two classes, four regions and two images. Validation/test never fit.',
            'inventory':data,'checkpoint_created':False}
    if len(data['training'])>NIGHT_TRAINING_PROFILE['max_training_regions']:
        raise ValueError('Training set exceeds this small-head budget; no labels were silently dropped')
    if len(data['validation'])>NIGHT_TRAINING_PROFILE['max_validation_regions']:
        raise ValueError('Validation set exceeds this small-head budget; no labels were silently dropped')
    views={r['view'] for r in data['training']}
    if None in views or len(views)!=1:raise ValueError('Assign a verified single camera view to every training image before fitting')
    if any(row['view'] not in views for row in data['validation']):
        raise ValueError('Validation images must use the same verified camera view as training images')
    training_plan={'algorithm':'regularized-multinomial-linear-downstream-head',
        'fitter':'linear-600-l2-0.03-v1','feature_schema':FEATURE_SCHEMA,
        'preprocessing_config':{'feature_schema':FEATURE_SCHEMA,'maximum_source_image_pixels':24_000_000},
        'class_schema':data['classes'],'resource_profile':NIGHT_TRAINING_PROFILE,
        'review_completion':data['review_completion'],
        'base_checkpoint':None,
        'base_checkpoint_note':'Trained from scratch as a downstream light-patch head; YOLOX and base context rules remain unchanged.'}
    manifest={'training':data['training'],'validation':data['validation'],**training_plan}
    evidence=json.dumps(manifest,sort_keys=True);fingerprint=hashlib.sha256(evidence.encode()).hexdigest()
    dataset_version=fingerprint
    saved_evidence_hash=(resume_checkpoint or {}).get('evidence_hash')
    checkpoint_state=(resume_checkpoint or {}) if (resume_checkpoint or {}).get('evidence_hash')==fingerprint else {}
    if saved_evidence_hash and saved_evidence_hash!=fingerprint:
        with library.connect(mission) as c:
            c.execute("UPDATE night_training SET status='failed',finished=?,error=? WHERE evidence_hash=? AND status IN ('running','failed')",
                (now(),'Training evidence changed; the old checkpoint was discarded.',saved_evidence_hash))
    with library.connect(mission) as c:
        old=c.execute('SELECT * FROM night_training WHERE evidence_hash=?',(fingerprint,)).fetchone()
        if old and old['status']=='completed':return {'status':old['status'],'id':old['id'],'reused_evidence':True}
        if old:
            identifier=old['id']
            c.execute("UPDATE night_training SET status='running',error=NULL,finished=NULL WHERE id=?",(identifier,))
        else:
            identifier=str(uuid.uuid4());c.execute('INSERT INTO night_training(id,status,progress,created,evidence_hash,evidence) VALUES (?,?,?,?,?,?)',
                (identifier,'running',0,now(),fingerprint,evidence))
    try:
        import numpy as np
        from PIL import Image
        from .night import patch_features
        def features(records,role):
            vectors=list(checkpoint_state.get(f'{role}_vectors',[]))
            if len(vectors)>len(records):vectors=[]
            for index,row in enumerate(records[len(vectors):],start=len(vectors)):
                if yield_checkpoint:yield_checkpoint()
                with Image.open(library.image_path(mission,row['image_id'])) as im:
                    if im.width*im.height>24_000_000:raise ValueError('Training image exceeds adapter pixel budget')
                    rgb=np.asarray(im.convert('RGB'))
                vectors.append([float(value) for value in patch_features(rgb,row['box'])])
                if save_checkpoint and (len(vectors)%64==0 or len(vectors)==len(records)):
                    checkpoint_state[f'{role}_vectors']=vectors
                    checkpoint_state.update({'evidence_hash':fingerprint,'stage':'features'})
                    save_checkpoint(checkpoint_state)
            return np.asarray(vectors,dtype=np.float64)
        x=features(data['training'],'training')
        validation_x=features(data['validation'],'validation') if data['validation'] else np.empty((0,x.shape[1]))
        classes=data['classes'];y=np.asarray([classes.index(r['class']) for r in data['training']])
        saved_fit=checkpoint_state if checkpoint_state.get('stage') in ('fit','ready') else {}
        if saved_fit:
            mean=np.asarray(saved_fit['mean']);scale=np.asarray(saved_fit['scale'])
            weights=np.asarray(saved_fit['weights']);bias=np.asarray(saved_fit['bias'])
            start_step=int(saved_fit.get('next_step',600))
        else:
            mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<.001]=1
            weights=np.zeros((x.shape[1],len(classes)));bias=np.zeros(len(classes));start_step=0
        z=(x-mean)/scale;targets=np.eye(len(classes))[y]
        for step in range(start_step,600):
            if yield_checkpoint:yield_checkpoint()
            logits=z@weights+bias;logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
            diff=(p-targets)/len(z);weights-=.05*(z.T@diff+.03*weights);bias-=.05*diff.sum(axis=0)
            if step%60==0:
                with library.connect(mission) as c:c.execute('UPDATE night_training SET progress=? WHERE id=?',(int(step/6),identifier))
            if save_checkpoint and ((step+1)%20==0 or step==599):
                checkpoint_state.update({'evidence_hash':fingerprint,'stage':'fit',
                    'training_vectors':x.tolist(),'validation_vectors':validation_x.tolist(),
                    'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),
                    'bias':bias.tolist(),'next_step':step+1})
                save_checkpoint(checkpoint_state)
        if save_checkpoint:
            checkpoint_state.update({'evidence_hash':fingerprint,'stage':'ready',
                'training_vectors':x.tolist(),'validation_vectors':validation_x.tolist(),
                'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),
                'bias':bias.tolist(),'next_step':600})
            save_checkpoint(checkpoint_state)
        model={'feature_schema':FEATURE_SCHEMA,'classes':classes,'mean':mean.tolist(),'scale':scale.tolist(),
               'weights':weights.tolist(),'bias':bias.tolist(),'fitting':'regularized multinomial linear head; YOLOX frozen',
               'evidence_hash':fingerprint,'dataset_version':dataset_version,
               'resource_profile':NIGHT_TRAINING_PROFILE,
               'base_checkpoint':None,
               'scope':'mission-local fixed-view light patches','view':next(iter(views)),'calibrated':False}
        head=Head(model,identifier)
        report={'training_regions':len(x),'training_images':data['eligible_training_images'],'classes':classes,
            'dataset_version':dataset_version,'base_checkpoint':None,
            'algorithm':training_plan['algorithm'],'preprocessing_config':training_plan['preprocessing_config'],
            'resource_profile':NIGHT_TRAINING_PROFILE,'review_completion':data['review_completion'],
            'promotion':'candidate only; must be selected explicitly for a new inference run','test_labels_used':False,'validation':None}
        if data['validation']:
            predicted=[head.predict(v)[0] for v in validation_x];actual=[r['class'] for r in data['validation']]
            metrics={}
            for label in sorted(set(actual)|set(predicted)):
                tp=sum(a==label and b==label for a,b in zip(actual,predicted));fp=sum(a!=label and b==label for a,b in zip(actual,predicted));fn=sum(a==label and b!=label for a,b in zip(actual,predicted))
                precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
                metrics[label]={'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,'support':actual.count(label)}
            report['validation']={'regions':len(actual),'per_class':metrics,'scope':'operator-selected patch labels, not whole-image vehicle detection accuracy','selection_bias_warning':True}
        else:report['validation_unavailable_reason']='No independent human validation labels'
        folder=library.location(mission)/'models'/'night'/identifier;folder.mkdir(parents=True)
        artifact=folder/'checkpoint.json';artifact.write_text(json.dumps(model,indent=2))
        with library.connect(mission) as c:c.execute("UPDATE night_training SET status='completed',progress=100,finished=?,artifact=?,artifact_hash=?,report=? WHERE id=?",(now(),str(artifact.relative_to(library.location(mission))),digest(artifact),json.dumps(report),identifier))
        return {'status':'completed','id':identifier,'report':report}
    except Exception as exc:
        with library.connect(mission) as c:c.execute("UPDATE night_training SET status='failed',finished=?,error=? WHERE id=?",(now(),str(exc),identifier))
        raise
