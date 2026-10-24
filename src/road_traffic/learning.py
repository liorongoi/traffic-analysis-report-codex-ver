"""Versioned, mission-local light correction head. Does not fine-tune YOLOX."""
import hashlib
import json
from pathlib import Path
import uuid
from datetime import datetime
from .store import now,digest
from .night import NIGHT_CLASSES,FEATURE_SCHEMA


def prepare(library,mission):
    with library.connect(mission) as c:
        c.executescript('''CREATE TABLE IF NOT EXISTS night_training(
          id TEXT PRIMARY KEY,status TEXT,progress INTEGER,created TEXT,finished TEXT,evidence_hash TEXT UNIQUE,
          evidence TEXT,artifact TEXT,artifact_hash TEXT,report TEXT,error TEXT);
          CREATE TRIGGER IF NOT EXISTS immutable_night_artifact BEFORE UPDATE OF artifact,artifact_hash,evidence,evidence_hash ON night_training
          WHEN OLD.status='completed' BEGIN SELECT RAISE(ABORT,'immutable trained candidate'); END;''')


def inventory(library,mission):
    """Only explicit human classes from latest revisions; protect hashes and adjacent hour groups."""
    with library.connect(mission) as c:
        rows=[dict(r) for r in c.execute('''SELECT reviews.*,images.sha256,images.name,images.capture_utc,images.provenance FROM reviews
          JOIN images ON images.id=reviews.image_id WHERE reviews.revision=(SELECT MAX(r.revision) FROM reviews r WHERE r.image_id=reviews.image_id)''')]
        protected=[dict(r) for r in c.execute("SELECT DISTINCT images.sha256,images.name,images.capture_utc,images.provenance FROM reviews JOIN images ON images.id=reviews.image_id WHERE role IN ('validation','test')")]
    def group(r):return (Path(r['name']).name.split('_')[0] if '_' in r['name'] else 'unknown-view',r['capture_utc'][:13])
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
                'view':Path(r['name']).name.split('_')[0] if json.loads(r['provenance']).get('preset')=='fintraffic' else None,
                'annotation_index':i,'class':label,'box':a['box'],'role':r['role'],'prediction_id':a.get('prediction_id')}
            # Avoid overweighting duplicate overlapping rectangles; latest annotation owns overlap.
            from .detector import iou
            eligible=[old for old in eligible if iou(old['box'],a['box'])<=0]
            eligible.append(candidate)
        result[r['role']].extend(eligible)
    for role in ('training','validation','test'):
        unique={}
        for row in sorted(result[role],key=lambda r:r['review_id']):unique[(row['sha256'],tuple(round(v,6) for v in row['box']))]=row
        result[role]=list(unique.values())
    result['eligible_training_images']=len({r['sha256'] for r in result['training']})
    result['classes']=sorted({r['class'] for r in result['training']})
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


def train(library,mission):
    prepare(library,mission);data=inventory(library,mission)
    # Technical fitting requirements, not an automatic retraining policy or quality guarantee.
    if data['eligible_training_images']<2 or len(data['classes'])<2 or len(data['training'])<4:
        return {'status':'needs_labels','message':'Need explicit training labels (confidence >=3) for at least two classes, four regions and two images. Validation/test never fit.',
            'inventory':data,'checkpoint_created':False}
    views={r['view'] for r in data['training']}
    if None in views or len(views)!=1:raise ValueError('Assign a verified single camera view before fitting; only explicit Fintraffic views are currently supported')
    if len(data['training'])>5000:raise ValueError('Training set exceeds this small-head budget; no labels were silently dropped')
    for m in library.missions():
        with library.connect(m['id']) as c:
            exists=c.execute("SELECT 1 FROM sqlite_master WHERE name='inference_runs'").fetchone()
            if exists and c.execute("SELECT 1 FROM inference_runs WHERE status IN ('queued','running')").fetchone():raise ValueError('Finish active inference before fitting a light head')
    evidence=json.dumps({'training':data['training'],'validation':data['validation'],'feature_schema':FEATURE_SCHEMA,'fitter':'linear-600-l2-0.03-v1'},sort_keys=True);fingerprint=hashlib.sha256(evidence.encode()).hexdigest()
    with library.connect(mission) as c:
        old=c.execute('SELECT * FROM night_training WHERE evidence_hash=?',(fingerprint,)).fetchone()
        if old:return {'status':old['status'],'id':old['id'],'reused_evidence':True}
        identifier=str(uuid.uuid4());c.execute('INSERT INTO night_training(id,status,progress,created,evidence_hash,evidence) VALUES (?,?,?,?,?,?)',
            (identifier,'running',0,now(),fingerprint,evidence))
    try:
        import numpy as np
        from PIL import Image
        from .night import patch_features
        def features(records):
            vectors=[]
            for row in records:
                with Image.open(library.image_path(mission,row['image_id'])) as im:
                    if im.width*im.height>24_000_000:raise ValueError('Training image exceeds adapter pixel budget')
                    rgb=np.asarray(im.convert('RGB'))
                vectors.append(patch_features(rgb,row['box']))
            return np.asarray(vectors,dtype=np.float64)
        x=features(data['training']);classes=data['classes'];y=np.asarray([classes.index(r['class']) for r in data['training']])
        mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<.001]=1
        z=(x-mean)/scale;weights=np.zeros((x.shape[1],len(classes)));bias=np.zeros(len(classes));targets=np.eye(len(classes))[y]
        for step in range(600):
            logits=z@weights+bias;logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
            diff=(p-targets)/len(z);weights-=.05*(z.T@diff+.03*weights);bias-=.05*diff.sum(axis=0)
            if step%60==0:
                with library.connect(mission) as c:c.execute('UPDATE night_training SET progress=? WHERE id=?',(int(step/6),identifier))
        model={'feature_schema':FEATURE_SCHEMA,'classes':classes,'mean':mean.tolist(),'scale':scale.tolist(),
               'weights':weights.tolist(),'bias':bias.tolist(),'fitting':'regularized multinomial linear head; YOLOX frozen',
               'evidence_hash':fingerprint,'scope':'mission-local fixed-view light patches','view':next(iter(views)),'calibrated':False}
        head=Head(model,identifier)
        report={'training_regions':len(x),'training_images':data['eligible_training_images'],'classes':classes,
            'promotion':'candidate only; must be selected explicitly for a new inference run','test_labels_used':False,'validation':None}
        if data['validation']:
            predicted=[head.predict(v)[0] for v in features(data['validation'])];actual=[r['class'] for r in data['validation']]
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
