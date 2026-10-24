"""Separate local worker. Durable results/progress; browser closure does not stop inference."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import uuid
from .store import Library,now,digest
from .inference_store import prepare


def execute(library,mission,run,pack):
    prepare(library,mission)
    with library.connect(mission) as conn:
        job=conn.execute('SELECT * FROM inference_runs WHERE id=?',(run,)).fetchone()
        if not job or job['status']!='queued':raise ValueError('Only a queued run can start')
        config=json.loads(job['config'])
        conn.execute("UPDATE inference_runs SET status='running',stage='loading_model',worker_pid=? WHERE id=?",(os.getpid(),run))
    try:
        import cv2
        import numpy as np
        from PIL import Image
        from .detector import VehicleDetector,light_mask,night_proposals,illumination,VERSION
        from .night import propose
        head=None
        if config.get('night_head_id'):
            from .learning import load_head
            head,_=load_head(library,mission,config['night_head_id'],config['night_head_hash'])
        detector=VehicleDetector(pack,config['threshold'],config['tiled'])
        if config['adapter']!=VERSION:raise ValueError('Adapter version changed; start a new run')
        if detector.sha!=config['model_sha256']:raise ValueError('Model changed after run approval')
        with library.connect(mission) as conn:
            inputs=[dict(r) for r in conn.execute('''SELECT images.*,COALESCE(iv.view_id,'') view_id,
                COALESCE(iv.view_revision,0) view_revision FROM images
                JOIN inference_inputs ii ON images.id=ii.image_id
                LEFT JOIN inference_input_views iv ON iv.inference_id=ii.inference_id AND iv.image_id=ii.image_id
                WHERE ii.inference_id=? ORDER BY capture_utc,name''',(run,))]
            heldout={r[0] for r in conn.execute("SELECT DISTINCT image_id FROM reviews WHERE role IN ('validation','test')")}
        if head and any(json.loads(r['provenance']).get('preset')!='fintraffic' or Path(r['name']).stem.split('_')[0]!=head.data.get('view') for r in inputs):
            raise ValueError('Selected light correction head does not match every source view; use an appropriate mission/view or select None')
        def load(row):
            with Image.open(library.image_path(mission,row['id'])) as image:
                if image.width*image.height>24_000_000:raise ValueError('Image exceeds this adapter pixel budget')
                return np.asarray(image.convert('RGB'))
        # Background references stay within one manual view assignment, or an
        # explicit Fintraffic source-view key. Unassigned camera files do not mix.
        refs=[];masks=[];reference_grays=[];shape=None;view=None;anchor=None
        def row_view_key(row):
            if row.get('view_id'):
                return 'manual:'+row['view_id']
            provenance=json.loads(row['provenance'])
            if provenance.get('preset')=='fintraffic':
                return 'fintraffic:'+Path(row['name']).stem.split('_')[0]
            return None
        with library.connect(mission) as conn:conn.execute("UPDATE inference_runs SET stage='building_light_background' WHERE id=?",(run,))
        if config.get('night_lights'):
            for row in inputs:
                with library.connect(mission) as conn:
                    if conn.execute('SELECT stop_requested FROM inference_runs WHERE id=?',(run,)).fetchone()[0]:break
                provenance=json.loads(row['provenance'])
                candidate_view=row_view_key(row)
                if candidate_view is None or row['id'] in heldout:continue
                rgb=load(row);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                if not illumination(rgb)['low_light'] or gray.size>2_000_000:continue
                if shape is None:
                    shape=gray.shape;view=candidate_view;anchor=cv2.resize(gray,(160,90)).astype(float).ravel()
                if gray.shape!=shape or candidate_view!=view:continue
                signature=cv2.resize(gray,(160,90)).astype(float).ravel()
                similarity=float(np.corrcoef(anchor,signature)[0,1])
                if not np.isfinite(similarity) or similarity<.8:continue
                refs.append({'image_id':row['id'],'sha256':row['sha256']});masks.append(light_mask(gray));reference_grays.append(gray)
                if len(refs)>=32:break
        reference_folder=library.location(mission)/'inference'/run
        reference_folder.mkdir(parents=True,exist_ok=True)
        counts=np.sum(masks,axis=0,dtype=np.uint16) if masks else None
        background_path=reference_folder/'light-background.npz'
        stack=np.asarray(reference_grays,dtype=np.uint8)
        median=np.median(stack,axis=0) if len(stack) else np.zeros((0,0),np.float32)
        mad=np.median(np.abs(stack.astype(np.float32)-median),axis=0) if len(stack) else median
        np.savez_compressed(background_path,median=median,mad=mad,counts=counts if counts is not None else np.zeros((0,0),np.uint16),reference_ids=np.asarray([r['image_id'] for r in refs]))
        background={'path':str(background_path.relative_to(library.location(mission))),'sha256':digest(background_path),
            'references':refs,'view':view,'min_reference_images':5,'persistence_threshold':.6,
            'rule':'leave-target-out persistence plus novelty >= max(25,3*MAD) above median; held-out references excluded; similarity >= 0.8'}
        refindex={r['image_id']:i for i,r in enumerate(refs)}
        for number,row in enumerate(inputs):
            with library.connect(mission) as conn:
                if conn.execute('SELECT stop_requested FROM inference_runs WHERE id=?',(run,)).fetchone()[0]:
                    conn.execute("UPDATE inference_runs SET status='cancelled',stage='cancelled',finished=? WHERE id=?",(now(),run));return
                conn.execute("UPDATE inference_runs SET stage='detecting' WHERE id=?",(run,))
            start=time.perf_counter()
            details={'adapter':VERSION,'model_sha256':detector.sha,'device':'CPUExecutionProvider','input_sha256':row['sha256'],
                'threshold':config['threshold'],'tiled':config['tiled'],'light_background':background,
                'view_id':row.get('view_id') or None,'view_assignment_revision':row.get('view_revision',0),
                'human_confirmation':False,'confidence_note':'Detector scores are not calibrated accuracy.'}
            predictions=[];error=None
            try:
                rgb=load(row);h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                details.update({'width':w,'height':h,**illumination(rgb)})
                predictions=detector.infer(rgb)
                if config.get('night_lights') and details['low_light']:
                    persistent=None;light_context=None;usable=counts is not None and gray.shape==shape and row_view_key(row)==view
                    similarity=float(np.corrcoef(anchor,cv2.resize(gray,(160,90)).astype(float).ravel())[0,1]) if usable else 0
                    if usable and np.isfinite(similarity) and similarity>=.8:
                        n=len(refs);local=counts.copy()
                        if row['id'] in refindex:local-=masks[refindex[row['id']]];n-=1
                        if n>=5:
                            base_median,base_mad=median,mad
                            if row['id'] in refindex:
                                subset=np.delete(stack,refindex[row['id']],axis=0)
                                base_median=np.median(subset,axis=0)
                                base_mad=np.median(np.abs(subset.astype(np.float32)-base_median),axis=0)
                            persistent=cv2.dilate((local>=n*.6).astype(np.uint8),np.ones((7,7),np.uint8))
                            novelty=(gray.astype(np.float32)-base_median)>=np.maximum(25,3*base_mad)
                            persistent[~novelty]=1
                            light_context={'frequency':local.astype(np.float32)/n,'novelty':novelty}
                    lights,light_details=propose(rgb,light_context,predictions,head)
                    predictions.extend(lights);details['lights']=light_details
                    details['lights']['background_used']=light_context is not None
                    details['night_head_id']=config.get('night_head_id')
                    details['night_head_hash']=config.get('night_head_hash')
                    details['lights']['view_similarity']=similarity
                    details['lights']['leave_out_image_id']=row['id'] if row['id'] in refindex else None
                for item in predictions:item['prediction_id']=str(uuid.uuid4())
                details['detections']=sum(x['evidence']=='yolox-s' for x in predictions)
                details['lights_count']=sum(x['kind']=='light' for x in predictions)
            except Exception as exc:
                predictions=[];error=f'{type(exc).__name__}: {exc}'
            elapsed=time.perf_counter()-start
            with library.connect(mission) as conn:
                conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(run,row['id'],json.dumps(predictions),json.dumps(details),elapsed,error))
                conn.execute('UPDATE inference_runs SET done=? WHERE id=?',(number+1,run))
            print(json.dumps({'run':run,'done':number+1,'total':len(inputs),'detections':details.get('detections',0),'lights':details.get('lights_count',0),'elapsed':round(elapsed,3),'error':error}),flush=True)
        with library.connect(mission) as conn:
            failures=conn.execute('SELECT COUNT(*) FROM predictions WHERE inference_id=? AND error IS NOT NULL',(run,)).fetchone()[0]
            conn.execute('UPDATE inference_runs SET status=?,stage=?,finished=? WHERE id=?',('completed_with_errors' if failures else 'completed','finished',now(),run))
    except Exception as exc:
        with library.connect(mission) as conn:
            conn.execute("UPDATE inference_runs SET status='failed',stage='failed',error=?,finished=? WHERE id=?",(f'{type(exc).__name__}: {exc}',now(),run))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--mission',required=True);p.add_argument('--run',required=True);p.add_argument('--pack',required=True)
    args=p.parse_args();execute(Library(args.workspace),args.mission,args.run,args.pack)
