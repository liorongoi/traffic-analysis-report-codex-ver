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
        from .regions import filter_proposals
        head=None
        if config.get('night_head_id'):
            from .learning import load_head
            head,_=load_head(library,mission,config['night_head_id'],config['night_head_hash'])
        detector=VehicleDetector(pack,config['threshold'],config['tiled'],config.get('cpu_threads',2))
        if config['adapter']!=VERSION:raise ValueError('Adapter version changed; start a new run')
        if detector.sha!=config['model_sha256']:raise ValueError('Model changed after run approval')
        with library.connect(mission) as conn:
            inputs=[dict(r) for r in conn.execute('''SELECT images.*,COALESCE(iv.view_id,'') view_id,
                COALESCE(iv.view_revision,0) view_revision FROM images
                JOIN inference_inputs ii ON images.id=ii.image_id
                LEFT JOIN inference_input_views iv ON iv.inference_id=ii.inference_id AND iv.image_id=ii.image_id
                WHERE ii.inference_id=? ORDER BY capture_utc,name''',(run,))]
            heldout={r[0] for r in conn.execute("SELECT DISTINCT image_id FROM reviews WHERE role IN ('validation','test')")}
            completed={r[0] for r in conn.execute('SELECT image_id FROM predictions WHERE inference_id=?',(run,))}
        pending_inputs=[row for row in inputs if row['id'] not in completed]
        if head and any(json.loads(r['provenance']).get('preset')!='fintraffic' or Path(r['name']).stem.split('_')[0]!=head.data.get('view') for r in inputs):
            raise ValueError('Selected light correction head does not match every source view; use an appropriate mission/view or select None')
        def load(row):
            with Image.open(library.image_path(mission,row['id'])) as image:
                if image.width*image.height>24_000_000:raise ValueError('Image exceeds this adapter pixel budget')
                return np.asarray(image.convert('RGB'))
        # Background references stay in separate manual/source camera views.
        def row_view_key(row):
            if row.get('view_id'):
                return 'manual:'+row['view_id']
            provenance=json.loads(row['provenance'])
            if provenance.get('preset')=='fintraffic':
                return 'fintraffic:'+Path(row['name']).stem.split('_')[0]
            return None
        with library.connect(mission) as conn:
            if conn.execute("SELECT status FROM inference_runs WHERE id=?",(run,)).fetchone()['status']!='running':return
            conn.execute("UPDATE inference_runs SET stage='building_light_background' WHERE id=?",(run,))
        reference_folder=library.location(mission)/'inference'/run
        reference_folder.mkdir(parents=True,exist_ok=True)
        from .light_background import build_view_backgrounds,load_background
        def stopped():
            with library.connect(mission) as conn:
                state=conn.execute('SELECT status,stop_requested FROM inference_runs WHERE id=?',(run,)).fetchone()
            return not state or state['status']!='running' or bool(state['stop_requested'])
        if config.get('night_lights'):
            backgrounds=build_view_backgrounds(inputs,heldout,row_view_key,load,reference_folder,
                stopped,cv2_module=cv2,light_mask_fn=light_mask,illumination_fn=illumination)
        else:
            backgrounds={}
        pending_by_view={}
        for row in pending_inputs:
            pending_by_view.setdefault(row_view_key(row),[]).append(row)
        ordered_pending=[row for group in pending_by_view.values() for row in group]
        number=len(completed)
        active_view_key=object()
        for row in ordered_pending:
            view_key=row_view_key(row)
            if view_key!=active_view_key:
                active_view_key=view_key
                background_info=backgrounds.get(view_key)
                background_data=load_background(background_info) if background_info else None
                refs=background_info['references'] if background_info else []
                refindex={ref['image_id']:i for i,ref in enumerate(refs)}
                counts=background_data['counts'] if background_data else None
                median=background_data['median'] if background_data else None
                mad=background_data['mad'] if background_data else None
                stack=background_data['stack'] if background_data else None
                masks=background_data['masks'] if background_data else None
                anchor=background_data['anchor'] if background_data else None
                shape=median.shape if median is not None else None
            background=({**background_info,'path':str(background_info['path'].relative_to(library.location(mission)))}
                if background_info else None)
            with library.connect(mission) as conn:
                state=conn.execute('SELECT status,stop_requested FROM inference_runs WHERE id=?',(run,)).fetchone()
                if not state or state['status']!='running':return
                if state['stop_requested']:
                    conn.execute("UPDATE inference_runs SET status='cancelled',stage='cancelled',finished=? WHERE id=?",(now(),run));return
                conn.execute("UPDATE inference_runs SET stage='detecting' WHERE id=?",(run,))
            start=time.perf_counter()
            details={'adapter':VERSION,'model_sha256':detector.sha,'device':'CPUExecutionProvider','cpu_threads':detector.cpu_threads,
                'resource_profile':config.get('resource_profile'),'input_sha256':row['sha256'],
                'threshold':config['threshold'],'tiled':config['tiled'],'light_background':background,
                'view_id':row.get('view_id') or None,'view_assignment_revision':row.get('view_revision',0),
                'human_confirmation':False,'confidence_note':'Detector scores are not calibrated accuracy.'}
            predictions=[];error=None
            try:
                rgb=load(row);h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                details.update({'width':w,'height':h,**illumination(rgb)})
                predictions=detector.infer(rgb)
                if config.get('night_lights') and details['low_light']:
                    persistent=None;light_context=None;usable=counts is not None and gray.shape==shape
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
                view_regions=config.get('view_regions_snapshot',{}).get(row.get('view_id') or '',[])
                predictions,region_summary=filter_proposals(predictions,view_regions)
                details['view_regions']={'regions':[{'id':r['id'],'revision':r['revision'],'kind':r['kind']} for r in view_regions],
                    'summary':region_summary,'policy':'exclusions override inclusion polygons'}
                for item in predictions:item['prediction_id']=str(uuid.uuid4())
                details['detections']=sum(x['evidence']=='yolox-s' for x in predictions)
                details['lights_count']=sum(x['kind']=='light' for x in predictions)
            except Exception as exc:
                predictions=[];error=f'{type(exc).__name__}: {exc}'
            elapsed=time.perf_counter()-start
            with library.connect(mission) as conn:
                conn.execute('BEGIN IMMEDIATE')
                state=conn.execute('SELECT status FROM inference_runs WHERE id=?',(run,)).fetchone()
                if not state or state['status']!='running':
                    conn.rollback();return
                conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?)',(run,row['id'],json.dumps(predictions),json.dumps(details),elapsed,error))
                conn.execute('UPDATE inference_runs SET done=? WHERE id=?',(number+1,run))
            number+=1
            print(json.dumps({'run':run,'done':number,'total':len(inputs),'detections':details.get('detections',0),'lights':details.get('lights_count',0),'elapsed':round(elapsed,3),'error':error}),flush=True)
        with library.connect(mission) as conn:
            failures=conn.execute('SELECT COUNT(*) FROM predictions WHERE inference_id=? AND error IS NOT NULL',(run,)).fetchone()[0]
            conn.execute("UPDATE inference_runs SET status=?,stage='finished',finished=? WHERE id=? AND status='running'",
                ('completed_with_errors' if failures else 'completed',now(),run))
    except Exception as exc:
        with library.connect(mission) as conn:
            conn.execute("UPDATE inference_runs SET status='failed',stage='failed',error=?,finished=? WHERE id=?",(f'{type(exc).__name__}: {exc}',now(),run))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--mission',required=True);p.add_argument('--run',required=True);p.add_argument('--pack',required=True)
    args=p.parse_args();execute(Library(args.workspace),args.mission,args.run,args.pack)
