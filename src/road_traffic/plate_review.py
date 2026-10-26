"""Plate-region proposals, crop files, and append-only human review.

This module intentionally has no OCR dependency, text field, or recognition step.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODEL_SHA256 = 'e5350bb2152eb44fa129b2438d1e39bbbf3c27accfe51cd893ce6e17d028d0b9'
MODEL_NAME = 'plate_yolox_tiny_640.onnx'


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _paths(library, mission):
    settings=library.settings(mission)
    if settings['input_type'] not in ('images','videos') or settings['analysis_mode']!='license_plate':
        raise ValueError('Plate crop review requires a plate-crop mission')
    if not settings['output_root']:
        raise ValueError('Set the mission output folder before saving plate crops')
    root=Path(library._configured_directory(settings['output_root'],'output',writable=True))
    if library.root.is_relative_to(root) or root.is_relative_to(library.root):
        raise ValueError('Mission data and crop output folders must be separate')
    parent=root/'plate-crops'
    if parent.is_symlink() or (hasattr(parent,'is_junction') and parent.is_junction()):
        raise ValueError('Plate crop output cannot use linked folders')
    parent.mkdir(exist_ok=True)
    folder=parent/mission
    if folder.is_symlink() or (hasattr(folder,'is_junction') and folder.is_junction()):
        raise ValueError('Plate crop output cannot use linked folders')
    folder.mkdir(exist_ok=True)
    db=folder/'review.sqlite3'
    if db.is_symlink() or (hasattr(db,'is_junction') and db.is_junction()):
        raise ValueError('Plate crop review database cannot use a linked file')
    return root,folder,db


def _open_connection(db):
    conn=sqlite3.connect(db,timeout=20)
    try:
        conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS crops(
                id TEXT PRIMARY KEY, image_id TEXT NOT NULL, source_name TEXT NOT NULL,
                source_sha256 TEXT NOT NULL, crop_file TEXT NOT NULL UNIQUE,
                crop_sha256 TEXT NOT NULL, bbox TEXT NOT NULL, confidence REAL NOT NULL,
                model_sha256 TEXT NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reviews(
                id INTEGER PRIMARY KEY AUTOINCREMENT, crop_id TEXT NOT NULL REFERENCES crops(id),
                revision INTEGER NOT NULL, decision TEXT NOT NULL CHECK(decision IN ('approved','rejected')),
                operator TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(crop_id,revision));
            CREATE INDEX IF NOT EXISTS reviews_latest ON reviews(crop_id,revision DESC);
            CREATE TRIGGER IF NOT EXISTS immutable_plate_reviews_update BEFORE UPDATE ON reviews
                BEGIN SELECT RAISE(ABORT,'plate crop reviews are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_plate_reviews_delete BEFORE DELETE ON reviews
                BEGIN SELECT RAISE(ABORT,'plate crop reviews are append-only'); END;
        ''')
        columns={row['name'] for row in conn.execute('PRAGMA table_info(crops)')}
        for name, declaration in (
            ('source_kind', "TEXT NOT NULL DEFAULT 'image'"), ('video_source_id','TEXT'),
            ('frame_index','INTEGER'), ('pts','INTEGER'), ('time_base_num','INTEGER'),
            ('time_base_den','INTEGER'), ('capture_utc','TEXT'), ('context_file','TEXT')):
            if name not in columns:conn.execute(f'ALTER TABLE crops ADD COLUMN {name} {declaration}')
        return conn
    except BaseException:
        conn.close()
        raise


@contextmanager
def _connect(db):
    conn=_open_connection(db)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _model_pack(pack):
    pack=Path(pack).resolve(strict=True)
    manifest=json.loads((pack/'manifest.json').read_text())
    model=pack/MODEL_NAME
    actual=_sha(model)
    expected=manifest.get('sha256')
    if expected!=MODEL_SHA256 or actual!=MODEL_SHA256:
        raise ValueError('Plate detector checksum mismatch; no download fallback is allowed')
    return model,actual


class PlateDetector:
    def __init__(self,pack,threshold=.3):
        import cv2
        import numpy as np
        import onnxruntime as ort
        if not .01<=threshold<=.9:raise ValueError('Invalid plate detector threshold')
        self.cv,self.np=cv2,np
        self.model,self.sha=_model_pack(pack)
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.model),sess_options=opts,providers=['CPUExecutionProvider'])
        if self.session.get_providers()!=['CPUExecutionProvider']:
            raise RuntimeError('Plate detector must run on the validated CPU provider')
        self.input=self.session.get_inputs()[0].name
        self.threshold=threshold
        self.grids=[]
        for stride in (8,16,32):
            y,x=np.meshgrid(np.arange(640//stride),np.arange(640//stride),indexing='ij')
            self.grids.append(np.stack([x,y],axis=-1).reshape(-1,2))
        self.grid=np.concatenate(self.grids).astype(np.float32)
        self.strides=np.concatenate([np.full((len(g),1),s,dtype=np.float32) for g,s in zip(self.grids,(8,16,32))])

    def detect(self, image):
        """Return only normalized plate boxes, confidence and detector provenance."""
        cv,np=self.cv,self.np
        h,w=image.shape[:2]
        scale=min(640/w,640/h)
        rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
        resized=cv.resize(image,(rw,rh),interpolation=cv.INTER_LINEAR)
        canvas=np.full((640,640,3),114,dtype=np.float32)
        px=(640-rw)//2;py=(640-rh)//2
        canvas[py:py+rh,px:px+rw]=resized
        tensor=np.ascontiguousarray(canvas.transpose(2,0,1)[None])
        raw=np.asarray(self.session.run(None,{self.input:tensor})[0])
        if raw.shape not in ((1,8400,6),(8400,6)):
            raise ValueError(f'Unexpected plate detector output schema: {raw.shape}')
        raw=raw[0] if raw.ndim==3 else raw
        # Exported YOLOX graph provides decoded pixel-space center boxes.
        scores=raw[:,4]*raw[:,5]
        boxes=[]
        for index in np.argsort(scores)[::-1]:
            row=raw[index];score=float(scores[index])
            if not math.isfinite(score) or score<self.threshold:break
            if not np.isfinite(row[:4]).all():continue
            cx,cy,bw,bh=map(float,row[:4])
            x1=max(0,min(w,(cx-bw/2-px)/scale));y1=max(0,min(h,(cy-bh/2-py)/scale))
            x2=max(0,min(w,(cx+bw/2-px)/scale));y2=max(0,min(h,(cy+bh/2-py)/scale))
            if x2<=x1 or y2<=y1:continue
            box=[x1/w,y1/h,(x2-x1)/w,(y2-y1)/h]
            if any(_iou(box,old['box'])>.45 for old in boxes):continue
            boxes.append({'box':box,'confidence':score,'model_sha256':self.sha})
            if len(boxes)>=200:break
        return boxes


def _iou(a,b):
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,right-left)*max(0,bottom-top)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-12)


def propose(library,mission,image_id,pack,threshold=.3):
    from PIL import Image
    source=library.image_path(mission,image_id)
    with library.connect(mission) as conn:
        row=conn.execute('SELECT name,sha256 FROM images WHERE id=?',(image_id,)).fetchone()
    if not row:raise ValueError('Image not found in mission')
    root,folder,db=_paths(library,mission)
    detector=PlateDetector(pack,threshold)
    with Image.open(source) as im:
        im=im.convert('RGB')
        boxes=detector.detect(__import__('numpy').asarray(im))
        crops=[]
        for item in boxes:
            x,y,bw,bh=item['box'];crop_id=str(uuid.uuid4())
            coords=(max(0,int(x*im.width)),max(0,int(y*im.height)),min(im.width,int((x+bw)*im.width)),min(im.height,int((y+bh)*im.height)))
            if coords[2]<=coords[0] or coords[3]<=coords[1]:continue
            filename=f'{crop_id}.jpg';target=folder/filename;tmp=folder/(filename+'.tmp')
            im.crop(coords).save(tmp,format='JPEG',quality=95)
            tmp.replace(target)
            crops.append((crop_id,filename,target,item,coords))
    created=datetime.now(timezone.utc).isoformat()
    inserted=[]
    try:
        with _connect(db) as conn:
            existing=conn.execute('SELECT id FROM crops WHERE image_id=? AND source_sha256=? AND model_sha256=?',(image_id,row['sha256'],detector.sha)).fetchall()
            if existing:
                for item in crops:(folder/item[1]).unlink(missing_ok=True)
                return list_crops(library,mission,image_id)
            for crop_id,filename,path,item,coords in crops:
                box=item['box']
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                    crop_sha256,bbox,confidence,model_sha256,created) VALUES (?,?,?,?,?,?,?,?,?,?)''',(
                    crop_id,image_id,row['name'],row['sha256'],filename,_sha(path),
                    json.dumps({'x':coords[0],'y':coords[1],'width':coords[2]-coords[0],'height':coords[3]-coords[1]}),
                    item['confidence'],detector.sha,created))
                inserted.append(crop_id)
    except Exception:
        for crop_id,filename,path,item,coords in crops:path.unlink(missing_ok=True)
        raise
    return list_crops(library,mission,image_id)


def propose_video_frame(library,mission,source_id,decoded,detector):
    """Save plate-region crops from one decoded frame, without reading plate text."""
    from PIL import Image
    from . import video_store
    import numpy as np
    _,folder,db=_paths(library,mission)
    source=video_store.source_detail(library,mission,source_id)
    image_id=f"video:{source_id}:{decoded.index}"
    with _connect(db) as conn:
        existing=conn.execute('''SELECT id FROM crops WHERE source_kind='video'
            AND video_source_id=? AND frame_index=? AND model_sha256=?''',
            (source_id,decoded.index,detector.sha)).fetchone()
        if existing:return []
    frame=np.asarray(decoded.image)
    boxes=detector.detect(frame)
    origin=datetime.fromisoformat(source['recording_start_utc']) if source['recording_start_utc'] else None
    utc=decoded.utc_at(origin)
    created=datetime.now(timezone.utc).isoformat()
    saved=[]
    try:
        with Image.fromarray(frame).convert('RGB') as image:
            for item in boxes:
                x,y,bw,bh=item['box'];crop_id=str(uuid.uuid4())
                coords=(max(0,int(x*image.width)),max(0,int(y*image.height)),
                    min(image.width,int((x+bw)*image.width)),min(image.height,int((y+bh)*image.height)))
                if coords[2]<=coords[0] or coords[3]<=coords[1]:continue
                crop_name=f'{crop_id}.jpg';context_name=f'{crop_id}-context.jpg'
                crop_path=folder/crop_name;context_path=folder/context_name
                image.crop(coords).save(crop_path,format='JPEG',quality=95)
                image.save(context_path,format='JPEG',quality=88)
                saved.append((crop_id,crop_name,context_name,crop_path,context_path,item,coords))
        with _connect(db) as conn:
            for crop_id,crop_name,context_name,crop_path,context_path,item,coords in saved:
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                    crop_sha256,bbox,confidence,model_sha256,created,source_kind,video_source_id,
                    frame_index,pts,time_base_num,time_base_den,capture_utc,context_file)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
                    crop_id,image_id,source['original_name'],source['sha256'],crop_name,_sha(crop_path),
                    json.dumps({'x':coords[0],'y':coords[1],'width':coords[2]-coords[0],
                        'height':coords[3]-coords[1]}),item['confidence'],detector.sha,created,'video',
                    source_id,decoded.index,decoded.timestamp.pts,decoded.timestamp.time_base_num,
                    decoded.timestamp.time_base_den,utc.isoformat() if utc else None,context_name))
    except Exception:
        for _,_,_,crop_path,context_path,_,_ in saved:
            crop_path.unlink(missing_ok=True);context_path.unlink(missing_ok=True)
        raise
    return [crop_id for crop_id,*_ in saved]


def list_crops(library,mission,image_id=None):
    root,folder,db=_paths(library,mission)
    with _connect(db) as conn:
        sql='''SELECT c.*,r.decision,r.operator,r.revision FROM crops c LEFT JOIN reviews r ON r.id=(
            SELECT id FROM reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)'''
        params=()
        if image_id:sql+=' WHERE c.image_id=?';params=(image_id,)
        sql+=' ORDER BY c.created,c.source_name,c.id'
        return [dict(row) for row in conn.execute(sql,params).fetchall()]


def crop_file(library,mission,crop_id):
    root,folder,db=_paths(library,mission)
    with _connect(db) as conn:row=conn.execute('SELECT crop_file,crop_sha256 FROM crops WHERE id=?',(crop_id,)).fetchone()
    if not row:raise ValueError('Plate crop not found in mission')
    name=Path(row['crop_file'])
    if name.name!=row['crop_file'] or name.suffix.lower()!='.jpg':raise ValueError('Invalid plate crop path')
    candidate=folder/name
    if candidate.is_symlink() or (hasattr(candidate,'is_junction') and candidate.is_junction()):raise ValueError('Plate crop path escapes its folder')
    path=candidate.resolve(strict=True)
    if not path.is_relative_to(folder.resolve(strict=True)):raise ValueError('Plate crop path escapes its folder')
    if _sha(path)!=row['crop_sha256']:raise ValueError('Saved plate crop checksum mismatch')
    return path


def context_file(library,mission,crop_id):
    _,folder,db=_paths(library,mission)
    with _connect(db) as conn:row=conn.execute('SELECT context_file FROM crops WHERE id=?',(crop_id,)).fetchone()
    if not row or not row['context_file']:raise ValueError('Plate crop context frame not found in mission')
    name=Path(row['context_file'])
    if name.name!=row['context_file'] or name.suffix.lower()!='.jpg':raise ValueError('Invalid plate context path')
    candidate=folder/name
    if candidate.is_symlink() or (hasattr(candidate,'is_junction') and candidate.is_junction()):
        raise ValueError('Plate context path cannot use linked files')
    path=candidate.resolve(strict=True)
    if not path.is_relative_to(folder.resolve(strict=True)):raise ValueError('Plate context path escapes its folder')
    return path


def review(library,mission,crop_id,expected_revision,decision,operator):
    if decision not in ('approved','rejected'):raise ValueError('Choose approved or rejected')
    if not isinstance(operator,str) or not 1<=len(operator.strip())<=120:raise ValueError('Operator must contain 1–120 characters')
    _,_,db=_paths(library,mission)
    with _connect(db) as conn:
        if not conn.execute('SELECT 1 FROM crops WHERE id=?',(crop_id,)).fetchone():raise ValueError('Plate crop not found in mission')
        row=conn.execute('SELECT revision FROM reviews WHERE crop_id=? ORDER BY revision DESC LIMIT 1',(crop_id,)).fetchone()
        current=row['revision'] if row else 0
        if expected_revision!=current:raise ValueError('Plate crop changed since it was opened; reload before reviewing')
        revision=current+1
        conn.execute('INSERT INTO reviews(crop_id,revision,decision,operator,created) VALUES (?,?,?,?,?)',
            (crop_id,revision,decision,operator.strip(),datetime.now(timezone.utc).isoformat()))
    return {'crop_id':crop_id,'revision':revision,'decision':decision}
