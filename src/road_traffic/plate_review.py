"""Local plate-region detection, image-chip extraction, and human crop approval.

This module intentionally does not recognize plate characters or digits.
"""
from __future__ import annotations
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .ort_session import create_session

MODEL_SHA256 = 'e5350bb2152eb44fa129b2438d1e39bbbf3c27accfe51cd893ce6e17d028d0b9'
MODEL_NAME = 'plate_yolox_tiny_640.onnx'


def _laplacian_variance(cv,np,rgb_crop):
    """Return a focus hint for review; no quality threshold is implied."""
    if rgb_crop is None or getattr(rgb_crop,'size',0)==0:
        return None
    gray=cv.cvtColor(rgb_crop,cv.COLOR_RGB2GRAY)
    value=float(np.var(cv.Laplacian(gray,cv.CV_64F)))
    return value if math.isfinite(value) else None


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _paths(library,mission):
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


def _mission_country(library,mission):
    with library.connect(mission) as conn:
        row=conn.execute('SELECT country FROM mission WHERE id=?',(mission,)).fetchone()
    return row['country'] if row else None


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
                model_sha256 TEXT NOT NULL, created TEXT NOT NULL,
                detector_config_sha256 TEXT NOT NULL DEFAULT '',
                mission_country_context TEXT,
                country_context_basis TEXT NOT NULL DEFAULT 'unknown');
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
        for name,declaration in (
            ('source_kind',"TEXT NOT NULL DEFAULT 'image'"),('video_source_id','TEXT'),
            ('frame_index','INTEGER'),('pts','INTEGER'),('time_base_num','INTEGER'),
            ('time_base_den','INTEGER'),('capture_utc','TEXT'),('context_file','TEXT'),
            ('detector_config_sha256',"TEXT NOT NULL DEFAULT ''"),
            ('blur_laplacian_variance','REAL'),('mission_country_context','TEXT'),
            ('country_context_basis',"TEXT NOT NULL DEFAULT 'unknown'")):
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
    if manifest.get('sha256')!=MODEL_SHA256 or actual!=MODEL_SHA256:
        raise ValueError('Plate detector checksum mismatch; no download fallback is allowed')
    return model,actual


def _plate_runtime_modules(*names):
    import importlib
    try:return [importlib.import_module(name) for name in names]
    except (ImportError,OSError) as exc:
        missing=f"module '{exc.name}'" if getattr(exc,'name',None) else str(exc)[:160]
        raise ValueError(f'The offline runtime is missing a required plate-detection dependency ({missing}). Repair or reinstall the app before running plate analysis.') from exc


class PlateDetector:
    """Single-class local plate-region detector. It never reads plate characters."""
    TILE_SIZE=768
    TILE_OVERLAP=128

    def __init__(self,pack,threshold=.18,*,tiled=True):
        cv2,np,ort=_plate_runtime_modules('cv2','numpy','onnxruntime')
        if not .01<=threshold<=.9:raise ValueError('Invalid plate detector threshold')
        if type(tiled) is not bool:raise ValueError('Plate tile scanning must be a boolean')
        self.cv,self.np=cv2,np
        self.model,self.sha=_model_pack(pack)
        self.session=create_session(ort,self.model,2)
        self.input=self.session.get_inputs()[0].name
        self.threshold,self.tiled=float(threshold),tiled
        config={'version':'plate-yolox-tiny-v1','model_sha256':self.sha,
            'threshold':self.threshold,'tiled':self.tiled}
        self.config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()

    @staticmethod
    def _tile_starts(length,tile,overlap):
        if length<=tile:return [0]
        step=tile-overlap
        starts=list(range(0,length-tile+1,step))
        last=length-tile
        if not starts or starts[-1]!=last:starts.append(last)
        return starts

    def _detect_single(self,image):
        """Run one full-image or high-resolution tile inference."""
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
            left,top=int(box[0]*w),int(box[1]*h)
            right,bottom=int((box[0]+box[2])*w),int((box[1]+box[3])*h)
            blur=_laplacian_variance(cv,np,image[top:bottom,left:right])
            boxes.append({'box':box,'confidence':score,'model_sha256':self.sha,
                'blur_laplacian_variance':blur})
            if len(boxes)>=200:break
        return boxes

    def detect(self,image):
        """Return normalized plate boxes, confidence and detector provenance.

        Large frames use overlapping 768-pixel tiles so small plates keep more
        source detail than after shrinking the entire frame to 640. Boxes are
        mapped to original-frame coordinates and overlapping duplicates are
        suppressed before crop review.
        """
        if not hasattr(image,'shape') or len(image.shape)!=3 or image.shape[2]!=3:
            raise ValueError('Plate detection requires an RGB image')
        h,w=image.shape[:2]
        if not self.tiled or max(h,w)<=self.TILE_SIZE:
            return self._detect_single(image)
        detections=[]
        xs=self._tile_starts(w,self.TILE_SIZE,self.TILE_OVERLAP)
        ys=self._tile_starts(h,self.TILE_SIZE,self.TILE_OVERLAP)
        for y in ys:
            for x in xs:
                tile=image[y:min(y+self.TILE_SIZE,h),x:min(x+self.TILE_SIZE,w)]
                tile_h,tile_w=tile.shape[:2]
                for item in self._detect_single(tile):
                    bx,by,bw,bh=item['box']
                    item=dict(item)
                    item['box']=[(bx*tile_w+x)/w,(by*tile_h+y)/h,
                        bw*tile_w/w,bh*tile_h/h]
                    detections.append(item)
        result=[]
        for item in sorted(detections,key=lambda row:row['confidence'],reverse=True):
            if any(_iou(item['box'],prior['box'])>.45 for prior in result):continue
            result.append(item)
            if len(result)>=200:break
        return result


def _iou(a,b):
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,right-left)*max(0,bottom-top)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-12)


def propose(library,mission,image_id,pack,threshold=.18,tiled=True):
    from PIL import Image
    import numpy as np
    source=library.image_path(mission,image_id)
    country=_mission_country(library,mission)
    with library.connect(mission) as conn:
        row=conn.execute('SELECT name,sha256 FROM images WHERE id=?',(image_id,)).fetchone()
    if not row:raise ValueError('Image not found in mission')
    _,folder,db=_paths(library,mission)
    detector=PlateDetector(pack,threshold,tiled=tiled)
    with Image.open(source) as im:
        im=im.convert('RGB');boxes=detector.detect(np.asarray(im));saved=[]
        for item in boxes:
            x,y,bw,bh=item['box'];crop_id=str(uuid.uuid4())
            coords=(max(0,int(x*im.width)),max(0,int(y*im.height)),
                min(im.width,int((x+bw)*im.width)),min(im.height,int((y+bh)*im.height)))
            if coords[2]<=coords[0] or coords[3]<=coords[1]:continue
            filename=f'{crop_id}.jpg';target=folder/filename;tmp=folder/(filename+'.tmp')
            im.crop(coords).save(tmp,format='JPEG',quality=95);tmp.replace(target)
            saved.append((crop_id,filename,target,item,coords))
    created=datetime.now(timezone.utc).isoformat()
    try:
        with _connect(db) as conn:
            existing=conn.execute('''SELECT id FROM crops WHERE image_id=? AND source_sha256=?
                AND model_sha256=? AND detector_config_sha256=?''',
                (image_id,row['sha256'],detector.sha,detector.config_sha256)).fetchone()
            if existing:
                for _,filename,*_ in saved:(folder/filename).unlink(missing_ok=True)
                return list_crops(library,mission,image_id)
            for crop_id,filename,path,item,coords in saved:
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,crop_sha256,
                    bbox,confidence,blur_laplacian_variance,model_sha256,created,detector_config_sha256,
                    mission_country_context,country_context_basis) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
                    crop_id,image_id,row['name'],row['sha256'],filename,_sha(path),
                    json.dumps({'x':coords[0],'y':coords[1],'width':coords[2]-coords[0],
                        'height':coords[3]-coords[1]}),item['confidence'],item.get('blur_laplacian_variance'),
                    detector.sha,created,detector.config_sha256,country,
                    'mission_configuration' if country else 'unknown'))
    except Exception:
        for _,_,path,*_ in saved:path.unlink(missing_ok=True)
        raise
    return list_crops(library,mission,image_id)


def propose_video_frame(library,mission,source_id,decoded,detector):
    """Save plate chips and their source-frame/PTS context without reading text."""
    from PIL import Image
    from . import video_store
    import numpy as np
    _,folder,db=_paths(library,mission);source=video_store.source_detail(library,mission,source_id)
    country=_mission_country(library,mission);image_id=f"video:{source_id}:{decoded.index}"
    with _connect(db) as conn:
        if conn.execute("""SELECT 1 FROM crops WHERE source_kind='video' AND video_source_id=?
            AND frame_index=? AND model_sha256=? AND detector_config_sha256=?""",
            (source_id,decoded.index,detector.sha,detector.config_sha256)).fetchone():return []
    frame=np.asarray(decoded.image);boxes=detector.detect(frame)
    origin=datetime.fromisoformat(source['recording_start_utc']) if source['recording_start_utc'] else None
    utc=decoded.utc_at(origin);created=datetime.now(timezone.utc).isoformat();saved=[]
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
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,crop_sha256,
                    bbox,confidence,blur_laplacian_variance,model_sha256,created,detector_config_sha256,
                    source_kind,video_source_id,
                    frame_index,pts,time_base_num,time_base_den,capture_utc,context_file,
                    mission_country_context,country_context_basis)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
                    crop_id,image_id,source['original_name'],source['sha256'],crop_name,_sha(crop_path),
                    json.dumps({'x':coords[0],'y':coords[1],'width':coords[2]-coords[0],
                        'height':coords[3]-coords[1]}),item['confidence'],item.get('blur_laplacian_variance'),
                    detector.sha,created,detector.config_sha256,'video',source_id,decoded.index,decoded.timestamp.pts,
                    decoded.timestamp.time_base_num,decoded.timestamp.time_base_den,utc.isoformat() if utc else None,
                    context_name,country,'mission_configuration' if country else 'unknown'))
    except Exception:
        for _,_,_,crop_path,context_path,*_ in saved:
            crop_path.unlink(missing_ok=True);context_path.unlink(missing_ok=True)
        raise
    return [item[0] for item in saved]


_CROP_FIELDS=('id','image_id','source_name','source_sha256','crop_file','crop_sha256','bbox','confidence',
    'model_sha256','created','detector_config_sha256','mission_country_context','country_context_basis','source_kind','video_source_id',
    'frame_index','pts','time_base_num','time_base_den','capture_utc','context_file','blur_laplacian_variance')


def _crop_query(where=''):
    fields=','.join('c.'+name for name in _CROP_FIELDS)
    return f'''SELECT {fields},r.decision,r.operator,r.revision FROM crops c LEFT JOIN reviews r ON r.id=(
        SELECT id FROM reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1){where}'''


def list_crops(library,mission,image_id=None):
    _,_,db=_paths(library,mission);where=' WHERE c.image_id=?' if image_id else ''
    params=(image_id,) if image_id else ()
    with _connect(db) as conn:
        return [dict(row) for row in conn.execute(_crop_query(where)+' ORDER BY c.created,c.source_name,c.id',params).fetchall()]


def list_crops_page(library,mission,offset=0,limit=40,image_id=None):
    """Fetch one bounded crop-review page and its filtered total count."""
    if isinstance(offset,bool) or not isinstance(offset,int) or offset<0:
        raise ValueError('Plate page offset must be a nonnegative integer')
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=100:
        raise ValueError('Plate page size must be between 1 and 100')
    _,_,db=_paths(library,mission);where=' WHERE c.image_id=?' if image_id else ''
    params=(image_id,) if image_id else ()
    with _connect(db) as conn:
        total=conn.execute('SELECT COUNT(*) FROM crops c'+where,params).fetchone()[0]
        items=[dict(row) for row in conn.execute(_crop_query(where)+' ORDER BY c.created,c.source_name,c.id LIMIT ? OFFSET ?',
            [*params,limit,offset]).fetchall()]
    return {'items':items,'offset':offset,'limit':limit,'total':total}


PLATE_EXPORT_FIELDS=('crop_id','mission_country_context','country_context_basis','source_kind','source_name',
    'source_sha256','image_id','video_source_id','frame_index','pts','time_base_num','time_base_den','capture_utc',
    'bbox','detector_confidence','blur_laplacian_variance','detector_model_sha256','detector_config_sha256','crop_file','crop_sha256',
    'crop_review_decision','crop_review_revision','crop_review_operator','crop_reviewed_at')


def export_crops_csv(library,mission):
    """Write an atomic UTF-8 crop-review snapshot without overwriting prior exports."""
    _,folder,db=_paths(library,mission);export_id=str(uuid.uuid4())
    target=folder/f'plate-review-{mission[:8]}-{export_id}.csv';temporary=folder/f'.{target.name}.tmp'
    query='''SELECT c.id AS crop_id,c.mission_country_context,c.country_context_basis,c.source_kind,c.source_name,
        c.source_sha256,c.image_id,c.video_source_id,c.frame_index,c.pts,c.time_base_num,c.time_base_den,
        c.capture_utc,c.bbox,c.confidence AS detector_confidence,c.blur_laplacian_variance,
        c.model_sha256 AS detector_model_sha256,c.detector_config_sha256,c.crop_file,c.crop_sha256,
        r.decision AS crop_review_decision,r.revision AS crop_review_revision,r.operator AS crop_review_operator,
        r.created AS crop_reviewed_at FROM crops c LEFT JOIN reviews r ON r.id=(
        SELECT id FROM reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)
        ORDER BY c.created,c.source_name,c.id'''
    count=0
    try:
        with _connect(db) as conn,temporary.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=PLATE_EXPORT_FIELDS,extrasaction='ignore');writer.writeheader()
            for raw in conn.execute(query):
                row=dict(raw)
                for key,value in row.items():
                    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):row[key]="'"+value
                writer.writerow(row);count+=1
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
        try:
            fd=os.open(folder,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        except OSError:pass
        return {'id':export_id,'path':str(target),'row_count':count,'sha256':_sha(target),'format':'csv'}
    except BaseException:
        temporary.unlink(missing_ok=True);target.unlink(missing_ok=True);raise


def crop_file(library,mission,crop_id):
    _,folder,db=_paths(library,mission)
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
