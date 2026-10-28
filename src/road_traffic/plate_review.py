"""Plate localization, chip extraction, OCR suggestions, and append-only human review.

OCR is a local proposal only. The recognizer supports Latin plus Arabic/Persian
letters, while Persian plate digits and target-country plate accuracy remain unverified.
Operators separately accept, correct, reject, or mark suggestions unreadable.
"""
from __future__ import annotations
from contextlib import contextmanager
import csv
import hashlib
import importlib
import json
import os
from .ort_session import create_session
import math
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODEL_SHA256 = 'e5350bb2152eb44fa129b2438d1e39bbbf3c27accfe51cd893ce6e17d028d0b9'
MODEL_NAME = 'plate_yolox_tiny_640.onnx'
OCR_MODEL_NAME = 'plate_ocr_arabic_ppocrv3_mobile.onnx'
OCR_DICT_NAME = 'plate_ocr_arabic_dict.json'
OCR_MODEL_SHA256 = '0394ca6c66c72d882b199aaeff34b3b448328592e14a0f95b36a398a339b2142'
OCR_DICT_SHA256 = '45cb8e46aded3f9680d1b382676d17ea163ef8072da93b2d3fd349cf68520fb5'
OCR_GLOBAL_MODEL_NAME = 'plate_ocr_global_cct_s_v2.onnx'
OCR_GLOBAL_PLATE_CONFIG_NAME = 'plate_ocr_global_cct_s_v2_plate_config.yaml'
OCR_GLOBAL_MODEL_CONFIG_NAME = 'plate_ocr_global_cct_s_v2_model_config.yaml'
OCR_GLOBAL_MODEL_SHA256 = '384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b'
OCR_GLOBAL_PLATE_CONFIG_SHA256 = '0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6'
OCR_GLOBAL_MODEL_CONFIG_SHA256 = '244c3c712205750e9747bca4707cd553a84005f41959745bff676b0fca8faba3'


def _plate_runtime_modules(role,*names):
    try:
        return [importlib.import_module(name) for name in names]
    except (ImportError,OSError) as exc:
        missing=f"module '{exc.name}'" if getattr(exc,'name',None) else str(exc)[:160]
        raise ValueError(f'The offline runtime is missing a required {role} dependency ({missing}). Repair or reinstall the app before running plate analysis.') from exc


def _laplacian_variance(cv,np,rgb_crop):
    """Return the raw focus score used as a review hint; no quality threshold is implied."""
    if rgb_crop is None or getattr(rgb_crop,'size',0)==0:
        return None
    gray=cv.cvtColor(rgb_crop,cv.COLOR_RGB2GRAY)
    value=float(np.var(cv.Laplacian(gray,cv.CV_64F)))
    return value if math.isfinite(value) else None

_PLATE_DIGITS = str.maketrans({
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
})


def normalize_plate_text(value):
    """Build a comparison key while leaving original OCR and reviewed text untouched.

    Unicode compatibility forms and Arabic/Persian digit shapes are folded for
    search. Letter order and leading zeros are preserved; separators are omitted.
    """
    if not isinstance(value, str):
        return ''
    normalized = unicodedata.normalize('NFKC', value).translate(_PLATE_DIGITS).casefold()
    return ''.join(char for char in normalized
        if unicodedata.category(char) not in ('Cf', 'Cc') and char.isalnum())


def _global_latin_ocr_applicable(country):
    """Avoid presenting a Latin-only alternative as useful for Iranian-script plates."""
    value=unicodedata.normalize('NFKC',str(country or '')).strip().casefold()
    return value not in ('iran','ir','irn','ایران','إيران')


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
                model_sha256 TEXT NOT NULL, created TEXT NOT NULL,
                ocr_text TEXT NOT NULL DEFAULT '', ocr_confidence REAL NOT NULL DEFAULT 0,
                ocr_model_sha256 TEXT NOT NULL DEFAULT '',
                ocr_search_text TEXT NOT NULL DEFAULT '');
            CREATE TABLE IF NOT EXISTS reviews(
                id INTEGER PRIMARY KEY AUTOINCREMENT, crop_id TEXT NOT NULL REFERENCES crops(id),
                revision INTEGER NOT NULL, decision TEXT NOT NULL CHECK(decision IN ('approved','rejected')),
                operator TEXT NOT NULL, created TEXT NOT NULL, UNIQUE(crop_id,revision));
            CREATE TABLE IF NOT EXISTS text_reviews(
                id INTEGER PRIMARY KEY AUTOINCREMENT, crop_id TEXT NOT NULL REFERENCES crops(id),
                revision INTEGER NOT NULL, decision TEXT NOT NULL CHECK(decision IN ('accepted','corrected','rejected','unreadable')),
                verified_text TEXT NOT NULL DEFAULT '', normalized_text TEXT NOT NULL DEFAULT '',
                operator TEXT NOT NULL, created TEXT NOT NULL,
                UNIQUE(crop_id,revision));
            CREATE INDEX IF NOT EXISTS text_reviews_latest ON text_reviews(crop_id,revision DESC);
            CREATE TRIGGER IF NOT EXISTS immutable_plate_text_reviews_update BEFORE UPDATE ON text_reviews
                BEGIN SELECT RAISE(ABORT,'plate text reviews are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_plate_text_reviews_delete BEFORE DELETE ON text_reviews
                BEGIN SELECT RAISE(ABORT,'plate text reviews are append-only'); END;
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
            ('time_base_den','INTEGER'), ('capture_utc','TEXT'), ('context_file','TEXT'),
            ('blur_laplacian_variance','REAL'),
            ('ocr_text', "TEXT NOT NULL DEFAULT ''"), ('ocr_confidence','REAL NOT NULL DEFAULT 0'),
            ('ocr_model_sha256', "TEXT NOT NULL DEFAULT ''"),
            ('ocr_search_text', "TEXT NOT NULL DEFAULT ''"),
            ('ocr_alternative_text', "TEXT NOT NULL DEFAULT ''"),
            ('ocr_alternative_confidence', 'REAL NOT NULL DEFAULT 0'),
            ('ocr_alternative_model_sha256', "TEXT NOT NULL DEFAULT ''"),
            ('ocr_alternative_search_text', "TEXT NOT NULL DEFAULT ''")):
            if name not in columns:conn.execute(f'ALTER TABLE crops ADD COLUMN {name} {declaration}')
        text_columns={row['name'] for row in conn.execute('PRAGMA table_info(text_reviews)')}
        if 'normalized_text' not in text_columns:
            conn.execute("ALTER TABLE text_reviews ADD COLUMN normalized_text TEXT NOT NULL DEFAULT ''")
            # This migration adds a derived index only; user decisions and source
            # text remain unchanged. Recreate the append-only guard around it.
            conn.execute('DROP TRIGGER IF EXISTS immutable_plate_text_reviews_update')
            for row in conn.execute('SELECT id,verified_text FROM text_reviews').fetchall():
                conn.execute('UPDATE text_reviews SET normalized_text=? WHERE id=?',
                    (normalize_plate_text(row['verified_text']),row['id']))
            conn.execute('''CREATE TRIGGER immutable_plate_text_reviews_update BEFORE UPDATE ON text_reviews
                BEGIN SELECT RAISE(ABORT,'plate text reviews are append-only'); END''')
        for row in conn.execute("SELECT id,ocr_text FROM crops WHERE ocr_search_text='' AND ocr_text<>''").fetchall():
            conn.execute('UPDATE crops SET ocr_search_text=? WHERE id=?',
                (normalize_plate_text(row['ocr_text']),row['id']))
        for row in conn.execute("SELECT id,ocr_alternative_text FROM crops WHERE ocr_alternative_search_text='' AND ocr_alternative_text<>''").fetchall():
            conn.execute('UPDATE crops SET ocr_alternative_search_text=? WHERE id=?',
                (normalize_plate_text(row['ocr_alternative_text']),row['id']))
        conn.execute('CREATE INDEX IF NOT EXISTS crops_ocr_search ON crops(ocr_search_text)')
        conn.execute('CREATE INDEX IF NOT EXISTS crops_ocr_alternative_search ON crops(ocr_alternative_search_text)')
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


class PlateTextRecognizer:
    """CPU CTC recognizer for Latin and Arabic/Persian letters; outputs need review."""
    def __init__(self,pack):
        np,ort=_plate_runtime_modules('plate OCR','numpy','onnxruntime')
        pack=Path(pack).resolve(strict=True)
        model=pack/OCR_MODEL_NAME;dictionary=pack/OCR_DICT_NAME
        if _sha(model)!=OCR_MODEL_SHA256 or _sha(dictionary)!=OCR_DICT_SHA256:
            raise ValueError('Plate OCR model or matched dictionary checksum mismatch')
        manifest=json.loads((pack/'manifest.json').read_text())
        artifacts=manifest.get('ocr_artifacts',{})
        if artifacts.get(OCR_MODEL_NAME,{}).get('sha256')!=OCR_MODEL_SHA256 or artifacts.get(OCR_DICT_NAME,{}).get('sha256')!=OCR_DICT_SHA256:
            raise ValueError('Plate OCR manifest does not match the pinned recognizer and dictionary')
        chars=json.loads(dictionary.read_text(encoding='utf-8'))
        if not isinstance(chars,list) or len(chars)!=161 or not all(isinstance(c,str) and len(c)==1 for c in chars):
            raise ValueError('Plate OCR character dictionary is invalid')
        self.charset=['']+chars+[' ']
        self.session=create_session(ort,model,2)
        inputs=self.session.get_inputs();outputs=self.session.get_outputs()
        if len(inputs)!=1 or len(outputs)!=1:
            raise ValueError('Unexpected plate OCR model input/output schema')
        self.input=inputs[0].name;self.output=outputs[0].name
        self.sha=OCR_MODEL_SHA256;self.np=np

    def read(self,image):
        from PIL import Image
        np=self.np;image=image.convert('RGB');width,height=image.size
        if width<1 or height<1:raise ValueError('Cannot read an empty plate crop')
        resized_width=max(8,min(320,int(round(width*48/height))//8*8))
        resized=image.resize((resized_width,48),Image.Resampling.BILINEAR)
        tensor=np.asarray(resized,dtype=np.float32)[...,::-1].transpose(2,0,1)[None]
        tensor=np.ascontiguousarray((tensor/255.0-0.5)/0.5)
        raw=np.asarray(self.session.run([self.output],{self.input:tensor})[0])
        if raw.ndim==3:
            if raw.shape[0]!=1:raise ValueError(f'Unexpected plate OCR batch output: {raw.shape}')
            raw=raw[0]
        if raw.ndim!=2 or raw.shape[-1]!=len(self.charset):
            raise ValueError(f'Unexpected plate OCR output schema: {raw.shape}')
        sums=raw.sum(axis=-1,keepdims=True)
        if not (np.all(raw>=-1e-5) and np.all(raw<=1.00001) and np.allclose(sums,1,atol=1e-3)):
            shifted=raw-raw.max(axis=-1,keepdims=True);exp=np.exp(shifted);raw=exp/exp.sum(axis=-1,keepdims=True)
        indices=raw.argmax(axis=-1);scores=raw.max(axis=-1);out=[];confidence=[];previous=-1
        for index,score in zip(indices,scores):
            index=int(index)
            if index!=previous and index!=0 and index<len(self.charset):
                char=self.charset[index]
                if char.isalnum():out.append(char.upper() if char.isascii() else char);confidence.append(float(score))
            previous=index
        text=''.join(out)
        if any('\u0600'<=char<='\u06ff' or '\u0750'<=char<='\u077f' for char in text):
            groups=[];latin=''
            for char in text:
                if char.isascii() and (char.isalnum() or char in ' :*./%+-'):
                    latin+=char
                else:
                    if latin:groups.append(latin);latin=''
                    groups.append(char)
            if latin:groups.append(latin)
            text=''.join(reversed(groups))
        return { 'text':text, 'confidence':float(sum(confidence)/len(confidence)) if confidence else 0.0, 'model_sha256':self.sha }


class GlobalPlateTextRecognizer:
    """Latin-script global plate OCR candidate; output remains an unverified suggestion."""
    alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
    max_slots=10

    def __init__(self,pack):
        cv2,np,ort=_plate_runtime_modules('global-plate OCR','cv2','numpy','onnxruntime')
        pack=Path(pack).resolve(strict=True)
        model=pack/OCR_GLOBAL_MODEL_NAME
        plate_config=pack/OCR_GLOBAL_PLATE_CONFIG_NAME
        model_config=pack/OCR_GLOBAL_MODEL_CONFIG_NAME
        expected={
            OCR_GLOBAL_MODEL_NAME:OCR_GLOBAL_MODEL_SHA256,
            OCR_GLOBAL_PLATE_CONFIG_NAME:OCR_GLOBAL_PLATE_CONFIG_SHA256,
            OCR_GLOBAL_MODEL_CONFIG_NAME:OCR_GLOBAL_MODEL_CONFIG_SHA256,
        }
        manifest=json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
        artifacts=manifest.get('ocr_artifacts',{})
        for name,sha in expected.items():
            entry=artifacts.get(name,{})
            if entry.get('sha256')!=sha or _sha(pack/name)!=sha:
                raise ValueError(f'Global plate OCR artifact checksum mismatch: {name}')
        plate_text=plate_config.read_text(encoding='utf-8')
        if ("alphabet: '"+self.alphabet+"'") not in plate_text or not all(
            token in plate_text for token in ('max_plate_slots: 10','img_height: 64',
                'img_width: 128','keep_aspect_ratio: false','image_color_mode: rgb')):
            raise ValueError('Global plate OCR preprocessing or alphabet configuration is unexpected')
        if 'model: cct' not in model_config.read_text(encoding='utf-8'):
            raise ValueError('Global plate OCR architecture configuration is unexpected')
        self.cv=cv2;self.np=np;self.sha=OCR_GLOBAL_MODEL_SHA256
        self.session=create_session(ort,model,2)
        inputs=self.session.get_inputs();outputs={item.name:item for item in self.session.get_outputs()}
        if len(inputs)!=1 or inputs[0].name!='input' or inputs[0].shape[1:]!=[64,128,3]:
            raise ValueError('Unexpected global plate OCR input schema')
        if 'plate' not in outputs or outputs['plate'].shape[1:]!=[10,len(self.alphabet)]:
            raise ValueError('Unexpected global plate OCR output schema')
        self.input=inputs[0].name;self.output='plate'

    def read(self,image):
        if hasattr(image,'convert'):
            image=self.np.asarray(image.convert('RGB'))
        if not isinstance(image,self.np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or not image.size:
            raise ValueError('Global plate OCR requires a non-empty RGB plate image')
        tensor=self.cv.resize(image,(128,64),interpolation=self.cv.INTER_LINEAR)[None]
        raw=self.np.asarray(self.session.run([self.output],{self.input:tensor})[0],dtype=self.np.float32)
        if raw.shape!=(1,self.max_slots,len(self.alphabet)) or not self.np.isfinite(raw).all():
            raise ValueError(f'Unexpected global plate OCR output schema: {raw.shape}')
        probs=raw[0]
        if not (self.np.all(probs>=-1e-5) and self.np.all(probs<=1.00001)
                and self.np.allclose(probs.sum(axis=-1),1,atol=1e-3)):
            shifted=probs-probs.max(axis=-1,keepdims=True)
            exp=self.np.exp(shifted);probs=exp/exp.sum(axis=-1,keepdims=True)
        indices=probs.argmax(axis=-1);scores=probs.max(axis=-1)
        chars=[self.alphabet[int(i)] for i in indices]
        kept=[(char,float(score)) for char,score in zip(chars,scores) if char!='_']
        text=''.join(char for char,_ in kept)
        confidence=sum(score for _,score in kept)/len(kept) if kept else 0.0
        return {'text':text,'confidence':float(confidence),'model_sha256':self.sha}


class PlateDetector:
    def __init__(self,pack,threshold=.3):
        cv2,np,ort=_plate_runtime_modules('plate-detection','cv2','numpy','onnxruntime')
        if not .01<=threshold<=.9:raise ValueError('Invalid plate detector threshold')
        self.cv,self.np=cv2,np
        self.model,self.sha=_model_pack(pack)
        self.session=create_session(ort,self.model,2)
        self.input=self.session.get_inputs()[0].name
        self.threshold=threshold
        self.recognizer=PlateTextRecognizer(pack)
        self.global_recognizer=GlobalPlateTextRecognizer(pack)
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
            left,top=int(box[0]*w),int(box[1]*h)
            right,bottom=int((box[0]+box[2])*w),int((box[1]+box[3])*h)
            blur_score=_laplacian_variance(cv,np,image[top:bottom,left:right])
            boxes.append({'box':box,'confidence':score,'model_sha256':self.sha,
                'blur_laplacian_variance':blur_score})
            if len(boxes)>=200:break
        return boxes


def _iou(a,b):
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,right-left)*max(0,bottom-top)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-12)


def propose(library,mission,image_id,pack,threshold=.3):
    from PIL import Image
    source=library.image_path(mission,image_id)
    global_ocr_enabled=_global_latin_ocr_applicable(library.settings(mission).get('country'))
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
            ocr=detector.recognizer.read(im.crop(coords))
            alternative=detector.global_recognizer.read(im.crop(coords)) if global_ocr_enabled else {
                'text':'','confidence':0.0,'model_sha256':''}
            im.crop(coords).save(tmp,format='JPEG',quality=95)
            tmp.replace(target)
            crops.append((crop_id,filename,target,item,coords,ocr,alternative))
    created=datetime.now(timezone.utc).isoformat()
    inserted=[]
    try:
        with _connect(db) as conn:
            existing=conn.execute('SELECT id FROM crops WHERE image_id=? AND source_sha256=? AND model_sha256=?',(image_id,row['sha256'],detector.sha)).fetchall()
            if existing:
                for item in crops:(folder/item[1]).unlink(missing_ok=True)
                return list_crops(library,mission,image_id)
            for crop_id,filename,path,item,coords,ocr,alternative in crops:
                box=item['box']
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                    crop_sha256,bbox,confidence,blur_laplacian_variance,model_sha256,created,ocr_text,ocr_confidence,ocr_model_sha256,
                    ocr_search_text,ocr_alternative_text,ocr_alternative_confidence,ocr_alternative_model_sha256,
                    ocr_alternative_search_text)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(crop_id,image_id,row['name'],row['sha256'],filename,_sha(path),
                    json.dumps({'x':coords[0],'y':coords[1],'width':coords[2]-coords[0],'height':coords[3]-coords[1]}),
                    item['confidence'],item.get('blur_laplacian_variance'),detector.sha,created,ocr['text'],ocr['confidence'],ocr['model_sha256'],
                    normalize_plate_text(ocr['text']),alternative['text'],alternative['confidence'],alternative['model_sha256'],
                    normalize_plate_text(alternative['text'])) )
                inserted.append(crop_id)
    except Exception:
        for crop_id,filename,path,item,coords,ocr,alternative in crops:path.unlink(missing_ok=True)
        raise
    return list_crops(library,mission,image_id)


def propose_video_frame(library,mission,source_id,decoded,detector):
    """Save plate chips from one decoded frame and attach OCR suggestions."""
    from PIL import Image
    from . import video_store
    import numpy as np
    _,folder,db=_paths(library,mission)
    source=video_store.source_detail(library,mission,source_id)
    global_ocr_enabled=_global_latin_ocr_applicable(library.settings(mission).get('country'))
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
                ocr=detector.recognizer.read(image.crop(coords))
                alternative=detector.global_recognizer.read(image.crop(coords)) if global_ocr_enabled else {
                    'text':'','confidence':0.0,'model_sha256':''}
                image.crop(coords).save(crop_path,format='JPEG',quality=95)
                image.save(context_path,format='JPEG',quality=88)
                saved.append((crop_id,crop_name,context_name,crop_path,context_path,item,coords,ocr,alternative))
        with _connect(db) as conn:
            for crop_id,crop_name,context_name,crop_path,context_path,item,coords,ocr,alternative in saved:
                conn.execute('''INSERT INTO crops(id,image_id,source_name,source_sha256,crop_file,
                    crop_sha256,bbox,confidence,blur_laplacian_variance,model_sha256,created,source_kind,video_source_id,
                    frame_index,pts,time_base_num,time_base_den,capture_utc,context_file,
                    ocr_text,ocr_confidence,ocr_model_sha256,ocr_search_text,ocr_alternative_text,
                    ocr_alternative_confidence,ocr_alternative_model_sha256,ocr_alternative_search_text)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
                    crop_id,image_id,source['original_name'],source['sha256'],crop_name,_sha(crop_path),
                    json.dumps({'x':coords[0],'y':coords[1],'width':coords[2]-coords[0],
                        'height':coords[3]-coords[1]}),item['confidence'],item.get('blur_laplacian_variance'),detector.sha,created,'video',
                    source_id,decoded.index,decoded.timestamp.pts,decoded.timestamp.time_base_num,
                    decoded.timestamp.time_base_den,utc.isoformat() if utc else None,context_name,
                    ocr['text'],ocr['confidence'],ocr['model_sha256'],normalize_plate_text(ocr['text']),
                    alternative['text'],alternative['confidence'],alternative['model_sha256'],
                    normalize_plate_text(alternative['text'])))
    except Exception:
        for _,_,_,crop_path,context_path,_,_,_,_ in saved:
            crop_path.unlink(missing_ok=True);context_path.unlink(missing_ok=True)
        raise
    return [crop_id for crop_id,*_ in saved]


def list_crops(library,mission,image_id=None,search=None):
    root,folder,db=_paths(library,mission)
    with _connect(db) as conn:
        sql='''SELECT c.*,r.decision,r.operator,r.revision,
            tr.decision AS text_decision,tr.verified_text,tr.normalized_text,
            tr.revision AS text_revision,tr.operator AS text_operator
            FROM crops c LEFT JOIN reviews r ON r.id=(
            SELECT id FROM reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)
            LEFT JOIN text_reviews tr ON tr.id=(SELECT id FROM text_reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)'''
        conditions=[];params=[]
        if image_id:conditions.append('c.image_id=?');params.append(image_id)
        if search is not None and (not isinstance(search,str) or len(search)>64):
            raise ValueError('Plate text search must be a string up to 64 characters')
        key=normalize_plate_text(search)
        if search is not None and search.strip() and not key:
            return []
        if key:
            conditions.append('(instr(c.ocr_search_text,?)>0 OR instr(c.ocr_alternative_search_text,?)>0 OR instr(tr.normalized_text,?)>0)')
            params.extend((key,key,key))
        if conditions:sql+=' WHERE '+' AND '.join(conditions)
        sql+=' ORDER BY c.created,c.source_name,c.id'
        return [dict(row) for row in conn.execute(sql,params).fetchall()]


def list_crops_page(library,mission,offset=0,limit=40,image_id=None,search=None):
    """Fetch one bounded crop-review page and its filtered total count."""
    if isinstance(offset,bool) or not isinstance(offset,int) or offset<0:
        raise ValueError('Plate page offset must be a nonnegative integer')
    if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=100:
        raise ValueError('Plate page size must be between 1 and 100')
    root,folder,db=_paths(library,mission)
    if search is not None and (not isinstance(search,str) or len(search)>64):
        raise ValueError('Plate text search must be a string up to 64 characters')
    key=normalize_plate_text(search)
    if search is not None and search.strip() and not key:
        return {'items':[],'offset':offset,'limit':limit,'total':0}
    conditions=[];params=[]
    if image_id:conditions.append('c.image_id=?');params.append(image_id)
    if key:
        conditions.append('(instr(c.ocr_search_text,?)>0 OR instr(c.ocr_alternative_search_text,?)>0 OR instr(tr.normalized_text,?)>0)')
        params.extend((key,key,key))
    where=' WHERE '+' AND '.join(conditions) if conditions else ''
    joins=''' FROM crops c LEFT JOIN reviews r ON r.id=(
        SELECT id FROM reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)
        LEFT JOIN text_reviews tr ON tr.id=(
        SELECT id FROM text_reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)'''
    count_joins=''' FROM crops c LEFT JOIN text_reviews tr ON tr.id=(
        SELECT id FROM text_reviews x WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1)''' if key else ' FROM crops c'
    with _connect(db) as conn:
        total=conn.execute('SELECT COUNT(*)'+count_joins+where,params).fetchone()[0]
        sql=('''SELECT c.*,r.decision,r.operator,r.revision,
            tr.decision AS text_decision,tr.verified_text,tr.normalized_text,
            tr.revision AS text_revision,tr.operator AS text_operator''' + joins + where +
            ' ORDER BY c.created,c.source_name,c.id LIMIT ? OFFSET ?')
        items=[dict(row) for row in conn.execute(sql,[*params,limit,offset]).fetchall()]
    return {'items':items,'offset':offset,'limit':limit,'total':total}


def _edit_distance(actual,predicted):
    """Return Levenshtein distance after the same plate-search normalization."""
    left=normalize_plate_text(actual);right=normalize_plate_text(predicted)
    if not left:return len(right)
    if not right:return len(left)
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def _normalized_edit_similarity(actual,predicted):
    """Return character-level similarity after the same plate-search normalization."""
    left=normalize_plate_text(actual);right=normalize_plate_text(predicted)
    if not left:return 1.0 if not right else 0.0
    return max(0.0,1.0-_edit_distance(left,right)/max(len(left),len(right)))


def _near_duplicate_components(values, min_similarity=0.9):
    """Group normalized plate strings linked by edit similarity at the threshold.

    A BK-tree avoids comparing every pair when a mission has many distinct
    verified plates. Components use transitive closure and are intended only
    to reduce repeated near-duplicate weighting in exploratory summaries.
    """
    if not 0 < min_similarity <= 1:
        raise ValueError('Minimum plate-string similarity must be greater than zero and at most one')
    strings=set()
    for value in values:
        normalized=normalize_plate_text(value)
        if normalized:strings.add(normalized)
    strings=sorted(strings)
    if not strings:return []
    parent=list(range(len(strings)))
    def find(index):
        while parent[index]!=index:
            parent[index]=parent[parent[index]]
            index=parent[index]
        return index
    def union(left,right):
        left_root=find(left);right_root=find(right)
        if left_root!=right_root:
            parent[max(left_root,right_root)]=min(left_root,right_root)

    max_distance=int((1.0-min_similarity)*max(map(len,strings))+1e-12)
    if max_distance<=1:
        # At the plate-string lengths covered by a one-edit threshold, shared
        # delete-one signatures find every substitution or insertion/deletion
        # neighbor in near-linear time without scanning all string pairs.
        signature_index={}
        for index,value in enumerate(strings):
            signatures={value}
            signatures.update(value[:position]+value[position+1:] for position in range(len(value)))
            candidates=set()
            for signature in signatures:candidates.update(signature_index.get(signature,()))
            for other in candidates:
                if _normalized_edit_similarity(value,strings[other])>=min_similarity:union(index,other)
            for signature in signatures:signature_index.setdefault(signature,[]).append(index)
    else:
        # Longer labels may allow two or more edits at the paper's relative
        # threshold; use the metric tree for those uncommon non-plate strings.
        root=None;max_length_seen=0
        def matches(query,radius):
            if root is None:return []
            found=[];stack=[root]
            while stack:
                node=stack.pop()
                distance=_edit_distance(query,node['value'])
                if distance<=radius and _normalized_edit_similarity(query,node['value'])>=min_similarity:
                    found.append(node['index'])
                low=distance-radius;high=distance+radius
                stack.extend(child for edge,child in node['children'].items() if low<=edge<=high)
            return found
        for index,value in enumerate(strings):
            radius=int((1.0-min_similarity)*max(len(value),max_length_seen)+1e-12)
            for other in matches(value,radius):union(index,other)
            if root is None:root={'value':value,'index':index,'children':{}}
            else:
                node=root
                while True:
                    distance=_edit_distance(value,node['value'])
                    child=node['children'].get(distance)
                    if child is None:
                        node['children'][distance]={'value':value,'index':index,'children':{}}
                        break
                    node=child
            max_length_seen=max(max_length_seen,len(value))
    groups={}
    for index,value in enumerate(strings):groups.setdefault(find(index),[]).append(value)
    return [groups[key] for key in sorted(groups)]


def _ocr_metrics(rows,text_field):
    exact=0;covered=0;similarities=[];edit_errors=0;reference_characters=0
    for row in rows:
        actual=row.get('normalized_text') or normalize_plate_text(row.get('verified_text',''))
        predicted=row.get(text_field) or ''
        if predicted:covered+=1
        if predicted and normalize_plate_text(predicted)==actual:exact+=1
        similarities.append(_normalized_edit_similarity(actual,predicted))
        edit_errors+=_edit_distance(actual,predicted)
        reference_characters+=len(actual)
    count=len(rows)
    return {'reviewed_crops':count,'candidate_coverage':covered/count if count else 0.0,
        'exact_matches':exact,'exact_match_rate':exact/count if count else 0.0,
        'mean_normalized_edit_similarity':sum(similarities)/count if count else 0.0,
        'character_error_rate':edit_errors/reference_characters if reference_characters else 0.0,
        'edit_errors':edit_errors,'reference_characters':reference_characters}


def ocr_quality_report(library,mission):
    """Compare pinned OCR candidates with explicit human text decisions.

    This is an exploratory report, not a held-out accuracy claim. It reports
    crop-level metrics and a deterministic one-crop-per-normalized-string view
    so repeated observations cannot inflate the exact-string summary. A
    separate 0.90 near-duplicate grouping follows the supplied paper's
    leakage-control lesson, but does not create a protected split.
    """
    with library.connect(mission) as conn:
        mission_row=conn.execute('SELECT country FROM mission WHERE id=?',(mission,)).fetchone()
    reviewed=[row for row in list_crops(library,mission)
        if row.get('decision')=='approved' and row.get('text_decision') in ('accepted','corrected')]
    result={'mission_id':mission,'country':mission_row['country'] if mission_row else None,'reviewed_crops':len(reviewed),
        'scope':'Exploratory metrics on human-verified approved crop observations; not a protected held-out evaluation. Crop-level observations may be correlated. Exact-string and 0.90 near-duplicate summaries use one deterministic crop per group; near-duplicate groups use transitive closure and are not a substitute for a protected split.',
        'models':[]}
    choices=(('primary','ocr_text','ocr_model_sha256'),
        ('global_latin','ocr_alternative_text','ocr_alternative_model_sha256'))
    for label,text_field,hash_field in choices:
        model_hashes=sorted({row.get(hash_field) for row in reviewed if row.get(hash_field)})
        for model_hash in model_hashes:
            rows=[row for row in reviewed if row.get(hash_field)==model_hash]
            grouped={}
            for row in rows:
                actual=row.get('normalized_text') or normalize_plate_text(row.get('verified_text',''))
                # Stable selection makes the summary reproducible and prevents a
                # frequently observed string from receiving extra weight.
                key=(row.get('crop_sha256') or '',row.get('id') or '')
                if actual not in grouped or key<(grouped[actual].get('crop_sha256') or '',grouped[actual].get('id') or ''):
                    grouped[actual]=row
            grouped_rows=list(grouped.values())
            components=_near_duplicate_components(grouped)
            near_duplicate_rows=[]
            for component in components:
                candidates=[grouped[value] for value in component]
                near_duplicate_rows.append(min(candidates,key=lambda row:(row.get('crop_sha256') or '',row.get('id') or '')))
            result['models'].append({'candidate':label,'model_sha256':model_hash,
                **_ocr_metrics(rows,text_field),
                'unique_verified_strings':len(grouped_rows),
                'string_grouped':_ocr_metrics(grouped_rows,text_field),
                'near_duplicate_similarity_threshold':0.9,
                'near_duplicate_groups':len(components),
                'near_duplicate_grouped':_ocr_metrics(near_duplicate_rows,text_field)})
    return result


PLATE_EXPORT_FIELDS=('crop_id','source_kind','source_name','source_sha256','image_id','video_source_id',
    'frame_index','pts','time_base_num','time_base_den','capture_utc','bbox','detector_confidence',
    'blur_laplacian_variance','detector_model_sha256','crop_file','crop_sha256','crop_review_decision','crop_review_revision',
    'crop_review_operator','crop_reviewed_at','ocr_primary_text','ocr_primary_confidence',
    'ocr_primary_model_sha256','ocr_alternative_text','ocr_alternative_confidence',
    'ocr_alternative_model_sha256','text_decision','verified_text','normalized_text',
    'text_review_revision','text_review_operator','text_reviewed_at')


def export_crops_csv(library,mission):
    """Write an atomic, UTF-8 plate-crop review snapshot without overwriting prior exports."""
    _,folder,db=_paths(library,mission)
    export_id=str(uuid.uuid4())
    target=folder/f'plate-review-{mission[:8]}-{export_id}.csv'
    temporary=folder/f'.{target.name}.tmp'
    query='''SELECT c.id AS crop_id,c.source_kind,c.source_name,c.source_sha256,c.image_id,c.video_source_id,
        c.frame_index,c.pts,c.time_base_num,c.time_base_den,c.capture_utc,c.bbox,
        c.confidence AS detector_confidence,c.blur_laplacian_variance,c.model_sha256 AS detector_model_sha256,c.crop_file,c.crop_sha256,
        r.decision AS crop_review_decision,r.revision AS crop_review_revision,r.operator AS crop_review_operator,
        r.created AS crop_reviewed_at,c.ocr_text AS ocr_primary_text,c.ocr_confidence AS ocr_primary_confidence,
        c.ocr_model_sha256 AS ocr_primary_model_sha256,c.ocr_alternative_text,c.ocr_alternative_confidence,
        c.ocr_alternative_model_sha256,tr.decision AS text_decision,tr.verified_text,tr.normalized_text,
        tr.revision AS text_review_revision,tr.operator AS text_review_operator,tr.created AS text_reviewed_at
        FROM crops c LEFT JOIN reviews r ON r.id=(SELECT id FROM reviews x WHERE x.crop_id=c.id
        ORDER BY x.revision DESC LIMIT 1) LEFT JOIN text_reviews tr ON tr.id=(SELECT id FROM text_reviews x
        WHERE x.crop_id=c.id ORDER BY x.revision DESC LIMIT 1) ORDER BY c.created,c.source_name,c.id'''
    count=0
    try:
        with _connect(db) as conn, temporary.open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=PLATE_EXPORT_FIELDS,extrasaction='ignore')
            writer.writeheader()
            for raw in conn.execute(query):
                row=dict(raw)
                for key,value in row.items():
                    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
                        row[key]="'"+value
                writer.writerow(row);count+=1
            stream.flush();os.fsync(stream.fileno())
        if target.exists():raise FileExistsError('Plate export filename already exists')
        os.replace(temporary,target)
        try:
            fd=os.open(folder,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        except OSError:
            pass
        digest=_sha(target)
        return {'id':export_id,'path':str(target),'row_count':count,'sha256':digest,'format':'csv'}
    except BaseException:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


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


def review_text(library,mission,crop_id,expected_revision,decision,verified_text,operator):
    allowed=('accepted','corrected','rejected','unreadable')
    if decision not in allowed:raise ValueError('Choose accepted, corrected, rejected, or unreadable')
    if not isinstance(operator,str) or not 1<=len(operator.strip())<=120:raise ValueError('Operator must contain 1–120 characters')
    if not isinstance(verified_text,str) or len(verified_text)>64:raise ValueError('Plate text must be a string up to 64 characters')
    value=verified_text.strip()
    if decision in ('accepted','corrected') and not value:raise ValueError('Enter or correct the plate text before confirming it')
    if decision in ('rejected','unreadable'):value=''
    _,_,db=_paths(library,mission)
    with _connect(db) as conn:
        if not conn.execute('SELECT 1 FROM crops WHERE id=?',(crop_id,)).fetchone():raise ValueError('Plate crop not found in mission')
        row=conn.execute('SELECT revision FROM text_reviews WHERE crop_id=? ORDER BY revision DESC LIMIT 1',(crop_id,)).fetchone()
        current=row['revision'] if row else 0
        if expected_revision!=current:raise ValueError('Plate text changed since it was opened; reload before reviewing')
        revision=current+1
        normalized=normalize_plate_text(value) if decision in ('accepted','corrected') else ''
        conn.execute('''INSERT INTO text_reviews(crop_id,revision,decision,verified_text,normalized_text,operator,created)
            VALUES (?,?,?,?,?,?,?)''',
            (crop_id,revision,decision,value,normalized,operator.strip(),datetime.now(timezone.utc).isoformat()))
    return {'crop_id':crop_id,'revision':revision,'decision':decision,'verified_text':value,
        'normalized_text':normalized}
