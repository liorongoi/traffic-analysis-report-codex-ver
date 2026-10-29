"""Local ONNX vehicle detector and experimental, separately labelled night-light proposals."""
import hashlib
import json
from pathlib import Path
import math
from .ort_session import create_session

VEHICLES={1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}
VERSION='yolox-s-coco+night-lights-v6+region-mask-v1'
SUPPORTED_LABELS=set(VEHICLES.values())


def model_pack(root):
    root=Path(root).resolve(strict=True)
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/'model.onnx'
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=manifest['files']['model.onnx']['sha256']:raise ValueError('Model checksum mismatch; no download fallback')
    return path,actual


def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);right=min(a[0]+a[2],b[0]+b[2]);bottom=min(a[1]+a[3],b[1]+b[3])
    inter=max(0,right-x)*max(0,bottom-y)
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-12)


def suppress(items,threshold=.45):
    result=[]
    # Vehicle classes share NMS: a car/truck duplicate must not become two vehicles.
    for item in sorted(items,key=lambda x:x['score'],reverse=True):
        if not any(iou(item['box'],r['box'])>threshold for r in result):result.append(item)
    return result


class VehicleDetector:
    def __init__(self,pack,threshold=.18,tiled=True,cpu_threads=2):
        import cv2
        import numpy as np
        import onnxruntime as ort
        if not .01<=threshold<=.9:raise ValueError('Invalid confidence threshold')
        if type(cpu_threads) is not int or not 1<=cpu_threads<=64:raise ValueError('CPU thread limit must be between 1 and 64')
        self.cv,self.np=cv2,np
        self.path,self.sha=model_pack(Path(pack))
        metadata=json.loads((self.path.parent/'manifest.json').read_text())
        trained_classes=metadata.get('classes')
        if trained_classes is not None:
            if (not isinstance(trained_classes,list) or not trained_classes
                    or any(label not in SUPPORTED_LABELS for label in trained_classes)
                    or len(set(trained_classes))!=len(trained_classes)):
                raise ValueError('Invalid vehicle class list in model manifest')
            self.class_labels={index:label for index,label in enumerate(trained_classes)}
        else:
            self.class_labels=VEHICLES
        self.session=create_session(ort,self.path,cpu_threads)
        self.input=self.session.get_inputs()[0].name
        self.threshold,self.tiled,self.cpu_threads=threshold,tiled,cpu_threads
        grids=[];strides=[]
        for stride in (8,16,32):
            y,x=np.meshgrid(np.arange(640//stride),np.arange(640//stride),indexing='ij')
            grid=np.stack([x,y],axis=-1).reshape(-1,2)
            grids.append(grid);strides.append(np.full((len(grid),1),stride))
        self.grid=np.concatenate(grids).astype(np.float32);self.strides=np.concatenate(strides).astype(np.float32)

    def pass_image(self,rgb,offset=(0,0),whole=None,coordinate_scale=1.0):
        cv,np=self.cv,self.np
        h,w=rgb.shape[:2];ratio=min(640/w,640/h)
        canvas=np.full((640,640,3),114,dtype=np.float32)
        resized=cv.resize(rgb,(int(w*ratio),int(h*ratio)),interpolation=cv.INTER_LINEAR)
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        tensor=np.ascontiguousarray(canvas.transpose(2,0,1)[None])
        raw=self.session.run(None,{self.input:tensor})[0][0]
        if raw.ndim!=2 or raw.shape[0]!=8400 or raw.shape[1]<6:
            raise ValueError('Unexpected YOLOX output schema')
        centers=(raw[:,:2]+self.grid)*self.strides
        sizes=np.exp(np.clip(raw[:,2:4],-16,16))*self.strides
        classes=np.argmax(raw[:,5:],axis=1)
        scores=raw[:,4]*np.max(raw[:,5:],axis=1)
        mask=np.isin(classes,list(self.class_labels))&(scores>=self.threshold)&np.isfinite(scores)
        ow,oh=whole or (w,h)
        result=[]
        for center,size,score,cls in zip(centers[mask],sizes[mask],scores[mask],classes[mask]):
            x1=max(0,float((center[0]-size[0]/2)/ratio));y1=max(0,float((center[1]-size[1]/2)/ratio))
            x2=min(w,float((center[0]+size[0]/2)/ratio));y2=min(h,float((center[1]+size[1]/2)/ratio))
            if x2<=x1 or y2<=y1:continue
            box=[(x1/coordinate_scale+offset[0])/ow,(y1/coordinate_scale+offset[1])/oh,
                (x2-x1)/(coordinate_scale*ow),(y2-y1)/(coordinate_scale*oh)]
            result.append({'kind':'candidate','suggested_kind':'vehicle','label':self.class_labels[int(cls)],'box':box,
                'score':float(score),'origin':'model','evidence':'yolox-s','coordinate_system':'normalized-original-image'})
        return result

    def infer(self,rgb):
        h,w=rgb.shape[:2]
        result=self.pass_image(rgb)
        # Overlapping native-resolution tiles improve opportunity to detect distant objects.
        # This is not a claim of measured recall improvement.
        if self.tiled and w>800:
            size=min(768,w,h)
            xs=sorted(set([0,max(0,(w-size)//2),max(0,w-size)]))
            ys=sorted(set([0,max(0,h-size)]))
            for y in ys:
                for x in xs:result.extend(self.pass_image(rgb[y:y+size,x:x+size],(x,y),(w,h)))
        return suppress(result)

    def infer_night_top(self,rgb):
        """Search the far/top image band at a bounded zoom for tiny vehicle proposals."""
        import cv2
        import numpy as np
        h,w=rgb.shape[:2]
        band_h=max(1,int(h*.55))
        band=rgb[:band_h,:]
        source_pixels=band_h*w
        # Bound the zoom input even for maximum-sized source images.
        scale=min(2.0,math.sqrt(2_500_000/max(1,source_pixels)))
        if scale<1.12:
            return []
        enlarged=cv2.resize(band,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        proposals=self.pass_image(enlarged,whole=(w,h),coordinate_scale=scale)
        for item in proposals:
            item['evidence']='yolox-s-night-top-zoom'
            item['proposal_context']='upper-image detail pass; distant object hypothesis requiring review'
        return proposals


def light_mask(gray):
    import cv2
    import numpy as np
    # Local contrast avoids labelling every bright road surface as a light.
    local=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    # Keep isolated saturated points and solid glare cores as hypotheses too;
    # connected-component bounds in night.propose prevent unbounded blobs.
    return (((gray>=215)&(local>=35))|(gray>=248)).astype(np.uint8)


def night_proposals(rgb,persistent,vehicles):
    import cv2
    import numpy as np
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    mask=light_mask(gray)
    raw_count=int(mask.sum())
    if persistent is not None:
        mask[persistent>0]=0
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    components=[]
    for label in range(1,count):
        x,y,bw,bh,area=map(int,stats[label])
        if not 2<=area<=1500 or bw>100 or bh>100:continue
        box=[x/w,y/h,bw/w,bh/h]
        cx,cy=centers[label]
        if any(v['box'][0]-.015<=cx/w<=v['box'][0]+v['box'][2]+.015 and v['box'][1]-.015<=cy/h<=v['box'][1]+v['box'][3]+.015 for v in vehicles):continue
        components.append({'box':box,'center':[float(cx),float(cy)],'area':area,'peak':int(gray[labels==label].max())})
    # Explicit pairing hypothesis, not asserted identity or confirmed vehicle count.
    pairs=[];used=set()
    for i,a in enumerate(components):
        if i in used:continue
        options=[]
        for j,b in enumerate(components):
            if j<=i or j in used:continue
            dx=abs(a['center'][0]-b['center'][0]);dy=abs(a['center'][1]-b['center'][1])
            if 3<=dx<=min(60,w*.06) and dy<=max(3,dx*.25) and max(a['area'],b['area'])/min(a['area'],b['area'])<=4:
                options.append((dy+dx*.03,j))
        group=[a]
        if options:
            j=min(options)[1];used.add(j);group.append(components[j])
        used.add(i)
        left=min(c['box'][0] for c in group);top=min(c['box'][1] for c in group)
        right=max(c['box'][0]+c['box'][2] for c in group);bottom=max(c['box'][1]+c['box'][3] for c in group)
        pairs.append({'kind':'light','label':'possible headlight pair' if len(group)==2 else 'unresolved light',
            'box':[left,top,right-left,bottom-top],'origin':'night-light-heuristic','evidence':'nonpersistent-bright-component',
            'light_components':group,'pairing':'hypothesis' if len(group)==2 else 'single',
            'coordinate_system':'normalized-original-image'})
    return pairs,{'bright_pixels':raw_count,'new_light_components':len(components),'proposals':len(pairs),
        'truncated':False,'warning':'Light candidates may be reflections or background changes; not confirmed vehicles.'}


def illumination(rgb):
    import cv2
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    saturation=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[:,:,1]
    mean_luma=float(gray.mean());mean_saturation=float(saturation.mean())
    monochrome=mean_saturation<5
    # Label this as a review hint, not astronomical truth: underexposed daytime
    # scenes can look night-like and bright urban nights can look day-like.
    if mean_luma<85 or (monochrome and mean_luma<175):
        period='night_like'
    elif mean_luma>=140 or (mean_luma>=115 and mean_saturation>=12):
        period='day_like'
    else:
        period='uncertain'
    return {'mean_luma':mean_luma,'mean_saturation':mean_saturation,
        'low_light':mean_luma<95 or monochrome,'scene_period':period,
        'scene_period_basis':'heuristic mean-luminance and saturation; review hint, not capture-time ground truth',
        'illumination_rule':'mean luma <95 OR mean saturation <5; routes monochrome night camera mode to light proposals'}
