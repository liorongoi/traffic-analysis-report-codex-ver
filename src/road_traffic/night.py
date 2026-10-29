"""Inspectable light hypotheses. No streetlight/reflection guess is ground truth."""
import math
FEATURE_SCHEMA='light-patch-v1'
NIGHT_CLASSES=('headlight','streetlight','reflection','other')


def patch_features(rgb,box):
    import cv2
    import numpy as np
    h,w=rgb.shape[:2];x,y,bw,bh=box
    x1=max(0,int(x*w));y1=max(0,int(y*h));x2=min(w,max(x1+1,int(math.ceil((x+bw)*w))));y2=min(h,max(y1+1,int(math.ceil((y+bh)*h))))
    patch=rgb[y1:y2,x1:x2]
    if not patch.size:raise ValueError('Empty light patch')
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY).astype(float)
    saturation=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)[:,:,1]
    margin=max(4,int(max(x2-x1,y2-y1)*.5))
    outer=cv2.cvtColor(rgb[max(0,y1-margin):min(h,y2+margin),max(0,x1-margin):min(w,x2+margin)],cv2.COLOR_RGB2GRAY)
    core=gray>=215
    return [float(v) for v in [x+bw/2,y+bh/2,bw,bh,math.log((bw*w+1)/(bh*h+1)),gray.mean()/255,
        gray.std()/128,np.percentile(gray,95)/255,core.mean(),saturation.mean()/255,
        (gray.mean()-float(np.median(outer)))/255, np.mean(np.abs(np.diff(gray,axis=0)))/255 if gray.shape[0]>1 else 0,
        np.mean(np.abs(np.diff(gray,axis=1)))/255 if gray.shape[1]>1 else 0]]


def classify_component(component):
    """Conservative, explainable subtype; persistent lights remain visible for review."""
    reasons=[];persistence=component.get('persistence');novelty=component.get('novelty')
    aspect=component['aspect'];fill=component['fill']
    if component.get('large_blob'):
        return 'uncertain_light',['large connected light/glare blob; it may obscure a vehicle and is not a vehicle box']
    if component.get('single_pixel'):
        return 'uncertain_light',['isolated bright point; could be a distant vehicle light, star, sensor artifact, or reflection']
    if aspect>=3.0 and component['area']>=8:
        return 'reflection_suspect',['elongated luminous structure; could also be a light trail']
    if persistence is not None and persistence>=.5:
        return 'streetlight_suspect',['recurs at a fixed location; could also be stationary vehicle lighting']
    if component.get('near_recurrence',0)>=.5:
        return 'uncertain_light',['near a recurring bright fixture; a glare change or nearby vehicle cannot be distinguished reliably']
    if novelty is not None and novelty>=.5 and aspect<2.5 and fill>=.25:
        return 'headlight_candidate',['compact new light relative to this view background; identity unconfirmed']
    return 'uncertain_light',['insufficient evidence to distinguish vehicle light, fixture or reflection']


def propose(rgb,context,vehicles,head=None):
    import cv2
    import numpy as np
    from .detector import light_mask
    h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    mask=light_mask(gray)
    # Join JPEG speckles within the same luminous source, without linking separated headlights.
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    components=[];covered=0
    nearby=cv2.dilate(context['frequency'],np.ones((11,11),np.uint8)) if context else None
    max_area=max(5000,int(h*w*.06))
    for label in range(1,count):
        x,y,bw,bh,area=map(int,stats[label])
        single_pixel=area==1
        large_blob=area>5000 or bw>250 or bh>250
        # Preserve sparse pinpoints and bounded connected glare blobs as review
        # hypotheses. Oversized regions stay excluded to avoid turning whole
        # illuminated road/sky areas into one candidate.
        if area<1 or area>max_area or bw>max(250,int(w*.4)) or bh>max(250,int(h*.4)):continue
        if single_pixel and int(gray[y:y+bh,x:x+bw].max())<250:continue
        cx,cy=centers[label];box=[x/w,y/h,bw/w,bh/h]
        if any(v['box'][0]-.008<=cx/w<=v['box'][0]+v['box'][2]+.008 and v['box'][1]-.008<=cy/h<=v['box'][1]+v['box'][3]+.008 for v in vehicles):
            covered+=1;continue
        pixels=labels[y:y+bh,x:x+bw]==label
        recurrence=float(context['frequency'][y:y+bh,x:x+bw][pixels].mean()) if context else None
        novelty=float(context['novelty'][y:y+bh,x:x+bw][pixels].mean()) if context else None
        c={'box':box,'center':[float(cx),float(cy)],'area':area,'aspect':max(bw,bh)/max(1,min(bw,bh)),
           'fill':area/(bw*bh),'near_recurrence':float(nearby[y:y+bh,x:x+bw][pixels].mean()) if nearby is not None else 0,'persistence':recurrence,'novelty':novelty,'peak':int(gray[y:y+bh,x:x+bw][pixels].max())}
        c.update(single_pixel=single_pixel,large_blob=large_blob)
        subtype,reasons=classify_component(c);c.update(subtype=subtype,reasons=reasons)
        components.append(c)
    # Mutual-nearest pairing only among compact novel sources; recurring fixtures are not paired.
    closest={}
    for i,a in enumerate(components):
        if a['subtype']!='headlight_candidate' or a.get('large_blob') or a.get('single_pixel'):continue
        options=[]
        for j,b in enumerate(components):
            if i==j or b['subtype']!='headlight_candidate' or b.get('large_blob') or b.get('single_pixel'):continue
            dx=abs(a['center'][0]-b['center'][0]);dy=abs(a['center'][1]-b['center'][1])
            if 3<=dx<=min(60,w*.06) and dy<=max(2,dx*.18) and max(a['area'],b['area'])/min(a['area'],b['area'])<=2.5:
                # A bright bridge between peaks is more consistent with a streak than separated headlights.
                left,right=sorted([int(a['center'][0]),int(b['center'][0])]);mid_y=int((a['center'][1]+b['center'][1])/2)
                gap=gray[max(0,mid_y-1):min(h,mid_y+2),left+2:right-1]
                if gap.size and float(np.median(gap))>min(a['peak'],b['peak'])-max(35,.2*min(a['peak'],b['peak'])):continue
                options.append((dy+dx*.04,j))
        if options:closest[i]=min(options)[1]
    result=[];used=set()
    for i,c in enumerate(components):
        if i in used:continue
        group=[c];j=closest.get(i)
        if j is not None and closest.get(j)==i and j not in used:group.append(components[j]);used.add(j)
        used.add(i)
        left=min(a['box'][0] for a in group);top=min(a['box'][1] for a in group)
        right=max(a['box'][0]+a['box'][2] for a in group);bottom=max(a['box'][1]+a['box'][3] for a in group)
        box=[left,top,right-left,bottom-top]
        subtype=c['subtype'];features=patch_features(rgb,box)
        item={'kind':'light','label':('possible headlight pair' if len(group)==2 else subtype.replace('_',' ')),
            'box':box,'origin':'night-light-heuristic','evidence':'contextual-light-v4','light_subtype':subtype,
            'reasons':c['reasons']+(['mutual-nearest similar compact sources; pairing is a hypothesis'] if len(group)==2 else []),
            'light_components':group,'pairing':'hypothesis' if len(group)==2 else 'single','night_features':features,
            'feature_schema':FEATURE_SCHEMA,'coordinate_system':'normalized-original-image'}
        if head:
            label,scores=head.predict(features)
            item['correction_head']={'suggestion':label,'scores':scores,'calibrated':False,'model_id':head.identifier}
            # A learned candidate does not become a confirmed vehicle or erase the context rule.
        result.append(item)
    order={'headlight_candidate':0,'uncertain_light':1,'reflection_suspect':2,'streetlight_suspect':3}
    result.sort(key=lambda a:order[a['light_subtype']])
    return result,{'proposals':len(result),'subtypes':{k:sum(a['light_subtype']==k for a in result) for k in order},
        'covered_by_detector':covered,'background_available':context is not None,'truncated':False,
        'single_point_hypotheses':sum(any(component.get('single_pixel') for component in a['light_components']) for a in result),
        'large_blob_hypotheses':sum(any(component.get('large_blob') for component in a['light_components']) for a in result),
        'warning':'Lights can be distant-vehicle cues, fixed fixtures, reflections, sensor artifacts, or glare blobs. Every light and pair remains an unconfirmed review hypothesis, never a vehicle box or count.'}
