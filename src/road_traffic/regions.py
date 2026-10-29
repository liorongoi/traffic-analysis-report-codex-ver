"""Normalized original-image ROI polygons and proposal filtering."""
import math


def validate_polygon(points):
    if not isinstance(points,list) or len(points)<3:
        raise ValueError('A region polygon needs at least three points')
    polygon=[]
    for point in points:
        if not isinstance(point,(list,tuple)) or len(point)!=2:
            raise ValueError('Each polygon point must be an [x,y] pair')
        x,y=point
        if isinstance(x,bool) or isinstance(y,bool) or not isinstance(x,(int,float)) or not isinstance(y,(int,float)):
            raise ValueError('Polygon coordinates must be numbers')
        if not math.isfinite(x) or not math.isfinite(y) or not 0<=x<=1 or not 0<=y<=1:
            raise ValueError('Polygon coordinates must be normalized between 0 and 1')
        polygon.append((float(x),float(y)))
    if len(set(polygon))<3 or any(polygon[i]==polygon[(i+1)%len(polygon)] for i in range(len(polygon))):
        raise ValueError('Polygon points must be distinct')
    area=sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1]
        for i in range(len(polygon)))
    if abs(area)<1e-10:
        raise ValueError('Polygon area must be greater than zero')
    for i in range(len(polygon)):
        a,b=polygon[i],polygon[(i+1)%len(polygon)]
        for j in range(i+1,len(polygon)):
            if j==i+1 or (i==0 and j==len(polygon)-1):
                continue
            c,d=polygon[j],polygon[(j+1)%len(polygon)]
            if _segments_intersect(a,b,c,d):
                raise ValueError('Polygon edges must not cross or touch')
    return [[x,y] for x,y in polygon]


def _orientation(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _on_segment(a,b,p):
    return min(a[0],b[0])-1e-10<=p[0]<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=p[1]<=max(a[1],b[1])+1e-10


def _segments_intersect(a,b,c,d):
    o1,o2,o3,o4=_orientation(a,b,c),_orientation(a,b,d),_orientation(c,d,a),_orientation(c,d,b)
    if ((o1>0 and o2<0 or o1<0 and o2>0) and (o3>0 and o4<0 or o3<0 and o4>0)):
        return True
    return ((abs(o1)<1e-10 and _on_segment(a,b,c)) or (abs(o2)<1e-10 and _on_segment(a,b,d)) or
        (abs(o3)<1e-10 and _on_segment(c,d,a)) or (abs(o4)<1e-10 and _on_segment(c,d,b)))


def point_in_polygon(point, polygon):
    x,y=point;inside=False
    for i,(x1,y1) in enumerate(polygon):
        x2,y2=polygon[(i+1)%len(polygon)]
        cross=(x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        if abs(cross)<1e-10 and _on_segment((x1,y1),(x2,y2),(x,y)):
            return True
        if (y1>y)!=(y2>y) and x<(x2-x1)*(y-y1)/(y2-y1)+x1:
            inside=not inside
    return inside


def includes_box(box, regions):
    """Apply include union and exclusion precedence to a normalized [x,y,w,h] box."""
    if not regions:
        return True
    x,y,w,h=box
    center=(x+w/2,y+h/2)
    included=[r for r in regions if r['kind']=='include' and r.get('role','analysis')!='road_surface']
    excluded=[r for r in regions if r['kind']=='exclude' and r.get('role','analysis')!='road_surface']
    if any(point_in_polygon(center,r['polygon']) for r in excluded):
        return False
    return not included or any(point_in_polygon(center,r['polygon']) for r in included)


def filter_proposals(proposals, regions):
    road_regions=[r for r in regions if r.get('role')=='road_surface']
    if not regions:
        return proposals, {'input':len(proposals),'excluded':0,'included':0}
    kept=[item for item in proposals if includes_box(item['box'],regions)]
    for item in kept:
        if item.get('kind')=='vehicle' or item.get('evidence','').startswith('yolox-s'):
            item['road_relation']=classify_road_relation(item['box'],road_regions)
    return kept, {'input':len(proposals),'excluded':len(proposals)-len(kept),'included':len(kept),
        'coordinate_system':'normalized-original-image','exclusion_overrides_inclusion':True,
        'road_surface_regions':len(road_regions),'road_relation_basis':'operator-drawn or operator-approved road polygons; heuristic bbox geometry'}


def classify_road_relation(box, road_regions):
    """Relate a vehicle box to operator road polygons using bottom-center contact geometry."""
    if not road_regions:
        return {'label':'unknown','basis':'no approved road-surface polygon'}
    x,y,w,h=box; contact=(x+w/2,y+h)
    if any(point_in_polygon(contact,r['polygon']) for r in road_regions):
        return {'label':'on_road','basis':'vehicle box bottom-center lies inside an approved road polygon'}
    # A box crossing the road edge is near/on boundary even when its bottom-center
    # falls just outside due to an imperfect detector box.
    points=[(x,y),(x+w,y),(x,y+h),(x+w,y+h),(x+w/2,y+h/2)]
    if any(point_in_polygon(p,r['polygon']) for p in points for r in road_regions):
        return {'label':'near_road','basis':'vehicle box intersects an approved road polygon'}
    tolerance=max(.01,min(.06,h*.5))
    if _distance_to_regions(contact,road_regions)<=tolerance:
        return {'label':'near_road','basis':'bottom-center within a bounded normalized distance of an approved road polygon'}
    return {'label':'off_road','basis':'vehicle box is outside the approved road polygons and proximity band'}


def mask_analysis_regions(rgb, regions, cv2_module=None):
    """Neutralize pixels outside operator-approved analysis inclusion polygons."""
    includes=[r for r in regions if r.get('kind')=='include' and r.get('role','analysis')=='analysis']
    if not includes:
        return rgb, {'applied':False,'include_regions':0,'policy':'full frame'}
    if cv2_module is None:
        import cv2 as cv2_module
    import numpy as np
    height,width=rgb.shape[:2]
    mask=np.zeros((height,width),dtype=np.uint8)
    for region in includes:
        polygon=np.asarray([[round(x*(width-1)),round(y*(height-1))] for x,y in region['polygon']],dtype=np.int32)
        cv2_module.fillPoly(mask,[polygon],1)
    prepared=rgb.copy()
    prepared[mask==0]=114
    return prepared, {'applied':True,'include_regions':len(includes),
        'policy':'pixels outside analysis inclusion polygons replaced with neutral 114 before model inference'}


def _distance_to_regions(point, regions):
    if any(point_in_polygon(point,r['polygon']) for r in regions):
        return 0.0
    best=float('inf')
    px,py=point
    for region in regions:
        poly=region['polygon']
        for i,(x1,y1) in enumerate(poly):
            x2,y2=poly[(i+1)%len(poly)];dx=x2-x1;dy=y2-y1
            t=max(0.0,min(1.0,((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy or 1)))
            best=min(best,math.hypot(px-(x1+t*dx),py-(y1+t*dy)))
    return best
