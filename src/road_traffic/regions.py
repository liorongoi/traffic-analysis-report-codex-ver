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
    included=[r for r in regions if r['kind']=='include']
    excluded=[r for r in regions if r['kind']=='exclude']
    if any(point_in_polygon(center,r['polygon']) for r in excluded):
        return False
    return not included or any(point_in_polygon(center,r['polygon']) for r in included)


def filter_proposals(proposals, regions):
    if not regions:
        return proposals, {'input':len(proposals),'excluded':0,'included':0}
    kept=[item for item in proposals if includes_box(item['box'],regions)]
    return kept, {'input':len(proposals),'excluded':len(proposals)-len(kept),'included':len(kept),
        'coordinate_system':'normalized-original-image','exclusion_overrides_inclusion':True}
