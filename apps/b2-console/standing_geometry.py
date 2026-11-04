"""B2 foot positions relative to each hip, in the body frame (SDK leg order).

The stance envelope is obtained from the existing +/-0.25 rad joint tolerance
around each accepted stance. It recognizes equivalent supporting geometry,
without requiring every thigh to return to one particular joint solution.
Training physics and skill acceptance are not changed by this runtime check.
"""
from functools import lru_cache
import math


def _trig_bounds(lo,hi,cos=False):
    fn=math.cos if cos else math.sin
    offset=0. if cos else math.pi/2
    values=[fn(lo),fn(hi)]
    for k in range(math.ceil((lo-offset)/math.pi),math.floor((hi-offset)/math.pi)+1):
        values.append(fn(offset+k*math.pi))
    return min(values),max(values)


def _sum(a,b):return a[0]+b[0],a[1]+b[1]
def _mul(a,b):
    values=[x*y for x in a for y in b]
    return min(values),max(values)
def _scale(a,k):return _mul(a,(k,k))


def _planar_bounds(t,c,cos=False):
    """Extrema on all four edges, keeping thigh/knee coupling intact."""
    tl,th=t-.25,t+.25;cl,ch=c-.25,c+.25
    points=[(a,b) for a in (tl,th) for b in (cl,ch)]
    offset=0. if cos else math.pi/2
    for b in (cl,ch):
        lo,hi=tl+b/2,th+b/2
        for k in range(math.ceil((lo-offset)/math.pi),math.floor((hi-offset)/math.pi)+1):
            points.append((offset+k*math.pi-b/2,b))
    for a in (tl,th):
        lo,hi=a+cl,a+ch
        for k in range(math.ceil((lo-offset)/math.pi),math.floor((hi-offset)/math.pi)+1):
            points.append((a,offset+k*math.pi-a))
    fn=math.cos if cos else math.sin
    values=[-.35*(fn(a)+fn(a+b)) for a,b in points]
    return min(values),max(values)


def foot_position(q,side):
    h,t,c=q
    x=-.35*(math.sin(t)+math.sin(t+c))
    z=-.35*(math.cos(t)+math.cos(t+c))
    y=side*.11973
    return x,y*math.cos(h)-z*math.sin(h),y*math.sin(h)+z*math.cos(h)


@lru_cache(maxsize=32)
def _bounds(reference):
    result=[]
    for i,side in zip(range(0,12,3),(-1,1,-1,1)):
        h,t,c=reference[i:i+3]
        sh=_trig_bounds(h-.25,h+.25);ch=_trig_bounds(h-.25,h+.25,True)
        x=_planar_bounds(t,c);z=_planar_bounds(t,c,True)
        y=_sum(_scale(ch,side*.11973),_scale(_mul(z,sh),-1))
        z=_sum(_scale(sh,side*.11973),_mul(z,ch))
        result.append((x,y,z))
    return result


def standing_feet(q,references):
    positions=[foot_position(q[i:i+3],side) for i,side in zip(range(0,12,3),(-1,1,-1,1))]
    envelopes=[_bounds(tuple(r)) for r in references]
    # Each leg must fit one full accepted stance envelope, not independently
    # combine an x from one stance with a z from another.
    valid=all(any(all(lo-1e-6<=v<=hi+1e-6 for v,(lo,hi) in zip(p,e[leg]))
                  for e in envelopes) for leg,p in enumerate(positions))
    return valid,positions


def tilted_support(q,quaternion):
    """Four supporting feet on one level plane, including a tilted body.

    Used only for leaving the tilted policy. Motion/freshness checks still
    come from posture_checks; this does not relax the upright entry gate.
    """
    if len(q)!=12 or len(quaternion)!=4 or not all(math.isfinite(x) for x in [*q,*quaternion]):return False
    norm=math.sqrt(sum(x*x for x in quaternion))
    if not .9<norm<1.1:return False
    w,x,y,z=[v/norm for v in quaternion]
    up=(-2*(w*y-x*z),2*(w*x+y*z),1-2*(x*x+y*y))
    roll=math.atan2(up[1],up[2]);pitch=math.atan2(-up[0],math.hypot(up[1],up[2]))
    if max(abs(roll),abs(pitch))>math.radians(20)+.14:return False
    heights=[]
    for leg,(ox,oy,side) in enumerate(((.3285,-.072,-1),(.3285,.072,1),(-.3285,-.072,-1),(-.3285,.072,1))):
        foot=foot_position(q[3*leg:3*leg+3],side)
        heights.append(sum(a*b for a,b in zip(up,(ox+foot[0],oy+foot[1],foot[2]))))
    return all(-.75<h<-.30 for h in heights) and max(heights)-min(heights)<=.06
