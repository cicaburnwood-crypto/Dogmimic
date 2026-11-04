"""Finite CPU kinematic check; this does NOT train or initialize the Actor."""
import json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from bootstrap import HERE
from common import write_json
from contract import Q0,ACTION_SCALE

def main():
    meta=json.loads((HERE/'model.json').read_text());m=mujoco.MjModel.from_xml_path(str(HERE/'b2_attitude.xml'));d=mujoco.MjData(m)
    feet=[m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    d.qpos[:]=[0,0,meta['standing_height_m'],1,0,0,0]+Q0;mujoco.mj_forward(m,d)
    fixed=d.geom_xpos[feet].copy();fixed[:,2]=.032
    lo=np.maximum(m.jnt_range[1:,0]+.02,np.array(Q0)-np.array(ACTION_SCALE))
    hi=np.minimum(m.jnt_range[1:,1]-.02,np.array(Q0)+np.array(ACTION_SCALE))
    rows=[]
    for axis in range(2):
        for degree in [-30.,-20.,-10.,0.,10.,20.,30.]:
            rp=np.zeros(3);rp[axis]=degree;quat=Rotation.from_euler('xyz',rp,degrees=True).as_quat()[[3,0,1,2]]
            def residual(q):
                d.qpos[:]=[0,0,meta['standing_height_m'],*quat,*q];mujoco.mj_forward(m,d)
                return (d.geom_xpos[feet]-fixed).ravel()
            opt=least_squares(residual,Q0,bounds=(lo,hi),max_nfev=160,xtol=1e-10,gtol=1e-10,ftol=1e-10)
            residual(opt.x)
            foot_error=np.linalg.norm(d.geom_xpos[feet]-fixed,axis=-1)
            rows.append(dict(axis=['roll','pitch'][axis],degrees=degree,max_foot_error_m=float(foot_error.max()),joint_limit_margin_rad=float(np.minimum(opt.x-lo,hi-opt.x).min()),q=opt.x.tolist(),geometrically_reachable=bool(foot_error.max()<.003)))
    result=dict(type='kinematic_feasibility_only',actor_initialized_from_IK=False,all_reachable=all(r['geometrically_reachable'] for r in rows),conditions=rows)
    write_json(HERE/'feasibility.json',result);print(json.dumps(result))

if __name__=='__main__':main()
