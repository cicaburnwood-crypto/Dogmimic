"""Mounted A3 kinematics, sampled reach and conservative staged path checks.
No dynamics / obstacle perception claim: only configured robot collision meshes.
"""
import json, math
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import least_squares

MODEL=Path(__file__).resolve().parents[2]/'assets/simulation'
NOMINAL=.4959882021194876
MOUNT=np.array([.16200000095367434,0,.09701000010661316])
TCP_FROM_CONTROLLER=.131

def tilt_limit(h):return np.deg2rad(np.interp(h,[-.2,-.15,-.1,.05],[5,10,20,20]))
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi

class Geometry:
    def __init__(self):
        self.m=mujoco.MjModel.from_xml_path(str(MODEL/'b2_a3.xml'));self.d=mujoco.MjData(self.m)
        self.meta=json.loads((MODEL/'model.json').read_text());self.q0=np.array(self.meta['q0'])
        self.feet=[self.m.geom(x+'_foot_collision').id for x in ['FL','FR','RL','RR']]
        self.arm_lo=self.m.jnt_range[13:19,0].copy();self.arm_hi=self.m.jnt_range[13:19,1].copy()
        self.site=self.m.site('arm_tcp').id;self.armbody=self.m.body('a3_link6').id
        self.set([0,0,0,0],self.meta['arm_home'],self.q0)
        self.anchors=self.d.geom_xpos[self.feet].copy();self.eq=ConvexHull(self.anchors[:,:2]).equations
        self.cloud=None;self.solutions=None;self.tree=None
    def set(self,body,q,legs=None,root_xy=(0,0)):
        r,p,y,h=body;d=self.d
        d.qpos[:3]=[*root_xy,NOMINAL+h];d.qpos[3:7]=Rot.from_euler('xyz',[r,p,y]).as_quat()[[3,0,1,2]]
        d.qpos[7:19]=self.q0 if legs is None else legs;d.qpos[19:25]=q
        mujoco.mj_forward(self.m,d)
    def ee(self,body,q,legs=None,xy=(0,0)):
        self.set(body,q,legs,xy);return self.d.site_xpos[self.site].copy()
    def controller_flange(self,q):
        self.set([0,0,0,0],q)
        R=self.d.xmat[self.armbody].reshape(3,3).copy()
        p=self.d.xpos[self.armbody].copy()-np.array([0,0,NOMINAL])-MOUNT-R[:,2]*.006
        return p,Rot.from_matrix(R).as_quat()
    def measured(self,dog):
        s=dog['robot_state'];l=s['locomotion_inputs'];o=s['base_odometry']
        if not l['valid'] or not o['valid']:raise ValueError('狗端 IMU / 编码器 / 里程计未就绪')
        legs=np.array(l['q_rad']).reshape(4,3)[[1,0,3,2]].ravel()
        r,p=np.deg2rad(l['imu_rpy_deg'][:2]);yaw=np.deg2rad(o['rpy_deg'][2])
        self.set([r,p,0,0],self.meta['arm_home'],legs)
        height=float(np.median(NOMINAL-self.d.geom_xpos[self.feet,2]+.032))
        return dict(rpy=[float(r),float(p),float(yaw)],height=height,legs=legs.tolist(),position=o['position_m'])
    def leg_ik(self,body,enforce=True):
        r,p,y,h=body
        if enforce and (not -.200001<=h<=.050001 or max(abs(r),abs(p))>tilt_limit(h)+1e-6):return None
        self.set([r,p,0,h],self.meta['arm_home'])
        for _ in range(45):
            err=(self.anchors-self.d.geom_xpos[self.feet]).ravel()
            if np.max(abs(err))<.0015:break
            J=[]
            for g in self.feet:
                jp=np.zeros((3,self.m.nv));mujoco.mj_jacGeom(self.m,self.d,jp,None,g);J.append(jp[:,6:18])
            J=np.concatenate(J);dq=np.linalg.solve(J.T@J+np.eye(12)*1e-4,J.T@err)
            self.d.qpos[7:19]=np.clip(self.d.qpos[7:19]+np.clip(dq,-.12,.12),self.m.jnt_range[1:13,0]+.001,self.m.jnt_range[1:13,1]-.001)
            mujoco.mj_forward(self.m,self.d)
        if np.max(abs(self.anchors-self.d.geom_xpos[self.feet]))>.003:return None
        return self.d.qpos[7:19].copy()
    def safe(self,body,q,legs):
        self.set(body,q,legs)
        for c in self.d.contact:
            a,b=int(c.geom1),int(c.geom2)
            if (self.m.geom_bodyid[a]==0 and b in self.feet) or (self.m.geom_bodyid[b]==0 and a in self.feet):continue
            if c.dist<0:return False
        # Ground clearance at the modeled tool tip and static COM margin.
        if self.d.site_xpos[self.site,2]<.04:return False
        co=self.d.subtree_com[1,:2]@Rot.from_euler('z',-body[2]).as_matrix()[:2,:2].T
        if np.max(self.eq[:,:2]@co+self.eq[:,2])>-.015:return False
        return True
    def sample(self):
        from scipy.stats import qmc
        aq=qmc.Sobol(6,scramble=True,seed=731).random_base2(7)*(self.arm_hi-self.arm_lo)+self.arm_lo
        rows=[];points=[]
        for h in [0.,.05,-.1,-.15,-.2]:
            lim=tilt_limit(h)
            for r in [0,-lim,lim]:
                for p in [0,-lim,lim]:
                    body=[r,p,0,h];legs=self.leg_ik(body)
                    if legs is None:continue
                    for q in aq:
                        if not self.safe(body,q,legs):continue
                        pos=self.d.site_xpos[self.site].copy()
                        for yaw in np.deg2rad(np.arange(-180,180,30)):
                            b=[r,p,yaw,h];points.append(Rot.from_euler('z',yaw).apply(pos));rows.append([*b,*q,*legs])
        self.cloud=np.array(points);self.solutions=np.array(rows);self.tree=cKDTree(self.cloud)
        return self.cloud
    def solve_arm(self,body,target,q0,legs,xy=(0,0)):
        def residual(q):return self.ee(body,q,legs,xy)-target
        sol=least_squares(residual,np.clip(q0,self.arm_lo+1e-7,self.arm_hi-1e-7),bounds=(self.arm_lo,self.arm_hi),max_nfev=70,gtol=1e-8)
        if np.linalg.norm(sol.fun)>.003 or not self.safe(body,sol.x,legs):return None
        return sol.x
    def path(self,start,b,q0,q1,startlegs):
        # Stage 1: body moves with arm stationary, first reduce tilt before crouching.
        neutral=[0,0,start[2],float(np.clip(start[3],-.2,.05))]
        height=[0,0,b[2],b[3]]
        segments=[(start,neutral),(neutral,height),(height,b)]
        for a,z in segments:
            steps=max(2,int(max(np.max(np.abs(np.array(z[:3])-a[:3]))/np.deg2rad(2),abs(z[3]-a[3])/.005))+1)
            for t in np.linspace(0,1,steps):
                body=np.array(a)*(1-t)+np.array(z)*t;legs=self.leg_ik(body,enforce=False)
                if legs is None or not self.safe(body,q0,legs):return False
        legs=self.leg_ik(b)
        steps=max(3,int(np.max(abs(q1-q0))/np.deg2rad(1))+1)
        for t in np.linspace(0,1,steps):
            if not self.safe(b,q0*(1-t)+q1*t,legs):return False
        return True
    def plan(self,target,start,q0,legs):
        target=np.asarray(target,dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError('请输入有限 XYZ 坐标')
        _,idx=self.tree.query(target,k=min(90,len(self.cloud)))
        candidates=[(np.array(start),np.array(q0),np.array(legs))] if -.2<=start[3]<=.05 else []
        # Try arm-only / minimal body adjustment before moving the carrier.
        for body in ([start[0],start[1],start[2],float(np.clip(start[3],-.2,.05))],[0,0,start[2],0],[0,0,start[2],.05]):
            leg=self.leg_ik(body)
            if leg is not None:
                for seed in (q0,self.meta['arm_home']):candidates.append((np.array(body),np.array(seed),leg))
        for row in self.solutions[np.atleast_1d(idx)]:candidates.append((row[:4],row[4:10],row[10:22]))
        candidates.sort(key=lambda c:abs(wrap(c[0][2]-start[2]))+2*np.linalg.norm(c[0][:2]-start[:2])+3*abs(c[0][3]-start[3])+.1*np.linalg.norm(c[1]-q0))
        for body,seed,leg in candidates:
            q=self.solve_arm(body,target,seed,leg)
            if q is not None and self.path(start,body,np.array(q0),q,legs):
                return dict(body=body.tolist(),joints=q.tolist(),legs=leg.tolist(),target=target.tolist(),predicted=self.ee(body,q,leg).tolist(),position_error_m=float(np.linalg.norm(self.ee(body,q,leg)-target)))
        raise ValueError('未找到通过机体包络、支撑及分段碰撞采样的路径；请换点。点云是候选，不代表每点均有安全过渡路径。')
