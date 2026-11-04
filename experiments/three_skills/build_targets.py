"""Original procedural targets, not extracted from the SuperMimic web demo."""
import json, re, argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation
from mjlab.asset_zoo.robots import get_g1_robot_cfg

HERE=Path(__file__).resolve().parent
OUT=HERE/'targets'
OUT.mkdir(exist_ok=True)
DT=.02

def smooth(u):
    u=np.clip(u,0,1)
    return u*u*u*(10-15*u+6*u*u)

def save(name,q,meta,m):
    q=np.asarray(q)
    vel=np.zeros((len(q),m.nv))
    vel[:,:3]=np.gradient(q[:,:3],DT,axis=0)
    rot=Rotation.from_quat(q[:,[4,5,6,3]])
    vel[1:-1,3:6]=(rot[2:]*rot[:-2].inv()).as_rotvec()/(2*DT)
    vel[:,6:]=np.gradient(q[:,7:],DT,axis=0)
    assert np.isfinite(q).all() and np.isfinite(vel).all()
    assert np.max(np.abs(vel[0]))<1e-5
    limits=m.jnt_range[1:]
    assert (q[:,7:]>=limits[:,0]-1e-5).all() and (q[:,7:]<=limits[:,1]+1e-5).all()
    np.savez_compressed(OUT/(name+'.npz'),qpos=q.astype('float32'),qvel=vel.astype('float32'),starts=np.array([0]),lengths=np.array([len(q)]),names=np.array([name]),fps=np.array(50.))
    meta.update(name=name,frames=len(q),duration_s=(len(q)-1)*DT,source='original procedural target',pretrained_policy=False,reference_motion=True,max_joint_velocity=float(abs(vel[:,6:]).max()))
    (OUT/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)

def build_b2_model():
    src=HERE/'vendor/unitree_mujoco/unitree_robots/b2/b2.xml'
    root=ET.parse(src).getroot()
    root.find('compiler').set('meshdir',str(src.parent/'assets'))
    ET.SubElement(root,'option',timestep='.005',integrator='implicitfast',iterations='10',ls_iterations='20')
    wb=root.find('worldbody')
    ET.SubElement(wb,'geom',name='floor',type='plane',size='0 0 .1',friction='1 .005 .0001',rgba='.3 .35 .4 1')
    ET.SubElement(wb,'light',pos='0 0 4',dir='0 0 -1',directional='true')
    joints={j.get('name'):j for j in root.findall('.//joint') if j.get('name')}
    for act in root.find('actuator'):
        force=act.get('ctrlrange')
        act.tag='position'
        act.set('kp','250');act.set('kv','8')
        act.set('forcerange',force)
        act.set('ctrlrange',joints[act.get('joint')].get('range'))
    for side in ('FL','FR','RL','RR'):
        calf=root.find(f'.//body[@name="{side}_calf"]')
        for g in calf.findall('geom'):
            if g.get('type')=='sphere':
                g.set('name',side+'_foot_collision');g.set('friction','1 .005 .0001')
    for tag in ('keyframe','sensor'):
        e=root.find(tag)
        if e is not None:root.remove(e)
    output=HERE/'b2_train.xml'
    ET.ElementTree(root).write(output,encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(output))

def b2():
    m=build_b2_model();d=mujoco.MjData(m)
    feet=[m.geom(x+'_foot_collision').id for x in ('FL','FR','RL','RR')]
    q0=np.r_[np.zeros(3),1.,0.,0.,0.,np.tile([0.,.7,-1.4],4)]
    def fk(q):
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        return d.geom_xpos[feet].copy(),d.subtree_com[1].copy()
    p,com=fk(q0);q0[2]+=.032-p[:,2].min()
    p0,c0=fk(q0);anchor=p0[2:].mean(0);initial_dx=c0[0]-anchor[0]
    qs=[];maxerr=0.;minfront=1.
    for t in np.arange(0,9.32,DT):
        u=smooth((t-.8)/2.5)
        q=q0.copy();pitch=-np.deg2rad(82)*u
        q[3:7]=Rotation.from_euler('y',pitch).as_quat()[[3,0,1,2]]
        q[[8,11]]=.7-1.05*u;q[[9,12]]=-1.4-.7*u
        def place(thigh):
            qq=q.copy();qq[[14,17]]=thigh;qq[:3]=0
            p,c=fk(qq);qq[:3]=anchor-p[2:].mean(0)
            p,c=fk(qq)
            return qq,p,c
        def error(thigh):
            qq,p,c=place(thigh)
            return (c[0]-anchor[0]-(1-u)*initial_dx)**2
        opt=minimize_scalar(error,bounds=(.2,3.2),method='bounded',options={'xatol':1e-10})
        q,p,c=place(opt.x)
        maxerr=max(maxerr,float(np.max(np.linalg.norm(p[2:]-p0[2:],axis=-1))))
        minfront=min(minfront,float(p[:2,2].min()-.032))
        qs.append(q)
    assert maxerr<.002 and minfront>-.005,(maxerr,minfront)
    save('b2_hindstand',qs,dict(robot='Unitree B2',target_pitch_deg=82,initial_hold_s=.8,rise_s=2.5,final_hold_s=6.,hind_foot_max_drift_m=maxerr,minimum_front_foot_clearance_m=minfront,final_com_support_error_m=abs(c[0]-anchor[0]),joint_count=12,model_source_commit='1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d'),m)

def g1():
    cfg=get_g1_robot_cfg();m=cfg.spec_fn().compile();d=mujoco.MjData(m)
    q0=np.r_[0.,0.,.76,1.,0.,0.,0.,np.zeros(29)]
    for j in range(1,m.njnt):
        for pat,v in cfg.init_state.joint_pos.items():
            if re.fullmatch(pat,m.joint(j).name):q0[m.jnt_qposadr[j]]=v
    feet=[m.body(s+'_ankle_roll_link').id for s in ('left','right')]
    geoms=[[i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(s+'_foot') and 'collision' in (m.geom(i).name or '')] for s in ('left','right')]
    legs=[[m.joint(s+'_'+j+'_joint').id for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')] for s in ('left','right')]
    def fk(q):
        d.qpos[:]=q;mujoco.mj_kinematics(m,d)
        return d.xpos[feet].copy(),d.xmat[feet].reshape(2,3,3).copy()
    p0,_=fk(q0)
    for side in range(2):
        low=min(m.geom_pos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix()[2,2]) for g in geoms[side])
        p0[side,2]=-low-.0005
    def joint(q,name,v):q[m.jnt_qposadr[m.joint(name+'_joint').id]]=v
    def ik(q,p,r):
        err=0.
        for side in range(2):
            adr=m.jnt_qposadr[legs[side]];lim=m.jnt_range[legs[side]]
            def fun(x):
                qq=q.copy();qq[adr]=x;pp,rr=fk(qq)
                return np.r_[8*(pp[side]-p[side]),Rotation.from_matrix(r[side].T@rr[side]).as_rotvec()]
            fit=least_squares(fun,np.clip(q[adr],lim[:,0]+1e-4,lim[:,1]-1e-4),bounds=(lim[:,0]+1e-5,lim[:,1]-1e-5),max_nfev=65)
            q[adr]=fit.x;err=max(err,np.linalg.norm(fun(fit.x)[:3])/8)
        return q,err
    # Funky Chicken: bent-elbow wing flaps, shoulder/chest pulses, hip bounce,
    # alternating knee/heel accents. Feet stay supported, avoiding a sliding dance.
    dance=[];maxerr=0.
    for t in np.arange(0,9.02,DT):
        q=q0.copy();u=smooth((t-.6)/.6)*(1-smooth((t-7.8)/.6))
        ph=2*np.pi*1.6*(t-.6);p=p0.copy();r=np.tile(np.eye(3),(2,1,1))
        q[1]=u*.018*np.sin(ph/2);q[2]=.75-u*(.025+.018*np.cos(2*ph))
        joint(q,'waist_yaw',u*.18*np.sin(ph/2));joint(q,'waist_pitch',u*.10*np.sin(ph))
        for si,side in enumerate(('left','right')):
            sign=1 if si==0 else -1
            joint(q,side+'_shoulder_roll',sign*(.2+u*(.6+.22*np.sin(ph))))
            joint(q,side+'_shoulder_pitch',.2+u*.25*np.cos(ph))
            joint(q,side+'_shoulder_yaw',sign*u*.18*np.sin(ph))
            joint(q,side+'_elbow',.6+u*(.65+.18*np.cos(ph)))
            joint(q,side+'_wrist_pitch',-u*.25)
            p[si,2]+=u*.018*max(0,sign*np.sin(ph/2))
        q,e=ik(q,p,r);dance.append(q);maxerr=max(maxerr,e)
    assert maxerr<.012,maxerr
    save('g1_funky_chicken',dance,dict(robot='G1',style='Funky Chicken inspired original choreography',beat_bpm=96,max_foot_ik_error_m=maxerr,official_clip_reproduction=False),m)
    # Original 360-degree vertical jump target: planted countermovement, full
    # rotation in flight, two-foot landing, recovery. No pre-existing motion.
    jump=[];maxerr=0.;clear=[];angles=[]
    take=1.6;air=.66;land=take+air;omega=2*np.pi/air
    for t in np.arange(0,4.62,DT):
        q=q0.copy();p=p0.copy();r=np.tile(np.eye(3),(2,1,1));theta=0.
        if t<take:
            if t>.7:q[2]=.76-.13*smooth((t-.7)/.5)
            if t>take-.18:
                u=(t-take+.18)/.18
                q[2]=.63*(2*u**3-3*u**2+1)+.79*(-2*u**3+3*u**2)+(.5*9.81*air)*.18*(u**3-u**2)
            if t>take-.25:
                tau=take-t;theta=-omega*tau*(1-tau/.25)**2
                q[3:7]=Rotation.from_euler('z',theta).as_quat()[[3,0,1,2]]
            twist=-.30*smooth((t-.8)/.55)+.30*smooth((t-1.35)/.25)
            joint(q,'waist_yaw',twist)
        elif t<land:
            a=(t-take)/air;theta=2*np.pi*a
            q[2]=.79+.5*9.81*(t-take)*(air-(t-take))
            rz=Rotation.from_euler('z',theta);q[3:7]=rz.as_quat()[[3,0,1,2]]
            p=rz.apply(p0);p[:,2]+=(q[2]-.76)+.08*np.sin(np.pi*a)
            r[:]=rz.as_matrix()
        else:
            theta=2*np.pi;q[2]=.79-.12*smooth((t-land)/.22)+.09*smooth((t-land-.22)/.65)
            if t<land+.35:
                tau=t-land;theta+=omega*tau*(1-tau/.35)**2
                q[3:7]=Rotation.from_euler('z',theta).as_quat()[[3,0,1,2]]
        swing=smooth((t-1.15)/.45)*(1-smooth((t-land)/.6))
        for si,side in enumerate(('left','right')):
            joint(q,side+'_shoulder_pitch',.2-.9*swing)
            joint(q,side+'_elbow',.6+.45*swing)
        q,e=ik(q,p,r);jump.append(q);maxerr=max(maxerr,e);clear.append(p[:,2].min()-p0[:,2].min());angles.append(theta)
    assert maxerr<.015,maxerr
    save('g1_jump_full_turn',jump,dict(robot='G1',target_turn_deg=360,planned_takeoff_s=take,planned_landing_s=land,flight_s=air,target_root_rise_m=9.81*air**2/8,max_foot_ik_error_m=maxerr,official_clip_reproduction=False),m)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robot',choices=['g1','b2','all'],default='all');a=p.parse_args()
    if a.robot in ('b2','all'):b2()
    if a.robot in ('g1','all'):g1()
