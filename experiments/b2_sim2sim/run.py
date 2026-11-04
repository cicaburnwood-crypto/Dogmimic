"""Frozen B2 deployment-policy comparison. No training, DDS, SSH or robot control.

MuJoCo and mjlab share MJCF; Bullet imports explicit inertias and collision shapes.
Bullet cannot represent MuJoCo's generalized armature exactly: its results are
labelled a model-transfer diagnostic, never equivalent-physics acceptance.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE/'inputs'), str(HERE/'deps')]
from runtime import Controller, rotation
from pd_adapter import saturated_pd_target
INPUTS = HERE/'inputs'
META = json.loads((INPUTS/'hindstand/policy.json').read_text())
PERM = np.array(META['policy_to_sdk'])
MEASURED = json.loads((INPUTS/'measured_start.json').read_text())
REF = dict(np.load(INPUTS/'hindstand/reference.npz'))
FOOT_NAMES = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs():
    for name, sha in json.loads((INPUTS/'manifest.json').read_text()).items():
        assert digest(INPUTS/name) == sha, name
    assert META['weights_sha256'] == '47e2a0ee384147c3682ce6c260d5d08985f8ea2e46fbdb358b5428e1ee91950c'


def original_metrics():
    p = ROOT/'experiments/b2_cycle/evaluate_cycle.py'
    assert digest(p) == '5e9612aaae20e7c5611f6a8a1d3aaa0e5d5e7c930ec03baf11c593a9301a99bd'
    tree = ast.parse(p.read_text())
    nodes = [x for x in tree.body if
             (isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CRITERIA' for t in x.targets))
             or (isinstance(x, ast.FunctionDef) and x.name in ('metrics_for_window', 'stable_mask'))]
    ns = {'np': np}; exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)
    return ns


def model_for(kg, case, dt):
    m = mujoco.MjModel.from_xml_path(str(INPUTS/f'b2_{kg}kg.xml'))
    m.opt.timestep = dt
    m.geom_friction[:, 0] = case.get('friction', 1.)
    m.body_mass[:] *= case.get('mass', 1.)
    m.body_inertia[:] *= case.get('mass', 1.)
    if 'armature_override' in case:
        m.dof_armature[:] = case['armature_override']
    mujoco.mj_setConst(m, mujoco.MjData(m))
    assert m.nq == 19 and m.nv == 18 and m.nu == 12
    assert [m.joint(j).name for j in range(1, 13)] == META['policy_joint_names']
    return m


def initial_states(m, case, n):
    qs, vs = [], []
    for i in range(n):
        q = np.zeros(19); v = np.zeros(18)
        if case['start'] == 'reference':
            q[:3] = REF['pos'][0]
            q[3:7] = Rotation.from_matrix(REF['rot'][0]).as_quat()[[3,0,1,2]]
            q[7:] = REF['q'][0]; v[:3] = REF['v'][0]
            v[3:6] = REF['rot'][0].T @ REF['w'][0]; v[6:] = REF['qd'][0]
        else:
            q[:3] = [0, 0, .6]
            q[3:7] = MEASURED['quaternion']; q[3:7] /= np.linalg.norm(q[3:7])
            q[7:] = np.array(MEASURED['q_sdk'])[PERM]
            v[3:6] = MEASURED['gyro']; v[6:] = np.array(MEASURED['dq_sdk'])[PERM]
        if i:
            q[7:] += np.random.default_rng(20261103+i).normal(0, .006, 12)
        if case['start'] != 'reference':
            d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m,d)
            ids = [m.geom(x[:2]+'_foot_collision').id for x in FOOT_NAMES]
            # No trustworthy real world height was measured. Explicitly align
            # the lowest collision sphere to the floor; do not silently settle.
            q[2] -= min(d.geom_xpos[g,2]-m.geom_size[g,0] for g in ids)
        qs.append(q); vs.append(v)
    return np.array(qs), np.array(vs)


class Cpu:
    def __init__(self, m, q, v, **kw):
        self.m = m; self.ds = [mujoco.MjData(m) for _ in q]
        self.act_to_joint=np.array([int(m.actuator_trnid[i,0])-1 for i in range(m.nu)])
        for d,qi,vi in zip(self.ds,q,v):
            d.qpos[:] = qi; d.qvel[:] = vi; mujoco.mj_forward(m,d)
    def state(self):
        return np.array([d.qpos.copy() for d in self.ds]), np.array([d.qvel.copy() for d in self.ds])
    def step(self, wire, force):
        for d, u in zip(self.ds, wire):
            d.ctrl[:] = u[self.act_to_joint]; d.xfrc_applied[:] = 0; d.xfrc_applied[1,:3] = force
            mujoco.mj_step(self.m,d)
        return np.array([d.actuator_force.copy()[np.argsort(self.act_to_joint)] for d in self.ds])
    def close(self): pass


class Warp:
    def __init__(self, m, q, v, **kw):
        import torch
        from mjlab.sim import MujocoCfg, SimulationCfg
        from mjlab.sim.sim import Simulation
        torch.set_num_threads(1)
        self.torch = torch
        self.act_to_joint=np.array([int(m.actuator_trnid[i,0])-1 for i in range(m.nu)])
        self.sim = Simulation(len(q), SimulationCfg(nconmax=64, njmax=400,
            mujoco=MujocoCfg(timestep=float(m.opt.timestep), iterations=int(m.opt.iterations),
                            ls_iterations=int(m.opt.ls_iterations),
                            tolerance=float(m.opt.tolerance), ls_tolerance=float(m.opt.ls_tolerance),
                            impratio=float(m.opt.impratio),
                            cone='elliptic' if int(m.opt.cone) else 'pyramidal')),
            m, device='cuda:0')
        self.sim.reset()
        self.sim.data.qpos[:] = torch.as_tensor(q, device='cuda:0', dtype=torch.float32)
        self.sim.data.qvel[:] = torch.as_tensor(v, device='cuda:0', dtype=torch.float32)
        self.sim.forward()
    def state(self):
        return self.sim.data.qpos.cpu().numpy().copy(), self.sim.data.qvel.cpu().numpy().copy()
    def step(self, wire, force):
        t = self.torch
        self.sim.data.ctrl[:] = t.as_tensor(wire[:,self.act_to_joint], device='cuda:0', dtype=t.float32)
        self.sim.data.xfrc_applied[:] = 0
        self.sim.data.xfrc_applied[:,1,:3] = t.as_tensor(force, device='cuda:0', dtype=t.float32)
        self.sim.step()
        return self.sim.data.actuator_force.cpu().numpy().copy()[:,np.argsort(self.act_to_joint)]
    def close(self):
        del self.sim
        import gc; gc.collect(); self.torch.cuda.empty_cache()


def fmt(x): return ' '.join(f'{v:.17g}' for v in np.asarray(x).reshape(-1))
def rpy(q): return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_euler('xyz')


def export_urdf(m, path):
    """Transfer all dynamic bodies, inertias, joints and collision primitives."""
    root = ET.Element('robot', name='b2_frozen_mjcf')
    for b in range(1,m.nbody):
        link = ET.SubElement(root,'link',name=m.body(b).name)
        inertial = ET.SubElement(link,'inertial')
        ET.SubElement(inertial,'origin',xyz=fmt(m.body_ipos[b]),rpy='0 0 0')
        ET.SubElement(inertial,'mass',value=str(m.body_mass[b]))
        R = rotation(m.body_iquat[b]).astype(float)
        I = R @ np.diag(m.body_inertia[b]) @ R.T
        ET.SubElement(inertial,'inertia',**{k:str(I[i,j]) for k,i,j in
            [('ixx',0,0),('iyy',1,1),('izz',2,2),('ixy',0,1),('ixz',0,2),('iyz',1,2)]})
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=b or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
            typ=int(m.geom_type[g]); size=m.geom_size[g]
            assert typ in (2,5,6), ('Unsupported collision shape',g,typ)
            for tag in ('collision','visual'):
                node=ET.SubElement(link,tag)
                ET.SubElement(node,'origin',xyz=fmt(m.geom_pos[g]),rpy=fmt(rpy(m.geom_quat[g])))
                geo=ET.SubElement(node,'geometry')
                if typ==2: ET.SubElement(geo,'sphere',radius=str(size[0]))
                elif typ==5: ET.SubElement(geo,'cylinder',radius=str(size[0]),length=str(2*size[1]))
                elif typ==6: ET.SubElement(geo,'box',size=fmt(2*size))
        if b>1:
            j=int(m.body_jntadr[b]); assert m.body_jntnum[b]==1 and int(m.jnt_type[j])==3
            assert np.linalg.norm(m.jnt_pos[j]) < 1e-10
            joint=ET.SubElement(root,'joint',name=m.joint(j).name,type='revolute')
            ET.SubElement(joint,'parent',link=m.body(int(m.body_parentid[b])).name)
            ET.SubElement(joint,'child',link=m.body(b).name)
            ET.SubElement(joint,'origin',xyz=fmt(m.body_pos[b]),rpy=fmt(rpy(m.body_quat[b])))
            ET.SubElement(joint,'axis',xyz=fmt(m.jnt_axis[j]))
            ET.SubElement(joint,'limit',lower=str(m.jnt_range[j,0]),upper=str(m.jnt_range[j,1]),effort='300',velocity='1000')
            ET.SubElement(joint,'dynamics',damping=str(m.dof_damping[int(m.jnt_dofadr[j])]),friction='0')
    ET.indent(root); ET.ElementTree(root).write(path,encoding='unicode')


class Bullet:
    def __init__(self,m,q,v,out,case,**kw):
        import pybullet as p
        self.p=p; self.m=m; self.clients=[]; self.bodies=[]; self.audits=[]
        path=out/'b2.urdf'; export_urdf(m,path)
        for qi,vi in zip(q,v):
            c=p.connect(p.DIRECT);self.clients.append(c)
            p.setGravity(0,0,-9.81,physicsClientId=c)
            p.setTimeStep(float(m.opt.timestep),physicsClientId=c)
            p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,physicsClientId=c)
            floor=p.createMultiBody(0,p.createCollisionShape(p.GEOM_PLANE,physicsClientId=c),physicsClientId=c)
            robot=p.loadURDF(str(path),[0,0,.6],flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION|p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,physicsClientId=c)
            self.bodies.append(robot)
            assert p.getNumJoints(robot,physicsClientId=c)==12
            names=[p.getJointInfo(robot,j,physicsClientId=c)[1].decode() for j in range(12)]
            assert names==META['policy_joint_names'],names
            mu=math.sqrt(case.get('friction',1.))
            p.changeDynamics(floor,-1,lateralFriction=mu,restitution=0,physicsClientId=c)
            for j in range(-1,12):
                p.changeDynamics(robot,j,lateralFriction=mu,restitution=0,linearDamping=0,angularDamping=0,physicsClientId=c)
            for sig in m.exclude_signature:
                b1=int(sig)>>16; b2=int(sig)&65535
                p.setCollisionFilterPair(robot,robot,b1-2,b2-2,0,physicsClientId=c)
            dyn=p.getDynamicsInfo(robot,-1,physicsClientId=c)
            bodyQ=qi[3:7][[1,2,3,0]]
            cp,cq=p.multiplyTransforms(qi[:3].tolist(),bodyQ.tolist(),dyn[3],dyn[4])
            p.resetBasePositionAndOrientation(robot,cp,cq,physicsClientId=c)
            R=rotation(qi[3:7]); omega=R@vi[3:6]
            p.resetBaseVelocity(robot,vi[:3]+np.cross(omega,R@m.body_ipos[1]),omega,physicsClientId=c)
            for j in range(12):p.resetJointState(robot,j,float(qi[j+7]),float(vi[j+6]),physicsClientId=c)
            p.setJointMotorControlArray(robot,range(12),p.VELOCITY_CONTROL,forces=[0.]*12,physicsClientId=c)
            d=mujoco.MjData(m);d.qpos[:]=qi;mujoco.mj_forward(m,d)
            fk=max(np.linalg.norm(np.array(p.getLinkState(robot,j,computeForwardKinematics=1,physicsClientId=c)[4])-d.xpos[j+2]) for j in range(12))
            mass=sum(p.getDynamicsInfo(robot,j,physicsClientId=c)[0] for j in range(-1,12))
            assert fk<2e-6 and abs(mass-m.body_mass.sum())<1e-8,(fk,mass)
            self.audits.append(dict(fk_max_error_m=float(fk),mass_kg=float(mass)))
        save(out/'bullet_transfer_audit.json',dict(passed_geometry_mass=True,trials=self.audits,
             unsupported_mujoco_armature=m.dof_armature.tolist(),
             differences=['Generalized armature 0.1 is not represented by Bullet.',
                          'Free-root damping is applied as external forces; joint damping comes from URDF.',
                          'Contact solver and compliance differ; no fitted contact parameters.',
                          'Bullet applies explicit PD torque; MuJoCo uses implicitfast position actuators.'],
             equivalent_physics_acceptance=False))
    def state(self):
        p=self.p;qs=[];vs=[]
        for c,b in zip(self.clients,self.bodies):
            cp,cq=p.getBasePositionAndOrientation(b,physicsClientId=c)
            dyn=p.getDynamicsInfo(b,-1,physicsClientId=c)
            ip,iq=p.invertTransform(dyn[3],dyn[4]);pos,quat=p.multiplyTransforms(cp,cq,ip,iq)
            lin,ang=p.getBaseVelocity(b,physicsClientId=c);quat=np.array(quat)[[3,0,1,2]];R=rotation(quat)
            lin=np.array(lin)-np.cross(ang,R@self.m.body_ipos[1])
            js=p.getJointStates(b,range(12),physicsClientId=c)
            qs.append(np.r_[pos,quat,[j[0] for j in js]])
            vs.append(np.r_[lin,R.T@ang,[j[1] for j in js]])
        return np.array(qs),np.array(vs)
    def step(self,wire,force):
        p=self.p;q,v=self.state();taus=[]
        for i,(c,b) in enumerate(zip(self.clients,self.bodies)):
            ctrl=np.clip(wire[i],self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
            tau=np.clip(250*(ctrl-q[i,7:])-8*v[i,6:],self.m.actuator_forcerange[:,0],self.m.actuator_forcerange[:,1])
            p.setJointMotorControlArray(b,range(12),p.TORQUE_CONTROL,forces=tau,physicsClientId=c)
            R=rotation(q[i,3:7])
            # MuJoCo xfrc_applied is a COM wrench. Keep the push at the COM;
            # generalized free-joint linear damping acts at the root origin.
            cp,_=p.getBasePositionAndOrientation(b,physicsClientId=c)
            p.applyExternalForce(b,-1,force,cp,p.WORLD_FRAME,physicsClientId=c)
            p.applyExternalForce(b,-1,-v[i,:3],q[i,:3],p.WORLD_FRAME,physicsClientId=c)
            p.applyExternalTorque(b,-1,-R@v[i,3:6],p.WORLD_FRAME,physicsClientId=c)
            p.stepSimulation(physicsClientId=c);taus.append(tau)
        return np.array(taus)
    def close(self):
        for c in self.clients:self.p.disconnect(c)


def kinematics(m, q, v):
    feet=[];com=[];cv=[];roots=[];incl=[]
    for qi,vi in zip(q,v):
        d=mujoco.MjData(m);d.qpos[:]=qi;d.qvel[:]=vi;mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
        feet.append([d.xpos[m.body(nm).id]+d.xmat[m.body(nm).id].reshape(3,3)@np.array([0,0,-.35]) for nm in FOOT_NAMES])
        com.append(d.subtree_com[1].copy());cv.append(d.subtree_linvel[1].copy())
        R=rotation(qi[3:7]);roots.append(np.r_[qi[:3],qi[3:7],qi[7:],vi[:3],R@vi[3:6],vi[6:]])
        incl.append(np.degrees(np.arcsin(np.clip(R[2,0],-1,1))))
    return dict(state=np.array(roots),feet=np.array(feet),com_pos=np.array(com),com_v=np.array(cv),inclination=np.array(incl))


def summarise(arr,sub,initial):
    ns=original_metrics();crit=ns['CRITERIA'];t=arr['time_s'];stand=(t>=4)&(t<=9.3);end=t>=13
    sm=ns['metrics_for_window'](arr,stand);fm=ns['metrics_for_window'](arr,end)
    feet=arr['feet'];rot=np.array([[rotation(q[3:7]) for q in step] for step in arr['qpos']])
    up=rot[:,:,2,0];rzz=rot[:,:,2,2];alive=arr['alive'];complete=alive.all(0)
    hind=((feet[:,:,2:,2]>=.020)&(feet[:,:,2:,2]<=.045)).all(-1)
    allfeet=((feet[:,:,:,2]>=.020)&(feet[:,:,:,2]<=.045)).all(-1)
    hold=((up>np.sin(np.deg2rad(75)))&(feet[:,:,:2,2].min(-1)>.35)&hind&alive)[stand].mean(0)
    returned=((rzz>np.cos(np.deg2rad(8)))&allfeet&alive)[end].mean(0)
    drift=np.linalg.norm(feet[stand,:,2:,:2]-feet[np.flatnonzero(stand)[0],:,2:,:2],axis=-1).max(axis=(0,2))
    cycle=complete&(hold>=crit['hold_pose_fraction_min'])&(returned>=crit['return_pose_fraction_min'])
    task=cycle&ns['stable_mask'](sm)&(drift<=crit['foot_xy_drift_max'])&ns['stable_mask'](fm)
    speed=np.max(np.abs(sub['dq']),axis=(0,2));power=np.max(sub['power'],axis=0)
    act=(speed<=6)&(power<=1501)
    rows=[]
    for i in range(len(complete)):
        first=np.flatnonzero(~alive[:,i]);pre=sub['time_s']<=1.8345823288
        rows.append(dict(trial=i,completed=bool(complete[i]),cycle=bool(cycle[i]),task_pass=bool(task[i]),
            actuator_pass=bool(act[i]),joint_pass=bool(task[i]&act[i]),
            first_termination_s=float(t[first[0]]) if len(first) else None,
            stand_pose_fraction=float(hold[i]),return_pose_fraction=float(returned[i]),
            hind_foot_xy_drift_m=float(drift[i]),peak_speed_rad_s=float(speed[i]),peak_power_w=float(power[i]),
            pre_real_failure_peak_speed_rad_s=float(abs(sub['dq'][pre,i]).max()),
            pre_real_failure_peak_power_w=float(sub['power'][pre,i].max()),
            stand={k:v[i].tolist() for k,v in sm.items()},returned={k:v[i].tolist() for k,v in fm.items()}))
    return dict(trials=rows,counts={k:sum(r[k] for r in rows) for k in ('completed','cycle','task_pass','actuator_pass','joint_pass')},
                criteria=crit,speed_limit_rad_s=6,power_limit_w=1501,full_acceptance=False,
                scope='Paired diagnostic subset. Not 48/96 conditions x 64 trials, and not hardware acceptance.')


def run_case(backend, kg, case, n, dt, out):
    out.mkdir(parents=True,exist_ok=True);start=time.time();m=model_for(kg,case,dt)
    q,v=initial_states(m,case,n);initq=q.copy();initv=v.copy()
    sim={'mujoco':Cpu,'mjlab':Warp,'bullet':Bullet}[backend](m,q,v,out=out,case=case)
    controllers=[Controller(INPUTS/'hindstand') for _ in range(n)]
    target=q[:,7:].copy();initial_target=target.copy();history=[];alive=np.ones(n,bool)
    prev=kinematics(m,q,v)
    rec={k:[] for k in ('state','feet','com_pos','com_v','com_acc','center_acc','inclination','alive','qpos','qvel','target')}
    sub={k:[] for k in ('time_s','dq','tau','power','pitch','target_clipped')}
    steps=round(.02/dt);assert abs(steps*dt-.02)<1e-9
    for frame in range(775):
        for i,c in enumerate(controllers):
            q_sdk=np.empty(12);q_sdk[PERM]=q[i,7:];dq_sdk=np.empty(12);dq_sdk[PERM]=v[i,6:]
            u,_=c.step(q_sdk,dq_sdk,v[i,3:6],q[i,3:7],root_position=q[i,:3],advance_reference=True)
            alpha=min(1.,frame*.02/.5) if case.get('blend',True) else 1.
            target[i]=initial_target[i]*(1-alpha)+u[PERM]*alpha
        history.append(target.copy());delay=round(case.get('delay_ms',0)/20)
        effective=history[max(0,len(history)-1-delay)] if frame>=delay else initial_target
        for k in range(steps):
            wire=[]
            for i in range(n):
                w,_,_=saturated_pd_target(effective[i],q[i,7:],v[i,6:],250,8,META['torque_limits'],META['torque_envelope'])
                wire.append(w)
            wire=np.array(wire);now=frame*.02+k*dt
            # Same finite 16 N pulse for every backend, on world X at trunk COM.
            force=np.array([case.get('push_n',0.) if 1.5<=now<1.8 else 0.,0.,0.])
            tau=sim.step(wire,force);q,v=sim.state()
            sub['time_s'].append(now+dt);sub['dq'].append(v[:,6:].copy());sub['tau'].append(tau)
            sub['power'].append(abs(tau*v[:,6:]).sum(-1))
            sub['pitch'].append([np.degrees(np.arcsin(np.clip(rotation(qi[3:7])[2,0],-1,1))) for qi in q])
            sub['target_clipped'].append(np.any((wire<m.actuator_ctrlrange[:,0])|(wire>m.actuator_ctrlrange[:,1]),axis=1))
        s=kinematics(m,q,v);s['com_acc']=(s['com_v']-prev['com_v'])/.02;s['center_acc']=(v[:,:3]-initv[:,:3] if frame==0 else v[:,:3]-rec['qvel'][-1][:,:3])/.02
        for i in range(n):
            R=rotation(q[i,3:7]);ez=abs(q[i,2]-REF['pos'][frame+1,2]);tilt=abs(R[2,2]-REF['rot'][frame+1,2,2])
            hard=ez>.32 or tilt>.8 or abs(R[2,1])>.65 or q[i,2]<.23 or abs(v[i,6:]).max()>300 or abs(v[i,:3]).max()>50
            alive[i]&=not hard
        for key in ('state','feet','com_pos','com_v','com_acc','center_acc','inclination'):rec[key].append(s[key])
        rec['alive'].append(alive.copy());rec['qpos'].append(q.copy());rec['qvel'].append(v.copy());rec['target'].append(target.copy());prev=s
    sim.close();arr={k:np.array(a) for k,a in rec.items()};arr['time_s']=(np.arange(775)+1)*.02
    sub={k:np.array(a) for k,a in sub.items()};result=summarise(arr,sub,initq)
    result.update(backend=backend,kg=kg,case=case,physics_dt=dt,policy_dt=.02,substeps=steps,
                  measured_height_assumption='Lowest foot sphere aligned to floor; no settling; no trustworthy measured root height.',
                  elapsed_s=time.time()-start,model_mass_kg=float(m.body_mass.sum()),model_xml_sha256=digest(INPUTS/f'b2_{kg}kg.xml'),
                  weights_sha256=META['weights_sha256'],bullet_model_equivalent=False if backend=='bullet' else None)
    np.savez_compressed(out/'rollout.npz',**arr,initial_qpos=initq,initial_qvel=initv)
    np.savez_compressed(out/'substeps.npz',**sub);save(out/'result.json',result)
    print(json.dumps(dict(backend=backend,kg=kg,case=case['name'],counts=result['counts'],elapsed_s=result['elapsed_s'])),flush=True)
    return result


CASES=[dict(name='reference',start='reference'),dict(name='measured',start='measured'),
       dict(name='friction04',start='measured',friction=.4),
       dict(name='mass11',start='measured',mass=1.1),
       dict(name='push16',start='measured',push_n=16),
       dict(name='delay20',start='measured',delay_ms=20),
       dict(name='combined',start='measured',friction=.4,mass=1.1,push_n=16,delay_ms=20)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['mujoco','mjlab','bullet'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--trials',type=int,default=4)
    p.add_argument('--case',action='append');p.add_argument('--kg',type=int,choices=[0,5],action='append');p.add_argument('--dt',type=float,default=.002)
    a=p.parse_args();verify_inputs();rows=[];start=time.time()
    for kg in a.kg or [0,5]:
        for case in CASES:
            if a.case and case['name'] not in a.case:continue
            d=a.out/a.backend/f'{kg}kg_{case["name"]}'
            rows.append(run_case(a.backend,kg,case,a.trials,a.dt,d))
            save(a.out/f'{a.backend}_progress.json',dict(completed=len(rows),results=rows,elapsed_s=time.time()-start))
    verify_inputs();save(a.out/f'{a.backend}_DONE.json',dict(completed=len(rows),elapsed_s=time.time()-start))

if __name__=='__main__':main()
