"""CPU export and parity checks against the actual training implementations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,re,argparse
from pathlib import Path
import numpy as np
import torch
from runtime import Actor,Controller,rotation,sha256
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--skill',choices=['jump_60cm','funky_chicken'],required=True)
p.add_argument('--checkpoint');p.add_argument('--checkpoint-sha256');p.add_argument('--evaluation');p.add_argument('--engine');p.add_argument('--out');p.add_argument('--label');a=p.parse_args()
project=Path(a.project);out=Path(a.out) if a.out else Path(__file__).parent/'models'/a.skill;out.mkdir(parents=True,exist_ok=True)
jump=a.skill=='jump_60cm'
engine=Path(a.engine) if a.engine else project/('code' if jump else 'experiments/three_skills/engine')
sys.path.insert(0,str(engine))
from supermimic.networks import Policy
from supermimic.task import Task
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
from mjlab.entity.entity import Entity
torch.set_num_threads(1)
source=project/('deployments/g1_actions/sources/jump_60cm.pt' if jump else 'runs/g1_funky_chicken_20261030_211659/best.pt')
expected='ab1d2b97010a646418ba113ffd9793f1371ac3972f5a612f9dd2c4921db2af1d' if jump else '1d9fc183c87e455b7a439c9598796c417fdb802a0fecd2a55cfd7e93554525f6'
if a.checkpoint:
 assert a.checkpoint_sha256 and a.evaluation,'Explicit checkpoint requires its hash and evaluation'
 source=Path(a.checkpoint);expected=a.checkpoint_sha256
assert sha256(source)==expected
ck=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
w={k:v.numpy() for k,v in ck['policy'].items()};np.savez(out/'weights.npz',**w)
obsdim=len(w['obs_norm.mean']);goaldim=len(w['goal_norm.mean'])
def widths(name):
 ids=sorted(int(k.split('.')[1]) for k in w if k.startswith(name+'.') and k.endswith('.weight'))
 return [w[f'{name}.{i}.weight'].shape[0] for i in ids]
enc,dec=widths('encoder'),widths('decoder')
policy=Policy(obsdim,goaldim,latent=enc[-1],enc_hidden=enc[:-1],dec_hidden=dec[:-1]).eval();policy.load_state_dict(ck['policy'])
robot=Entity(get_g1_robot_cfg())
import mujoco
robot.spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[10,10,.1])
m=robot.spec.compile();m.opt.timestep=.005;root=m.body('pelvis').id
jids=[m.body_jntadr[root+1+i] for i in range(29)]
names=[m.joint(i).name for i in jids]
sdk=[f'{s}_{n}_joint' for s in ('left','right') for n in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]+['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']+[f'{s}_{n}_joint' for s in ('left','right') for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
qadr=m.jnt_qposadr[jids]
act=[int(np.where(m.actuator_trnid[:,0]==j)[0][0]) for j in jids]
scale=[next(v for pat,v in G1_ACTION_SCALE.items() if re.fullmatch(pat,n)) for n in names]
refsource=project/('data/processed/g1_standing_jump_060m.npz' if jump else 'experiments/three_skills/targets/g1_funky_chicken.npz')
with np.load(refsource,allow_pickle=False) as r:
 qp,qv=r['qpos'],r['qvel'];assert float(r['fps'])==50
 reference=dict(pos=qp[:,:3],rot=np.stack([rotation(q) for q in qp[:,3:7]]),quat=qp[:,3:7],q=qp[:,7:],v=qv[:,:3],w=qv[:,3:6],qd=qv[:,6:])
np.savez(out/'reference.npz',**reference)
meta=dict(skill=a.skill,iteration=ck['it'],checkpoint=str(source),checkpoint_sha256=expected,
 weights_sha256=sha256(out/'weights.npz'),reference_sha256=sha256(out/'reference.npz'),reference_source_sha256=sha256(refsource),
 obs_dim=obsdim,goal_dim=goaldim,history=obsdim//93,future=goaldim//73,policy_dt=.02,
 policy_joint_names=names,sdk_joint_names=sdk,policy_to_sdk=[sdk.index(n) for n in names],
 q0=m.key_qpos[0,qadr].tolist(),action_scale=scale,joint_limits=m.jnt_range[jids].tolist(),
 kp=m.actuator_gainprm[act,0].tolist(),kd=(-m.actuator_biasprm[act,2]).tolist(),torque_limits=m.actuator_forcerange[act,1].tolist(),
 duration_s=(len(qp)-1)*.02,reference_frames=len(qp),hardware_validated=False,root_position_required=True,
 action_output='unbounded linear decoder; q_target=q0+scale*action, as trained',robot='G1 29DOF, unlocked waist',
 source='Our trained SuperMimic policy, procedural reference; not an official policy')
meta.update(label=a.label or f'{a.skill} {ck["it"]}',physical_eval_passed=False,
 reference_alignment=ck.get('reference_alignment','episode_initial_root'))
if meta['reference_alignment']=='episode_initial_foot_center_xy':
 chains=[]
 for side in ('left','right'):
  body=m.body(side+'_ankle_roll_link').id;chain=[];b=body
  while b!=root:
   j=int(m.body_jntadr[b]);assert m.body_jntnum[b]==1 and np.allclose(m.jnt_pos[j],0)
   chain.append(dict(policy_index=jids.index(j),offset_pos=m.body_pos[b].tolist(),
     offset_rot=rotation(m.body_quat[b]).tolist(),axis=m.jnt_axis[j].tolist()))
   b=int(m.body_parentid[b])
  geoms=[i for i in range(m.ngeom) if (m.geom(i).name or '').split('/')[-1].startswith(side+'_foot') and 'collision' in (m.geom(i).name or '')]
  assert len(geoms)==7 and all(m.geom_bodyid[i]==body for i in geoms)
  chains.append(dict(joints=list(reversed(chain)),geom_center=m.geom_pos[geoms].mean(0).tolist()))
 meta['foot_center_kinematics']=dict(chains=chains,source='Compiled training model; mean of seven collision centers per foot; XY only')
(out/'policy.json').write_text(json.dumps(meta,indent=2)+'\n')
evalpath=Path(a.evaluation) if a.evaluation else project/('deployments/g1_actions/sources/jump_eval.json' if jump else 'runs/g1_funky_chicken_20261030_211659/best_eval.json')
if a.checkpoint:assert json.loads(evalpath.read_text())['checkpoint_sha256']==expected
(out/'simulation_eval.json').write_bytes(evalpath.read_bytes())
rng=np.random.default_rng(20261101)
o=(w['obs_norm.mean']+rng.normal(size=(128,obsdim))*np.sqrt(w['obs_norm.var']+.0001)).astype('f')
g=(w['goal_norm.mean']+rng.normal(size=(128,goaldim))*np.sqrt(w['goal_norm.var']+.0001)).astype('f')
with torch.no_grad():oracle=policy(torch.from_numpy(o),torch.from_numpy(g)).numpy()
actor=Actor(out);err=max(float(np.max(abs(actor(o,g)-oracle))),max(float(np.max(abs(actor(x[None],y[None])[0]-z))) for x,y,z in zip(o,g,oracle)))
assert err<3e-5,err
c=Controller(out);c.anchor_position=reference['pos'][0];c.anchor_rotation=np.eye(3,dtype='f');errors=[]
for frame in range(0,len(qp)-1,7):
 ids=np.minimum(frame+np.arange(1,meta['future']+1),len(qp)-1)
 pos=reference['pos'][frame]+rng.normal(0,.03,3).astype('f');R=reference['rot'][frame]
 ref={k:torch.from_numpy(reference[k][ids][None]) for k in ('pos','rot','q','qd','v','w')}
 goal=Task.goal(None,{'pos':torch.from_numpy(pos[None]),'rot':torch.from_numpy(R[None])},ref)[0].numpy()
 errors.append(float(np.max(abs(c.goal(pos,R,frame)-goal))))
assert max(errors)<3e-6,max(errors)
np.savez(out/'parity_inputs.npz',obs=o,goal=g,expected_action=oracle)
import mujoco
mujoco.mj_saveModel(m,str(out/'training_model.mjb'),None)
audit=dict(passed=True,cases=128,actor_max_error=err,goal_cases=len(errors),goal_max_error=max(errors),oracle='Original CPU float32 training Policy and Task.goal')
if meta['reference_alignment']=='episode_initial_foot_center_xy':
 from supermimic.kinematics import TreeKinematics
 kin=TreeKinematics(m,'cpu',root)
 feet=kin.body_index(('left_ankle_roll_link','right_ankle_roll_link'))
 local=torch.tensor([v['geom_center'] for v in chains],dtype=torch.float32)
 def center(pos,R,q):
  P,O=kin.forward(torch.from_numpy(pos[None]),torch.from_numpy(R[None]),torch.from_numpy(q[None]))
  return (P[:,feet]+torch.einsum('bsij,sj->bsi',O[:,feet],local)).mean(1)[0].numpy()
 nominal=center(reference['pos'][0],reference['rot'][0],reference['q'][0]);alignment_errors=[];foot_errors=[]
 inputs=[];expected_goals=[]
 for i in range(128):
  pos=reference['pos'][0]+rng.normal(0,.05,3).astype('f');q=reference['q'][0]+rng.normal(0,.2,29).astype('f')
  R=reference['rot'][0];physical=center(pos,R,q);c.reset()
  foot_errors.append(float(np.max(abs(c.foot_center(pos,R,q)-physical))))
  state_pos=pos.copy();state_pos[:2]-=(physical-nominal)[:2]
  ids=np.arange(1,meta['future']+1)
  ref={k:torch.from_numpy(reference[k][ids][None]) for k in ('pos','rot','q','qd','v','w')}
  goal=Task.goal(None,{'pos':torch.from_numpy(state_pos[None]),'rot':torch.from_numpy(R[None])},ref)[0].numpy()
  actual=c.goal(pos,R,0,q);alignment_errors.append(float(np.max(abs(actual-goal))))
  inputs.append(np.r_[pos,q]);expected_goals.append(goal)
 assert max(foot_errors)<3e-6 and max(alignment_errors)<3e-6,(max(foot_errors),max(alignment_errors))
 np.savez(out/'alignment_inputs.npz',inputs=inputs,rotation=reference['rot'][0],expected_goal=expected_goals)
 audit.update(foot_center_cases=len(foot_errors),foot_center_max_error=max(foot_errors),initial_alignment_max_error=max(alignment_errors))
(out/'export_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({**meta,'audit':audit}))
