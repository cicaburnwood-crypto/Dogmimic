"""CPU integration audit of coordinate alignment, including partial reset routing."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).parent;sys.path.insert(0,str(BASE/'engine'))
import torch,mujoco,numpy as np
from robust_env import RobustEnv,JumpEnv
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import quat_to_mat

torch.set_num_threads(1)
m=mujoco.MjModel.from_binary_path('/tmp/g1_reference_alignment_audit.mjb')
root=next(i for i in range(m.nbody) if m.body(i).name.split('/')[-1]=='pelvis')
raw=np.load('data/processed/g1_standing_jump_060m.npz')['qpos'][0]
e=RobustEnv.__new__(RobustEnv);e.device='cpu';e.num_envs=2;e.kin=FastTreeKinematics(m,'cpu',root_body=root)
e.root_qadr=int(m.jnt_qposadr[m.body_jntadr[root]]);e.root_vadr=int(m.jnt_dofadr[m.body_jntadr[root]])
h=[m.body_jntadr[root+1+i] for i in range(29)];e.q_adr=torch.tensor(m.jnt_qposadr[h]);e.qd_adr=torch.tensor(m.jnt_dofadr[h]);e.frame=torch.zeros(2,dtype=torch.long)
qpos=torch.tensor(np.repeat(raw[None],2,0));qvel=torch.zeros(2,m.nv)
qpos[1,e.q_adr]=torch.tensor(json.loads((BASE/'measured_start.json').read_text())['q']);qpos[1,2]+=.027
e.sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos,qvel=qvel));e.reference_xy_offset=torch.zeros(2,2)
e.align_feet=e.kin.body_index(('left_ankle_roll_link','right_ankle_roll_link'))
gids=[[i for i in range(m.ngeom) if (m.geom(i).name or '').split('/')[-1].startswith(side+'_foot') and 'collision' in (m.geom(i).name or '')] for side in ('left','right')]
e.align_local=torch.tensor(m.geom_pos[gids].mean(1),dtype=torch.float32)
ref=dict(pos=torch.tensor(raw[:3])[None],quat=torch.tensor(raw[3:7])[None],q=torch.tensor(raw[7:])[None])
e.lib=SimpleNamespace(frame=lambda f:{k:v.expand(len(f),*v.shape[1:]) for k,v in ref.items()})
original=JumpEnv._finish_resets;captured={}
def finish(self,ids,force=False):captured['state']=self.read_state();captured['ids']=ids.clone()
JumpEnv._finish_resets=finish
before=qpos.clone()
try:
 e._finish_resets(torch.arange(2),True)
 aligned=e._foot_center(e.read_state());nom=e._foot_center(dict(pos=ref['pos'].expand(2,-1),rot=quat_to_mat(ref['quat']).expand(2,-1,-1),q=ref['q'].expand(2,-1)))
 assert torch.allclose(aligned,nom,atol=2e-7)
 assert torch.equal(qpos,before)
 assert e.reference_xy_offset[0].abs().max()<1e-7
 saved=e.reference_xy_offset[1].clone();e._finish_resets(torch.tensor([0]));assert torch.equal(saved,e.reference_xy_offset[1])
 assert torch.equal(captured['state']['v'],torch.zeros(2,3))
 print(json.dumps(dict(passed=True,scope='CPU environment integration, no physical policy rollout',offset_xy=e.reference_xy_offset.tolist(),aligned_foot_center_max_error=float((aligned-nom).abs().max()),physical_qpos_unchanged=True,partial_reset_preserves_other_episode=True)))
finally:JumpEnv._finish_resets=original
