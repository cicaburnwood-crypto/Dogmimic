import sys,json
from pathlib import Path
p=Path(__file__).resolve().parent;v5=p.parent/'g1_jump_robust_v5_candidate'
sys.path.insert(1,str(v5/'engine'));sys.path.insert(2,str(v5))
import torch
from supermimic.motion_lib import MotionLibrary
from robust_env import RobustEnv
from supermimic.rotations import quat_to_mat
torch.set_num_threads(2);torch.manual_seed(11)
lib=MotionLibrary('data/processed/g1_standing_jump_060m.npz','cpu');e=RobustEnv(lib,'cpu',4,evaluate=False)
e.reset();ids=torch.arange(4);e.reset_envs(ids,torch.zeros(4,dtype=torch.long));e._finish_resets(ids,force=True)
physical=e.sim.data.qpos[:,e.q_adr].clone()
assert torch.equal(e.queue,physical[:,None,:].expand(-1,9,-1))
r=lib.frame(e.frame);nom=dict(pos=r['pos'],rot=quat_to_mat(r['quat']),q=r['q'])
assert torch.allclose(e._foot_center(e.read_state()),e._foot_center(nom),atol=2e-6)
for _ in range(8):
 _,info=e.step(torch.zeros(4,29));assert torch.isfinite(e.sim.data.qpos).all();assert torch.isfinite(info['reward']).all()
keep=e.queue[1:].clone();offset=e.reference_xy_offset[1:].clone()
e.reset_envs(ids[:1],torch.zeros(1,dtype=torch.long));e._finish_resets(ids[:1],force=True)
assert torch.equal(e.queue[1:],keep) and torch.equal(e.reference_xy_offset[1:],offset)
print(json.dumps(dict(passed=True,device='cpu',real_simulator=True,steps=8,environments=4,initial_queue_parity=True,foot_frame_alignment=True,partial_reset_preserves_other_envs=True,scope='Integration only; not skill acceptance')))
