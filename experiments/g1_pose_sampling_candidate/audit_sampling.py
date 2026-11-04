import ast,json,torch
from pathlib import Path
from types import SimpleNamespace as NS
from pose_sampling import sample_pose_blend
class Parent:
 def reset_envs(self,ids,frames=None):pass
p=Path(__file__).resolve().parent
node=next(n for n in ast.parse((p/'robust_env.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RobustEnv')
node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='reset_envs'];ns=dict(torch=torch,JumpEnv=Parent,sample_pose_blend=sample_pose_blend)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_candidate_reset','exec'),ns)
n=100000;torch.manual_seed(11);torch.set_num_threads(2)
e=ns['RobustEnv'].__new__(ns['RobustEnv']);e.eval_mode=False;e.device='cpu';e.frame=torch.zeros(n,dtype=torch.long);e.frame[n//2:]=100;e.q_adr=torch.arange(7,36);e.root_qadr=0;e.measured=torch.ones(29);e.queue=torch.zeros(n,9,29)
# Parent-provided pose noise must be preserved, not overwritten by endpoint sampling.
noise=torch.randn(n,29)*.02
qpos=torch.zeros(n,36);qpos[:,7:]=noise
e.sim=NS(data=NS(qpos=qpos,qvel=torch.ones(n,35)));e.lib=NS(frame=lambda x:{'q':torch.zeros(len(x),29)})
e.reset_envs(torch.arange(n));delta=e.sim.data.qpos[:,7:]-noise;blend=delta[:n//2,0]
assert (delta[n//2:]==0).all()
assert torch.allclose(delta[:n//2],blend[:,None].expand(-1,29),atol=2e-7)
assert torch.equal(e.queue,e.sim.data.qpos[:,None,7:].expand(-1,9,-1))
assert (e.sim.data.qvel[:n//2]==0).all() and (e.sim.data.qvel[n//2:]==1).all()
native=int(torch.isclose(blend,torch.ones_like(blend),atol=1e-7).sum());ref=int((blend==0).sum())
assert 12000<native<13000 and 12000<ref<13000
print(json.dumps(dict(passed=True,start_samples=n//2,native_endpoint=native,reference_endpoint=ref,nonstart_unchanged=True,parent_joint_noise_preserved=True,queue_matches_physical_pose=True,scope='Actual candidate reset with tensor parent; physical rollout not tested')))
