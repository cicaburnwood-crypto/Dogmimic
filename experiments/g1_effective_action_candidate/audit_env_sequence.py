import ast,json,torch
from pathlib import Path
from types import SimpleNamespace as NS
from actuator_sequence import consumed_targets
root=Path(__file__).resolve().parents[2]
source=root/'experiments/g1_pose_sampling_candidate/robust_env.py'
cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RobustEnv')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='physics_step'];ns=dict(torch=torch,JumpEnv=object)
exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_physics_step','exec'),ns)
torch.manual_seed(11);n=9;e=ns['RobustEnv'].__new__(ns['RobustEnv']);e.num_envs=n;e.device='cpu';e.extra=torch.zeros(n,9);e.extra[:,4]=torch.arange(n)/8;e.extra[:,6:]=torch.randn(n,3)
e.frame=torch.full((n,),50);e.torso_body=1;e.decimation=4;e.queue=torch.randn(n,9,29);e.queue_index=5;e.act_to_joint=torch.randperm(29)
e.sim=NS(data=NS(xfrc_applied=torch.zeros(n,2,6),ctrl=torch.zeros(n,29)));record=[]
e.sim.step=lambda:record.append(e.sim.data.ctrl.clone())
xi=torch.zeros(n,304);xi[:,37]=e.extra[:,4];xi[:,42:303]=e.queue[:,(torch.arange(9)+5)%9].flatten(1)
a=torch.randn(n,29);q0=torch.zeros(29);scale=.7
expected=consumed_targets(xi,a,q0,scale);e.physics_step(q0+scale*a)
assert torch.equal(torch.stack(record,1),expected[:,:,e.act_to_joint])
print(json.dumps(dict(passed=True,actual_environment_method=True,all_delays=True,nonzero_ring_index=True,nonidentity_joint_mapping=True,scope='Actual physics_step with captured simulator control buffers, no dynamics accuracy claim')))
