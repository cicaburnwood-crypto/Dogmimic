"""Finite five-step physical simulation smoke test. Never connects to hardware."""
from bootstrap import HERE,ROOT
from joint import *
from arm import ArmController
import hashlib,json
from supermimic import networks
for name,digest in [('v18_best24500.pt','d5540cc2f1a8a283c38c9e644c9693d2b6317158d9beb2172f6d67c0594da703'),('p4_20000.pt','e0314388419456d6597e5b44502508dbac00da0019592da4e79bf3a6c992457b')]:
 assert hashlib.sha256((ROOT/'checkpoints'/name).read_bytes()).hexdigest()==digest
old=ArmController.envelope
ArmController.envelope=lambda self,speed:(old(self,speed)[0],old(self,speed)[1]*1.25)
torch.set_num_threads(2);networks.AMP['enabled']=False;torch.manual_seed(24500)
e=JointEnv('cuda',1,randomize=False,compile=False,autoreset=False);e.reset()
ck=torch.load(ROOT/'checkpoints/v18_best24500.pt',map_location='cuda',weights_only=False)
assert ck['it']==24000 and ck['wm_arm_normalization_version']==1
manager=Manager(99,e.arm).cuda();manager.load_state_dict(ck['manager']);manager.eval()
lower=make_policy().cuda();lower.load_state_dict(torch.load(ROOT/'checkpoints/p4_20000.pt',map_location='cuda',weights_only=False)['policy']);lower.eval()
p,r=ee(e.arm,e.state);goal=torch.cat([p,r.flatten(1)],1);last=torch.zeros(1,12,device='cuda')
with torch.no_grad():
 for _ in range(5):
  u=manager(e.state,e.context,goal,last,e.xi);assert torch.isfinite(u).all()
  e.upper=u;e.set_target(commands(u));s,info=e.step(lower(*e.policy_inputs()));last=u
  assert not bool(info['terminated'][0]),'simulation terminated during smoke'
  assert all(torch.isfinite(v).all() for v in s.values() if isinstance(v,torch.Tensor))
dist,ang,_=errors(e.arm,s,goal)
print('SMOKE_RESULT',json.dumps({'steps':5,'position_error_m':float(dist[0]),'orientation_deg':float(torch.rad2deg(ang[0])),'hardware_io':False,'acceptance_test':False}))
