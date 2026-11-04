"""CPU method-level integration audit; no claim of MuJoCo rollout validation."""
import ast,json
from pathlib import Path
import torch
from actuator_filter import filtered_target,torque_envelope,advance_context
root=Path(__file__).resolve().parent
class Parent:
    def advance_context(self,s,frame,xi):return xi[...,:30]+1
    def transition_analytics(self,*args):return {'r':torch.zeros(2),'terms':{}}
tree=ast.parse((root/'robust_env.py').read_text())
node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RobustTask')
ns=dict(torch=torch,StageTask=Parent,filtered_target=filtered_target,torque_envelope=torque_envelope,SPEED=6.,POWER=1500.,ACCEL=180.)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_RobustTask','exec'),ns)
task=ns['RobustTask']();task.q0=torch.zeros(12);task.action_scale=.7;task.qmin=torch.full((12,),-2.);task.qmax=-task.qmin
task.cfg=type('Cfg',(),{'dt':.02})()
xi=torch.zeros(2,43);xi[:,30]=1.;xi[:,31:]=.1
nxt=task.advance_context({},torch.zeros(2),xi)
assert torch.equal(nxt[:,30:],xi[:,30:]) and torch.equal(nxt[:,:30],xi[:,:30]+1)
a=torch.ones(2,12);state={'q':torch.zeros(2,12),'qd':torch.zeros(2,12)}
seen=[]
def capture(target,*args):
    seen.append(target.clone());return torque_envelope(target,*args)
ns['torque_envelope']=capture
out=task.transition_analytics(state,state,None,None,None,a,a,xi)
expected=advance_context(xi,a,task.q0,task.action_scale,task.qmin,task.qmax)[:,31:]
assert torch.equal(seen[0],expected) and torch.isfinite(out['r']).all()
# Execute actual WM input adapter against an identity parent, no GPU imports.
class WMParent:
    def __init__(self,task,feat_dim,xi_dim):self.dim=xi_dim
    def inputs(self,s,a,xi,feats=None):return xi
class Recovery:
    @staticmethod
    def canonical_context(s,xi):return xi[:,:30]+2
tr=ast.parse((root/'train_guarded.py').read_text())
wmnode=next(x for x in tr.body if isinstance(x,ast.ClassDef) and x.name=='FilteredWM')
wms=dict(torch=torch,WorldModel=WMParent,RecoveryWorldModel=Recovery)
exec(compile(ast.Module(body=[wmnode],type_ignores=[]),'actual_FilteredWM','exec'),wms)
wm=wms['FilteredWM'](None,1,43);inp=wm.inputs({},a,xi)
assert wm.dim==34 and inp.shape==(2,34)
assert torch.equal(inp[:,:21],xi[:,:21]+2) and torch.equal(inp[:,21:],xi[:,30:])
print(json.dumps(dict(passed=True,actual_task_preserves_filter_memory=True,actual_reward_target_matches_imagined_filter=True,actual_wm_context_dim=34,scope='Actual extracted task and WM adapter methods on CPU mocks; physical env reset/step and GPU trainer smoke still pending')))
