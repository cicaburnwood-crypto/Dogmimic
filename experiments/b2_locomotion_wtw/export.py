"""Export only this experiment's actor; no simulation/WM dependency on the robot."""
import argparse,json,hashlib,shutil,time
from pathlib import Path
import torch
from torch import nn
from policy import make_policy
from contract import *
from gait_clock import GOAL_DIM,FREQUENCY,SWING_HEIGHT

class DeploymentActor(nn.Module):
    def __init__(self,policy):
        super().__init__()
        self.encoder=policy.encoder;self.decoder=policy.decoder
        self.register_buffer('obs_mean',policy.obs_norm.mean.clone())
        self.register_buffer('obs_std',policy.obs_norm.std.clone())
        self.register_buffer('goal_mean',policy.goal_norm.mean.clone())
        self.register_buffer('goal_std',policy.goal_norm.std.clone())
    def forward(self,observation:torch.Tensor,command:torch.Tensor)->torch.Tensor:
        # No command clamp; feasible translation handling occurs in the shared controller.
        z=torch.tanh(self.encoder((command-self.goal_mean)/self.goal_std))
        o=((observation-self.obs_mean)/self.obs_std).clamp(-10.,10.)
        return torch.tanh(self.decoder(torch.cat([z,o],-1)))

def export_checkpoint(checkpoint,out,evaluation=None):
    from supermimic import networks
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['task_kind']=='b2_proprioceptive_locomotion_wtw_v4' and ck['source_checkpoint'] is None
    p=make_policy().cpu();p.load_state_dict(ck['policy']);p.eval()
    net=DeploymentActor(p).eval();script=torch.jit.script(net)
    rng=torch.Generator().manual_seed(971)
    obs=torch.randn(256,420,generator=rng);cmd=torch.randn(256,GOAL_DIM,generator=rng)*3
    oldamp=networks.AMP['enabled'];networks.AMP['enabled']=False
    with torch.no_grad():err=float((p(obs,cmd)-script(obs,cmd)).abs().max())
    networks.AMP['enabled']=oldamp
    assert err<1e-6,err
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    script.save(str(out/'policy.ts.tmp'));(out/'policy.ts.tmp').replace(out/'policy.ts')
    meta=dict(task=ck['task_kind'],checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              iteration=ck['it'],policy_sha256=hashlib.sha256((out/'policy.ts').read_bytes()).hexdigest(),
              random_initialization=True,official_policy=False,reference_motion=False,policy_dt=DT,history=HISTORY,obs_per_frame=OBS,
              q0=Q0,action_scale=ACTION_SCALE,kp=KP,kd=KD,policy_joint_names=POLICY_JOINTS,policy_to_sdk=POLICY_TO_SDK,
              joint_limits=ck['model_meta']['joint_limits'],command=dict(max_norm_xy=2.,max_abs_y=.6,additional_x_cap=None,yaw_cap=None),
              linear_acceleration=LINEAR_ACCELERATION,angular_acceleration=ANGULAR_ACCELERATION,
              proprioception_only=True,goal_layout="vx,vy,wz,clock_FL,clock_FR,clock_RL,clock_RR",gait_frequency_hz=FREQUENCY,footswing_height=SWING_HEIGHT,cpu_export_max_error=err,hardware_validated=False,
              physical_eval_passed=bool(evaluation and evaluation['all_core_95']),model_mass_kg=ck['model_meta']['mass_kg'],exported_unix=time.time())
    (out/'policy.json').write_text(json.dumps(meta,indent=2)+'\n')
    here=Path(__file__).parent
    for name in ('contract.py','gait_clock.py'):
        if (here/name).exists():shutil.copy2(here/name,out/name)
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(export_checkpoint(a.checkpoint,a.out)))
