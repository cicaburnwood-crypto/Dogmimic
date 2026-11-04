import argparse
from pathlib import Path
import torch
from bootstrap import HERE
from common import write_json,sha256
from contract import Q0,ACTION_SCALE,KP,KD,POLICY_JOINTS,POLICY_TO_SDK,DT,OBS,HISTORY,GOAL_DIM
from policy import make_policy
from supermimic import networks

def export(checkpoint,output):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    policy=make_policy();policy.load_state_dict(ck['policy']);policy.eval()
    inputs=(torch.randn(3,OBS*HISTORY),torch.randn(3,GOAL_DIM)*.2)
    # The trainer enables CUDA AMP globally; CPU deployment export is FP32.
    amp_before=networks.AMP['enabled']
    networks.AMP['enabled']=False
    try:
        with torch.no_grad():
            artifact=torch.jit.trace(policy,inputs);error=float((artifact(*inputs)-policy(*inputs)).abs().max())
    finally:
        networks.AMP['enabled']=amp_before
    assert error<1e-6
    artifact.save(str(output))
    write_json(Path(output).with_suffix('.json'),dict(checkpoint=str(checkpoint),source_sha256=sha256(checkpoint),iteration=ck['it'],
        backend='vibe',task='b2_attitude_v1',dt=DT,history=HISTORY,observation_order='per frame: q-q0, qd, body gyro, projected gravity, previous action; 10 frames oldest first',
        goal_order=['desired_roll_rad','desired_pitch_rad','desired_roll_rate','desired_pitch_rate','requested_roll_rad','requested_pitch_rad'],
        q0=Q0,action_scale=ACTION_SCALE,kp=KP,kd=KD,joints=POLICY_JOINTS,policy_to_sdk=POLICY_TO_SDK,
        command_interface='command_interface.AttitudeCommands',export_max_error=error,physical_acceptance=False))
    return error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(a.checkpoint,a.output)

