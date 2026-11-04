"""Paper-style deterministic first-frame rollout, not physical acceptance."""
import argparse
import os
import time
from pathlib import Path
import torch
import numpy as np
from common import ROOT,write_json,sha256
from sm_original.motion_lib import MotionLibrary
from sm_original.task import TaskCfg,OBS_DIM,GOAL_FRAME_DIM,state_to_flat
from sm_original.networks import Policy
from sm_original import networks
from environments import make_env

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();robot=os.environ['SM_RESET_ROBOT'];torch.set_num_threads(2);os.chdir(ROOT)
    cp=Path(a.checkpoint);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(cp,map_location='cuda',weights_only=False);assert ck['robot']==robot
    networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True
    cfg=TaskCfg();pol=Policy(OBS_DIM*cfg.history,GOAL_FRAME_DIM*cfg.future).to('cuda')
    pol.load_state_dict(ck['policy']);pol.eval()
    lib=MotionLibrary(ck['cfg']['motions'],'cuda')
    env=make_env(robot,lib,'cuda',1,compile=False,evaluate=True)
    rows=[];started=time.time()
    for load in ([0.,5.] if robot=='B2' else [None]):
        torch.manual_seed(20261103)
        if load is not None:env.fixed_load=load
        env.reset();ids=torch.zeros(1,device='cuda',dtype=torch.long)
        env.reset_envs(ids,frames=ids);env._finish_resets(ids,force=True)
        if load is not None:env.push_vectors.zero_()
        frames=[];ret=0.;failed=False;T=int(lib.lengths[0])-1
        for t in range(T):
            s,info=env.step(pol(*env.policy_inputs()))
            ret+=float(info['reward'][0]);frames.append(state_to_flat(s)[0].cpu().numpy())
            if bool(info['done'][0]):
                failed=bool(info['terminated'][0]);break
        row=dict(load_kg=load,normalized_tracking_return=ret/(T*cfg.max_step_reward),
                 completed=bool(not failed and len(frames)==T),steps=len(frames),total_steps=T)
        rows.append(row)
        np.savez_compressed(out/f'load_{load}.npz',state=np.stack(frames),dt=np.array(.02))
    write_json(out/'result.json',dict(robot=robot,it=ck['it'],checkpoint_sha256=sha256(cp),
        domain='restored_training_physics_nominal',physical_acceptance=False,friction=1.,
        noise=False,push=False,conditions=rows,elapsed_seconds=time.time()-started))

if __name__=='__main__':main()
