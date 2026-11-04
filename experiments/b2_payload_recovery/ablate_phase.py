import argparse,json,time
from pathlib import Path
import torch
from supermimic import networks
from cycle_env import CycleLibrary
from evaluate_pushes import schedule
from evaluate_payloads import run_condition,compact
from phase_policy import from_checkpoint
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--mode',choices=('no_hind','no_quad','return_only'),required=True);p.add_argument('--out',required=True);a=p.parse_args()
torch.set_num_threads(2);networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
ck=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
if a.mode=='return_only':
    from stage_policy import from_checkpoint
policy=from_checkpoint(ck)
if a.mode!='return_only':
    head=policy.hind_head if a.mode=='no_hind' else policy.quad_head
    with torch.no_grad():head.weight.zero_();head.bias.zero_()
root=Path.cwd();lib=CycleLibrary(str(root/'experiments/b2_cycle/b2_full_cycle.npz'),'cuda')
planned,_=schedule(64,lib.num_frames-1,.02,20261110);results=[];start=time.time()
for kg in (0,5,10,15):
    path=Path(a.checkpoint).parent/'models'/f'b2_payload_{kg}kg.xml'
    r,_,_=run_condition(policy,lib,path,64,20261109,planned,True)
    r.update(payload_kg=kg);results.append(compact(r))
    print(a.mode,kg,'complete',round(64*r['completion_rate']),'strict',round(64*r['success_rate']),'reasons',r['failure_reason_counts'],flush=True)
Path(a.out).write_text(json.dumps(dict(checkpoint=a.checkpoint,mode=a.mode,conditions=results,elapsed_seconds=time.time()-start),indent=2)+'\n')
