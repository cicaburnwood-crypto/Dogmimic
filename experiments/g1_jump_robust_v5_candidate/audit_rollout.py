import sys,json,importlib.util,hashlib,time
from pathlib import Path
BASE=Path(__file__).parent;ROOT=BASE.parents[1];sys.path.insert(0,str(BASE/'engine'))
import torch
from supermimic import networks
from supermimic.motion_lib import MotionLibrary
from supermimic.evaluate import load_supermimic_policy
from robust_env import RobustEnv
from evaluate_jump import run_trials
old=ROOT/'experiments/g1_jump_robust_v4_candidate/robust_env.py';spec=importlib.util.spec_from_file_location('unaligned_robust_env',old);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.set_num_threads(2);networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True
cp=ROOT/'runs/g1_jump_robust_v4_20261101_193010/best.pt';policy,ck=load_supermimic_policy(str(cp),'cuda');lib=MotionLibrary('data/processed/g1_standing_jump_060m.npz','cuda');rows=[]
for name,cls in [('unaligned',mod.RobustEnv),('aligned',RobustEnv)]:
 env=cls(lib,'cuda',32,evaluate=True)
 for case in [dict(name='nominal',friction=.8),dict(name='native_start',measured=True)]:
  env.case=case;r,tr=run_trials(policy,lib,env,seed=20261101+(80000 if case['name']=='native_start' else 0))
  rows.append(dict(frame=name,case=case['name'],passed=sum(t['success'] for t in r['trials']),complete=sum(t['complete'] for t in r['trials']),nominal=r['nominal'],trials=r['trials']));print(name,case['name'],rows[-1]['passed'],flush=True)
 del env
 torch.cuda.empty_cache()
out=ROOT/'runs/g1_alignment_rollout_audit';out.mkdir(exist_ok=True)
(out/'report.json').write_text(json.dumps(dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),unchanged_evaluator_sha256=hashlib.sha256((BASE/'evaluate_jump.py').read_bytes()).hexdigest(),rows=rows,completed=time.time(),physical_acceptance_complete=False),indent=2))
