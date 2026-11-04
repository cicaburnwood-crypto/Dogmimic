import sys,os,time,json,shutil,hashlib,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'engine'))
import torch
from supermimic.motion_lib import MotionLibrary
from supermimic.evaluate import load_supermimic_policy
from supermimic import networks
from robust_env import RobustEnv
from evaluate_jump import run_trials
out=Path(sys.argv[1]);networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True
cases=[dict(name='nominal',friction=.8)]
cases += [dict(name='mu_'+str(v),friction=v) for v in [.15,.2,.3,.4,.6,1.,1.5]]
cases += [dict(name='native_start',measured=True),dict(name='delay40',delay_ms=40),dict(name='mass125',mass=1.25),dict(name='weak_motors',effort=.7,kp=.8,kd=.7),dict(name='stiff_motors',kp=1.2,kd=1.3),dict(name='asymmetric',friction=.4,asym=.3),dict(name='pitch_plus',tilt=.06),dict(name='pitch_minus',tilt=-.06),dict(name='push_forward',force=[25,0,0]),dict(name='push_back',force=[-25,0,0]),dict(name='push_side',force=[0,25,0]),dict(name='combined_low',friction=.2,measured=True,delay_ms=20,mass=1.15,effort=.85,force=[0,15,0]),dict(name='combined_high',friction=1.5,measured=True,delay_ms=40,mass=1.25,kp=.8,kd=1.3)]
(out/'eval_cases.json').write_text(json.dumps(cases,indent=2));lib=MotionLibrary('data/processed/g1_standing_jump_060m.npz','cuda');env=RobustEnv(lib,'cuda',32,evaluate=True)
last=-1;best=(-1,-1,-1);(out/'eval').mkdir(exist_ok=True)
def write(name,obj):
 p=out/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
while True:
 req=out/'eval_request.json'
 if req.exists() and json.loads(req.read_text())['it']!=last:
  shutil.copyfile(out/'candidate.pt',out/'evaluating.pt');policy,ck=load_supermimic_policy(str(out/'evaluating.pt'),'cuda');it=int(ck['it']);start=time.time();results=[]
  for idx,case in enumerate(cases):
   env.case=case;r,traj=run_trials(policy,lib,env,seed=20261101+idx*10000+it)
   r.update(case=case,passed=sum(x['success'] for x in r['trials']),completed=sum(x['complete'] for x in r['trials']));results.append(r)
   write('eval_progress.json',dict(it=it,cases_done=len(results),cases_total=len(cases),current_case=case['name'],updated=time.time()))
  score=(min(x['passed'] for x in results),sum(x['passed'] for x in results),sum(x['completed'] for x in results))
  result=dict(it=it,passed=score[1],total=len(cases)*32,worst_case_passed=score[0],completed=score[2],all_cases_95=all(x['passed']>=31 for x in results),nominal_passed=results[0]['passed'],cases=results,eval_seconds=time.time()-start,finished=time.time(),checkpoint_sha256=hashlib.sha256((out/'evaluating.pt').read_bytes()).hexdigest())
  write('eval/it_%07d.json'%it,result);write('latest_eval.json',result)
  with (out/'eval.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in result.items() if k!='cases'})+'\n')
  if it==0:write('initial_eval.json',result)
  elif score>best:
   best=score;shutil.copyfile(out/'evaluating.pt',out/'best.tmp.pt');(out/'best.tmp.pt').replace(out/'best.pt');write('best_eval.json',result)
  print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True);last=it
 if (out/'DONE').exists() and req.exists() and json.loads(req.read_text())['it']==last:
  write('EVAL_DONE.json',dict(time=time.time(),it=last));break
 time.sleep(5)
