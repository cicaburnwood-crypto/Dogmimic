"""Read-only compact status and failure attribution for the two current training runs."""
import json,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None

def last_records(p,n):
 if not p.exists():return []
 with p.open('rb') as f:
  size=f.seek(0,2);f.seek(max(0,size-524288));data=f.read().decode(errors='replace')
 rows=[]
 for line in data.splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 return rows[-n:]

def summarize(d,kind):
 if not d:return None
 out={k:v for k,v in d.items() if k in ['it','passed','total','completed','nominal_passed','worst_case_passed','full','all_passed','all_cases_95','checkpoint_sha256','sha256']}
 if kind=='B2':
  out['conditions']=[]
  for r in d.get('conditions',[]):
   row={k:r.get(k) for k in ['kg','friction','mass_scale','asymmetric','success_rate','completion_rate','cycle_success_rate','stable_hindstand_rate','stable_return_rate','peak_speed','peak_mechanical_power','actuator_pass']}
   trials=r.get('trials_detail',[]);limit=r.get('criteria',{}).get('foot_xy_drift_max')
   row['hind_foot_drift_failures']=sum(t.get('hind_foot_xy_drift_m',0)>limit for t in trials if isinstance(t.get('hind_foot_xy_drift_m'),(int,float))) if limit is not None else None
   row['trials_with_drift_measurement']=sum(isinstance(t.get('hind_foot_xy_drift_m'),(int,float)) for t in trials)
   for phase in ['stand','return']:
    metrics=r.get(phase+'_metrics_completed_trials') or {}
    row[phase+'_metrics_completed_trials']={k:metrics.get(k) for k in ['com_position_std_norm','com_acc_rms_xyz','center_acc_rms_xyz']}
   out['conditions'].append(row)
 else:
  cases=[]
  for c in d.get('cases',[]):
   trials=c.get('trials',[])
   if not trials:continue
   causes=dict(incomplete=sum(not t['complete'] for t in trials),no_landing=sum(t['landing_frame']<0 for t in trials),distance=sum(abs(t['landing_distance_m']-.6)>.1 for t in trials),pre_shift=sum(t['pre_takeoff_foot_shift_m'] is None or t['pre_takeoff_foot_shift_m']>=.08 for t in trials),slide=sum(t['post_landing_shift_m'] is None or t['post_landing_shift_m']>=.1 for t in trials),unsettled=sum(not t['settled'] for t in trials))
   cases.append(dict(name=c['case']['name'],passed=c['passed'],total=len(trials),failure_counts=causes,median_landing_distance=statistics.median(t['landing_distance_m'] for t in trials)))
  out['cases']=cases
 return out

def status(kind,folder,launch_name="latest_launch.json"):

 launch=read(ROOT/'experiments'/folder/launch_name)
 if not launch:return dict(error='missing launch',kind=kind)
 run=Path(launch['out']);processes=[]
 for key in ['pid','train_pid','eval_pid']:
  if key not in launch:continue
  proc=Path('/proc')/str(launch[key])
  try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  except FileNotFoundError:cmd=''
  processes.append(dict(role=key,pid=launch[key],exists=proc.exists(),matches_run=run.name in cmd,command=cmd))
 logs=last_records(run/'log.jsonl',5)
 return dict(kind=kind,experiment=folder,run=str(run),checked_unix=time.time(),processes=processes,elapsed_s=time.time()-launch.get('launch_unix',launch.get('started',time.time())),train=({k:logs[-1].get(k) for k in ['it','iter_s','wm_err1','wm_errk','ep_term_frac','pi_loss']} if logs else None),log_age_s=time.time()-(run/'log.jsonl').stat().st_mtime if (run/'log.jsonl').exists() else None,eval_progress=read(run/'eval_progress.json'),stop_requested=(run/'STOP').exists(),latest=summarize(read(run/'latest_eval.json'),kind),best=summarize(read(run/'best_eval.json'),kind),guard=read(run/'GUARD_STOP.json'),flags={f:(run/f).exists() for f in ['STOP','DONE','EVAL_DONE.json','TARGET_MET.json','accepted.pt']})
if __name__=='__main__':print(json.dumps([status('B2','b2_light_payload'),status('G1','g1_boundary_repair')],ensure_ascii=False))
