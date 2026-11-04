"""Measure complete launch-to-exit time and reject accidental checkpoint reuse."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpus',required=True);p.add_argument('--cold-cache',action='store_true')
args,extra=p.parse_known_args()
if '--resume' in extra:p.error('Fresh runs cannot resume')
root=Path(__file__).resolve().parents[2];out=(root/args.out).resolve();out.mkdir(parents=True,exist_ok=False)
start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,COMMAND_LAUNCH_UNIX=str(start))
if args.cold_cache:env['COMMAND_COLD_CACHE']=str(out/'cold_cache')
cmd=['bash','experiments/command_locomotion/train.sh','--out',str(out),*extra]
rec=dict(start_unix=start,gpus=args.gpus,world=len(args.gpus.split(',')),command=cmd,fresh_parameters=True,
         reference_motions=False,pretrained_policy=False,fresh_replay=True,cold_cache=args.cold_cache,
         sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'experiments/command_locomotion').glob('*.py')})
(out/'launch.json').write_text(json.dumps(rec,indent=2))
with (out/'console.log').open('w') as log:
    process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    rec['pid']=process.pid;(out/'launch.json').write_text(json.dumps(rec,indent=2))
    code=process.wait()
rec.update(end_unix=time.time(),total_wall_seconds=time.time()-start,returncode=code)
(out/'launch_timing.json').write_text(json.dumps(rec,indent=2))
raise SystemExit(code)
