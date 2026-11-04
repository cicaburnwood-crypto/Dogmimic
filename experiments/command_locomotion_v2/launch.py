"""Separate continuation clock and source lineage; never overwrite an old run."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--gpus',required=True)
p.add_argument('--init-checkpoint',required=True)
p.add_argument('--cache-root',default='')
args,extra=p.parse_known_args()
root=Path(__file__).resolve().parents[2];out=(root/args.out).resolve()
source=(root/args.init_checkpoint).resolve()
if not source.is_file():p.error('Missing initialization checkpoint')
if '--resume' in extra:p.error('This launcher starts a new continuation phase')
out.mkdir(parents=True,exist_ok=False);start=time.time()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,COMMAND_LAUNCH_UNIX=str(start))
# Reuse compiler caches; a continuation is not a new cold-start benchmark.
env.pop('COMMAND_COLD_CACHE',None)
if args.cache_root:env['COMMAND_CACHE_ROOT']=str((root/args.cache_root).resolve())
cmd=['bash','experiments/command_locomotion_v2/train.sh','--out',str(out),'--init-checkpoint',str(source),*extra]
rec=dict(start_unix=start,gpus=args.gpus,world=len(args.gpus.split(',')),command=cmd,
         initialization='continued_from_own_reference_free_checkpoint',reference_motions=False,external_pretraining=False,
         source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
         fresh_replay=True,cold_cache=False,reused_compiler_cache=env.get('COMMAND_CACHE_ROOT'),
         sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'experiments/command_locomotion_v2').iterdir() if f.is_file()})
(out/'launch.json').write_text(json.dumps(rec,indent=2))
with (out/'console.log').open('w') as log:
    process=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    rec['pid']=process.pid;(out/'launch.json').write_text(json.dumps(rec,indent=2))
    code=process.wait()
rec.update(end_unix=time.time(),total_wall_seconds=time.time()-start,returncode=code)
(out/'launch_timing.json').write_text(json.dumps(rec,indent=2))
raise SystemExit(code)
