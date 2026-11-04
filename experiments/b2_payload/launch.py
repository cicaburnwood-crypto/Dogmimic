import hashlib,json,os,shutil,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
audit=json.loads((HERE/'audit.json').read_text())
assert audit['passed'] and audit['timed_force_replay_verified'] and audit['real_imagined_reward_parity']
assert audit['heading_frame_force_context_verified']
source=ROOT/'runs/b2_stable_cycle_v2_20261030_214533/best.pt'
assert source.exists()
out=ROOT/'runs'/time.strftime('b2_payload_stable_cycle_v3_%Y%m%d_%H%M%S')
out.mkdir(exist_ok=False)
snapshot=out/'source';snapshot.mkdir()
for path in HERE.glob('*.py'):shutil.copy2(path,snapshot/path.name)
shutil.copy2(HERE/'train.sh',snapshot/'train.sh')
shutil.copy2(HERE/'audit.json',out/'audit.json')
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='6',B2_INIT_CHECKPOINT=str(source),PYTHONUNBUFFERED='1',B2_WM_PREFIT_UPDATES='100')
cmd=['bash','experiments/b2_payload/train.sh','--out',str(out),'--seed','20261111']
with (out/'console.log').open('ab',buffering=0) as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                          start_new_session=True,close_fds=True)
meta=dict(gpu=6,pid=proc.pid,out=str(out),launch_unix=time.time(),command=cmd,
          source_checkpoint=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          payload_range_kg=[5,15],payload_distribution='continuous uniform, resampled on each episode reset',
          force_range_n=[8,16],push_duration_s=[.12,.24],full_cycle_pushes=5,detached=True,
          max_iterations=30000,max_wall_hours=3,eval_every_iterations=2000,checkpoint_every_iterations=1000,
          strict_target='>=95% and nominal pass, all 0/5/10/15 kg + pushes, two consecutive evaluations',
          code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()})
(out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
(HERE/'latest_launch.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
