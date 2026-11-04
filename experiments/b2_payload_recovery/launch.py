import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
audit=json.loads((HERE/'audit.json').read_text());assert audit['passed']
stage_variant=os.environ.get('B2_RECOVERY_VARIANT')=='v6'
phase_variant=os.environ.get('B2_RECOVERY_VARIANT')=='v5' or stage_variant
if phase_variant:
    assert json.loads((HERE/'audit_phase.json').read_text())['passed']
    if stage_variant:assert json.loads((HERE/'audit_stage.json').read_text())['passed']
    previous=json.loads((HERE/'latest_launch.json').read_text())
    assert (Path(previous['out'])/'DONE').exists(),'Previous guarded pilot has not stopped'
prior=json.loads((ROOT/'experiments/b2_payload/latest_launch.json').read_text())
assert (Path(prior['out'])/'DONE').exists(),'Old training has not finished saving'
source=ROOT/'runs/b2_stable_cycle_v2_20261030_214533/best.pt'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='643e02a1a091ff85f724956dc45e41f6369a542dcea6e2ec52d0f090eff25f70'
prefix='b2_payload_stage_recovery_v6_' if stage_variant else 'b2_payload_phase_recovery_v5_' if phase_variant else 'b2_payload_recovery_v4_'
out=ROOT/'runs'/time.strftime(prefix+'%Y%m%d_%H%M%S');out.mkdir(exist_ok=False)
snap=out/'source';snap.mkdir()
for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
shutil.copy2(HERE/'train.sh',snap/'train.sh')
for name in ('audit.json','diagnosis.json'):shutil.copy2(HERE/name,out/name)
if phase_variant:shutil.copy2(HERE/'audit_phase.json',out/'audit_phase.json')
if stage_variant:shutil.copy2(HERE/'audit_stage.json',out/'audit_stage.json')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6',B2_INIT_CHECKPOINT=str(source),PYTHONUNBUFFERED='1',B2_EVAL_TRIALS='64')
if phase_variant:env['B2_TRAIN_ENTRY']='train_phase.py'
if stage_variant:env['B2_TRAIN_ENTRY']='train_stage.py'
cmd=['bash','experiments/b2_payload_recovery/train.sh','--out',str(out),'--seed','20261115' if stage_variant else '20261114' if phase_variant else '20261113']
with (out/'console.log').open('ab',buffering=0) as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
meta=dict(gpu=6,pid=proc.pid,out=str(out),launch_unix=time.time(),command=cmd,source_checkpoint=str(source),
          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),previous_failed_run=prior['out'],detached=True,
          payload_range_kg=[5,15],force_range_n=[8,16],force_duration_s=[.12,.24],full_cycle_pushes=5,
          max_joint_correction_rad=.04,actor_learning_rate=1e-5,world_model_learning_rate=1e-4,horizon=8,
          eval_every_iterations=500,checkpoint_every_iterations=500,max_iterations=30000,max_wall_hours=3,
          automatic_rollback_on_movement_regression=True,max_rejected_updates=4,
          code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir()})
if phase_variant:meta.update(variant='v5',hind_joint_correction_rad=.008,hind_actor_learning_rate=1e-6,
                             separate_stance_heads=True,correction_precision='FP32',teacher_features_frozen=True,previous_guarded_pilot=previous['out'])
if stage_variant:meta.update(variant='v6',reference_phase_command=True,initial_stance_isolated_from_return_head=True)
(out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n');(HERE/'latest_launch.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
