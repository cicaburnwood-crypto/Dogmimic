"""Launch one finite, detached continuation after implementation validation."""
import argparse,datetime,json,os,subprocess,shutil
from pathlib import Path
from bootstrap import HERE,ROOT
from common import write_json,sha256
from sampling import MIXTURE


def main(validation):
    validation=Path(validation).resolve()
    check=json.loads((validation/'VALIDATED.json').read_text())
    assert check['real_updates']==3 and check['production_batch'] and check['engine_learning_files_unchanged']
    assert check['exact_actor_wm_continuation'] and check['prefit_frozen_actor'] and check['reversal_schedule_verified']
    compiled=json.loads((validation/'compiled_check.json').read_text())
    assert compiled['iterations']==10 and compiled['updates']>=4 and compiled['actor_updates']>=2 and compiled['finite']
    assert compiled['num_envs']==1024 and compiled['compile_wm'] and compiled['export_max_error']<1e-6
    baseline=json.loads((validation/'baseline_eval/result.json').read_text())
    assert baseline['it']==20000 and baseline['total']==1312 and not baseline['full']
    source=Path(check['source']['path'])
    assert sha256(source)==check['source']['sha256']==baseline['checkpoint_sha256']
    pinned=json.loads((HERE/'unchanged_contract.json').read_text())
    assert all(sha256(HERE/name)==h for name,h in pinned.items())
    previous_path=HERE/'latest_launch.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text());proc=Path('/proc')/str(previous['pid'])
        if proc.exists() and previous['out'].encode() in (proc/'cmdline').read_bytes():
            raise RuntimeError('This continuation is already active')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    out=ROOT/'runs'/('b2_tilt_locomotion_repair_v2_'+now.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    archive=out/'source_archive';archive.mkdir()
    for name in ['best_eval.json','latest_eval.json','timing.json','launch.json']:
        shutil.copy2(source.parent/name,archive/name)
    write_json(archive/'weights.json',dict(best_path=str(source),best_sha256=sha256(source),
        last_path=str(source.parent/'last.pt'),last_sha256=sha256(source.parent/'last.pt'),original_run_unchanged=True))
    sources={str(p.relative_to(HERE)):sha256(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}
    write_json(out/'frozen_contract.json',dict(evaluation=json.loads((HERE/'evaluation_manifest.json').read_text()),
        sources=sources,model_sha256=sha256(HERE/'b2_tilt_locomotion.xml'),source_checkpoint=check['source'],
        initialization='Actor/WM continuation; new critic/replay/optimizers',hardware_validated=False))
    shutil.copy2(HERE/'REPAIR_NOTES.txt',out/'REPAIR_NOTES.txt')
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SM_','B2_','G1_','TORCHELASTIC_')) or k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
               PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='2')
    cmd=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'train.py'),'--out',str(out)]
    with (out/'train.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(task='b2_tilt_locomotion_v1',revision='boundary_repair_v2',backend='vibe',out=str(out),
        pid=proc.pid,session_id=proc.pid,cmd=cmd,cwd=str(ROOT),gpus=[6],started=now.isoformat(),
        budget=40000,wall_hours=6,actor_warmup_iters=1000,eval_every=2000,learning_rate=5e-5,
        source_checkpoint=check['source'],initialization='own best20000 Actor/WM; new critic/replay/optimizers',
        sampling_mixture=MIXTURE,old_tasks_restarted=False,periodic_automation=False,hardware_connected=False,
        validation=str(validation/'VALIDATED.json'),compiled_validation=str(validation/'compiled_check.json'),
        source_baseline_validation=str(validation/'baseline_eval/result.json'),
        command_limits=dict(vx=[-4,4],vy=[-1,1],xy_norm=4,roll_degrees=[-20,20],pitch_degrees=[-20,20],
                            simultaneous_tilt=True,yaw_accumulated_angle=None),
        yaw_training=dict(max_rate_rad_s=2,speed_dependent_sampling='min(2,2/max(1,speed))',interface_rate_clipped=False))
    write_json(out/'launch.json',launch);write_json(HERE/'latest_launch.json',launch)
    write_json(ROOT/'experiments/active_training_backend.json',dict(backend='vibe',task='b2_tilt_locomotion_v1',revision='boundary_repair_v2',
        launch=str(HERE/'latest_launch.json'),status_script=str(HERE/'status.py'),periodic_automation=False))
    print(json.dumps(launch,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation',required=True);main(p.parse_args().validation)
