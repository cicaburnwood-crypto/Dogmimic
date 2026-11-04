"""Launch one bounded detached vibe training run; never deploy or schedule."""
import argparse,datetime,json,os,subprocess
from pathlib import Path
from bootstrap import HERE,ROOT
from common import write_json,sha256

def main(validation):
    validation=Path(validation).resolve()
    check=json.loads((validation/'VALIDATED.json').read_text())
    assert check['real_updates']==3 and check['production_batch'] and check['engine_learning_files_unchanged']
    compiled=json.loads((validation/'compiled_check.json').read_text())
    assert compiled['iterations']==10 and compiled['updates']>=4 and compiled['finite']
    assert compiled['num_envs']==1024 and compiled['compile_wm'] and compiled['export_max_error']<1e-6
    probe=json.loads((validation/'interface_probe/result.json').read_text())
    assert len(probe['cases'])==2 and probe['total']==4
    # Avoid duplicate work; this is a job-launch check, not a robot operation lock.
    old=HERE/'latest_launch.json'
    if old.exists():
        previous=json.loads(old.read_text());pid=Path('/proc')/str(previous['pid'])
        if pid.exists() and previous['out'].encode() in (pid/'cmdline').read_bytes():raise RuntimeError('A training run is already active')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    out=ROOT/'runs'/('b2_tilt_locomotion_vibe_'+now.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SM_','B2_','G1_','TORCHELASTIC_')) or k in ('RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(k,None)
    env.update(CUDA_VISIBLE_DEVICES='6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='2')
    cmd=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'train.py'),'--out',str(out)]
    sources={str(p.relative_to(HERE)):sha256(p) for p in HERE.rglob('*.py') if '__pycache__' not in str(p)}
    write_json(out/'frozen_contract.json',dict(evaluation=json.loads((HERE/'evaluation_manifest.json').read_text()),sources=sources,
        model_sha256=sha256(HERE/'b2_tilt_locomotion.xml'),random_initialization=True,hardware_validated=False))
    with (out/'train.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(task='b2_tilt_locomotion_v1',backend='vibe',out=str(out),pid=proc.pid,session_id=proc.pid,cmd=cmd,cwd=str(ROOT),gpus=[6],
        started=now.isoformat(),budget=80000,wall_hours=6,initialization='random Actor/WM/critic',old_tasks_restarted=False,
        periodic_automation=False,hardware_connected=False,validation=str(validation/'VALIDATED.json'),compiled_validation=str(validation/'compiled_check.json'),
        command_limits=dict(vx=[-4,4],vy=[-1,1],xy_norm=4,roll_degrees=[-20,20],pitch_degrees=[-20,20],simultaneous_tilt=True,yaw_accumulated_angle=None),
        yaw_training=dict(max_rate_rad_s=2,speed_dependent_sampling='min(2,2/max(1,speed))',interface_rate_clipped=False))
    write_json(out/'launch.json',launch);write_json(HERE/'latest_launch.json',launch)
    write_json(ROOT/'experiments/active_training_backend.json',dict(backend='vibe',task='b2_tilt_locomotion_v1',launch=str(HERE/'latest_launch.json'),status_script=str(HERE/'status.py'),periodic_automation=False))
    print(json.dumps(launch,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validation',required=True);main(p.parse_args().validation)
