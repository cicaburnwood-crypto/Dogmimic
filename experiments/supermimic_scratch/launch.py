"""One finite, detached fresh run per robot. No recurring automation."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import tarfile
from zoneinfo import ZoneInfo
from common import HERE,ROOT,clean_environment,write_json,sha256

def main():
    validation=ROOT/'diagnostics/scratch_reset_20261103'
    for robot in ('B2','G1'):
        for rank in range(1 if robot=='B2' else 2):
            value=json.loads((validation/f'production_{robot}'/f'validation_rank{rank}.json').read_text())
            assert value['passed'] and len(value['updates'])==3
        assert (validation/f'strict_interface_{robot}'/'result.json').exists()
    stamp=datetime.datetime.now(ZoneInfo('Asia/Hong_Kong'))
    out=ROOT/'runs'/('supermimic_scratch_'+stamp.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    provenance=json.loads((HERE/'baseline_provenance.json').read_text())
    refs={'B2':'experiments/b2_cycle/b2_full_cycle.npz','G1':'data/processed/g1_standing_jump_060m.npz'}
    for robot,reference in refs.items():
        assert sha256(ROOT/reference)==provenance['reference_sha256'][robot]
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('latest_launch.json','monitor_state.json')]
    hashes={str(p.relative_to(HERE)):sha256(p) for p in files}
    write_json(out/'source_hashes.json',hashes)
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(HERE)))
    launch=dict(experiment='supermimic_scratch',out=str(out),started=stamp.isoformat(),
                initialization='all random; no historical weights or optimizer states',
                algorithm='SuperMimic online WM + H32 BPTT + critic',reward='paper Table A.2',
                supersedes='runs/supermimic_core_v3_20261103_001524',automation_recreated=False,
                hardware_connected=False,robots={},source_snapshot_sha256=sha256(out/'source_snapshot.tar.gz'))
    for robot,gpus,seed in [('B2','6',20261104),('G1','4,5',20261103)]:
        dest=out/robot;dest.mkdir()
        env=clean_environment(robot);env['CUDA_VISIBLE_DEVICES']=gpus
        env['TORCHINDUCTOR_COMPILE_THREADS']='2'
        cmd=[str(ROOT/'.venv/bin/python'),'-u'] if robot=='B2' else [str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=2']
        cmd += [str(HERE/'train.py'),'--motions',refs[robot],'--out',str(dest),'--seed',str(seed),
                '--num-envs','1024','--buffer-steps','512','--warmup-steps','64',
                '--wm-batch','1024','--pi-batch','2048','--wm-batches','10',
                '--k','8','--horizon','32','--use-critic','true','--utd','0.4',
                '--iterations','80000','--wall-hours','6','--lr','0.0002',
                '--lr-decay-iters','80000','--log-every','100','--ckpt-every','2000','--eval-every','4000']
        with (dest/'train.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch['robots'][robot]=dict(out=str(dest),pid=proc.pid,command=cmd,gpus=[int(i) for i in gpus.split(',')],seed=seed,
                                    iteration_budget=80000,hours_limit=6,reference=refs[robot],reference_sha256=sha256(ROOT/refs[robot]))
        write_json(out/'launch.json',launch)
        write_json(HERE/'latest_launch.json',launch)
    write_json(ROOT/'experiments/active_training.json',dict(current_experiment=str(HERE),launch_file=str(HERE/'latest_launch.json'),
               run=str(out),robots=['B2','G1'],automatic_monitor=False,updated=stamp.isoformat()))
    print(json.dumps(launch,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
