"""Finite detached v3 workflow. No recurring automation, no hardware connection."""
import argparse
import json
import os
import select
import shutil
import subprocess
import sys
import time
import traceback
import importlib.metadata
from pathlib import Path
from common import ROOT,DESIGN,read_json,write_json,sha256
from selection import improved,comparable

p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--g1-refit-pid',type=int);p.add_argument('--adopt-b2-eval-pid',type=int)
a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
m=read_json(DESIGN/'experiment_matrix.json');start=time.time();cost={'evaluation_seconds':46.55484437942505,'diagnostic_seconds':143.84724974632263}
status=dict(started=start,stage='preflight',pid=os.getpid(),robots={},automation_recreated=False,hardware_connected=False)
source={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'experiments/supermimic_core').glob('*.py')}

def save():
    status.update(updated=time.time(),costs=cost)
    write_json(out/'status.json',status)


def open_pid_handle(pid):
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    # Some standalone CPython builds omit the wrapper although Linux supports it.
    import ctypes,platform
    assert platform.machine() in ('x86_64','aarch64')
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.syscall(434,int(pid),0)
    if fd<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    return fd


def run(script,args,gpus,folder,kind,timeout):
    if (out/'STOP').exists():raise InterruptedError('User STOP')
    if kind=='evaluation' and cost['evaluation_seconds']>=14400:raise RuntimeError('GPU6 evaluation budget exhausted; acceptance incomplete')
    if kind=='diagnostic' and cost['diagnostic_seconds']>=3600:raise RuntimeError('GPU6 diagnostic budget exhausted')
    for file,digest in source.items():assert sha256(ROOT/file)==digest,('Running source changed',file)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1')
    if gpus=='4,5':cmd=[str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=2']
    else:cmd=[str(ROOT/'.venv/bin/python'),'-u']
    cmd += [str(ROOT/'experiments/supermimic_core'/script),*map(str,args)]
    begin=time.time()
    with open(folder/'worker.log','w') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status.update(stage=kind,child_pid=child.pid,child_command=cmd,child_output=str(folder));save()
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise RuntimeError(f'{script} finite wall budget exceeded')
    elapsed=time.time()-begin
    if kind in ('evaluation','diagnostic'):cost[kind+'_seconds']+=elapsed
    if code:raise RuntimeError(f'{script} exited {code}; see {folder}/worker.log')
    save()


def eval_checkpoint(robot,checkpoint,folder,pack='DEV-A',basic=False):
    args=['--robot',robot,'--checkpoint',checkpoint,'--out',folder,'--pack',pack]
    if basic:args+=['--basic']
    run('evaluation.py',args,'6',folder,'evaluation',3600)
    return read_json(Path(folder)/'result.json')


def do_diagnostics(robot,folder,wm=None):
    args=['--robot',robot,'--out',folder]
    if wm:args+=['--wm-checkpoint',wm]
    run('diagnostics.py',args,'6',folder,'diagnostic',1800)
    return read_json(Path(folder)/'result.json')


def wait_existing_refit():
    result=out/'G1/D0/wm_refit/result.json'
    if result.exists():return
    pid=a.g1_refit_pid
    assert pid
    actual=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    assert 'experiments/supermimic_core/wm_refit.py' in actual and str(out/'G1/D0/wm_refit') in actual
    status.update(stage='wait_for_existing_G1_WM_refit',child_pid=pid);save()
    fd=open_pid_handle(pid)
    try:
        readable,_,_=select.select([fd],[],[],1800)
        if not readable:raise RuntimeError('Existing finite G1 WM fit did not finish within budget')
    finally:os.close(fd)
    if not result.exists():raise RuntimeError('Existing G1 WM fit exited without a result; not restarted')

try:
    # Validate actual source files against the accepted, earlier read-only audit.
    audit=read_json(DESIGN/'source_audit.json')['files']
    verified={}
    for file,info in audit.items():
        if file.startswith(('experiments/','data/')) and info.get('exists'):
            assert sha256(ROOT/file)==info['sha256'],file
            verified[file]=info['sha256']
    for robot in ['G1','B2']:
        r=m['robots'][robot];assert sha256(r['initialization']['checkpoint'])==r['initialization']['sha256']
    tests=ROOT/'runs/supermimic_core_v3_audit'
    for name in ['G1_cpu','G1_cuda','G1_ddp','B2_cpu','B2_cuda','G1_fork','B2_fork']:
        assert read_json(tests/name/'report.json')['passed'],name
    write_json(out/'source_verification.json',dict(original=verified,new=source))
    dependencies={}
    for package in ['torch','mujoco','mujoco-warp','warp-lang','mjlab','numpy']:
        try:dependencies[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:dependencies[package]='unavailable'
    write_json(out/'execution_authorization.json',dict(user_instruction='执行训练',design_version='3.0',
        overrides_prior_design_only_flags=True,automatic_monitoring=False,robots=['B2_0kg_5kg','G1_jump60cm'],
        new_actor_training_conditional_on_D0=True,dependencies=dependencies,started=start))
    if (out/'source').exists():
        (out/'source').rename(out/f'source_prior_{int(time.time())}')
    shutil.copytree(ROOT/'experiments/supermimic_core',out/'source')
    save()
    # G1 WM fitting on4/5 can overlap the independent B2 initial evaluation on6.
    b2=out/'B2/D0';g1=out/'G1/D0'
    initial_file=b2/'initial_DEV_A/result.json'
    if a.adopt_b2_eval_pid and not initial_file.exists():
        pid=a.adopt_b2_eval_pid
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        assert str(ROOT/'experiments/supermimic_core/evaluation.py') in cmd and str(b2/'initial_DEV_A') in cmd
        status.update(stage='adopt_existing_B2_initial_evaluation',child_pid=pid,child_output=str(b2/'initial_DEV_A'));save()
        fd=open_pid_handle(pid)
        try:
            ready,_,_=select.select([fd],[],[],3600)
            if not ready:raise RuntimeError('Existing B2 evaluation exceeded finite time budget')
        finally:os.close(fd)
        if not initial_file.exists():raise RuntimeError('Existing B2 evaluation ended without result; no restart')
    if initial_file.exists():
        initial=read_json(initial_file)
        assert initial['checkpoint_sha256']==m['robots']['B2']['initialization']['sha256'] and initial['complete_coverage']
        cost['evaluation_seconds']+=initial['elapsed_seconds']
    else:initial=eval_checkpoint('B2',m['robots']['B2']['initialization']['checkpoint'],b2/'initial_DEV_A')
    bdiag=do_diagnostics('B2',b2/'diagnostics_before_refit')
    wait_existing_refit()
    gdiag=do_diagnostics('G1',g1/'diagnostics',g1/'wm_refit/wm_refit.pt')
    status['robots']['G1']=dict(stage='D0_passed' if gdiag['passed'] else 'stopped_model_credibility',diagnostics=str(g1/'diagnostics/result.json'))
    save()
    for robot in ['G1','B2']:
        root=out/robot;d0=root/'D0'
        if robot=='B2':
            if bdiag['passed']:
                shutil.copytree(d0/'diagnostics_before_refit',d0/'diagnostics')
                diag=bdiag
            elif bdiag.get('summaries'):
                run('wm_refit.py',['--robot',robot,'--out',d0/'wm_refit'],'6',d0/'wm_refit','WM_refit',1800)
                diag=do_diagnostics(robot,d0/'diagnostics',d0/'wm_refit/wm_refit.pt')
            else:diag=bdiag
            status['robots'][robot]=dict(stage='D0_passed' if diag['passed'] else 'stopped_model_credibility',diagnostics=str(d0/'diagnostics/result.json'));save()
        else:diag=gdiag
        if not diag['passed']:
            write_json(root/'STOPPED.json',dict(reason='D0 model-direction credibility not established after the single permitted WM fit',actor_updates=0,diagnostic=diag.get('summaries',diag)))
            continue
        pairs=[]
        for seed in [1202601,1202602]:
            run('train_pair.py',['--robot',robot,'--seed',seed,'--out',root/f'pair_{seed}','--d0',d0],
                '4,5' if robot=='G1' else '6',root/f'pair_{seed}','core_pilot',21600)
            pairs.append(read_json(root/f'pair_{seed}/result.json'))
            cost['evaluation_seconds']+=sum(x['eval_seconds'] for x in pairs[-1]['branches'].values())
        # Evaluate matching endpoints on both development packs. Never compare
        # a time-censored branch against a later checkpoint of its peer.
        comparisons=[]
        for pair in pairs:
            b=pair['branches'];same=b['C0']['endpoint']==b['C1']['endpoint'] and b['C1']['endpoint']>0
            checks=[]
            for pack,key in [('DEV-A','latest'),('DEV-B','secondary')]:
                if not same:checks.append(False);continue
                if pack=='DEV-B':
                    init_path=root/f'initial_{pack}/result.json'
                    init=read_json(init_path) if init_path.exists() else eval_checkpoint(robot,m['robots'][robot]['initialization']['checkpoint'],root/f'initial_{pack}',pack,basic=robot=='B2')
                else:init=b['C0']['initial']
                c0,c1=b['C0'][key],b['C1'][key]
                checks.append(bool(c0 and c1 and comparable(c1,c0) and improved(c1,c0) and improved(c1,init)))
            comparisons.append(dict(seed=pair['seed'],same_sample_endpoint=same,development_passes=checks,passed=all(checks)))
        qualified=all(x['passed'] for x in comparisons)
        write_json(root/'pilot_comparison.json',dict(qualified=qualified,comparisons=comparisons,conditional_candidate_not_selected=True))
        if qualified and robot=='B2':
            endpoint=pairs[0]['branches']['C1']['endpoint'];checkpoint=root/f'pair_1202601/C1/candidate_{endpoint:06d}.pt'
            full=eval_checkpoint(robot,checkpoint,root/'full_candidate_DEV_A')
            baseline=read_json(d0/'initial_DEV_A/result.json')
            qualified=improved(full,baseline)
        status['robots'][robot].update(stage='qualified_for_long_training' if qualified else 'pilot_not_promoted');save()
        if qualified:
            run('train_long.py',['--robot',robot,'--pair',root/'pair_1202601','--out',root/'long'],
                 '4,5' if robot=='G1' else '6',root/'long','long_training',21600)
            status['robots'][robot]['stage']='long_training_finished';save()
    status.update(stage='finished',finished=time.time());save()
    write_json(out/'DONE',dict(finished=time.time(),robots=status['robots']))
except Exception as error:
    status.update(stage='stopped',error=str(error),traceback=traceback.format_exc(),finished=time.time());save()
    write_json(out/'ERROR.json',status)
    traceback.print_exc()
    sys.exit(1)
