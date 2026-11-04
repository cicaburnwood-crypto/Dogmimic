"""One matched C0/C1 seed, detached by supervisor; no automatic reward changes."""
import argparse
import dataclasses
import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from common import ROOT,DESIGN,setup,read_json,write_json,sha256,isolated_rng,tensor_digest
from selection import score,RegressionGuard


def train_pair(robot,seed,root,d0,quick=False):
    # This is a research precondition, not a new robot control lock.
    d0=Path(d0)
    proof=read_json(d0/'diagnostics/result.json')
    assert proof['passed'] and not proof.get('smoke') and proof['actor_updates']==0
    matrix=read_json(DESIGN/'experiment_matrix.json');m=matrix['robots'][robot]
    assert proof['source_sha256']==m['initialization']['sha256']
    Trainer,base,_=setup(robot)
    import torch
    from state import TrainerSnapshot,verify_replay,EnvironmentSnapshot
    torch.set_num_threads(2)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    batch=m['batches'];hp=m['hyperparameters']
    cfg=base.TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz' if robot=='B2' else 'data/processed/g1_standing_jump_060m.npz',
        out=str(root/'prepare'),seed=seed,num_envs=batch['envs_per_gpu'],buffer_steps=batch['buffer_steps'],
        k=8,horizon=32,wm_batch=batch['wm_batch'],pi_batch=batch['pi_batch'],wm_batches=batch['wm_batches'],
        lr=hp['base_lr'],lr_decay_iters=hp['lr_decay_iters'],action_noise=hp['action_noise_std'],
        use_critic=False,compile_wm=True,eval_every=0,wall_hours=6)
    if quick:
        raise ValueError('Training quick mode cannot bypass full D0 or launch a smoke as a formal run')
    start=time.time();t=Trainer(cfg,'cuda');t.core_phase='shared_prepare'
    refit=d0/'wm_refit/wm_refit.pt'
    refit_seconds=0.
    if refit.exists():
        c=torch.load(refit,map_location=t.device,weights_only=False)
        assert c['actor_updates']==0 and all(torch.equal(v,t.policy.state_dict()[k]) for k,v in c['policy'].items())
        assert sha256(refit)==proof['wm_source_sha256']
        t.wm.load_state_dict(c['wm'],strict=True)
        refit_seconds=read_json(d0/'wm_refit/result.json')['elapsed']
    # Record expanded configuration and the actual input/source identity on every rank.
    if t.rank==0:write_json(root/'manifest.json',dict(robot=robot,seed=seed,cfg=dataclasses.asdict(cfg),
        source_sha256=t.source_hash,source_iteration=t.source_iteration,world=t.world,
        plan_sha256=sha256(DESIGN/'PLAN.txt'),matrix_sha256=sha256(DESIGN/'experiment_matrix.json'),
        acceptance_sha256=m['acceptance']['sha256'],initialized=time.time()))
    common_start=time.time()
    for i in range(128):t.collect()
    for i in range(500):
        if (root/'STOP').exists():raise RuntimeError('User STOP during common preparation')
        t.it=i;t.collect();t.update_credit+=cfg.utd
        while t.update_credit>=1.:
            t.update_credit-=1.;t.update_wm();t.updates+=1
        if i%100==0 and t.rank==0:print(json.dumps(dict(phase='shared_WM_prefit',iteration=i,actor_updates=0)),flush=True)
        if time.time()-common_start>(1800-refit_seconds)/2:raise RuntimeError('Per-seed half of remaining common preparation time budget exhausted')
    torch.cuda.synchronize();common_seconds=time.time()-common_start;t.train_time=common_seconds
    t.it=500
    replay=verify_replay(t.env,t.policy,steps=8)
    assert replay['passed'],replay
    common=TrainerSnapshot(t)
    fork_digest=dict(policy=tensor_digest(t.policy.state_dict()),wm=tensor_digest(t.wm.state_dict()),
                     replay=tensor_digest(vars(t.buf)))
    if t.rank==0:write_json(root/'fork.json',dict(common_seconds=common_seconds,hashes=fork_digest,physical_replay=replay,actor_updates=0))
    common.save(root/f'common_rank{t.rank}.pt')
    t._barrier()

    def evaluate(checkpoint,folder,pack='DEV-A',full=False):
        t._barrier();r=None;begin=time.time()
        if t.rank==0:
            if robot=='B2':
                from evaluation import evaluate as run_eval
                with isolated_rng(t.device):r=run_eval(robot,checkpoint,folder,pack=pack,full=full)
                gc.collect();torch.cuda.empty_cache()
            else:
                cmd=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'experiments/supermimic_core/evaluation.py'),
                     '--robot',robot,'--checkpoint',str(checkpoint),'--out',str(folder),'--pack',pack]
                env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='6'
                for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','LOCAL_WORLD_SIZE']:
                    env.pop(k,None)
                with open(Path(folder).with_suffix('.log'),'w') as f:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
                r=read_json(Path(folder)/'result.json')
            assert r['checkpoint_sha256']==sha256(checkpoint)
        if t.world>1:
            import torch.distributed as dist
            data=[r];dist.broadcast_object_list(data,src=0);r=data[0]
        t._barrier()
        return r,time.time()-begin

    results={}
    for branch in ['C0','C1']:
        common.restore();cfg.use_critic=branch=='C1';t.core_phase=branch
        branch_root=root/branch;branch_root.mkdir(parents=True,exist_ok=True)
        assert fork_digest['policy']==tensor_digest(t.policy.state_dict())
        assert fork_digest['wm']==tensor_digest(t.wm.state_dict())
        assert fork_digest['replay']==tensor_digest(vars(t.buf))
        train_seconds=0.;eval_seconds=0.;prefit_seconds=0.;branch_start=time.time()
        if cfg.use_critic:
            before=tensor_digest(t.policy.state_dict());wm_before=tensor_digest(t.wm.state_dict());pstart=time.time()
            with isolated_rng(t.device):
                for i in range(100):
                    t.update_policy(critic_only=True)
                    if time.time()-pstart>600:raise RuntimeError('Critic prefit time cap reached')
            prefit_seconds=time.time()-pstart
            t.train_time=common_seconds+prefit_seconds
            assert before==tensor_digest(t.policy.state_dict()) and wm_before==tensor_digest(t.wm.state_dict())
        def save_candidate(loop):
            path=branch_root/f'candidate_{loop:06d}.pt'
            if t.rank==0:t.save(str(path))
            t._barrier();return path
        initial_cp=save_candidate(0)
        # Evaluate once per shared Actor; critic prefit cannot alter this baseline.
        if branch=='C0':initial,seconds=evaluate(initial_cp,branch_root/'eval_000000');eval_seconds+=seconds
        else:
            initial=copy_result(results['C0']['initial'],initial_cp)
            if t.rank==0:write_json(branch_root/'initial_eval.json',initial)
        best=None;bad=0;history=[];last=initial;last_cp=initial_cp;reason='iteration_budget';endpoint=0
        guard=RegressionGuard(initial)
        for loop in range(1,2001):
            if (root/'STOP').exists() or (root.parent/'STOP').exists():reason='user_stop';break
            if train_seconds+prefit_seconds>=1800:reason='training_time_budget';break
            if time.time()-branch_start>=21600:reason='wall_time_guard';break
            begin=time.time();t.it=500+loop;t.core_phase=branch
            lr=base.lr_at(cfg,t.it)
            for opt in (t.opt_wm,t.opt_pi,t.opt_v):
                for group in opt.param_groups:group['lr']=lr
            t.collect();t.update_credit+=cfg.utd;stats={}
            while t.update_credit>=1.:
                t.update_credit-=1.;stats.update(t.update_wm());stats.update(t.update_policy());t.updates+=1
            torch.cuda.synchronize();train_seconds+=time.time()-begin
            t.train_time=common_seconds+prefit_seconds+train_seconds
            endpoint=loop
            if loop%100==0 and t.rank==0:
                rec=dict(branch=branch,seed=seed,actor_enabled_loops=loop,actor_updates=t.actor_updates,
                    wm_gradient_steps=t.wm_gradient_steps,critic_fits=t.critic_fits,
                    critic_gradient_steps=t.critic_fits*cfg.critic_epochs*cfg.critic_minibatches,
                    physics_transitions=t.env_steps*t.world,train_seconds=train_seconds,prefit_seconds=prefit_seconds,
                    eval_seconds=eval_seconds,updated=time.time(),**stats)
                write_json(branch_root/'progress.json',rec)
                with open(branch_root/'metrics.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(rec),flush=True)
            if loop in (500,1000,2000):
                last_cp=save_candidate(loop);last,seconds=evaluate(last_cp,branch_root/f'eval_{loop:06d}');eval_seconds+=seconds
                stop_for_regression=guard.update(last,best or initial)
                if best is None or score(last)>score(best):
                    best=last
                    if t.rank==0:
                        shutil.copy2(last_cp,branch_root/'best.pt');write_json(branch_root/'best_eval.json',best)
                history.append(last)
                if t.rank==0:write_json(branch_root/'latest_eval.json',last)
                if stop_for_regression:
                    reason='persistent_physical_regression'
                    if t.rank==0:write_json(branch_root/'GUARD_STOP.json',dict(reason=reason,loop=loop,best=best['checkpoint_sha256']))
                    break
        # Preserve full endpoint state, never label a weights-only file as resumable.
        t.core_phase='pilot_finished';state=TrainerSnapshot(t);state.save(branch_root/f'endpoint_rank{t.rank}.pt');del state
        if t.rank==0:t.save(str(branch_root/'last.pt'))
        t._barrier()
        if last_cp.name!=f'candidate_{endpoint:06d}.pt':
            last_cp=save_candidate(endpoint);last,seconds=evaluate(last_cp,branch_root/f'eval_{endpoint:06d}');eval_seconds+=seconds
        if endpoint>0:
            secondary,seconds=evaluate(last_cp,branch_root/f'eval_{endpoint:06d}_DEV_B',pack='DEV-B');eval_seconds+=seconds
        else:secondary=None
        results[branch]=dict(initial=initial,latest=last,secondary=secondary,history=history,endpoint=endpoint,
            train_seconds=train_seconds,prefit_seconds=prefit_seconds,eval_seconds=eval_seconds,reason=reason,
            actor_updates=t.actor_updates,wm_gradient_steps=t.wm_gradient_steps,critic_fits=t.critic_fits)
        if t.rank==0:
            write_json(branch_root/'summary.json',results[branch]);write_json(branch_root/'DONE',dict(reason=reason,finished=time.time()))
        if reason=='user_stop':break
    if t.rank==0:write_json(root/'result.json',dict(robot=robot,seed=seed,common_seconds=common_seconds,branches=results,elapsed=time.time()-start))
    t._barrier()


def copy_result(source,checkpoint):
    import copy
    result=copy.deepcopy(source)
    result['checkpoint']=str(checkpoint);result['checkpoint_sha256']=sha256(checkpoint)
    result['reused_physical_baseline']='exactly equal Actor verified at common fork; no critic influence on action'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robot',required=True,choices=['B2','G1']);p.add_argument('--seed',required=True,type=int)
    p.add_argument('--out',required=True);p.add_argument('--d0',required=True);a=p.parse_args()
    train_pair(a.robot,a.seed,a.out,a.d0)
