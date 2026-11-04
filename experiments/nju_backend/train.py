"""NJU Trainer, with artifact/strict-evaluation management around its original run loop."""
import argparse
import os
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--robot',choices=['B2','G1'],required=True)
p.add_argument('--out',required=True);p.add_argument('--iterations',type=int,default=20000)
p.add_argument('--envs',type=int,default=1024);p.add_argument('--batch',type=int,default=512)
p.add_argument('--max-hours',type=float,default=6.);p.add_argument('--seed',type=int,default=20261103)
p.add_argument('--smoke',action='store_true');p.add_argument('--no-compile',action='store_true')
a=p.parse_args()
os.environ['SM_NJU_ROBOT']=a.robot
sys.path.insert(0,str(HERE/'backend_src'))
# Reuse NJU's pre-CUDA LOCAL_RANK pinning, required by MuJoCo Warp.
from supermimic.scripts import train as upstream_entrypoint
import json
import shutil
import subprocess
import time
import torch
import torch.distributed as dist
from common import ROOT,sha256,tensor_hash,write_json
from export_policy import export
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
from supermimic.world_model.heads import analytic_observation, flatten_history_term_major


class ManagedTrainer(Trainer):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.started=time.time();self.best=None;self.last_evaluated=-1;self.eval_seconds=0.
        self.initial_hashes={k:tensor_hash(getattr(self,k).state_dict()) for k in ('wm','policy','critic')}
        if self.is_main:
            write_json(self.log_dir/'initialization.json',dict(random_initialization=True,loaded_checkpoint=None,
                external_teacher=False,hashes=self.initial_hashes,backend=json.loads((HERE/'backend_provenance.json').read_text())))

    def barrier(self):
        if dist.is_initialized():dist.barrier()

    def save(self,name='ckpt_latest.pt'):
        super().save(name)
        if self.is_main:
            write_json(self.log_dir/'status.json',dict(robot=a.robot,iteration=self.iteration,updates=self.num_updates,
                budget=self.cfg.schedule.total_iterations,elapsed_seconds=time.time()-self.started,
                physical_eval_seconds=self.eval_seconds,updated=time.time(),pid=os.getpid()))
        # Immutable checkpoints are evaluated while both distributed workers wait.
        if name.startswith('ckpt_0') and not a.smoke:
            self.barrier()
            if self.is_main:self.evaluate_checkpoint(name,False)
            self.barrier()

    def evaluate_checkpoint(self,name,full=False,initial=False,pack='DEV-A'):
        ck=self.log_dir/name
        jit=self.log_dir/(ck.stem+'_policy.pt')
        export(self,ck,jit,a.robot)
        tag='initial' if initial else f'it_{self.iteration:07d}'+('_final' if full else '')
        if pack!='DEV-A':tag+='_'+pack
        dest=self.log_dir/'eval'/tag
        env=os.environ.copy()
        for k in list(env):
            if k.startswith(('TORCHELASTIC_', 'SM_')) or k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
                env.pop(k,None)
        env['PYTHONPATH']=str(HERE);env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
        # This child uses the prior pinned simulator solely for unchanged acceptance.
        cmd=[str(ROOT/'.venv/bin/python'),str(HERE/'evaluate.py'),'--robot',a.robot,'--checkpoint',str(jit),'--out',str(dest),'--pack',pack]
        if full:cmd.append('--full')
        t=time.time();dest.mkdir(parents=True,exist_ok=True)
        with (dest/'eval.log').open('w') as log:
            r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        self.eval_seconds+=time.time()-t
        if r.returncode:
            raise RuntimeError(f'Independent acceptance exited {r.returncode}: {dest}/eval.log')
        result=json.loads((dest/'result.json').read_text())
        result.update(source_checkpoint=str(ck),source_checkpoint_sha256=sha256(ck))
        if initial:
            write_json(self.log_dir/'initial_eval.json',result)
            return result
        write_json(self.log_dir/'latest_eval.json',result)
        self.last_evaluated=self.iteration
        # B2 comparison uses the same two nominal loads even for the larger final matrix.
        rows=result['conditions'][:2] if a.robot=='B2' else result['conditions']
        if a.robot=='B2':
            rank=(int(result['all_passed']),min(x['completed']/64 for x in rows),
                  min(x['passed']/64 for x in rows),sum(x['passed'] for x in rows),-max(x['peak_speed'] for x in rows))
        else:rank=(int(result['all_passed']),sum(x['passed'] for x in rows),sum(x['completed'] for x in rows))
        result['selection_rank']=rank
        if self.best is None or rank>tuple(self.best['selection_rank']):
            shutil.copy2(ck,self.log_dir/'best.pt');shutil.copy2(jit,self.log_dir/'best_policy.pt')
            result.update(best_sha256=sha256(self.log_dir/'best.pt'),best_policy_sha256=sha256(self.log_dir/'best_policy.pt'))
            self.best=result;write_json(self.log_dir/'best_eval.json',result)
        with (self.log_dir/'eval.jsonl').open('a') as f:
            f.write(json.dumps({k:result[k] for k in ('it','passed','total','completed','all_passed','full','selection_rank')})+'\n')
        print('STRICT_EVAL',a.robot,self.iteration,result['passed'],result['total'],'accepted',result['all_passed'],flush=True)
        return result

    def run(self):
        if a.smoke:
            from validate_adapter import validate_env
            checks=validate_env(self)
            if self.is_main:write_json(self.log_dir/'adapter_checks.json',checks)
        if not a.smoke:
            self.save('initial.pt');self.barrier()
            if self.is_main:self.evaluate_checkpoint('initial.pt',False,True)
            self.barrier()
        super().run()
        if not self.is_main:return
        ck=self.log_dir/'ckpt_latest.pt'
        if a.smoke:
            final_hashes={k:tensor_hash(getattr(self,k).state_dict()) for k in self.initial_hashes}
            assert self.num_updates>=3
            assert all(final_hashes[k]!=self.initial_hashes[k] for k in final_hashes)
            assert all(torch.isfinite(v).all() for mod in (self.wm,self.policy,self.critic) for v in mod.parameters())
            meta=export(self,ck,self.log_dir/'smoke_policy.pt',a.robot)
            write_json(self.log_dir/'VALIDATED.json',dict(robot=a.robot,iteration=self.iteration,updates=self.num_updates,
                horizon=self.cfg.policy_trainer.horizon,wm_window=self.cfg.wm_trainer.window,critic=self.cfg.policy_trainer.use_critic,
                initial_hashes=self.initial_hashes,final_hashes=final_hashes,export_error=meta['export_max_abs_error'],seconds=time.time()-self.started))
        else:
            train_end=time.time()
            write_json(self.log_dir/'DONE',dict(iteration=self.iteration,reason='budget' if self.iteration>=a.iterations else 'STOP_or_wall_time',time=train_end))
            result=self.evaluate_checkpoint('ckpt_latest.pt',True)
            if result['all_passed']:
                holds=[self.evaluate_checkpoint('ckpt_latest.pt',True,pack=pack) for pack in ('HOLD-A','HOLD-B')]
                if all(r['all_passed'] for r in holds):
                    shutil.copy2(ck,self.log_dir/'accepted.pt')
                    write_json(self.log_dir/'accepted_eval.json',dict(holds=holds,sha256=sha256(ck)))
            write_json(self.log_dir/'EVAL_DONE.json',dict(iteration=self.iteration,time=time.time(),accepted=(self.log_dir/'accepted.pt').exists()))
            write_json(self.log_dir/'timing.json',dict(training_and_periodic_evaluation_seconds=train_end-self.started,
                physical_evaluation_seconds=self.eval_seconds,pipeline_seconds=time.time()-self.started))


torch.set_num_threads(2)
cfg=TrainConfig(run_name=Path(a.out).name,log_root=Path(a.out).parent,seed=a.seed)
cfg.data.npz_dir=HERE/'data'/a.robot
cfg.env.num_envs=a.envs;cfg.env.episode_length_s=20. if a.robot=='B2' else 10.
cfg.env.replay_capacity=128 if a.smoke else 1000
cfg.wm_trainer.batch_size=a.batch;cfg.policy_trainer.batch_size=a.batch
cfg.schedule.total_iterations=a.iterations
cfg.schedule.max_hours=a.max_hours
cfg.schedule.checkpoint_interval=2000
cfg.schedule.log_interval=10 if a.smoke else 50
if a.no_compile:
    cfg.wm_trainer.compile=None;cfg.policy_trainer.compile=None
trainer=None
try:
    trainer=ManagedTrainer(cfg)
    trainer.run()
except BaseException as e:
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    write_json(out/f'ERROR_rank{os.environ.get("RANK","0")}.json',dict(error=repr(e),time=time.time(),iteration=getattr(trainer,'iteration',None)))
    raise
finally:
    if trainer is not None:trainer.env.close()
