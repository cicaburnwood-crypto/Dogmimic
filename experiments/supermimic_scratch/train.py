"""Fresh SuperMimic training with auditable initialization and independent eval."""
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from common import HERE,ROOT,clean_environment,sha256,tensor_hash,write_json
from sm_original.train import Trainer,parse_args
from environments import make_env


class ScratchTrainer(Trainer):
    def __init__(self,cfg,device='cuda'):
        assert not cfg.resume,'Fresh experiment cannot load a checkpoint'
        assert cfg.use_critic,'SuperMimic critic is part of this experiment'
        self.robot_kind=os.environ['SM_RESET_ROBOT']
        self.started_at=time.time()
        self.best_score=None;self.last_eval=None;self.eval_seconds=0.
        self.final_evaluation=False
        super().__init__(cfg,device)
        self._initialization=dict(
            actor='random',world_model='random',critic='random',replay='empty',optimizers='empty',
            external_policy=False,teacher=False,loaded_checkpoint=None,
            norm_counts={name:float(norm.count) for name,norm in
                         [('actor_obs',self.policy.obs_norm),('actor_goal',self.policy.goal_norm),
                          ('wm_input',self.wm.in_norm),('wm_output',self.wm.out_norm),('critic',self.critic.norm)]},
            module_hashes={name:tensor_hash(module.state_dict()) for name,module in
                           [('actor',self.policy),('world_model',self.wm),('critic',self.critic)]})
        assert all(x==0 for x in self._initialization['norm_counts'].values())
        assert self.buf.size==0 and all(len(o.state)==0 for o in (self.opt_wm,self.opt_pi,self.opt_v))
        if self.rank==0:
            self.save(str(Path(cfg.out)/'initial.pt'))
            write_json(Path(cfg.out)/'initialization.json',self._initialization)
            write_json(Path(cfg.out)/'config.json',dataclasses.asdict(cfg))
            self.status('initialized')

    def _make_env(self):
        env=make_env(self.robot_kind,self.lib,self.device,self.cfg.num_envs,self.cfg.gamma,
                     compile=self.cfg.compile_wm)
        self.tcfg=env.tcfg
        return env

    def _extra_state(self):
        return dict(experiment='supermimic_scratch',robot=self.robot_kind,
                    initialization=getattr(self,'_initialization',None),baseline_commit='072f0910dadc9d95c720e625b546120e9ce917c9',
                    task_cfg=dataclasses.asdict(self.tcfg),xi_dim=self.env.xi.shape[-1],
                    reward='SuperMimic Table A.2',teacher=False)

    def load(self,path):
        raise RuntimeError('Loading historical weights is disabled in the fresh-training entrypoint')

    def status(self,stage,**extra):
        if self.rank==0:
            write_json(Path(self.cfg.out)/'status.json',dict(robot=self.robot_kind,stage=stage,it=self.it,
                       actor_updates=self.updates,train_seconds=self.train_time,elapsed_seconds=time.time()-self.started_at,
                       pid=os.getpid(),cwd=str(ROOT),updated=time.time(),**extra))

    def collect(self,random=False):
        super().collect(random)
        if self.it%self.cfg.log_every==0:self.status('training')

    def update_policy(self):
        result=super().update_policy()
        if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
            raise FloatingPointError('Non-finite optimization statistics')
        return result

    def _should_stop(self):
        stopped=super()._should_stop()
        external=(Path(self.cfg.out)/'STOP').exists() or (Path(self.cfg.out).parent/'STOP').exists() or (Path(self.cfg.out)/'TARGET_MET.json').exists()
        wall=time.time()-self.started_at>=self.cfg.wall_hours*3600
        flag=torch.tensor([int(stopped or external or wall)],device=self.device)
        if self.world>1:
            import torch.distributed as dist
            dist.broadcast(flag,0)
        return bool(flag.item())

    def _physical_eval(self,checkpoint,full,pack='DEV-A'):
        out=Path(self.cfg.out)/'eval'/f'it_{self.it:07d}_{"full" if full else "basic"}_{pack}'
        env=clean_environment(self.robot_kind)
        # A standalone process on the rank-0 training GPU, while training waits.
        # No new GPUs, and neither robot's evaluator changes the other trainer.
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if visible:env['CUDA_VISIBLE_DEVICES']=visible.split(',')[0]
        cmd=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'evaluate.py'),'--robot',self.robot_kind,
             '--checkpoint',str(checkpoint),'--out',str(out),'--pack',pack]
        if full:cmd.append('--full')
        self.status('physical_evaluation',checkpoint=str(checkpoint),evaluation_output=str(out))
        out.mkdir(parents=True,exist_ok=True)
        torch.cuda.empty_cache()
        before=time.time()
        with (out/'process.log').open('w') as f:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200 if full and self.robot_kind=='B2' else 2400)
        self.eval_seconds+=time.time()-before
        if result.returncode:
            raise RuntimeError(f'Physical evaluator failed with code {result.returncode}: {out}/process.log')
        return json.loads((out/'result.json').read_text())

    def run_eval(self):
        out=Path(self.cfg.out)
        cp=out/('initial.pt' if self.it==0 else f'evaluated_{self.it:07d}.pt')
        if not cp.exists():self.save(str(cp))
        baseline=out/'baseline_eval'/f'it_{self.it:07d}'
        if not (baseline/'result.json').exists():
            baseline.mkdir(parents=True,exist_ok=True)
            env=clean_environment(self.robot_kind)
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if visible:env['CUDA_VISIBLE_DEVICES']=visible.split(',')[0]
            cmd=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'baseline_eval.py'),
                 '--checkpoint',str(cp),'--out',str(baseline)]
            self.status('baseline_evaluation',evaluation_output=str(baseline))
            torch.cuda.empty_cache();before=time.time()
            with (baseline/'process.log').open('w') as f:
                proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
            self.eval_seconds+=time.time()-before
            if proc.returncode:raise RuntimeError(f'Baseline evaluation failed: {baseline}/process.log')
        write_json(out/'latest_baseline_eval.json',json.loads((baseline/'result.json').read_text()))
        full=self.final_evaluation or self.robot_kind=='G1'
        result=self._physical_eval(cp,full)
        if self.it==0:
            write_json(out/'initial_eval.json',result)
        else:
            if self.robot_kind=='B2':
                basic=result['conditions'][:2]
                joint=[sum(t['success'] and v<=6. and p<=1501. for t,v,p in
                           zip(row['trials_detail'],row['trial_peak_speed'],row['trial_peak_power'])) for row in basic]
                score=(min(joint),sum(joint),min(row['passed'] for row in basic),
                       sum(row['passed'] for row in basic),sum(row['completed'] for row in basic),
                       -max(row['peak_mechanical_power']/1501.+row['peak_speed']/6. for row in basic))
                result['basic_joint_passed']=joint
            else:
                score=(min(row['passed'] for row in result['conditions']),result['passed'],result['completed'])
            if self.best_score is None or score>self.best_score:
                self.best_score=score
                shutil.copy2(cp,out/'best.pt')
                result['best_sha256']=sha256(out/'best.pt')
                write_json(out/'best_eval.json',result)
            write_json(out/'latest_eval.json',result)
        with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        self.last_eval=result
        print('PHYSICAL_EVAL',self.robot_kind,self.it,result['passed'],result['total'],
              'all_passed',result['all_passed'],flush=True)
        if result['all_passed']:
            # Independent fixed holdout seeds, unchanged full coverage and criteria.
            a=self._physical_eval(cp,True,'HOLD-A')
            b=self._physical_eval(cp,True,'HOLD-B')
            if a['all_passed'] and b['all_passed']:
                shutil.copy2(cp,out/'accepted.pt')
                write_json(out/'accepted_eval.json',dict(checkpoint_sha256=sha256(out/'accepted.pt'),evaluations=[a,b]))
                write_json(out/'TARGET_MET.json',dict(it=self.it,finished=time.time()))
        self.status('training')

    def run(self):
        self.status('initial_evaluation')
        if self.rank==0:self.run_eval()
        self._barrier()
        super().run()
        # Trainer's final checkpoint is already saved. Final B2 coverage is full.
        if self.rank==0:
            self.final_evaluation=True
            if self.robot_kind=='B2' or self.last_eval is None or self.last_eval['it']!=self.it:
                self.run_eval()
            write_json(Path(self.cfg.out)/'EVAL_DONE.json',dict(it=self.it,finished=time.time()))
            write_json(Path(self.cfg.out)/'timing.json',dict(started=self.started_at,finished=time.time(),
                train_seconds=self.train_time,evaluation_seconds=self.eval_seconds,
                pipeline_seconds=time.time()-self.started_at,it=self.it,actor_updates=self.updates))
            self.status('complete',accepted=(Path(self.cfg.out)/'accepted.pt').exists())
        self._barrier()


if __name__=='__main__':
    os.chdir(ROOT);torch.set_num_threads(2)
    cfg=parse_args()
    try:
        trainer=ScratchTrainer(cfg)
        trainer.run()
    except BaseException as exc:
        if int(os.environ.get('RANK','0'))==0:
            write_json(Path(cfg.out)/'ERROR.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
