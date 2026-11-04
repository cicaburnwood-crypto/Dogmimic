import dataclasses,json,os,time
from pathlib import Path
import torch
from supermimic import train as base
from train_phase import PhaseTrainer
from train_recovery import START
from stage_policy import StagePolicy
from stage_env import StageTask
from recovery_env import make_recovery_env

base.Policy=lambda obs,goal:StagePolicy(obs,goal)


class StageTrainer(PhaseTrainer):
    def _make_env(self):
        env=make_recovery_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma,task_class=StageTask)
        self.tcfg=env.tcfg
        return env

    def _extra_state(self):
        d=super()._extra_state()
        d.update(task_kind='b2_payload_stage_recovery_v6',
                 motion_gating='explicit reference phase command; return head disabled before descent',
                 reference_phase_command=True,actor_goal_dim=196,teacher_goal_dim=195)
        d['policy_architecture']['type']='stage_guarded_residual'
        return d


if __name__=='__main__':
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);first=(0 if gpu<4 else 16)+(gpu%4)*4
    os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
    cfg=base.parse_args();assert not cfg.use_critic
    trainer=StageTrainer(cfg)
    Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,gpu=gpu,cfg=dataclasses.asdict(cfg),**trainer._extra_state()),indent=2)+'\n')
    trainer.run_eval();trainer.run()
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,
                                                          train_seconds=trainer.train_time,iterations=trainer.it),indent=2)+'\n')
