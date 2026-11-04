import dataclasses,json,os,time
from pathlib import Path
import torch
from supermimic import train as base
from train_recovery import RecoveryTrainer,START
from phase_policy import PhasePolicy

base.Policy=lambda obs,goal:PhasePolicy(obs,goal)

class PhaseTrainer(RecoveryTrainer):
    def __init__(self,cfg,device='cuda'):
        super().__init__(cfg,device)
        self.opt_pi=torch.optim.Adam([dict(params=list(self.policy.hind_head.parameters()),lr_multiplier=.1),
                                      dict(params=list(self.policy.quad_head.parameters()),lr_multiplier=1.)],
                                     lr=cfg.lr,betas=(.9,.95),fused=True)
    def _extra_state(self):
        d=super()._extra_state()
        d.update(task_kind='b2_payload_phase_recovery_v5',
                 policy_architecture=dict(type='phase_guarded_residual',encoder=[512,256,128],decoder=[512,256,128],
                                          max_joint_delta_rad=float(self.policy.max_joint_delta_rad),hind_limit_rad=float(self.policy.max_hind_joint_delta_rad)),
                 actor_feature_network_frozen=True,correction_head_precision='float32',
                 separate_stance_heads=True,hind_head_learning_rate_multiplier=.1,
                 motion_gating='reference front thigh pose; corrections attenuated while reference joints move')
        return d

if __name__=='__main__':
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);first=(0 if gpu<4 else 16)+(gpu%4)*4
    os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
    cfg=base.parse_args();assert not cfg.use_critic
    trainer=PhaseTrainer(cfg)
    Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,gpu=gpu,cfg=dataclasses.asdict(cfg),**trainer._extra_state()),indent=2)+'\n')
    trainer.run_eval();trainer.run()
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,
                                                          train_seconds=trainer.train_time,iterations=trainer.it),indent=2)+'\n')
