import os,time,json,dataclasses
from pathlib import Path
import torch
from supermimic.train import Trainer,parse_args
from sprint_task import make_env

# Keep every rank on physical cores local to its selected GPU.
physical=int(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[int(os.environ.get('LOCAL_RANK','0'))])
local=physical%4
cores=set(range((0 if physical<4 else 16)+local*4,(0 if physical<4 else 16)+local*4+4))
os.sched_setaffinity(0,cores)
torch.set_num_threads(2)
START=time.time()

class SprintTrainer(Trainer):
    def run(self):
        # Checkpoints do not contain replay; refill it before any resumed update.
        if self.it>0 and self.buf.size<self.cfg.warmup_steps:
            for _ in range(self.cfg.warmup_steps):self.collect()
        super().run()
    def _make_env(self):
        env=make_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma);self.tcfg=env.tcfg
        return env
    def _extra_state(self):
        return dict(task_cfg=dataclasses.asdict(self.tcfg),world=self.world,wall_seconds=time.time()-START,launch_wall_seconds=time.time()-float(os.environ.get('SPRINT_LAUNCH_UNIX',START)),invalid_physics_transitions=int(self.env.invalid_physics_count.item()))
    @torch.no_grad()
    def collect(self,random=False):
        super().collect(random)
        # Exclude invalid simulator transitions from world-model fitting.
        self.buf.ep[(self.buf.ptr-1)%self.buf.L,self.env.last_step_invalid]=-1
    def _should_stop(self):
        stop=super()._should_stop()
        # A shared stop marker allows all ranks to end and save at the same iteration.
        requested=Path(self.cfg.out,'STOP').exists() if self.rank==0 else False
        if self.world>1:
            import torch.distributed as dist
            value=torch.tensor(int(requested),device=self.device);dist.broadcast(value,0);requested=bool(value.item())
        return stop or requested

if __name__=='__main__':
    cfg=parse_args()
    trainer=SprintTrainer(cfg)
    if trainer.rank==0:
        Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,world=trainer.world,gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),cfg=dataclasses.asdict(cfg)),indent=2)+'\n')
    trainer.run()
    if trainer.rank==0:
        Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,train_seconds=trainer.train_time,iterations=trainer.it,world=trainer.world),indent=2)+'\n')
    if trainer.world>1:
        import torch.distributed as dist
        dist.destroy_process_group()
