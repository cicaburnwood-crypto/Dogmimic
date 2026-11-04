"""SuperMimic dynamics/BPTT training with command rewards, entirely from scratch."""
import os,time,json,dataclasses,argparse,math
from pathlib import Path
import torch
from supermimic import networks
from supermimic.train import Trainer,TrainCfg
from supermimic.networks import WorldModel,Policy,Critic,run_mlp,apply_delta_c
from supermimic.task import XI_DIM,NJ,flat_to_state,state_to_flat
from task import CommandEnv,CommandCfg,OBS
from replay import CommandReplay
START=time.time()

def make_policy(history=5):
    return Policy(OBS*history,3,enc_hidden=(128,128),dec_hidden=(512,256,128))

@dataclasses.dataclass
class Config(TrainCfg):
    motions: str = ""
    wm_width: int = 2048
    start_speed: float = 1.
    max_speed: float = 3.
    curriculum_steps: int = 20000
    eval_every: int = 0
    wm_batches: int = 4
    critic_epochs: int = 2
    horizon: int = 32
    iterations: int = 30000
    ckpt_every: int = 2000
    seed: int = 20261031

class CommandTrainer(Trainer):
  def __init__(self, cfg: TrainCfg, device="cuda"):
    self.cfg = cfg
    self.world = int(os.environ.get("WORLD_SIZE", "1"))
    self.rank = int(os.environ.get("RANK", "0"))
    if self.world > 1:
      import datetime

      import torch.distributed as dist

      local = int(os.environ.get("LOCAL_RANK", "0"))
      torch.cuda.set_device(local)
      device = f"cuda:{local}"
      dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
      networks.DIST["world"] = self.world
      if cfg.lr_sqrt_world:
        # Paper: 2 GPUs use sqrt(2) x 2e-4.
        cfg.lr = cfg.lr * math.sqrt(self.world)
    self.device = device
    torch.manual_seed(cfg.seed)
    if cfg.tf32:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
    networks.AMP["enabled"] = cfg.amp
    if cfg.motions or cfg.clip_prefixes:
      raise ValueError("This task never accepts motion data")
    self.lib = None
    self.env = CommandEnv(device,cfg.num_envs,max_speed=cfg.start_speed)
    self.tcfg = self.env.tcfg
    self.task = self.env.task
    self.task.compile_sizes |= {cfg.wm_batch, cfg.pi_batch}
    s = self.env.reset()
    self.feat_dim = self.task.state_features(s).shape[-1]
    self.obs_dim = OBS * self.tcfg.history
    self.goal_dim = 3
    self.wm = WorldModel(self.task, self.feat_dim, XI_DIM, hidden=(cfg.wm_width,)*3).to(device)
    self.policy = make_policy(self.tcfg.history).to(device)
    self.critic_in = self.feat_dim + self.goal_dim + XI_DIM + 1
    self.critic = Critic(self.critic_in).to(device)
    self.opt_wm = torch.optim.Adam(self.wm.net.parameters(), lr=cfg.lr, fused=True)
    self.opt_pi = torch.optim.Adam(
      list(self.policy.encoder.parameters()) + list(self.policy.decoder.parameters()), lr=cfg.lr, betas=(0.9, 0.95), fused=True
    )
    self.opt_v = torch.optim.Adam(self.critic.net.parameters(), lr=cfg.lr, fused=True)
    if self.world > 1:
      import torch.distributed as dist

      for mod in (self.wm, self.policy, self.critic):
        for t in list(mod.parameters()) + list(mod.buffers()):
          dist.broadcast(t.data, 0)
      torch.manual_seed(cfg.seed + 1000 * self.rank)
    self.buf = CommandReplay(cfg.num_envs,cfg.buffer_steps,XI_DIM,device)
    self._wm_loss = torch.compile(self.wm_window_loss, dynamic=False) if cfg.compile_wm else self.wm_window_loss
    self.it = 0
    self.updates = 0
    self.env_steps = 0
    self.train_time = 0.0
    self.update_credit = 0.0
    self.ep_returns = []
    self.ep_norm_returns = []
    self.ep_terms = []
    self.ep_lens = []
    self.ep_step_rewards = []
    self.profile = bool(int(os.environ.get("SMIMIC_PROFILE", "0")))
    self.prof = {}
    os.makedirs(cfg.out, exist_ok=True)
    self.log_f = open(os.path.join(cfg.out, "log.jsonl" if self.rank == 0 else f"log_rank{self.rank}.jsonl"), "a")
    if cfg.resume:
      ck = torch.load(cfg.resume,map_location="cpu",weights_only=False)
      if ck.get("task_kind") != "reference_free_command_v1":
        raise ValueError("Only this experiment own checkpoints can be continued")
      self.load(cfg.resume)

  def _extra_state(self):
    return dict(task_kind='reference_free_command_v1',task_cfg=dataclasses.asdict(self.tcfg),
                no_motion_data=True,pretrained_policy=False,initial_velocity='zero',
                policy_architecture=dict(encoder=[128,128],decoder=[512,256,128]),
                world=self.world,wall_seconds=time.time()-START,
                launch_wall_seconds=time.time()-float(os.environ.get('COMMAND_LAUNCH_UNIX',START)),
                invalid_physics_transitions=int(self.env.invalid_physics_count.item()))

  def _should_stop(self):
    requested=Path(self.cfg.out,'STOP').exists()
    if self.world>1:
      import torch.distributed as dist
      t=torch.tensor(int(requested),device=self.device);dist.broadcast(t,0);requested=bool(t.item())
    return requested or super()._should_stop()

  def run(self):
    if self.it>0:
      for _ in range(self.cfg.warmup_steps):self.collect()
    super().run()

  @torch.no_grad()
  def collect(self,random=False):
    env=self.env
    f=min(1.,max(0.,(self.it-2000)/self.cfg.curriculum_steps))
    env.max_speed=self.cfg.start_speed+(self.cfg.max_speed-self.cfg.start_speed)*f
    obs,cmd=env.policy_inputs();cmd=cmd.clone()
    self.policy.obs_norm.update(obs);self.policy.goal_norm.update(cmd)
    a=self.policy(obs,cmd)+self.cfg.action_noise*torch.randn(env.num_envs,NJ,device=self.device)
    s=state_to_flat(env.state);xi,ep=env.xi.clone(),env.ep_id.clone()
    next_s,info=env.step(a)
    ep[env.last_step_invalid]=-1
    self.buf.add(s,a,state_to_flat(next_s),xi,cmd,ep)
    self.env_steps+=env.num_envs
    if 'done_ep_return' in info:
      self.ep_returns.append(info['done_ep_return'].mean().item())
      self.ep_norm_returns.append(info['done_norm_return'].mean().item())
      self.ep_terms.append(info['done_terminated'].float().mean().item())
      self.ep_lens.append(info['done_ep_len'].float().mean().item())
      self.ep_step_rewards.append((info['done_ep_return']/(info['done_ep_len']*self.tcfg.max_step_reward)).mean().item())

  def update_policy(self):
    cfg,task,wm=self.cfg,self.task,self.wm
    B,H=cfg.pi_batch,cfg.horizon
    s0,xi,cmd,hist,prev=self.buf.sample_starts(B,self.tcfg.history,task)
    bias=xi[:,1:1+NJ]
    for p in wm.net.parameters():p.requires_grad_(False)
    s=flat_to_state(s0)
    hist=hist+task.sample_obs_noise((B,self.tcfg.history))
    an=task.command_fast(s,cmd,prev,prev,bias)
    c=torch.ones(B,device=self.device);J=torch.zeros_like(c);rem=torch.ones_like(c)
    valid=torch.ones(B,H+1,device=self.device)
    rec={k:[] for k in ('x','r','d','m','c')};terms={}
    std,mean=wm.out_norm.std,wm.out_norm.mean
    for t in range(H):
      feats=an['feats']
      rec['x'].append(self.critic_input(feats,cmd,xi,rem).detach())
      a=self.policy(hist.flatten(1),cmd)
      delta=run_mlp(wm.net,wm.in_norm(torch.cat([feats,a,xi],-1)))*std+mean
      s1=apply_delta_c(s,delta)
      an=task.command_fast(s1,cmd,a,prev,bias)
      r,d=an['r'],an['soft']
      J=J+cfg.gamma**t*c*r
      rec['r'].append(r.detach());rec['d'].append(d.detach())
      rec['m'].append(rem);rec['c'].append(c.detach())
      for k,v in an['terms'].items():terms[k]=terms.get(k,0.)+v.detach().mean()/H
      c=c*(1-d)
      obs=an['obs']+task.sample_obs_noise((B,))
      hist=torch.cat([hist[:,1:],obs[:,None]],1)
      s,prev=s1,a
    xH=self.critic_input(an['feats'],cmd,xi,rem)
    if cfg.use_critic:J=J+cfg.gamma**H*c*self.critic(xH)
    loss=-J.mean()/H
    self.opt_pi.zero_grad(set_to_none=True);loss.backward()
    params=[p for g in self.opt_pi.param_groups for p in g['params']]
    self.sync_grads(params)
    gnorm=torch.nn.utils.clip_grad_norm_(params,1.)
    torch.nn.utils.clip_grad_value_(params,.1);self.opt_pi.step()
    for p in wm.net.parameters():p.requires_grad_(True)
    self.critic.zero_grad(set_to_none=True)
    st=dict(pi_J=J.mean().item(),pi_loss=loss.item(),pi_gnorm=gnorm.item(),img_survival=c.mean().item(),command_max=self.env.max_speed)
    st.update({f'img_r_{k}':v.item() for k,v in terms.items()})
    if cfg.use_critic:st.update(self.update_critic(rec,xH.detach(),valid))
    return st

def main():
  physical=int(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[int(os.environ.get('LOCAL_RANK','0'))])
  first=(0 if physical<4 else 16)+(physical%4)*4
  os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
  ap=argparse.ArgumentParser()
  for f in dataclasses.fields(Config):
    typ=f.type if not isinstance(f.type,str) else eval(f.type)
    ap.add_argument('--'+f.name.replace('_','-'),type=(lambda x:x.lower() in ('1','true','yes')) if typ is bool else typ,default=f.default)
  cfg=Config(**vars(ap.parse_args()));trainer=CommandTrainer(cfg)
  if trainer.rank==0:
    Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),cfg=dataclasses.asdict(cfg)),indent=2))
  trainer.run()
  if trainer.rank==0:
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,train_seconds=trainer.train_time,iterations=trainer.it,world=trainer.world),indent=2))
  if trainer.world>1:
    import torch.distributed as dist
    dist.destroy_process_group()

if __name__=='__main__':main()
