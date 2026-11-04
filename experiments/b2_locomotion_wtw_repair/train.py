"""SuperMimic dynamics/BPTT training with command rewards, entirely from scratch."""
import os,time,json,dataclasses,argparse,math,hashlib
from pathlib import Path
import torch
from supermimic import networks
from supermimic.train import Trainer,TrainCfg
from supermimic.networks import WorldModel,Policy,Critic,run_mlp,apply_delta_c
from supermimic.task import NJ,flat_to_state,state_to_flat
from supermimic.networks import state_delta_c,state_error
from task import XI_DIM
from contract import HISTORY
from policy import make_policy
from task import CommandEnv,CommandCfg,OBS
from replay import CommandReplay
from physics_head import PhysicsHead
from gait_clock import GOAL_DIM,policy_goal,advance_phase,FREQUENCY,SWING_HEIGHT
from wtw_rewards import CONFIG as PUBLIC_REWARD_CONFIG
START=time.time()

@dataclasses.dataclass
class Config(TrainCfg):
    motions: str = ""
    wm_width: int = 2048
    start_speed: float = .5
    max_speed: float = 2.
    curriculum_steps: int = 20000
    eval_every: int = 0
    wm_batches: int = 4
    critic_epochs: int = 2
    horizon: int = 32
    iterations: int = 80000
    ckpt_every: int = 2000
    seed: int = 20261104
    policy_lr_scale: float = .75
    yaw_std: float = .35
    max_yaw_std: float = 1.5
    command_advance_every: int = 200
    init_from: str = ""
    init_sha256: str = ""
    critic_warmup_updates: int = 400

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
    self.env = CommandEnv(device,cfg.num_envs,max_speed=cfg.start_speed,yaw_std=cfg.yaw_std)
    self.tcfg = self.env.tcfg
    self.task = self.env.task
    self.task.compile_sizes |= {cfg.wm_batch, cfg.pi_batch}
    s = self.env.reset()
    self.feat_dim = self.task.state_features(s).shape[-1]
    self.obs_dim = OBS * self.tcfg.history
    self.goal_dim = GOAL_DIM
    self.wm = WorldModel(self.task, self.feat_dim, XI_DIM, hidden=(cfg.wm_width,)*3).to(device)
    self.policy = make_policy(self.tcfg.history).to(device)
    self.critic_in = self.feat_dim + self.goal_dim + XI_DIM + 1
    self.critic = Critic(self.critic_in).to(device)
    self.physics = PhysicsHead(self.feat_dim+NJ+XI_DIM).to(device)
    self.dynamics_params=list(self.wm.net.parameters())+list(self.physics.net.parameters())
    self.opt_wm = torch.optim.Adam(self.dynamics_params, lr=cfg.lr, fused=True)
    self.opt_pi = torch.optim.Adam(
      list(self.policy.encoder.parameters()) + list(self.policy.decoder.parameters()), lr=cfg.lr, betas=(0.9, 0.95), fused=True
    )
    self.opt_v = torch.optim.Adam(self.critic.net.parameters(), lr=cfg.lr, fused=True)
    for g in self.opt_pi.param_groups:
      g['lr_scale']=cfg.policy_lr_scale;g['lr']*=cfg.policy_lr_scale
    if self.world > 1:
      import torch.distributed as dist

      for mod in (self.wm, self.physics, self.policy, self.critic):
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
      raise ValueError('Use explicit --init-from for a new recovery run, with fresh replay and optimizer state')
    self.origin=dict(source_checkpoint=None,parent_iteration=0,prior_iterations=0,prior_training_seconds=0.)
    if cfg.init_from:
      source=Path(cfg.init_from).resolve();digest=hashlib.sha256(source.read_bytes()).hexdigest()
      if not cfg.init_sha256 or digest!=cfg.init_sha256:raise ValueError('Initialization checkpoint SHA256 mismatch')
      ck=torch.load(source,map_location='cpu',weights_only=False)
      if ck['task_kind'] not in ('b2_proprioceptive_locomotion_wtw_v4','b2_proprioceptive_locomotion_wtw_repair_v5'):
        raise ValueError('Only this chat self-trained WTW lineage is permitted')
      if ck.get('official_policy',False) or ck.get('reference_policy',False) or not ck.get('no_motion_data',False):
        raise ValueError('Unexpected external-policy/motion provenance')
      self.wm.load_state_dict(ck['wm']);self.physics.load_state_dict(ck['physics_head']);self.policy.load_state_dict(ck['policy'])
      same_reward=ck['task_kind']=='b2_proprioceptive_locomotion_wtw_repair_v5' and ck.get('reward_config')==PUBLIC_REWARD_CONFIG
      if same_reward:self.critic.load_state_dict(ck['critic'])
      self.origin=dict(source_checkpoint=str(source),source_sha256=digest,parent_iteration=ck['it'],
                       prior_iterations=ck.get('prior_iterations',0)+ck['it'],
                       prior_training_seconds=ck.get('prior_training_seconds',0.)+ck['train_time'],
                       inherited_critic=same_reward,initialization='our own trained policy, world model and physics head; never an official policy',
                       replay='empty',optimizer_state='fresh')
      print(json.dumps(dict(initialization=self.origin)),flush=True)
      del ck

  def _extra_state(self):
    return dict(task_kind='b2_proprioceptive_locomotion_wtw_repair_v5',robot='B2',angular_command_clip=None,
                ground_truth_velocity_actor_input=False,pretrained_world_model=bool(self.cfg.init_from),reference_policy=False,official_policy=False,
                command_contract=dict(norm_xy_max=2.,abs_y_max=.6,x_component_limit=None,wz_limit=None),
                **self.origin,model_meta=self.env.model_meta,task_cfg=dataclasses.asdict(self.tcfg),
                no_motion_data=True,pretrained_policy=bool(self.cfg.init_from),initial_velocity='zero',trained_from_scratch_in_this_run=not bool(self.cfg.init_from),
                gait_reward='MIT Walk These Ways 19 published formulas, additive branch, documented B2 recovery scales and duty adaptation',
                reward_source_commit=PUBLIC_REWARD_CONFIG['source_commit'],reward_config=PUBLIC_REWARD_CONFIG,
                public_contact_gait=True,gait_frequency_hz=FREQUENCY,swing_height_m=SWING_HEIGHT,
                b2_morphology=dict(height=self.task.height,stance_width=self.task.width,stance_length=self.task.length,foot_radius=.032),
                physics_head=self.physics.state_dict(),physics_head_targets='applied torques12, heading-frame foot force12, limb collision8',
                training_strategy='SuperMimic world-model+BPTT, repair continuation of our self-trained WTW policy; no official policy or motion data',
                physical_gait_acceptance='200Hz solver foot normal forces and geometric clearance; gait.py',
                policy_architecture=dict(encoder=[128,128],decoder=[512,256,128]),
                world=self.world,wall_seconds=time.time()-START,
                launch_wall_seconds=time.time()-float(os.environ.get('COMMAND_LAUNCH_UNIX',START)),
                invalid_physics_transitions=int(self.env.invalid_physics_count.item()))

  def _should_stop(self):
    requested=Path(self.cfg.out,'STOP').exists()
    if self.world>1:
      import torch.distributed as dist
      t=torch.tensor(int(requested),device=self.device);dist.broadcast(t,0);requested=bool(t.item())
    stop=requested or time.time()-float(os.environ.get('COMMAND_LAUNCH_UNIX',START))>=self.cfg.wall_hours*3600
    if self.world>1:
      import torch.distributed as dist
      t=torch.tensor(int(stop),device=self.device);dist.broadcast(t,0);stop=bool(t.item())
    return stop

  def run(self):
    if self.it>0:
      for _ in range(self.cfg.warmup_steps):self.collect()
    super().run()

  @torch.no_grad()
  def collect(self,random=False):
    env=self.env
    if self.it % self.cfg.command_advance_every == 0:
      curriculum=Path(self.cfg.out,'curriculum.json')
      if curriculum.exists():
        rec=json.loads(curriculum.read_text());env.max_speed=float(rec['speed']);env.yaw_std=float(rec['yaw_std'])
    env.prepare_commands()
    obs,goal=env.policy_inputs();goal=goal.clone();cmd=env.commands.clone()
    self.policy.obs_norm.update(obs);self.policy.goal_norm.update(goal)
    a=(self.policy(obs,goal)+self.cfg.action_noise*torch.randn(env.num_envs,NJ,device=self.device)).clamp(-1.,1.)
    s=state_to_flat(env.state);xi,ep=env.xi.clone(),env.ep_id.clone()
    context=torch.cat([env.phase[:,None],env.reward_prev_contacts.float(),env.lastlast_action],-1).clone()
    next_s,info=env.step(a)
    ep[env.last_step_invalid]=-1
    self.buf.add(s,a,state_to_flat(next_s),xi,cmd,context,env.last_aux,ep)
    self.env_steps+=env.num_envs
    if 'done_ep_return' in info:
      self.ep_returns.append(info['done_ep_return'].mean().item())
      self.ep_norm_returns.append(info['done_norm_return'].mean().item())
      self.ep_terms.append(info['done_terminated'].float().mean().item())
      self.ep_lens.append(info['done_ep_len'].float().mean().item())
      self.ep_step_rewards.append((info['done_ep_return']/(info['done_ep_len']*self.tcfg.max_step_reward)).mean().item())

  def update_wm(self):
    stats={'wm_loss':0.,'wm_err1':0.,'wm_errk':0.,'physics_prediction_loss':0.}
    for bi in range(self.cfg.wm_batches):
      s0,A,S1,X,Y,mask=self.buf.sample_context_windows(self.cfg.wm_batch,self.cfg.k)
      if bi==0 and self.updates % self.cfg.norm_update_every==0:
        with torch.no_grad():
          previous=flat_to_state(s0);xs=[];ds=[];ys=[]
          for j in range(self.cfg.k):
            nxt=flat_to_state(S1[:,j]);xs.append(self.wm.inputs(previous,A[:,j],X[:,j])[mask])
            ds.append(state_delta_c(previous,nxt)[mask]);ys.append(Y[:,j,:24][mask]);previous=nxt
          x=torch.cat(xs);self.wm.in_norm.update(x);self.wm.out_norm.update(torch.cat(ds))
          self.physics.in_norm.update(x);self.physics.out_norm.update(torch.cat(ys))
      loss,e1,ek,auxloss=self._wm_loss(s0,A,S1,X,Y,mask.float())
      if not torch.isfinite(loss):raise FloatingPointError('Nonfinite world-model loss')
      self.opt_wm.zero_grad(set_to_none=True);loss.backward()
      self.sync_grads(self.dynamics_params);torch.nn.utils.clip_grad_norm_(self.dynamics_params,1.,error_if_nonfinite=True);self.opt_wm.step()
      for key,v in zip(stats,(loss.detach(),e1,ek,auxloss)):stats[key]+=float(v)/self.cfg.wm_batches
    return stats

  def wm_window_loss(self,s0,A,S1,X,Y,mask):
    s=flat_to_state(s0);truth_prev=s;errors=[];auxloss=0.
    for j in range(self.cfg.k):
      # Dynamics supervision uses recorded solver targets, never a gait loss.
      auxloss=auxloss+self.physics.loss(self.wm.inputs(truth_prev,A[:,j],X[:,j]),Y[:,j],mask)
      s=self.wm.step(s,A[:,j],X[:,j]);truth=flat_to_state(S1[:,j])
      errors.append((state_error(s,truth)/self.wm.out_norm.std).square().mean(-1))
      truth_prev=truth
    denom=mask.sum().clamp_min(1.)
    loss=(torch.stack(errors).sum(0)*mask).sum()/denom+auxloss
    return loss,(errors[0].detach()*mask).sum()/denom,(errors[-1].detach()*mask).sum()/denom,auxloss.detach()/self.cfg.k

  def update_policy(self):
    cfg,task,wm=self.cfg,self.task,self.wm
    B,H=cfg.pi_batch,cfg.horizon
    s0,xi,cmd,hist,prev,context=self.buf.sample_starts(B,self.tcfg.history,task)
    phase=context[:,0];contacts=context[:,1:5].bool();lastlast=context[:,5:]
    bias=xi[:,1:1+NJ]
    for p in self.dynamics_params:p.requires_grad_(False)
    s=flat_to_state(s0)
    hist=hist+task.sample_obs_noise((B,self.tcfg.history))
    feats=task.features_fast(s)
    c=torch.ones(B,device=self.device);J=torch.zeros_like(c);rem=torch.ones_like(c)
    valid=torch.ones(B,H+1,device=self.device)
    rec={k:[] for k in ('x','r','d','m','c')};terms={}
    std,mean=wm.out_norm.std,wm.out_norm.mean
    for t in range(H):
      goal=policy_goal(cmd,phase)
      rec['x'].append(self.critic_input(feats,goal,xi,rem).detach())
      a=self.policy(hist.flatten(1),goal)
      model_input=torch.cat([feats,a,xi],-1)
      delta=run_mlp(wm.net,wm.in_norm(model_input))*std+mean
      aux=self.physics(model_input)
      s1=apply_delta_c(s,delta);phase=advance_phase(phase,cmd)
      an=task.command_fast(s1,s,cmd,a,prev,lastlast,bias,phase,aux,contacts)
      r,d=an['r'],an['soft']
      J=J+cfg.gamma**t*c*r
      rec['r'].append(r.detach());rec['d'].append(d.detach())
      rec['m'].append(rem);rec['c'].append(c.detach())
      for k,v in an['terms'].items():terms[k]=terms.get(k,0.)+v.detach().mean()/H
      c=c*(1-d)
      obs=an['obs']+task.sample_obs_noise((B,))
      hist=torch.cat([hist[:,1:],obs[:,None]],1)
      xi=torch.cat([xi[:,:20],a],-1)
      s,lastlast,prev=s1,prev,a
      contacts=an['contacts'];feats=an['feats']
    xH=self.critic_input(an['feats'],policy_goal(cmd,phase),xi,rem)
    # The changed reward invalidates the old value function. Learn its targets
    # before allowing an uncalibrated terminal value to steer BPTT.
    if cfg.use_critic and (self.origin.get('inherited_critic',False) or self.updates>=cfg.critic_warmup_updates):
      J=J+cfg.gamma**H*c*self.critic(xH)
    loss=-J.mean()/H
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor loss')
    self.opt_pi.zero_grad(set_to_none=True);loss.backward()
    params=[p for g in self.opt_pi.param_groups for p in g['params']]
    self.sync_grads(params)
    gnorm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    torch.nn.utils.clip_grad_value_(params,.1);self.opt_pi.step()
    for p in self.dynamics_params:p.requires_grad_(True)
    self.critic.zero_grad(set_to_none=True)
    st=dict(pi_J=J.mean().item(),pi_loss=loss.item(),pi_gnorm=gnorm.item(),img_survival=c.mean().item(),command_max=self.env.max_speed,command_yaw_std=self.env.yaw_std)
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
  if trainer.rank==0:trainer.save(str(Path(cfg.out,'ckpt_0000000.pt')))
  try:
    trainer.run()
  except Exception as exc:
    if trainer.rank==0:
      Path(cfg.out,'GUARD_STOP.json').write_text(json.dumps(dict(reason='training exception',error=repr(exc),it=trainer.it,time=time.time()),indent=2)+'\n')
    raise
  if trainer.rank==0:
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,train_seconds=trainer.train_time,iterations=trainer.it,world=trainer.world,launch_wall_seconds=time.time()-float(os.environ.get('COMMAND_LAUNCH_UNIX',START))),indent=2))
  if trainer.world>1:
    import torch.distributed as dist
    dist.destroy_process_group()

if __name__=='__main__':main()
