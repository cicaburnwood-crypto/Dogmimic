from pathlib import Path
import json,torch
from pose_sampling import sample_pose_blend
from sampling import sample_domain
from jump_task import JumpEnv,JumpTask,JumpTaskCfg
from supermimic.sim_env import EnvCfg
from supermimic.rotations import quat_from_euler_xyz,quat_mul
BASE=Path(__file__).parent
class RobustEnv(JumpEnv):
 def __init__(self,lib,device,n,gamma=.99,evaluate=False):
  cfg=EnvCfg(num_envs=n,episode_length_s=10,friction_range=(.15,1.5),joint_pos_range=(-.06,.06),pose_range=((-0.02,.02),(-.02,.02),(-.003,.003),(-.06,.06),(-.06,.06),(-.08,.08)),vel_range=((-0.05,.05),(-.05,.05),(-.03,.03),(-.1,.1),(-.1,.1),(-.1,.1)),enc_bias_range=(-.02,.02),com_range=((-0.03,.03),(-.025,.025),(-.03,.03)),compile=not evaluate,obs_noise=not evaluate,reset_noise=not evaluate,set_const_every=1)
  super().__init__(lib,cfg,JumpTaskCfg(),device,gamma)
  old=self.task;self.task=JumpTask(self.kin,old.q0,torch.stack([old.qmin,old.qmax],-1),old.action_scale,self.tcfg,device,compile_sizes=() if evaluate else (n,))
  self.sim.expand_model_fields(('body_mass','body_inertia','actuator_gainprm','actuator_biasprm','actuator_forcerange'))
  self.defaults={key:torch.tensor(getattr(self.mj_model,key),device=device,dtype=torch.float32) for key in ['body_mass','body_inertia','actuator_gainprm','actuator_biasprm','actuator_forcerange']}
  self.extra=torch.zeros(n,9,device=device);self.extra[:,:4]=1
  self.queue=torch.zeros(n,9,29,device=device);self.queue_index=0
  self.eval_mode=evaluate;self.case={};self.progress=0.
  self.measured=torch.tensor(json.loads((BASE/'measured_start.json').read_text())['q'],device=device)
  self.left=torch.tensor([i for i in self.foot_geoms.tolist() if 'left' in self.mj_model.geom(i).name],device=device)
  self.right=torch.tensor([i for i in self.foot_geoms.tolist() if 'right' in self.mj_model.geom(i).name],device=device)
  self.reference_xy_offset=torch.zeros(n,2,device=device)
  self.align_feet=self.kin.body_index(('left_ankle_roll_link','right_ankle_roll_link'))
  geom_ids=[[i for i in range(self.mj_model.ngeom) if (self.mj_model.geom(i).name or '').split('/')[-1].startswith(side+'_foot') and 'collision' in (self.mj_model.geom(i).name or '')] for side in ('left','right')]
  assert [len(g) for g in geom_ids]==[7,7]
  self.align_local=torch.tensor(self.mj_model.geom_pos[geom_ids].mean(1),device=device,dtype=torch.float32)
 @property
 def xi(self):
  order=(torch.arange(9,device=self.device)+self.queue_index)%9
  chronological=self.queue[:,order].flatten(1)
  return torch.cat([super().xi,self.extra,chronological,self.frame[:,None].float()/200.],-1)
 def _randomize(self,ids):
  super()._randomize(ids);k=len(ids);dev=self.device
  x=torch.zeros(k,9,device=dev);x[:,:4]=1
  if self.eval_mode:
   c=self.case;self.friction[ids]=c.get('friction',.8);x[:,0]=c.get('mass',1);x[:,1]=c.get('kp',1);x[:,2]=c.get('kd',1);x[:,3]=c.get('effort',1);x[:,4]=c.get('delay_ms',0)/40;x[:,5]=c.get('asym',0);x[:,6:]=torch.tensor(c.get('force',[0,0,0]),device=dev)
  else:
   x, friction, self.domain_groups = sample_domain(k, dev, self.friction[ids])
   self.friction[ids] = friction
  self.extra[ids]=x
  gf=self.sim.model.geom_friction;gf[ids,:,0]=(self.friction[ids]*(1-x[:,5].abs()))[:,None]
  gf[ids[:,None],self.left[None],0]=(self.friction[ids]*(1+x[:,5])).clamp(.1,1.8)[:,None]
  gf[ids[:,None],self.right[None],0]=(self.friction[ids]*(1-x[:,5])).clamp(.1,1.8)[:,None]
  # Floor <= both feet: MuJoCo max mixing preserves each assigned foot coefficient.
  self.sim.model.body_mass[ids]=self.defaults['body_mass'][None]*x[:,0,None]
  self.sim.model.body_inertia[ids]=self.defaults['body_inertia'][None]*x[:,0,None,None]
  gain=self.defaults['actuator_gainprm'][None].repeat(k,1,1);gain[:,:,0]*=x[:,1,None];self.sim.model.actuator_gainprm[ids]=gain
  bias=self.defaults['actuator_biasprm'][None].repeat(k,1,1);bias[:,:,1]*=x[:,1,None];bias[:,:,2]*=x[:,2,None];self.sim.model.actuator_biasprm[ids]=bias
  self.sim.model.actuator_forcerange[ids]=self.defaults['actuator_forcerange'][None]*x[:,3,None,None]
  self._need_set_const=True
 def reset_envs(self,ids,frames=None):
  super().reset_envs(ids,frames)
  if not len(ids):return
  if not self.eval_mode:
   first=self.frame[ids]==0;blend=sample_pose_blend(first)
   q=self.sim.data.qpos[ids[:,None],self.q_adr[None]];ref=self.lib.frame(torch.zeros(len(ids),dtype=torch.long,device=self.device))['q']
   q+=(self.measured-ref)*blend[:,None]
   self.sim.data.qpos[ids[:,None],self.q_adr[None]]=q
   self.sim.data.qpos[ids,self.root_qadr+2]+=.027*blend
   self.sim.data.qvel[ids[first]]=0
  self.queue[ids]=self.sim.data.qpos[ids[:,None],self.q_adr[None]][:,None,:]
 def _sample_start_frames(self,k):
  frames,bins=super()._sample_start_frames(k);mask=torch.rand(k,device=self.device)<.7;frames[mask]=0;bins[mask]=0;return frames,bins
 def prepare_eval(self,pos,quat,q):
  c=self.case
  if c.get('measured'):q+=self.measured-self.lib.frame(torch.zeros(len(q),dtype=torch.long,device=self.device))['q'];pos[:,2]+=.027
  if c.get('tilt'):
   a=torch.full((len(q),),c['tilt'],device=self.device);quat=quat_mul(quat_from_euler_xyz(a*0,a,a*0),quat)
  self.queue[:]=q[:,None,:]
  return pos,quat,q
 def _foot_center(self,s):
  P,R=self.kin.forward(s['pos'],s['rot'],s['q'])
  return (P[:,self.align_feet]+torch.einsum('bsij,sj->bsi',R[:,self.align_feet],self.align_local)).mean(1)[:,:2]
 def read_state(self):
  s=super().read_state()
  if hasattr(self,'reference_xy_offset'):
   s['pos'][:,:2]=s['pos'][:,:2]-self.reference_xy_offset
  return s
 def _finish_resets(self,ids,force=False):
  # Set one constant coordinate translation per episode before goals/history/replay.
  # Physics qpos is never moved. Constant XY translation leaves evaluation distances,
  # velocities, heights, tilt and foot-drift criteria unchanged.
  if len(ids) and hasattr(self,'reference_xy_offset'):
   from supermimic.rotations import quat_to_mat
   physical=super().read_state()
   physical={k:v[ids] for k,v in physical.items()}
   ref=self.lib.frame(self.frame[ids]);nominal=dict(pos=ref['pos'],rot=quat_to_mat(ref['quat']),q=ref['q'])
   self.reference_xy_offset[ids]=self._foot_center(physical)-self._foot_center(nominal)
  return super()._finish_resets(ids,force)
 def physics_step(self,target):
  ids=torch.arange(self.num_envs,device=self.device);lag=(self.extra[:,4]*8).round().long()
  pulse=((self.frame>=48)&(self.frame<68)).float()[:,None]
  self.sim.data.xfrc_applied.zero_();self.sim.data.xfrc_applied[:,self.torso_body,:3]=self.extra[:,6:]*pulse
  for _ in range(self.decimation):
   self.queue[:,self.queue_index]=target
   delayed=self.queue[ids,(self.queue_index-lag)%9]
   self.sim.data.ctrl[:]=delayed[:,self.act_to_joint];self.sim.step();self.queue_index=(self.queue_index+1)%9
