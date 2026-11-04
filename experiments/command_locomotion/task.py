"""Reference-free G1 command task. No motion files, phases or expert policies."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from mjlab.managers.event_manager import RecomputeLevel
from supermimic.task import Task, TaskCfg, NJ
from supermimic.sim_env import TrackingSimEnv, EnvCfg
from supermimic.rotations import heading_mat, mtv

OBS = 96

@dataclass
class CommandCfg(TaskCfg):
    history: int = 5
    future: int = 0
    @property
    def max_step_reward(self):
        return 8.5 * self.dt

class CommandTask(Task):
    def __init__(self, *a, joint_names, **kw):
        super().__init__(*a, **kw)
        self.obs_noise = torch.cat([self.obs_noise, torch.full((3,), .05, device=self.device)])
        self.upper = torch.tensor([i for i,n in enumerate(joint_names) if any(x in n for x in ('waist','shoulder','elbow','wrist'))], device=self.device)
        self.feet = torch.tensor(self.kin.body_index(('left_ankle_roll_link','right_ankle_roll_link')),device=self.device)

    def obs_clean(self, s, prev_action, enc_bias):
        return torch.cat([super().obs_clean(s,prev_action,enc_bias),mtv(s['rot'],s['v'])],-1)

    def sample_obs_noise(self, shape):
        return (torch.rand(shape+(OBS,),device=self.device)*2-1)*self.obs_noise

    def command_analytics(self, s, cmd, action, prev, bias):
        v = mtv(heading_mat(s['rot']),s['v'])
        stopped = (cmd.square().sum(-1)<.01).float()
        e = (v[...,:2]-cmd[...,:2]).square().sum(-1)
        tilt = 1-s['rot'][...,2,2]
        qerr = (s['q']-self.q0).square()
        kin = self.body_kin(s)
        foot_z = kin[0][...,self.feet,2]
        foot_v = kin[2][...,self.feet,:2]
        contact = torch.exp(-((foot_z-.035)/.04).square())
        terms = dict(
            alive=torch.ones_like(e),
            velocity=4*torch.exp(-e/.5**2)-.5*e,
            yaw=.5*torch.exp(-(s['w'][...,2]-cmd[...,2]).square()/.5**2),
            stop=2*stopped*torch.exp(-v[...,:2].square().sum(-1)/.15**2),
            upright=-4*tilt,
            height=-15*(s['pos'][...,2]-.76).square(),
            angular=-.08*s['w'][...,:2].square().sum(-1),
            vertical=-.15*s['v'][...,2].square(),
            smooth=-.04*(action-prev).square().sum(-1),
            joint_vel=-.001*s['qd'].square().sum(-1),
            upper=-.15*qerr[...,self.upper].sum(-1),
            stand_posture=-.3*stopped*qerr.sum(-1),
            slip=-.15*(contact*foot_v.square().sum(-1)).sum(-1),
            limits=-2*(torch.relu(self.qmin-s['q'])+torch.relu(s['q']-self.qmax)).sum(-1),
        )
        hard=(s['pos'][...,2]<.45)|(tilt>.65)
        soft=1-(1-torch.sigmoid((.45-s['pos'][...,2])/.025))*(1-torch.sigmoid((tilt-.65)/.04))
        return dict(r=sum(terms.values())*self.cfg.dt,hard=hard,soft=soft,terms=terms,
                    obs=self.obs_clean(s,action,bias),feats=self.state_features(s,kin),goal=cmd)

    def command_fast(self,*args):
        return self._maybe_compiled('command',self.command_analytics,*args)

class CommandEnv(TrackingSimEnv):
    """Reuse the upstream physics setup; override every motion-dependent path.

    The base constructor only uses num_bins for legacy logging allocations.
    There is no MotionLibrary instance and no reference data is read.
    """
    def __init__(self, device, n, evaluate=False, max_speed=1.):
        tc=CommandCfg()
        ec=EnvCfg(num_envs=n,episode_length_s=24.,domain_rand=not evaluate,
                  obs_noise=not evaluate,reset_noise=not evaluate,compile=not evaluate,
                  friction_range=(.8,1.2),enc_bias_range=(-.003,.003),
                  com_range=((-0.005,.005),(-.005,.005),(-.005,.005)))
        super().__init__(SimpleNamespace(num_bins=1),ec,tc,device)
        self.lib=None
        old=self.task
        self.task=CommandTask(self.kin,old.q0,torch.stack([old.qmin,old.qmax],-1),old.action_scale,tc,device,
                              compile_sizes=() if evaluate else (n,),joint_names=self.joint_names)
        self.obs_hist=torch.zeros(n,tc.history,OBS,device=device)
        self.commands=torch.zeros(n,3,device=device)
        self.command_left=torch.zeros(n,dtype=torch.long,device=device)
        self.max_speed=max_speed
        self.eval_mode=evaluate
        self.invalid_physics_count=torch.zeros((),device=device,dtype=torch.long)
        self.last_step_invalid=torch.zeros(n,device=device,dtype=torch.bool)

    def reset_envs(self, ids, frames=None):
        k=len(ids)
        if not k:return
        self.sim.reset(ids);self._randomize(ids)
        pos=torch.zeros(k,3,device=self.device);pos[:,2]=.76
        quat=torch.zeros(k,4,device=self.device);quat[:,0]=1
        q=self.q0.expand(k,-1).clone()
        if self.cfg.reset_noise and not self.eval_mode:
            q+=(torch.rand_like(q)*2-1)*.025
        z=torch.zeros(k,3,device=self.device)
        self.write_state(ids,pos,quat,q,z,z,torch.zeros_like(q))
        self.frame[ids]=0;self.ep_len[ids]=0
        self.ep_id[ids]=torch.arange(self._next_ep,self._next_ep+k,device=self.device);self._next_ep+=k
        self.prev_action[ids]=0;self.ep_ret[ids]=0
        self.commands[ids]=0
        self.command_left[ids]=torch.randint(20,51,(k,),device=self.device)

    def _finish_resets(self, ids, force=False):
        self._steps_since_const+=1
        if self.cfg.domain_rand and self._need_set_const and (force or self._steps_since_const>=self.cfg.set_const_every):
            self.sim.recompute_constants(RecomputeLevel.set_const)
            self._need_set_const=False;self._steps_since_const=0
        self.state=self.read_state()
        if len(ids):
            s={k:v[ids] for k,v in self.state.items()}
            obs=self.task.obs_clean(s,self.prev_action[ids],self.enc_bias[ids])+self._noise(len(ids))
            self.obs_hist[ids]=obs[:,None].expand(-1,self.tcfg.history,-1)
        self.goal_buf=self.commands

    def _noise(self,k):
        return self.task.sample_obs_noise((k,)) if self.cfg.obs_noise and not self.eval_mode else torch.zeros(k,OBS,device=self.device)

    def set_command(self,cmd):
        self.commands[:]=torch.as_tensor(cmd,device=self.device)
        self.goal_buf=self.commands

    def step(self, action):
        self.sim.data.ctrl[:]=(self.q0+self.action_scale*action-self.enc_bias)[:,self.act_to_joint]
        for _ in range(self.decimation):self.sim.step()
        s=self.read_state()
        bad=~torch.stack([torch.isfinite(x).flatten(1).all(1) for x in s.values()]).all(0)
        self.invalid_physics_count+=bad.sum();self.last_step_invalid=bad
        for k,v in s.items():
            s[k]=torch.where(bad.reshape((-1,)+(1,)*(v.ndim-1)),self.state[k],v)
        an=self.task.command_fast(s,self.commands,action,self.prev_action,self.enc_bias)
        r=torch.where(bad,0.,an['r']);hard=an['hard']|bad
        self.ep_len+=1;self.frame+=1;self.ep_ret+=r
        truncated=(self.ep_len>=self.max_ep_len)&~hard;done=hard|truncated
        info=dict(reward=r,terminated=hard,truncated=truncated,done=done,terms=an['terms'])
        self.prev_action=action.clone()
        self.obs_hist=torch.cat([self.obs_hist[:,1:],(an['obs']+self._noise(self.num_envs))[:,None]],1)
        ids=done.nonzero().squeeze(-1)
        if len(ids):
            info.update(done_ep_return=self.ep_ret[ids].clone(),done_ep_len=self.ep_len[ids].clone(),
                        done_terminated=hard[ids].clone(),done_norm_return=self.ep_ret[ids]/(self.max_ep_len*self.tcfg.max_step_reward))
            self.reset_envs(ids)
        if not self.eval_mode:
            self.command_left-=1
            change=(self.command_left<=0).nonzero().squeeze(-1);k=len(change)
            if k:
                self.commands[change]=0
                go=(torch.rand(k,device=self.device)>.3).float()
                self.commands[change,0]=go*(.2+torch.rand(k,device=self.device)*(self.max_speed-.2))
                self.command_left[change]=torch.randint(100,251,(k,),device=self.device)
        self._finish_resets(ids)
        return s,info
