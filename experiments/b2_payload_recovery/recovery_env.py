"""Bounded movement costs anchored at the actual start of each hold window."""
from dataclasses import dataclass
import mujoco
import torch
from supermimic.sim_env import EnvCfg
from payload_env import PayloadCfg,PayloadTask,PayloadEnv
from evaluate_payloads import BASE_XML

XI_DIM=30  # 21 physical values + hind XY (4), return COM (3), valid flags (2)


@dataclass
class RecoveryCfg(PayloadCfg):
    w_hind_anchor:float=2.
    hind_anchor_scale:float=.03
    w_hind_speed:float=1.
    hind_speed_scale:float=.08
    w_loaded_hold_speed:float=.5
    hold_speed_scale:float=.05
    w_return_com_anchor:float=2.
    return_anchor_scale:float=.015
    w_return_com_speed:float=1.5
    return_speed_scale:float=.04
    w_return_feet_speed:float=.5
    return_feet_speed_scale:float=.08
    w_termination:float=20.


class RecoveryTask(PayloadTask):
    @staticmethod
    def bounded_cost(x,scale):
        norm=x.square().sum(-1)/(scale*scale)
        return norm/(1+norm)

    def advance_context(self,s,frame,xi):
        new_h=(frame>=200)&(frame<475)&(xi[...,28]<.5)
        new_c=(frame>=650)&(xi[...,29]<.5)
        feet=self.foot_positions(self.body_kin(s))[...,2:,:2].flatten(-2)
        com,_=self.centroid(s,com_offset=xi[...,13:16],payload_kg=xi[...,16]*15.)
        h=torch.where(new_h[...,None],feet,xi[...,21:25])
        c=torch.where(new_c[...,None],com,xi[...,25:28])
        valid=torch.stack([torch.maximum(xi[...,28],new_h.float()),torch.maximum(xi[...,29],new_c.float())],-1)
        return torch.cat([xi[...,:21],h,c,valid],-1)

    def stability_terms(self,s0,s1,ref,kin0,kin1,refkin,xi):
        terms,metrics=super().stability_terms(s0,s1,ref,kin0,kin1,refkin,xi)
        c=self.cfg;hold=ref['hold_weight']
        hind=hold*(ref['phase']==2);returned=hold*(ref['phase']==4)
        feet=self.foot_positions(kin1);fv=self.foot_velocities(kin1)
        anchor=xi[...,21:25].reshape(*xi.shape[:-1],2,2)
        # These replace all six v3 additions. Each cost has a strict upper bound;
        # feet are compared with their own stance-entry positions, not mocap feet.
        terms.update(
            hind_foot_anchor=-hind*xi[...,28]*c.w_hind_anchor*self.bounded_cost(feet[...,2:,:2]-anchor,c.hind_anchor_scale).mean(-1),
            hind_foot_speed=-hind*c.w_hind_speed*self.bounded_cost(fv[...,2:,:],c.hind_speed_scale).mean(-1),
            loaded_hold_speed=-hold*c.w_loaded_hold_speed*self.bounded_cost(metrics['com_velocity'],c.hold_speed_scale),
            return_com_anchor=-returned*xi[...,29]*c.w_return_com_anchor*self.bounded_cost(metrics['com_pos']-xi[...,25:28],c.return_anchor_scale),
            return_com_speed=-returned*c.w_return_com_speed*self.bounded_cost(metrics['com_velocity'],c.return_speed_scale),
            return_feet_speed=-returned*c.w_return_feet_speed*self.bounded_cost(fv,c.return_feet_speed_scale).mean(-1))
        return terms,metrics

    def transition_analytics(self,*args):
        out=super().transition_analytics(*args)
        penalty=-self.cfg.w_termination*out['soft']
        out['terms']['termination_avoidance']=penalty
        out['r']=out['r']+self.cfg.dt*penalty
        return out


class RecoveryEnv(PayloadEnv):
    def __init__(self,lib,cfg,task_cfg,device,gamma=.99,task_class=RecoveryTask,**kw):
        super().__init__(lib,cfg,task_cfg,device,gamma,**kw)
        old=self.task
        self.task=task_class(self.kin,old.q0,torch.stack([old.qmin,old.qmax],-1),old.action_scale,
                              task_cfg,device,mj_model=mujoco.MjModel.from_xml_path(str(BASE_XML)),
                              compile_sizes=(cfg.num_envs,) if cfg.compile else ())
        self.anchor_context=torch.zeros(self.num_envs,9,device=device)

    @property
    def xi(self):return torch.cat([super().xi,self.anchor_context],-1)

    def _finish_resets(self,ids,force=False):
        super()._finish_resets(ids,force)
        self.anchor_context[ids]=0.
        need=((self.frame>=200)&(self.frame<475)&(self.anchor_context[:,7]<.5))|((self.frame>=650)&(self.anchor_context[:,8]<.5))
        selected=need.nonzero().squeeze(-1)
        if len(selected):
            s={k:v[selected] for k,v in self.state.items()}
            self.anchor_context[selected]=self.task.advance_context(s,self.frame[selected],self.xi[selected])[:,21:]


def make_recovery_env(lib,device,n,gamma=.99,evaluate=False,fixed_loads=None,pushes=True,task_class=RecoveryTask):
    cfg=EnvCfg(num_envs=n,episode_length_s=20.,domain_rand=not evaluate,obs_noise=not evaluate,
               reset_noise=not evaluate,compile=not evaluate,set_const_every=1,
               friction_range=(.85,1.15),enc_bias_range=(-.003,.003),com_range=((-0.005,.005),)*3,
               pose_range=((-0.01,.01),(-.01,.01),(-.003,.003),(-.02,.02),(-.02,.02),(-.02,.02)),
               joint_pos_range=(-.02,.02),vel_range=((-0.05,.05),)*6)
    env=RecoveryEnv(lib,cfg,RecoveryCfg(),device,gamma,fixed_loads=fixed_loads,pushes=pushes,task_class=task_class)
    env.eval_mode=evaluate
    return env
