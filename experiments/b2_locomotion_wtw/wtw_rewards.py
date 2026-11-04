"""Vector port of the 19 active CoRLRewards in MIT Walk These Ways.

Upstream commit 0e7236bdc81ce855cbe3d70345a7899452bdeb1c; original code,
attribution and licenses are in vendor/walk_these_ways. Upstream configuration
is merged in upstream_reward_config.json, with per-setting source provenance.
No added locomotion penalties. audit_wtw.py compares every term to the original
class, including gradients. Only B2 morphology and a fixed public gait preset
are substituted. All orientation commands are zero in this locomotion task.
"""
import json
from pathlib import Path
import torch
from supermimic.rotations import mtv
from gait_clock import foot_phases,desired_contacts,FREQUENCY,SWING_HEIGHT
from contract import DT

CONFIG=json.loads(Path(__file__).with_name('upstream_reward_config.json').read_text())
SCALES=CONFIG['reward_scales']

def reward_terms(s,previous_s,cmd,action,prev,lastlast,bias,phase,aux,contacts_prev,
                 fp,fv,q0,action_scale,qmin,qmax,height,width,length):
    """aux: torque12, foot forces12 (world or heading frame), limb collision8.

    Force norms and vertical components are invariant to the heading rotation.
    B2 uses a .032 m foot radius, replacing upstream Go1's .02 m offset.
    """
    p=foot_phases(phase,cmd);desired=desired_contacts(p)
    v=mtv(s['rot'],s['v']);w=mtv(s['rot'],s['w'])
    forces=aux[...,12:24].reshape(*aux.shape[:-1],4,3)
    contact=forces[...,2]>1.
    filt=contact|contacts_prev.bool()
    target=(q0+action_scale*action-bias).clamp(qmin+.02,qmax-.02)
    last_target=(q0+action_scale*prev-bias).clamp(qmin+.02,qmax-.02)
    lastlast_target=(q0+action_scale*lastlast-bias).clamp(qmin+.02,qmax-.02)
    mid=(qmin+qmax)*.5;half=(qmax-qmin)*.5*.9
    # Equivalent to upstream quat_apply_yaw(quat_conjugate(q), relative_foot).
    # This twist yaw avoids nondifferentiable matrix-to-quaternion conversion.
    R=s['rot'];theta=torch.atan2(R[...,1,0]-R[...,0,1],R[...,0,0]+R[...,1,1])
    rel=fp-s['pos'][...,None,:];c=theta.cos()[...,None];sn=theta.sin()[...,None]
    feet_xy=torch.stack([c*rel[...,0]+sn*rel[...,1],-sn*rel[...,0]+c*rel[...,1]],-1)
    ph=(1-p*2).abs()-.5
    signs_x=p.new_tensor([1.,1.,-1.,-1.]);signs_y=p.new_tensor([1.,-1.,1.,-1.])
    dx=length*.5*signs_x+ph*cmd[...,0:1]*(.5/FREQUENCY)
    dy=width*.5*signs_y+ph*(cmd[...,2:3]*length*.5)*(.5/FREQUENCY)*signs_x
    triangle=1-(1-(p*2-1).clamp(0,1)*2).abs()
    terms=dict(
        tracking_lin_vel=torch.exp(-(cmd[...,:2]-v[...,:2]).square().sum(-1)/.25),
        tracking_ang_vel=torch.exp(-(cmd[...,2]-w[...,2]).square()/.25),
        lin_vel_z=v[...,2].square(),
        ang_vel_xy=w[...,:2].square().sum(-1),
        torques=aux[...,:12].square().sum(-1),
        dof_vel=s['qd'].square().sum(-1),
        dof_acc=((previous_s['qd']-s['qd'])/DT).square().sum(-1),
        collision=(aux[...,24:]>.5).to(aux.dtype).sum(-1),
        action_rate=(prev-action).square().sum(-1),
        tracking_contacts_shaped_force=-((1-desired)*(1-torch.exp(-forces.square().sum(-1)/100.))).mean(-1),
        tracking_contacts_shaped_vel=-(desired*(1-torch.exp(-fv.square().sum(-1)/10.))).mean(-1),
        jump=-(s['pos'][...,2]-height).square(),
        dof_pos_limits=(torch.relu(mid-half-s['q'])+torch.relu(s['q']-mid-half)).sum(-1),
        feet_slip=(filt*fv[...,:2].square().sum(-1)).sum(-1),
        feet_clearance_cmd_linear=((SWING_HEIGHT*triangle+.032-fp[...,2]).square()*(1-desired)).sum(-1),
        action_smoothness_1=((target-last_target).square()*(prev!=0)).sum(-1),
        action_smoothness_2=((target-2*last_target+lastlast_target).square()*(prev!=0)*(lastlast!=0)).sum(-1),
        raibert_heuristic=(torch.stack([dx,dy],-1)-feet_xy).square().sum((-1,-2)),
        orientation_control=R[...,2,:2].square().sum(-1),
    )
    return terms,contact

def combine(terms):
    weighted={k:terms[k]*(scale*DT) for k,scale in SCALES.items()}
    # In this pinned configuration every term has a fixed sign. These are the
    # exact positive and negative accumulators of upstream compute_reward.
    positive=weighted['tracking_lin_vel']+weighted['tracking_ang_vel']
    negative=sum(v for k,v in weighted.items() if k not in ('tracking_lin_vel','tracking_ang_vel'))
    return positive*torch.exp(negative/CONFIG['rewards']['sigma_rew_neg']),weighted
