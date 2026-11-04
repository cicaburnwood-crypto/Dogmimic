"""Independent precise stance corrections on frozen original actor features."""
import torch
from torch import nn
from torch.nn import functional as F
from supermimic.networks import Policy,amp_ctx

class PhasePolicy(nn.Module):
    def __init__(self,obs_dim,goal_dim,enc_hidden=(512,256,128),dec_hidden=(512,256,128),hind_limit=.008,quad_limit=.04):
        super().__init__()
        self.teacher=Policy(obs_dim,goal_dim,enc_hidden=enc_hidden,dec_hidden=dec_hidden)
        self.teacher.requires_grad_(False)
        self.teacher.obs_norm.until=0;self.teacher.goal_norm.until=0
        self.hind_head=nn.Linear(dec_hidden[-1],12);self.quad_head=nn.Linear(dec_hidden[-1],12)
        for head in (self.hind_head,self.quad_head):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.register_buffer('max_joint_delta_rad',torch.tensor(float(quad_limit)))
        self.register_buffer('max_hind_joint_delta_rad',torch.tensor(float(hind_limit)))
    @property
    def encoder(self):return self.hind_head
    @property
    def decoder(self):return self.quad_head
    @property
    def obs_norm(self):return self.teacher.obs_norm
    @property
    def goal_norm(self):return self.teacher.goal_norm
    def load_state_dict(self,state_dict,strict=True,assign=False):
        if 'encoder.0.weight' in state_dict:
            for head in (self.hind_head,self.quad_head):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
            return self.teacher.load_state_dict(state_dict,strict=strict,assign=assign)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)
    @staticmethod
    def gates(goal):
        # Reference joint targets already exist in the original actor input.
        # No hidden simulator mass, force or clock is exposed to the actor.
        front=(goal[...,10]+goal[...,13])*.5
        upright=((.7-front)/1.05).clamp(0,1)
        upright=upright.square()*(3-2*upright)
        still=torch.exp(-goal[...,21:33].square().mean(-1)/.15**2)
        return upright[...,None],still[...,None]
    def with_prior(self,obs,goal):
        z=self.teacher.encode(goal)
        x=torch.cat([z,self.teacher.obs_norm(obs)],-1)
        with amp_ctx():
            h=self.teacher.decoder[:-1](x)
            prior=self.teacher.decoder[-1](h).float()
        # FP32 heads resolve sub-milliradian adjustments instead of subtracting
        # two quantized BF16 actor outputs. The teacher remains exactly unchanged.
        with torch.autocast(device_type=obs.device.type,enabled=False):
            dh=F.linear(h.float(),self.hind_head.weight,self.hind_head.bias)
            dq=F.linear(h.float(),self.quad_head.weight,self.quad_head.bias)
        u,still=self.gates(goal)
        bh=self.max_hind_joint_delta_rad/.7;bq=self.max_joint_delta_rad/.7
        correction=still*(u*bh*torch.tanh(dh/bh)+(1-u)*bq*torch.tanh(dq/bq))
        return prior+correction,prior
    def forward(self,obs,goal):return self.with_prior(obs,goal)[0]

def from_checkpoint(ck,device='cuda'):
    a=ck['policy_architecture'];s=ck['policy']
    p=PhasePolicy(s['teacher.obs_norm.mean'].numel(),s['teacher.goal_norm.mean'].numel(),tuple(a['encoder']),tuple(a['decoder']),a['hind_limit_rad'],a['max_joint_delta_rad']).to(device)
    p.load_state_dict(s);return p
