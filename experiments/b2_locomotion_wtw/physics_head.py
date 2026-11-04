"""Supervised dynamics outputs for contact forces, torques and limb collisions.

These are model targets, never additional task reward terms or actor inputs.
"""
import torch
from torch import nn
from supermimic.networks import mlp,Normalizer,run_mlp

class PhysicsHead(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.net=mlp(in_dim,(512,512),32,act=nn.SiLU,out_gain=.01)
        self.in_norm=Normalizer(in_dim,eps=1e-3)
        self.out_norm=Normalizer(24,eps=1e-3,clip=None)
        with torch.no_grad():self.net[-1].bias[24:]=-6.
    def raw(self,x):
        return run_mlp(self.net,self.in_norm(x))
    def forward(self,x):
        z=self.raw(x)
        return torch.cat([z[...,:24]*self.out_norm.std+self.out_norm.mean,(z[...,24:]>0).float()],-1)
    def loss(self,x,target,mask):
        z=self.raw(x)
        continuous=(z[...,:24]-(target[...,:24]-self.out_norm.mean)/self.out_norm.std).square().mean(-1)
        binary=torch.nn.functional.binary_cross_entropy_with_logits(z[...,24:],target[...,24:],reduction='none').mean(-1)
        return ((continuous+binary)*mask).sum()/mask.sum().clamp_min(1.)
