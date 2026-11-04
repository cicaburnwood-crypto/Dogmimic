"""Pair root orientation with measured joint-pose blend, retaining reset noise."""
import torch
from supermimic.rotations import quat_mul


def paired_orientation(current, reference, measured, blend):
    def yaw(q):
        return torch.atan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2].square()+q[...,3].square()))
    measured = measured / measured.norm(dim=-1, keepdim=True)
    angle = yaw(reference)-yaw(measured)
    zero = torch.zeros_like(angle)
    alignment = torch.stack([torch.cos(angle/2),zero,zero,torch.sin(angle/2)],-1)
    target = quat_mul(alignment, measured.expand_as(reference))
    inverse = torch.cat([reference[...,:1],-reference[...,1:]],-1)
    delta = quat_mul(target,inverse)
    delta = torch.where(delta[...,:1]<0,-delta,delta)
    identity = torch.zeros_like(delta);identity[...,0]=1
    mixed = (1-blend[...,None])*identity+blend[...,None]*delta
    mixed = mixed / mixed.norm(dim=-1,keepdim=True)
    return quat_mul(mixed,current)
