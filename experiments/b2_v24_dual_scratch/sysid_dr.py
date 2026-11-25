"""Effective arm response domain; never changes physical safety bounds."""
import torch

def sample_arm_kd(n,device,profile,mode='auto',randomized=True):
    center=torch.tensor(profile['kd_center'],device=device,dtype=torch.float32)
    if mode=='identified':return center.expand(n,-1).clone()
    if mode=='legacy' or (mode=='auto' and not randomized):return torch.full((n,6),8.,device=device)
    if mode!='auto':raise ValueError(mode)
    cfg=profile['randomization'];draw=torch.rand(n,1,device=device)
    near=center*(1-cfg['center_relative_halfwidth']+2*cfg['center_relative_halfwidth']*torch.rand(n,6,device=device))
    wide=cfg['broad_lower']+(center*cfg['broad_upper_factor']-cfg['broad_lower'])*torch.rand(n,6,device=device)
    return torch.where(draw<cfg['legacy_fraction'],torch.full_like(near,8.),torch.where(draw<cfg['legacy_fraction']+cfg['center_fraction'],near,wide))
