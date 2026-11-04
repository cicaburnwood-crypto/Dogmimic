"""Proposed reset distribution; not used by active v5 training."""
import torch

def sample_pose_blend(first_frame, *, generator=None):
    # Keep continuous coverage while explicitly rehearsing both endpoint poses.
    selector=torch.rand(first_frame.shape,device=first_frame.device,generator=generator)
    continuous=torch.rand(first_frame.shape,device=first_frame.device,generator=generator)
    blend=torch.where(selector<.25,torch.ones_like(continuous),torch.where(selector<.5,torch.zeros_like(continuous),continuous))
    return blend*first_frame
