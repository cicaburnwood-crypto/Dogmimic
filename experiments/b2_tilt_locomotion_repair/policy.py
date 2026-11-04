import torch
from bootstrap import HERE
from contract import OBS,HISTORY,GOAL_DIM
from supermimic.networks import Policy

class BoundedPolicy(Policy):
    def forward(self,obs,goal):return torch.tanh(super().forward(obs,goal))

def make_policy():
    # Retain vibe's encoder/decoder and latent, use its default network widths.
    policy=BoundedPolicy(OBS*HISTORY,GOAL_DIM)
    policy.goal_norm.clip=None
    return policy

