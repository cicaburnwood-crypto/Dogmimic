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


def make_fresh_policy(seed,device):
    """Independent reproducible initialization; never accepts checkpoint weights."""
    devices=[torch.cuda.current_device()] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return make_policy().to(device)
