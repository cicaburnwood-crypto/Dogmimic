import torch
from supermimic.networks import Policy
from contract import OBS,HISTORY

class BoundedPolicy(Policy):
    def forward(self,obs,goal):
        return torch.tanh(super().forward(obs,goal))

def make_policy(history=HISTORY):
    p=BoundedPolicy(OBS*history,3,enc_hidden=(128,128),dec_hidden=(512,256,128))
    # Do not turn the normalizer into an implicit angular command clip.
    p.goal_norm.clip=None
    return p
