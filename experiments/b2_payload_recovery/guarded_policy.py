"""Keep the known-good feedback controller and learn a bounded correction."""
import torch
from torch import nn
from supermimic.networks import Policy


class GuardedPolicy(nn.Module):
    def __init__(self,obs_dim,goal_dim,enc_hidden=(512,256,128),dec_hidden=(512,256,128),max_joint_delta_rad=.04):
        super().__init__()
        self.student=Policy(obs_dim,goal_dim,enc_hidden=enc_hidden,dec_hidden=dec_hidden)
        self.teacher=Policy(obs_dim,goal_dim,enc_hidden=enc_hidden,dec_hidden=dec_hidden)
        self.teacher.requires_grad_(False)
        self.register_buffer('max_joint_delta_rad',torch.tensor(float(max_joint_delta_rad)))
        for p in (self.student,self.teacher):
            p.obs_norm.until=0;p.goal_norm.until=0

    @property
    def encoder(self):return self.student.encoder
    @property
    def decoder(self):return self.student.decoder
    @property
    def obs_norm(self):return self.student.obs_norm
    @property
    def goal_norm(self):return self.student.goal_norm

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if 'encoder.0.weight' in state_dict:
            self.student.load_state_dict(state_dict,strict=strict,assign=assign)
            return self.teacher.load_state_dict(state_dict,strict=strict,assign=assign)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def with_prior(self,obs,goal):
        prior=self.teacher(obs,goal)
        delta=self.student(obs,goal)-prior
        bound=self.max_joint_delta_rad/.7
        return prior+bound*torch.tanh(delta/bound),prior

    def forward(self,obs,goal):return self.with_prior(obs,goal)[0]


def from_checkpoint(checkpoint,device='cuda'):
    a=checkpoint['policy_architecture']
    if a.get('type')=='stage_guarded_residual':
        from stage_policy import from_checkpoint as load_stage
        return load_stage(checkpoint,device)
    if a.get('type')=='phase_guarded_residual':
        from phase_policy import from_checkpoint as load_phase
        return load_phase(checkpoint,device)
    state=checkpoint['policy']
    obs=state['student.obs_norm.mean'].numel();goal=state['student.goal_norm.mean'].numel()
    policy=GuardedPolicy(obs,goal,tuple(a['encoder']),tuple(a['decoder']),a['max_joint_delta_rad']).to(device)
    policy.load_state_dict(state)
    return policy
