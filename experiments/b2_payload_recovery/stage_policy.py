"""Phase-commanded corrections preserve the original initial four-foot stance."""
import torch
from torch.nn import functional as F
from supermimic.networks import amp_ctx
from phase_policy import PhasePolicy


def append_phase(goal, future):
    # The phase is part of the requested reference sequence, not simulator state.
    return torch.cat([goal, future['phase'][..., :1].float()], -1)


def install_phase_command(task):
    original = task.goal
    task.goal = lambda state, future: append_phase(original(state, future), future)


class StagePolicy(PhasePolicy):
    reference_phase_command = True

    def with_prior(self, obs, goal):
        dim = self.teacher.goal_norm.mean.numel()
        assert goal.shape[-1] == dim + 1, 'Reference phase command is required'
        original_goal, phase = goal[..., :dim], goal[..., dim:]
        z = self.teacher.encode(original_goal)
        x = torch.cat([z, self.teacher.obs_norm(obs)], -1)
        with amp_ctx():
            h = self.teacher.decoder[:-1](x)
            prior = self.teacher.decoder[-1](h).float()
        with torch.autocast(device_type=obs.device.type, enabled=False):
            dh = F.linear(h.float(), self.hind_head.weight, self.hind_head.bias)
            dq = F.linear(h.float(), self.quad_head.weight, self.quad_head.bias)
        upright, still = self.gates(original_goal)
        bh, bq = self.max_hind_joint_delta_rad / .7, self.max_joint_delta_rad / .7
        # Initial stance/rise cannot receive the return-controller correction.
        return_gate = (phase >= 3).to(goal.dtype)
        correction = still * (upright * bh * torch.tanh(dh / bh)
                              + return_gate * (1 - upright) * bq * torch.tanh(dq / bq))
        return prior + correction, prior


def from_checkpoint(ck, device='cuda'):
    a, s = ck['policy_architecture'], ck['policy']
    p = StagePolicy(s['teacher.obs_norm.mean'].numel(), s['teacher.goal_norm.mean'].numel(),
                    tuple(a['encoder']), tuple(a['decoder']), a['hind_limit_rad'],
                    a['max_joint_delta_rad']).to(device)
    p.load_state_dict(s)
    return p
