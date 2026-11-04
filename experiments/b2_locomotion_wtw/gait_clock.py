"""Walk These Ways diagonal-trot preset; internally generated, not motion data.

Phase convention and Gaussian contact targets follow pinned upstream
legged_robot.py:_step_contact_targets (duration=.5). Stop freezes all feet in
the stance phase to implement the user's stand/move/stand command interface.
"""
import math
import torch
from contract import DT

FREQUENCY = 2.0
SWING_HEIGHT = .09
GOAL_DIM = 7

def moving(cmd):
    return cmd.square().sum(-1) >= .0025

def advance_phase(phase, cmd):
    return torch.where(moving(cmd), torch.remainder(phase + DT * FREQUENCY, 1.), .25)

def foot_phases(phase, cmd):
    offsets = phase.new_tensor([.5, 0., 0., .5])
    return torch.where(moving(cmd)[..., None], torch.remainder(phase[..., None]+offsets, 1.), .25)

def desired_contacts(p):
    def cdf(x):
        return .5 * (1. + torch.erf(x / (.07 * math.sqrt(2.))))
    return cdf(p) * (1-cdf(p-.5)) + cdf(p-1) * (1-cdf(p-1.5))

def policy_goal(cmd, phase):
    return torch.cat([cmd, torch.sin(2*math.pi*foot_phases(phase,cmd))], -1)
