"""Walk These Ways diagonal-trot preset; internally generated, not motion data.

Phase convention and Gaussian contact targets follow pinned upstream
legged_robot.py:_step_contact_targets (duration=.65). Stop freezes all feet in
the stance phase to implement the user's stand/move/stand command interface.
"""
import math
import torch
from contract import DT

FREQUENCY = 2.0
SWING_HEIGHT = .06
DURATION = .65
GOAL_DIM = 7

def moving(cmd):
    return cmd.square().sum(-1) >= .0025

def advance_phase(phase, cmd):
    return torch.where(moving(cmd), torch.remainder(phase + DT * FREQUENCY, 1.), .25)

def foot_phases(phase, cmd):
    offsets = phase.new_tensor([.5, 0., 0., .5])
    return torch.where(moving(cmd)[..., None], torch.remainder(phase[..., None]+offsets, 1.), DURATION*.5)

def desired_contacts(p):
    p=warped_phases(p)
    def cdf(x):
        return .5 * (1. + torch.erf(x / (.07 * math.sqrt(2.))))
    return cdf(p) * (1-cdf(p-.5)) + cdf(p-1) * (1-cdf(p-1.5))

def policy_goal(cmd, phase):
    p=warped_phases(foot_phases(phase,cmd))
    p=torch.where(moving(cmd)[...,None],p,.25)
    return torch.cat([cmd, torch.sin(2*math.pi*p)], -1)

def warped_phases(p):
    # Exact public duty-duration time warp; diagonal pairs overlap in stance.
    return torch.where(p<DURATION,p*(.5/DURATION),.5+(p-DURATION)*(.5/(1-DURATION)))
