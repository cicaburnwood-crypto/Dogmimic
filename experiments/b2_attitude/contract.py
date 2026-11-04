"""All angles are radians internally. Body axes: x forward, y left, z up."""
import math
import torch

DT = .02
SIM_DT = .005
HISTORY = 10
OBS = 42
GOAL_DIM = 6  # desired roll,pitch; rate roll,pitch; requested roll,pitch
MAX_ANGLE = math.radians(30)
MAX_RATE = math.radians(30)
MAX_ACCEL = math.radians(90)
COMMAND_OMEGA = 5.
Q0 = [0., .85, -1.70] * 4
ACTION_SCALE = [.70, 1.1, 1.1] * 4
KP, KD = 250., 8.
POLICY_JOINTS = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
POLICY_TO_SDK = [3,4,5,0,1,2,9,10,11,6,7,8]

def angles(rot):
    return torch.stack([torch.atan2(rot[...,2,1], rot[...,2,2]),
                        torch.atan2(-rot[...,2,0], torch.sqrt(rot[...,2,1].square()+rot[...,2,2].square()+1e-12))], -1)

def angle_error(actual, target):
    return torch.atan2(torch.sin(actual-target), torch.cos(actual-target))

def advance_command(context):
    """Same smooth BODY command in simulation, imagination and inference.

    No joint target slew limiter is used. A new target may replace the old one
    at any step; desired attitude and rate remain continuous.
    """
    q, v, target = context[..., :2], context[..., 2:4], context[..., 4:6]
    acceleration = (COMMAND_OMEGA**2*(target-q)-2*COMMAND_OMEGA*v).clamp(-MAX_ACCEL, MAX_ACCEL)
    next_v = (v+DT*acceleration).clamp(-MAX_RATE, MAX_RATE)
    next_q = q + DT*next_v
    return torch.cat([next_q, next_v, target], -1)

def body_rate(context):
    roll=context[...,0]; rd,pd=context[...,2],context[...,3]
    return torch.stack([rd,pd*torch.cos(roll),-pd*torch.sin(roll)],-1)

def target_from_degrees(axis, degrees):
    if axis not in ('roll','pitch'): raise ValueError('axis must be roll or pitch')
    if not math.isfinite(degrees) or abs(degrees)>30.: raise ValueError('angle must be within [-30,30] degrees')
    result=[0.,0.];result[0 if axis=='roll' else 1]=math.radians(degrees)
    return result

