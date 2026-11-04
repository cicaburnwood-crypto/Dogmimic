"""Shared command/actuator contract; no learned weights or motion references."""
import math

DT = .02
SIM_DT = .005
HISTORY = 10
OBS = 42
Q0 = [0., .67, -1.3] * 4
ACTION_SCALE = [.45, .8, .8] * 4
KP = 250.
KD = 8.
MAX_TRANSLATION = 2.
MAX_LATERAL = .6
LINEAR_ACCELERATION = 1.5
ANGULAR_ACCELERATION = 3.
POLICY_JOINTS = [f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
POLICY_TO_SDK = [3,4,5,0,1,2,9,10,11,6,7,8]

def constrain_command(vx, vy, wz):
    """The feasible set is a radius-2 disk intersected with |vy| <= .6.

    vx has no additional component cap; wz is never clipped. All directions,
    including reverse and simultaneous translation/rotation, are supported.
    """
    if not all(math.isfinite(v) for v in (vx,vy,wz)):
        raise ValueError('Non-finite velocity command')
    vy = min(MAX_LATERAL, max(-MAX_LATERAL, vy))
    scale = min(1., MAX_TRANSLATION / max(math.hypot(vx,vy), 1e-12))
    return [vx*scale, vy*scale, wz]

def slew_command(previous, target, dt=DT):
    target = constrain_command(*target)
    dx,dy = target[0]-previous[0],target[1]-previous[1]
    scale = min(1., LINEAR_ACCELERATION*dt/max(math.hypot(dx,dy),1e-12))
    dz = target[2]-previous[2]
    return [previous[0]+scale*dx,previous[1]+scale*dy,
            previous[2]+min(ANGULAR_ACCELERATION*dt,max(-ANGULAR_ACCELERATION*dt,dz))]
