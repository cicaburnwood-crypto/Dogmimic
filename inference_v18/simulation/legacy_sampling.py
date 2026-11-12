"""Train-only continuous command coverage. Never imports the evaluation cases."""
import math
import torch
from contract import constrain_targets, yaw_sampling_bound

MIXTURE = dict(easy=0.30, continuous_full=0.30, boundary=0.40)


def _sign(k, device):
    return torch.where(torch.rand(k, device=device) < .5, -1., 1.)


def sample_targets(k, device, return_labels=False):
    mix = torch.rand(k, device=device)
    labels = torch.where(mix < .3, 0, torch.where(mix < .6, 1, 2))
    easy, hard = labels == 0, labels == 2
    # Continuous domain is retained, including arbitrary simultaneous angles.
    speed = 4. * torch.rand(k, device=device)
    speed[easy] = torch.rand(int(easy.sum()), device=device)
    speed[hard] = 3. + torch.rand(int(hard.sum()), device=device)
    edge = hard & (torch.rand(k, device=device) < .5)
    speed[edge] = 4.
    vy = (2 * torch.rand(k, device=device) - 1) * speed.clamp(max=1.)
    vx = torch.sqrt((speed.square() - vy.square()).clamp_min(0)) * _sign(k, device)
    mode = torch.rand(k, device=device)
    # Equal forward/backward exposure; longitudinal, diagonal, lateral and spin.
    straight = mode < .50
    side = (mode >= .80) & (mode < .90)
    spin = mode >= .90
    vx[straight] = speed[straight] * _sign(int(straight.sum()), device)
    vy[straight] = 0.
    vx[side | spin] = 0.
    vy[side] = speed[side].clamp(max=1.) * _sign(int(side.sum()), device)
    vy[spin] = 0.
    planar = torch.sqrt(vx.square() + vy.square())
    yaw = (2 * torch.rand(k, device=device) - 1) * yaw_sampling_bound(planar)
    yaw[easy] *= .3
    yaw[straight | side] = 0.
    yaw[~spin & (torch.rand(k, device=device) < .35)] = 0.
    yaw[spin & hard] = 2. * _sign(int((spin & hard).sum()), device)
    rp = (2 * torch.rand(k, 2, device=device) - 1) * math.radians(20.)
    rp[easy] *= .5
    nh = int(hard.sum())
    rp[hard] = torch.sign(rp[hard]) * torch.deg2rad(15. + 5. * torch.rand(nh, 2, device=device))
    corners = hard & (torch.rand(k, device=device) < .5)
    rp[corners] = torch.sign(rp[corners]) * math.radians(20.)
    # Easy/full samples retain level posture, isolated roll/pitch and stopping.
    tiltmode = torch.rand(k, device=device)
    rp[(~hard) & (tiltmode < .15)] = 0.
    rp[(~hard) & (tiltmode >= .15) & (tiltmode < .30), 0] = 0.
    rp[(~hard) & (tiltmode >= .30) & (tiltmode < .45), 1] = 0.
    result = torch.cat([vx[:, None], vy[:, None], yaw[:, None], rp], -1)
    stationary = (~hard) & (torch.rand(k, device=device) < .12)
    result[stationary, :3] = 0.
    result[(~hard) & (torch.rand(k, device=device) < .08)] = 0.
    result = constrain_targets(result)
    return (result, labels) if return_labels else result


def positive_huber(x):
    """Quadratic near zero, linear tail; x is a nonnegative normalized error."""
    return torch.where(x < 1., .5 * x.square(), x - .5)
