"""Frozen B2 tilted-locomotion inference. No robot transport or publishers."""
import math
import numpy as np
from runtime import Actor, rotation

DT = np.float32(.02)
MAX_ANGLE = np.float32(math.radians(20))


def constrain_request(requested):
    x = np.array(requested, dtype=np.float32, copy=True)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError('需要有效的 vx/vy/yaw-rate/roll/pitch 五维命令')
    if np.max(np.abs(x[3:])) > float(MAX_ANGLE)+1e-6:
        raise ValueError('roll/pitch 每轴不能超过 ±20°')
    x[0] = np.clip(x[0], -4., 4.)
    x[1] = np.clip(x[1], -1., 1.)
    x[:2] *= np.float32(min(1., 4./max(float(np.linalg.norm(x[:2])), 1e-8)))
    x[3:] = np.clip(x[3:], -MAX_ANGLE, MAX_ANGLE)
    return x


def policy_goal(context):
    speed = np.linalg.norm(context[:2])
    moving = speed > .05 or abs(float(context[2])) > .05
    duty = np.float32(.65)-np.float32(.1)*np.clip(speed, 0., 4.)/np.float32(4.)
    phases = np.remainder(context[12]+np.array([.5, 0., 0., .5], np.float32), 1.)
    if not moving:
        phases[:] = duty*np.float32(.5)
    warped = np.where(phases < duty, phases*(np.float32(.5)/duty),
                      np.float32(.5)+(phases-duty)*(np.float32(.5)/(1-duty)))
    clocks = np.sin(np.float32(2*math.pi)*warped)
    return np.concatenate((context[:12], clocks)).astype(np.float32)


def advance_command(context):
    x = context.copy()
    delta = context[7:10]-context[:3]
    x[:2] += delta[:2]*np.float32(min(1., float(np.float32(1.5)*DT)/max(float(np.linalg.norm(delta[:2])), 1e-8)))
    x[2] += np.clip(delta[2], -np.float32(3.)*DT, np.float32(3.)*DT)
    acceleration = np.clip(np.float32(25.)*(context[10:12]-context[3:5])-np.float32(10.)*context[5:7],
                           -np.float32(math.radians(90)), np.float32(math.radians(90)))
    x[5:7] = np.clip(context[5:7]+DT*acceleration, -np.float32(math.radians(30)), np.float32(math.radians(30)))
    x[3:5] = context[3:5]+DT*x[5:7]
    speed = np.linalg.norm(x[:2])
    if speed > .05 or abs(float(x[2])) > .05:
        frequency = np.float32(2.)+np.float32(.35)*np.clip(speed, 0., 4.)
        x[12] = np.remainder(context[12]+DT*frequency, np.float32(1.))
    else:
        x[12] = np.float32(.25)
    return x.astype(np.float32)


class TiltLocomotionController:
    def __init__(self, folder):
        self.actor = Actor(folder)
        self.meta = self.actor.meta
        if (self.meta.get('skill') != 'tilt_locomotion' or self.meta['obs_dim'] != 420
                or self.meta['goal_dim'] != 16 or self.meta['history'] != 10):
            raise ValueError('B2 倾斜行走模型格式不匹配')
        self.perm = np.asarray(self.meta['policy_to_sdk'], dtype=int)
        self.q0 = np.asarray(self.meta['q0'], dtype=np.float32)
        self.scale = np.asarray(self.meta['action_scale'], dtype=np.float32)
        self.limits = np.asarray(self.meta['joint_limits'], dtype=np.float32)
        self.reference = None
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(12, np.float32)
        self.context = np.zeros(13, np.float32)
        self.context[12] = .25
        self.history = None
        self.frame = 0

    def step(self, q_sdk, dq_sdk, gyro, quaternion, requested=(0., 0., 0., 0., 0.), **_):
        requested = constrain_request(requested)
        q = np.asarray(q_sdk, dtype=np.float32)[self.perm]
        dq = np.asarray(dq_sdk, dtype=np.float32)[self.perm]
        rot = rotation(quaternion)
        obs = np.concatenate((q-self.q0, dq, np.asarray(gyro, np.float32),
                              -rot[2, :], self.previous_action)).astype(np.float32)
        if obs.shape != (42,) or not np.isfinite(obs).all():
            raise ValueError('倾斜行走模型本体输入无效')
        self.history = (np.tile(obs, (10, 1)) if self.history is None
                        else np.concatenate((self.history[1:], obs[None])))
        self.context[7:12] = requested
        goal = policy_goal(self.context)
        action = self.actor(self.history.reshape(1, -1), goal[None])[0]
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('倾斜行走模型输出无效')
        raw = self.q0+self.scale*action
        target = np.clip(raw, self.limits[:, 0]+.02, self.limits[:, 1]-.02)
        sdk = np.empty(12, np.float32)
        sdk[self.perm] = target
        self.previous_action = action.copy()
        self.context = advance_command(self.context)
        self.frame += 1
        actual = [math.atan2(rot[2, 1], rot[2, 2]), math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2]))]
        return sdk, dict(action=action.tolist(), goal=goal.tolist(), requested=requested.tolist(),
                         requested_degrees=np.degrees(requested[3:]).tolist(),
                         desired_degrees=np.degrees(goal[3:5]).tolist(), actual_degrees=np.degrees(actual).tolist(),
                         target_clipped=bool(np.any(raw != target)))
