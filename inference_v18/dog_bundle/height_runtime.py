"""Numpy deployment of the frozen height policy and its training command contract.

No transport, motor publishers or dependency on the training host.
"""
import math
import numpy as np
from runtime import Actor, rotation
from tilt_runtime import JointTargetNotch

DT = np.float32(.02)
HEIGHT_MIN, HEIGHT_MAX = -.20, .05


def tilt_limit(height):
    h = float(height)
    return math.radians(max(5., min(20., 20. if h >= -.10 else
        10.+200.*(h+.15) if h >= -.15 else 5.+100.*(h+.20))))


def minimum_height_for_tilt(rp):
    degrees = float(np.max(np.abs(rp)))*180/math.pi
    return max(-.20, min(-.10, -.20 if degrees <= 5. else
        -.20+(degrees-5.)/100. if degrees <= 10. else -.15+(degrees-10.)/200.))


def constrain_request(requested, limits=None):
    x = np.asarray(requested, dtype=np.float32).copy()
    if x.shape == (5,):
        x = np.concatenate((x, np.zeros(1, np.float32)))
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('需要有效的 vx/vy/yaw-rate/roll/pitch/height 六维命令')
    x[0] = np.clip(x[0], -4., 4.)
    x[1] = np.clip(x[1], -1., 1.)
    x[:2] *= np.float32(min(1., 4./max(float(np.linalg.norm(x[:2])), 1e-8)))
    x[5] = np.clip(x[5], HEIGHT_MIN, HEIGHT_MAX)
    angle = np.float32(tilt_limit(x[5]))
    x[3:5] = np.clip(x[3:5], -angle, angle)
    if limits:
        x[0] = np.clip(x[0], -limits['xy_max_m_s'], limits['xy_max_m_s'])
        x[1] = np.clip(x[1], -limits['lateral_max_m_s'], limits['lateral_max_m_s'])
        x[:2] *= np.float32(min(1., limits['xy_max_m_s']/max(float(np.linalg.norm(x[:2])), 1e-8)))
        x[2] = np.clip(x[2], -limits['yaw_max_rad_s'], limits['yaw_max_rad_s'])
    return x


def policy_goal(context):
    speed = np.linalg.norm(context[:2])
    duty = np.float32(.65)-np.float32(.1)*np.clip(speed, 0., 4.)/np.float32(4.)
    phases = np.remainder(context[12]+np.array([.5, 0., 0., .5], np.float32), 1.)
    warp = np.where(phases < duty, phases*(np.float32(.5)/duty),
                    np.float32(.5)+(phases-duty)*(np.float32(.5)/(1-duty)))
    progress = np.clip(context[16], 0., 1.)
    blend = progress**2*(3.-2.*progress)
    clocks = 1.+blend*(np.sin(np.float32(2*math.pi)*warp)-1.)
    return np.concatenate((context[:12], clocks, context[13:17])).astype(np.float32)


def advance_command(context):
    x = context.copy()
    delta = context[7:10]-context[:3]
    x[:2] += delta[:2]*np.float32(min(1., float(np.float32(1.5)*DT)/max(float(np.linalg.norm(delta[:2])), 1e-8)))
    x[2] += np.clip(delta[2], -np.float32(3.)*DT, np.float32(3.)*DT)
    current_limit = np.float32(tilt_limit(context[13]))
    limit = min(np.float32(tilt_limit(context[15])), current_limit)
    target = np.clip(context[10:12], -limit, limit)
    acc = np.clip(25.*(target-context[3:5])-10.*context[5:7],
                  -math.radians(90), math.radians(90))
    velocity = np.clip(context[5:7]+DT*acc, -math.radians(30), math.radians(30))
    x[3:5] = np.clip(context[3:5]+DT*velocity, -current_limit, current_limit)
    x[5:7] = (x[3:5]-context[3:5])/DT
    floor = np.float32(minimum_height_for_tilt(x[3:5]))
    height_target = max(context[15], floor)
    height_acc = np.clip(16.*(height_target-context[13])-8.*context[14], -.15, .15)
    x[14] = np.clip(context[14]+DT*height_acc, -.06, .06)
    x[13] = np.clip(context[13]+DT*x[14], HEIGHT_MIN, HEIGHT_MAX)
    if x[13] < floor:
        x[13], x[14] = floor, 0.
    moving = np.linalg.norm(x[:2]) > .05 or abs(float(x[2])) > .05
    x[16] = np.clip(context[16]+(float(DT)/.4 if moving else -float(DT)/.4), 0., 1.)
    blend = x[16]**2*(3.-2.*x[16])
    speed = max(float(np.linalg.norm(x[:2])), .25*abs(float(x[2])))
    frequency = 1.4+.5*min(speed, 4.)
    x[12] = (context[12]+float(DT)*frequency*blend)%1. if blend > 0. else .25
    return x.astype(np.float32)


class HeightLocomotionController:
    def __init__(self, folder):
        self.actor = Actor(folder)
        self.meta = self.actor.meta
        if (self.meta.get('skill') != 'height_locomotion' or self.meta['obs_dim'] != 420
                or self.meta['goal_dim'] != 20 or self.meta['history'] != 10):
            raise ValueError('B2 高度行走模型格式不匹配')
        self.perm = np.asarray(self.meta['policy_to_sdk'], dtype=int)
        self.q0 = np.asarray(self.meta['q0'], dtype=np.float32)
        self.scale = np.asarray(self.meta['action_scale'], dtype=np.float32)
        self.limits = np.asarray(self.meta['joint_limits'], dtype=np.float32)
        self.target_filter_spec = self.meta['joint_target_filter']
        spec = self.target_filter_spec
        if spec not in (dict(kind='notch', center_hz=11., quality_factor=2., sample_hz=50.),
                         dict(kind='notch', center_hz=11., quality_factor=1., sample_hz=50.)):
            raise ValueError('高度策略只支持 11 Hz、Q=2 或候选 Q=1、50 Hz 陷波配置；配置支持不代表实机验证')
        self.target_filter = JointTargetNotch(spec['center_hz'], spec['quality_factor'], spec['sample_hz'])
        self.height_control_config = dict(self.meta['height_control'])
        self.motion_limits = dict(self.meta['deployment_limits'])
        self.reference = None
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(12, np.float32)
        self.context = np.zeros(17, np.float32)
        self.context[12] = .25
        self.history = None
        self.frame = 0
        self.target_filter.reset()

    def step(self, q_sdk, dq_sdk, gyro, quaternion, requested=(0., 0., 0., 0., 0., 0.), **_):
        requested = constrain_request(requested, self.motion_limits)
        q = np.asarray(q_sdk, dtype=np.float32)[self.perm]
        dq = np.asarray(dq_sdk, dtype=np.float32)[self.perm]
        rot = rotation(quaternion)
        obs = np.concatenate((q-self.q0, dq, np.asarray(gyro, np.float32),
                              -rot[2, :], self.previous_action)).astype(np.float32)
        if obs.shape != (42,) or not np.isfinite(obs).all():
            raise ValueError('高度行走模型本体输入无效')
        self.history = (np.tile(obs, (10, 1)) if self.history is None
                        else np.concatenate((self.history[1:], obs[None])))
        self.context[7:12], self.context[15] = requested[:5], requested[5]
        goal = policy_goal(self.context)
        action = self.actor(self.history.reshape(1, -1), goal[None])[0]
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('高度行走模型输出无效')
        raw = self.q0+self.scale*np.clip(action, -1., 1.)
        unfiltered = np.clip(raw, self.limits[:, 0]+.02, self.limits[:, 1]-.02)
        filtered = self.target_filter(unfiltered)
        target = np.clip(filtered, self.limits[:, 0]+.02, self.limits[:, 1]-.02)
        if np.any(target != filtered):
            self.target_filter.prime(target)
        sdk = np.empty(12, np.float32)
        sdk[self.perm] = target
        self.previous_action = action.copy()
        self.context = advance_command(self.context)
        self.frame += 1
        actual = [math.atan2(rot[2, 1], rot[2, 2]), math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2]))]
        return sdk, dict(command_adapter='height_gait_filter_v1', action=action.tolist(),
            goal=goal.tolist(), requested=requested.tolist(), requested_degrees=np.degrees(requested[3:5]).tolist(),
            desired_degrees=np.degrees(goal[3:5]).tolist(), actual_degrees=np.degrees(actual).tolist(),
            requested_height_m=float(requested[5]), desired_height_m=float(goal[16]),
            desired_height_rate_m_s=float(goal[17]), nominal_root_height_m=self.meta['nominal_root_height_m'],
            tilt_limit_degrees=math.degrees(tilt_limit(goal[16])),
            gait_blend=float(goal[19]**2*(3.-2.*goal[19])), gait_transition_seconds=.4,
            deployment_limits=self.motion_limits, target_filter=self.target_filter_spec,
            unfiltered_target_policy=unfiltered.tolist(),
            target_clipped=bool(np.any(raw != unfiltered) or np.any(filtered != target)))
