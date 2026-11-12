"""Frozen B2 tilted-locomotion inference. No robot transport or publishers."""
import math
import numpy as np
from runtime import Actor, rotation

DT = np.float32(.02)
MAX_ANGLE = np.float32(math.radians(20))


class JointTargetNotch:
    """Unity-DC biquad for the measured 10–12 Hz target oscillation.

    Operates only on Actor joint targets, not sensors, requested body attitude,
    PD gains or native/damping commands. Its first sample is passed unchanged.
    """
    def __init__(self, center_hz=11., quality_factor=2., sample_hz=50.):
        if not all(math.isfinite(v) for v in (center_hz,quality_factor,sample_hz)):
            raise ValueError('关节目标滤波参数必须有限')
        if not 0 < center_hz < sample_hz/2 or quality_factor <= 0 or center_hz/quality_factor >= sample_hz/2:
            raise ValueError('关节目标陷波参数超出有效频率范围')
        omega=2*math.pi*center_hz/sample_hz
        gain=1/(1+math.tan(omega/(2*quality_factor)))
        self.b=np.array([gain,-2*gain*math.cos(omega),gain],dtype=np.float64)
        self.a=np.array([1.,self.b[1],2*gain-1],dtype=np.float64)
        self.reset()

    def reset(self):
        self.z1=self.z2=None

    def prime(self, value):
        value=np.asarray(value,dtype=np.float64)
        self.z1=(1-self.b[0])*value
        self.z2=(self.b[2]-self.a[2])*value

    def __call__(self, value):
        x=np.asarray(value,dtype=np.float64)
        if self.z1 is None:self.prime(x)
        y=self.b[0]*x+self.z1
        self.z1=self.b[1]*x-self.a[1]*y+self.z2
        self.z2=self.b[2]*x-self.a[2]*y
        return y.astype(np.float32)


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


def policy_goal(context, gait_blend=None):
    """Original training goal if blend is None; continuous entry/exit otherwise."""
    speed = np.linalg.norm(context[:2])
    moving = speed > .05 or abs(float(context[2])) > .05
    duty = np.float32(.65)-np.float32(.1)*np.clip(speed, 0., 4.)/np.float32(4.)
    phases = np.remainder(context[12]+np.array([.5, 0., 0., .5], np.float32), 1.)
    if gait_blend is None and not moving:
        phases[:] = duty*np.float32(.5)
    warped = np.where(phases < duty, phases*(np.float32(.5)/duty),
                      np.float32(.5)+(phases-duty)*(np.float32(.5)/(1-duty)))
    clocks = np.sin(np.float32(2*math.pi)*warped)
    if gait_blend is not None:
        clocks = np.float32(1.)+np.float32(gait_blend)*(clocks-np.float32(1.))
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
        self.target_filter_spec = self.meta.get('joint_target_filter')
        self.target_filter = None
        if self.target_filter_spec:
            spec=self.target_filter_spec
            if spec.get('kind')!='notch' or abs(spec.get('sample_hz',0)-1/float(DT))>1e-4:
                raise ValueError('关节目标滤波器与控制周期不匹配')
            self.target_filter=JointTargetNotch(spec['center_hz'],spec['quality_factor'],spec['sample_hz'])
        self.reference = None
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(12, np.float32)
        self.context = np.zeros(13, np.float32)
        self.context[12] = .25
        self.history = None
        self.frame = 0
        self.gait_progress = 0.
        self.gait_transition_seconds = .4
        if self.target_filter is not None:self.target_filter.reset()

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
        moving = np.linalg.norm(self.context[:2]) > .05 or abs(float(self.context[2])) > .05
        # Crossing the velocity deadband must not swap stance clocks directly
        # to a full-amplitude swing: that caused a ~0.66 rad target jump on B2.
        # Reverse an unfinished transition continuously, without phase reset.
        self.gait_progress = float(np.clip(
            self.gait_progress + (1 if moving else -1)*float(DT)/self.gait_transition_seconds, 0., 1.))
        gait_blend = self.gait_progress**2*(3.-2.*self.gait_progress)
        goal = policy_goal(self.context, gait_blend)
        action = self.actor(self.history.reshape(1, -1), goal[None])[0]
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('倾斜行走模型输出无效')
        raw = self.q0+self.scale*action
        target = np.clip(raw, self.limits[:, 0]+.02, self.limits[:, 1]-.02)
        unfiltered_target=target.copy()
        if self.target_filter is not None:
            filtered=self.target_filter(target)
            target=np.clip(filtered,self.limits[:,0]+.02,self.limits[:,1]-.02)
            if np.any(target!=filtered):self.target_filter.prime(target)
        sdk = np.empty(12, np.float32)
        sdk[self.perm] = target
        self.previous_action = action.copy()
        phase = float(self.context[12])
        self.context = advance_command(self.context)
        # Retain phase until the swing amplitude is zero; restarting the phase
        # at the velocity threshold would reintroduce the stopping discontinuity.
        frequency = 2.+.35*min(float(np.linalg.norm(self.context[:2])), 4.)
        self.context[12] = np.float32((phase+float(DT)*frequency*gait_blend)%1. if gait_blend > 0. else .25)
        self.frame += 1
        actual = [math.atan2(rot[2, 1], rot[2, 2]), math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2]))]
        return sdk, dict(command_adapter='continuous_gait_v1', gait_blend=gait_blend,
                         gait_transition_seconds=self.gait_transition_seconds, action=action.tolist(), goal=goal.tolist(), requested=requested.tolist(),
                         requested_degrees=np.degrees(requested[3:]).tolist(),
                         desired_degrees=np.degrees(goal[3:5]).tolist(), actual_degrees=np.degrees(actual).tolist(),
                         target_filter=self.target_filter_spec,
                         unfiltered_target_policy=unfiltered_target.tolist(),
                         target_clipped=bool(np.any(raw != unfiltered_target) or np.any(target<=self.limits[:,0]+.02) or np.any(target>=self.limits[:,1]-.02)))
