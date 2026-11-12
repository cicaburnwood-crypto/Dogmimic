"""FP32 inference for the frozen B2 attitude Actor; no DDS or motor publisher."""
import math
import numpy as np
from runtime import Actor, rotation


def advance_command(context):
    """Numpy equivalent of the training contract, once per applied 20 ms step."""
    q, v, target = context[:2], context[2:4], context[4:6]
    acceleration = np.clip(25.*(target-q)-10.*v, -math.radians(90), math.radians(90))
    velocity = np.clip(v+np.float32(.02)*acceleration, -math.radians(30), math.radians(30))
    return np.concatenate((q+np.float32(.02)*velocity, velocity, target)).astype(np.float32)


class AttitudeController:
    def __init__(self, folder):
        self.actor = Actor(folder)
        self.meta = self.actor.meta
        if self.meta['skill'] != 'attitude' or self.meta['obs_dim'] != 420 or self.meta['goal_dim'] != 6:
            raise ValueError('B2 姿态模型格式不匹配')
        self.perm = np.asarray(self.meta['policy_to_sdk'], dtype=int)
        self.q0 = np.asarray(self.meta['q0'], dtype=np.float32)
        self.scale = np.asarray(self.meta['action_scale'], dtype=np.float32)
        self.limits = np.asarray(self.meta['joint_limits'], dtype=np.float32)
        self.reference = None
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.context = np.zeros(6, dtype=np.float32)
        self.history = None
        self.frame = 0

    def step(self, q_sdk, dq_sdk, gyro, quaternion, requested=(0., 0.), **_):
        target_angle = np.asarray(requested, dtype=np.float32)
        if target_angle.shape != (2,) or not np.isfinite(target_angle).all():
            raise ValueError('姿态命令必须为有效的 roll / pitch 弧度')
        if np.max(np.abs(target_angle)) > math.radians(30)+1e-7 or np.count_nonzero(target_angle) > 1:
            raise ValueError('每一步仅允许单轴 ±30° 内的姿态命令')
        q = np.asarray(q_sdk, dtype=np.float32)[self.perm]
        dq = np.asarray(dq_sdk, dtype=np.float32)[self.perm]
        R = rotation(quaternion)
        obs = np.concatenate((q-self.q0, dq, np.asarray(gyro, dtype=np.float32),
                              -R[2, :], self.previous_action)).astype(np.float32)
        if obs.shape != (42,) or not np.isfinite(obs).all():
            raise ValueError('姿态模型本体输入无效')
        if self.history is None:
            self.history = np.tile(obs, (self.meta['history'], 1))
        else:
            self.history = np.concatenate((self.history[1:], obs[None]))
        self.context[4:6] = target_angle
        goal = self.context.copy()
        action = self.actor(self.history.reshape(1, -1), goal[None])[0]
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError('姿态模型输出无效')
        raw = self.q0+self.scale*action
        target = np.clip(raw, self.limits[:, 0]+.02, self.limits[:, 1]-.02)
        sdk = np.empty(12, dtype=np.float32)
        sdk[self.perm] = target
        actual = [math.atan2(R[2, 1], R[2, 2]),
                  math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))]
        self.previous_action = action.copy()
        self.context = advance_command(self.context)
        self.frame += 1
        return sdk, dict(action=action.tolist(), goal=goal.tolist(),
                         requested_degrees=np.degrees(target_angle).tolist(),
                         desired_degrees=np.degrees(goal[:2]).tolist(),
                         actual_degrees=np.degrees(actual).tolist(),
                         target_clipped=bool(np.any(raw != target)))
