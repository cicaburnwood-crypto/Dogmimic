"""Portable inference for our B2 actors. This module never publishes motor commands."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or not .5 < norm < 1.5:
        raise ValueError('Invalid wxyz IMU quaternion')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], dtype=np.float32)


def constrain(command):
    v = np.array(command, dtype=np.float32, copy=True)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('Invalid velocity command')
    v[1] = np.clip(v[1], -.6, .6)
    v[:2] *= min(1., 2. / max(float(np.linalg.norm(v[:2])), 1e-9))
    return v


def slew(previous, requested):
    target = constrain(requested)
    delta = target - previous
    result = previous.copy()
    result[:2] += delta[:2] * min(1., .03/max(float(np.linalg.norm(delta[:2])), 1e-9))
    result[2] += np.clip(delta[2], -.06, .06)
    return result


def clock_goal(command, phase):
    if float(np.square(command).sum()) < .0025:
        clocks = np.ones(4, dtype=np.float32)
    else:
        p = np.remainder(np.float32(phase) + np.array([.5, 0., 0., .5], dtype=np.float32), 1.)
        p = np.where(p < .65, p*(.5/.65), .5+(p-.65)*(.5/.35))
        clocks = np.sin(np.float32(2*math.pi)*p)
    return np.concatenate([command, clocks]).astype(np.float32)


def remote_command(data, yaw_gain=2.):
    data = bytes(data)
    if len(data) != 40:
        raise ValueError('Wireless remote requires 40 bytes')
    lx, rx, ry, _, ly = struct.unpack_from('<5f', data, 4)
    if not all(math.isfinite(v) and abs(v) <= 1.05 for v in (lx, rx, ry, ly)):
        raise ValueError('Invalid remote axes')
    def dead(v):
        return 0. if abs(v) < .05 else math.copysign((abs(v)-.05)/.95, v)
    return {'command': constrain([2*dead(ly), -.6*dead(lx), -yaw_gain*dead(rx)]).tolist(),
            'deadman': bool(data[2] & 2), 'start': bool(data[2] & 4),
            'stop': bool(data[3] & 2)}


class Actor:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder/'policy.json').read_text())
        if sha256(self.folder/'weights.npz') != self.meta['weights_sha256']:
            raise ValueError('Actor weights hash mismatch')
        with np.load(self.folder/'weights.npz', allow_pickle=False) as data:
            self.w = {k: np.ascontiguousarray(data[k], dtype=np.float32) for k in data.files}
        self.prefix = 'teacher.' if self.meta['skill'] == 'hindstand' else ''

    def normalize(self, name, value, clip=True):
        prefix = self.prefix + name + '.'
        y = (value-self.w[prefix+'mean'])/np.sqrt(self.w[prefix+'var']+np.float32(.0001))
        return np.clip(y, -10., 10.) if clip else y

    def mlp(self, name, value, features=False):
        prefix = self.prefix + name + '.'
        indices = sorted(int(k[len(prefix):].split('.')[0]) for k in self.w
                         if k.startswith(prefix) and k.endswith('.weight'))
        if features:
            indices = indices[:-1]
        x = value
        for i in indices:
            x = x @ self.w[f'{prefix}{i}.weight'].T + self.w[f'{prefix}{i}.bias']
            if features or i != indices[-1]:
                x = np.maximum(x, 0.) + np.expm1(np.minimum(x, 0.))
        return x

    def __call__(self, observation, goal):
        obs = np.asarray(observation, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if obs.shape[-1] != self.meta['obs_dim'] or goal.shape[-1] != self.meta['goal_dim']:
            raise ValueError('Policy input dimensions do not match export')
        if not np.isfinite(obs).all() or not np.isfinite(goal).all():
            raise ValueError('Nonfinite actor input')
        original = goal[..., :195] if self.prefix else goal
        z = np.tanh(self.mlp('encoder', self.normalize('goal_norm', original, bool(self.prefix))))
        x = np.concatenate([z, self.normalize('obs_norm', obs)], axis=-1)
        if not self.prefix:
            return np.tanh(self.mlp('decoder', x))
        h = self.mlp('decoder', x, features=True)
        last = max(int(k.split('.')[2]) for k in self.w
                   if k.startswith('teacher.decoder.') and k.endswith('.weight'))
        prior = h @ self.w[f'teacher.decoder.{last}.weight'].T + self.w[f'teacher.decoder.{last}.bias']
        dh = h @ self.w['hind_head.weight'].T + self.w['hind_head.bias']
        dq = h @ self.w['quad_head.weight'].T + self.w['quad_head.bias']
        upright = np.clip((.7-(original[..., 10]+original[..., 13])*.5)/1.05, 0., 1.)
        upright = (upright*upright*(3-2*upright))[..., None]
        still = np.exp(-np.square(original[..., 21:33]).mean(-1)/.15**2)[..., None]
        bh, bq = self.w['max_hind_joint_delta_rad']/.7, self.w['max_joint_delta_rad']/.7
        return prior + still*(upright*bh*np.tanh(dh/bh)
                              + (goal[..., 195:] >= 3)*(1-upright)*bq*np.tanh(dq/bq))


class Controller:
    def __init__(self, folder):
        self.actor = Actor(folder)
        self.meta = self.actor.meta
        self.perm = np.array(self.meta['policy_to_sdk'], dtype=int)
        self.q0 = np.array(self.meta['q0'], dtype=np.float32)
        self.scale = np.array(self.meta['action_scale'], dtype=np.float32)
        self.limits = np.array(self.meta['joint_limits'], dtype=np.float32)
        self.reference = None
        if self.meta['skill'] == 'hindstand':
            path = Path(folder)/'reference.npz'
            if sha256(path) != self.meta['reference_sha256']:
                raise ValueError('Reference hash mismatch')
            with np.load(path, allow_pickle=False) as a:
                self.reference = {k:a[k] for k in a.files}
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.history = None
        self.command = np.zeros(3, dtype=np.float32)
        self.phase = .25
        self.frame = 0
        self.anchor_position = None
        self.anchor_rotation = None

    def reference_goal(self, root_position, root_rotation, frame):
        if root_position is None:
            raise ValueError('Hindstand requires fresh, verified root position; no fabricated position fallback')
        position = np.asarray(root_position, dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid root position estimate')
        ref = self.reference
        if self.anchor_position is None:
            self.anchor_position = position.copy()
            yaw = math.atan2(-root_rotation[0, 1], root_rotation[1, 1])
            c, s = math.cos(yaw), math.sin(yaw)
            self.anchor_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        ids = np.minimum(frame+np.arange(1, 6), len(ref['pos'])-1)
        A = self.anchor_rotation
        rp = (ref['pos'][ids]-ref['pos'][0]) @ A.T + self.anchor_position
        rr = A @ ref['rot'][ids]
        dp = (rp-position) @ root_rotation
        relative_rotation = root_rotation.T @ rr
        rot6d = relative_rotation[..., :, :2].transpose(0, 2, 1).reshape(5, 6)
        v_local = np.einsum('kji,kj->ki', ref['rot'][ids], ref['v'][ids])
        w_local = np.einsum('kji,kj->ki', ref['rot'][ids], ref['w'][ids])
        goal = np.concatenate([dp, rot6d, ref['q'][ids], ref['qd'][ids], v_local, w_local], -1).reshape(-1)
        return np.concatenate([goal, [float(ref['phase'][ids[0]])]]).astype(np.float32)

    def step(self, q_sdk, dq_sdk, gyro, quaternion, requested=(0., 0., 0.), root_position=None,
             advance_reference=False):
        q = np.asarray(q_sdk, dtype=np.float32)[self.perm]
        dq = np.asarray(dq_sdk, dtype=np.float32)[self.perm]
        R = rotation(quaternion)
        observation = np.concatenate([q-self.q0, dq, np.asarray(gyro, dtype=np.float32),
                                      -R[2, :], self.previous_action]).astype(np.float32)
        if not np.isfinite(observation).all():
            raise ValueError('Nonfinite proprioceptive input')
        if self.history is None:
            self.history = np.tile(observation, (self.meta['history'], 1))
        else:
            self.history = np.concatenate([self.history[1:], observation[None]])
        if self.reference is None:
            self.command = slew(self.command, requested)
            goal = clock_goal(self.command, self.phase)
        else:
            goal = self.reference_goal(root_position, R, self.frame)
        action = self.actor(self.history.reshape(1, -1), goal[None])[0]
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite policy output')
        self.previous_action = action.copy()
        if self.reference is None:
            self.phase = (self.phase+.04) % 1. if np.square(self.command).sum() >= .0025 else .25
        elif advance_reference:
            self.frame = min(self.frame+1, len(self.reference['pos'])-2)
        raw_target = self.q0+self.scale*action
        target = np.clip(raw_target, self.limits[:, 0]+.02, self.limits[:, 1]-.02)
        sdk = np.empty(12, dtype=np.float32)
        sdk[self.perm] = target
        return sdk, {'command':self.command.tolist(), 'reference_frame':self.frame,
                     'action':action.tolist(), 'target_clipped':bool(np.any(raw_target != target)),
                     'joint_delta_max_rad':float(np.max(np.abs(target-q)))}
