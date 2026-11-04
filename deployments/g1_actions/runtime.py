"""Portable inference for our G1 actors. This module never publishes motor commands."""
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


class Actor:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder/'policy.json').read_text())
        if sha256(self.folder/'weights.npz') != self.meta['weights_sha256']:
            raise ValueError('Actor weights hash mismatch')
        with np.load(self.folder/'weights.npz', allow_pickle=False) as data:
            self.w = {k: np.ascontiguousarray(data[k], dtype=np.float32) for k in data.files}
        self.prefix = ''

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
        z = np.tanh(self.mlp('encoder', self.normalize('goal_norm', goal)))
        return self.mlp('decoder', np.concatenate([z, self.normalize('obs_norm', obs)], axis=-1))

class Controller:
    """50 Hz reference tracking. All vectors in policy joint order unless named SDK."""
    def __init__(self, folder):
        self.actor = Actor(folder)
        self.meta = self.actor.meta
        path = Path(folder)/'reference.npz'
        if sha256(path) != self.meta['reference_sha256']:
            raise ValueError('Reference hash mismatch')
        with np.load(path, allow_pickle=False) as r:
            self.ref = {k:r[k] for k in r.files}
        self.perm = np.asarray(self.meta['policy_to_sdk'], dtype=int)
        self.q0 = np.asarray(self.meta['q0'], dtype=np.float32)
        self.scale = np.asarray(self.meta['action_scale'], dtype=np.float32)
        self.foot_kinematics = self.meta.get('foot_center_kinematics')
        self.reset()

    def reset(self):
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.history = None
        self.frame = 0
        self.anchor_position = None
        self.anchor_rotation = None

    def foot_center(self, position, R, q):
        """Same mean collision-geometry centers as the v11 training reset."""
        k = self.foot_kinematics
        if k is None:
            raise ValueError('This model has no exported foot-center kinematics')
        centers = []
        for chain in k['chains']:
            p = np.asarray(position, dtype=np.float32).copy()
            rot = np.asarray(R, dtype=np.float32).copy()
            for joint in chain['joints']:
                p = p + rot @ np.asarray(joint['offset_pos'], dtype=np.float32)
                axis = np.asarray(joint['axis'], dtype=np.float32)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
                angle = q[joint['policy_index']]
                turn = np.eye(3, dtype=np.float32) + np.sin(angle)*skew + (1-np.cos(angle))*(skew @ skew)
                rot = rot @ np.asarray(joint['offset_rot'], dtype=np.float32) @ turn
            centers.append(p + rot @ np.asarray(chain['geom_center'], dtype=np.float32))
        return np.mean(centers, axis=0)

    def goal(self, position, R, frame, q=None):
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Fresh root position is required')
        if self.anchor_position is None:
            self.anchor_position = position.copy()
            # Align reference heading, retaining roll/pitch tracking error.
            yaw = math.atan2(-R[0,1], R[1,1])
            R0 = self.ref['rot'][0]
            yaw -= math.atan2(-R0[0,1], R0[1,1])
            c,s = math.cos(yaw),math.sin(yaw)
            self.anchor_rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)
            if self.meta.get('reference_alignment') == 'episode_initial_foot_center_xy':
                if q is None:
                    raise ValueError('Initial joint positions are required for foot-center alignment')
                physical = self.foot_center(position, R, q)
                nominal = self.foot_center(self.ref['pos'][0], R0, self.ref['q'][0])
                self.anchor_position = self.anchor_rotation @ self.ref['pos'][0]
                self.anchor_position[:2] += (physical - self.anchor_rotation @ nominal)[:2]
                # Training translates XY only. Preserve the measured vertical error.
        ids = np.minimum(frame+np.arange(1,self.meta['future']+1), len(self.ref['pos'])-1)
        r = self.ref
        A = self.anchor_rotation
        rp = (r['pos'][ids]-r['pos'][0]) @ A.T + self.anchor_position
        rr = A @ r['rot'][ids]
        dp = (rp-position) @ R
        rot6d = (R.T @ rr)[...,:,:2].transpose(0,2,1).reshape(len(ids),6)
        v = np.einsum('kji,kj->ki',r['rot'][ids],r['v'][ids])
        w = np.einsum('kji,kj->ki',r['rot'][ids],r['w'][ids])
        return np.concatenate([dp,rot6d,r['q'][ids],r['qd'][ids],v,w],-1).reshape(-1).astype(np.float32)

    def step(self, q_sdk, dq_sdk, gyro, quaternion, root_position):
        q,dq = [np.asarray(x,dtype=np.float32) for x in (q_sdk,dq_sdk)]
        if q.shape!=(29,) or dq.shape!=(29,):
            raise ValueError('This export requires the 29-DOF G1 joint configuration')
        R = rotation(quaternion)
        observation = np.concatenate([q[self.perm]-self.q0,dq[self.perm],
                                      np.asarray(gyro,dtype=np.float32),-R[2],self.previous_action])
        if observation.shape!=(93,) or not np.isfinite(observation).all():
            raise ValueError('Invalid proprioceptive input')
        if self.history is None:
            self.history = np.tile(observation,(self.meta['history'],1))
        else:
            self.history = np.concatenate([self.history[1:],observation[None]])
        goal = self.goal(root_position,R,self.frame,q[self.perm])
        action = self.actor(self.history.reshape(1,-1),goal[None])[0]
        if not np.isfinite(action).all():
            raise ValueError('Invalid actor output')
        self.previous_action = action.copy()
        target = np.empty(29,dtype=np.float32)
        target[self.perm] = self.q0 + self.scale*action
        used_frame = self.frame
        self.frame = min(self.frame+1,len(self.ref['pos'])-1)
        return target, {'frame':used_frame,'progress':self.frame/(len(self.ref['pos'])-1),
                        'trajectory_complete':self.frame==len(self.ref['pos'])-1,
                        'action':action.tolist()}
