"""Robot-side inference. Dependencies: numpy, torch; no simulator or official policy."""
import json,math,struct,time
from pathlib import Path
import numpy as np
import torch
from contract import constrain_command,slew_command

def gravity_from_quaternion(quaternion):
    q=np.asarray(quaternion,dtype=np.float32)
    norm=float(np.linalg.norm(q))
    if not .5<norm<1.5:raise ValueError('Invalid IMU quaternion')
    w,x,y,z=q/norm
    return np.array([2*(w*y-x*z),-2*(y*z+w*x),-(1-2*(x*x+y*y))],dtype=np.float32)

def decode_remote(data,yaw_gain=2.):
    """Unitree wire format only; yaw_gain is a configurable stick scale, not a clip."""
    data=bytes(data)
    if len(data)!=40:raise ValueError('Expected 40-byte wireless remote payload')
    lx,rx,ry,unused,ly=struct.unpack_from('<5f',data,4)
    if not all(math.isfinite(v) and abs(v)<=1.05 for v in (lx,rx,ry,ly)):raise ValueError('Invalid joystick axes')
    def deadband(v):return 0. if abs(v)<.05 else math.copysign((abs(v)-.05)/.95,v)
    # SDK joystick axes: left vertical forward, left horizontal lateral, right horizontal yaw.
    # Both horizontal signs are explicit, verified on the actual remote before arming.
    command=constrain_command(2*deadband(ly),-.6*deadband(lx),-yaw_gain*deadband(rx))
    return dict(command=command,deadman=bool(data[2]&2),start=bool(data[2]&4),estop=bool(data[3]&2))

class Controller:
    def __init__(self,directory):
        directory=Path(directory);self.meta=json.loads((directory/'policy.json').read_text())
        self.policy=torch.jit.load(str(directory/'policy.ts'),map_location='cpu').eval()
        self.perm=np.array(self.meta['policy_to_sdk'],dtype=int)
        self.q0=np.array(self.meta['q0'],dtype=np.float32);self.scale=np.array(self.meta['action_scale'],dtype=np.float32)
        self.limits=np.array(self.meta['joint_limits'],dtype=np.float32)
        self.reset()
    def reset(self):
        self.previous_action=np.zeros(12,dtype=np.float32);self.history=None;self.command=[0.,0.,0.]
    @torch.no_grad()
    def step(self,q_sdk,dq_sdk,gyro,quaternion,target):
        q=np.asarray(q_sdk,dtype=np.float32)[self.perm];dq=np.asarray(dq_sdk,dtype=np.float32)[self.perm]
        gyro=np.asarray(gyro,dtype=np.float32);gravity=gravity_from_quaternion(quaternion)
        if not all(np.isfinite(v).all() for v in (q,dq,gyro,gravity)):raise ValueError('Non-finite observation')
        observation=np.concatenate([q-self.q0,dq,gyro,gravity,self.previous_action]).astype(np.float32)
        if self.history is None:self.history=np.tile(observation,(self.meta['history'],1))
        else:self.history=np.concatenate([self.history[1:],observation[None]],axis=0)
        self.command=slew_command(self.command,target,self.meta['policy_dt'])
        t=time.perf_counter()
        action=self.policy(torch.from_numpy(self.history.reshape(1,-1)),torch.tensor([self.command],dtype=torch.float32))[0].numpy()
        inference=time.perf_counter()-t
        if not np.isfinite(action).all():raise ValueError('Non-finite policy output')
        action=np.clip(action,-1.,1.);self.previous_action=action.copy()
        qtarget=np.clip(self.q0+self.scale*action,self.limits[:,0]+.02,self.limits[:,1]-.02)
        sdk=np.empty(12,dtype=np.float32);sdk[self.perm]=qtarget
        return sdk,dict(command=self.command.copy(),action=action.tolist(),inference_seconds=inference,tilt=math.acos(float(np.clip(-gravity[2],-1,1))))
