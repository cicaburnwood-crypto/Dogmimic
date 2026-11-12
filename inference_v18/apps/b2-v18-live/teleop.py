"""EE goal integration only. No robot, joint-command, or IK interface."""
import time
import numpy as np
from scipy.spatial.transform import Rotation

KEYS=frozenset(['KeyW','KeyA','KeyS','KeyD','KeyR','KeyF','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','KeyO','KeyP'])

class ControlFault(ValueError):
 def __init__(self,code,message,issues=None):
  super().__init__(message);self.detail={'code':code,'message':message,'issues':issues or []}

class KeyboardGoal:
 def __init__(self,clock=time.monotonic):
  self.clock=clock;self.enabled=False;self.owner=None;self.keys=set();self.seq=-1;self.position=None;self.rotation=None;self.stamp=0.;self.last=0.;self.revision=0;self.notice=None
 def enable(self,owner,s):
  self.enabled=True;self.owner=owner;self.keys=set();self.seq=-1;self.position=s['ee'].copy();self.rotation=s['eerot'].copy();self.stamp=self.last=self.clock();self.revision=0;self.notice=None
 def disable(self):self.enabled=False;self.keys=set();self.owner=None
 def input(self,owner,seq,keys):
  if not self.enabled or owner!=self.owner:raise ControlFault('KEYBOARD_OWNER','键盘会话未启用或不属于当前页面')
  if type(seq)!=int or seq<=self.seq:raise ControlFault('KEYBOARD_SEQUENCE','拒绝重复或倒序按键帧')
  if not isinstance(keys,list) or any(not isinstance(k,str) or k not in KEYS for k in keys):raise ControlFault('KEYBOARD_KEYS','包含未定义按键')
  # Never apply a newly received key to the elapsed time before its arrival.
  if set(keys)!=self.keys:self.last=self.clock()
  self.keys=set(keys);self.seq=seq;self.stamp=self.clock()
 def advance(self,s):
  now=self.clock()
  if not self.enabled:raise ControlFault('KEYBOARD_DISABLED','键盘控制已关闭')
  if now-self.stamp>.35:raise ControlFault('KEYBOARD_STALE','按键连接超时，已停止联合动作')
  dt=min(.04,max(0.,now-self.last));self.last=now
  k=self.keys
  xyz=np.array([int('KeyW' in k)-int('KeyS' in k),int('KeyA' in k)-int('KeyD' in k),int('KeyR' in k)-int('KeyF' in k)],float)
  xyz/=max(1.,np.linalg.norm(xyz))
  # Pitch about TCP X, yaw about reference/world Z, spin about TCP pointing Z.
  rot=np.array([int('ArrowDown' in k)-int('ArrowUp' in k),int('ArrowLeft' in k)-int('ArrowRight' in k),int('KeyO' in k)-int('KeyP' in k)],float)
  rot/=max(1.,np.linalg.norm(rot))
  pos=self.position+xyz*.05*dt
  angular_world=self.rotation@np.array([rot[0],0,rot[2]])+np.array([0,0,rot[1]])
  angular_world/=max(1.,np.linalg.norm(angular_world))
  orientation=Rotation.from_rotvec(angular_world*np.deg2rad(10)*dt).as_matrix()@self.rotation
  self.notice=None
  if np.linalg.norm(pos[:2])>3 or not .1<=pos[2]<=1.6:
   self.notice={'code':'TARGET_WORKSPACE','message':'EE 目标超出当前任务范围：水平半径≤3 m，Z 为 0.10–1.60 m；保留上一有效目标','issues':[]};return False
  lead=np.linalg.norm(pos-s['ee']);angle=Rotation.from_matrix(s['eerot'].T@orientation).magnitude()
  if lead>.10 or angle>np.deg2rad(15):
   self.notice={'code':'TARGET_TRACKING_LAG','message':f'EE 尚未跟上目标（{lead*100:.1f} cm / {np.rad2deg(angle):.1f}°），暂停累加，等待联合策略追踪；不是关节无解证明','issues':[]};return False
  changed=bool(np.linalg.norm(xyz)+np.linalg.norm(rot)>0 and dt>0)
  if changed:self.position=pos;self.rotation=orientation;self.revision+=1
  return changed
 def snapshot(self):
  return {'enabled':self.enabled,'owner':self.owner,'keys':sorted(self.keys),'seq':self.seq,'revision':self.revision,'xyz':self.position.tolist() if self.position is not None else None,'quaternion_xyzw':Rotation.from_matrix(self.rotation).as_quat().tolist() if self.rotation is not None else None,'linear_speed_m_s':.05,'angular_speed_deg_s':10,'notice':self.notice}
