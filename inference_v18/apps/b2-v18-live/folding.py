"""Explicit arm-only fold, using the previously measured and verified folded pose."""
import asyncio
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from teleop import ControlFault

FOLD_CONFIG=json.loads((Path(__file__).parent/'models/fold_pose.json').read_text())
FOLD_TARGET=np.array(FOLD_CONFIG['joints_rad'],dtype=float)

class FoldArmMixin:
 def fold_ready(self,moving=False):
  self.require_remote_idle();self.fresh();d=self.dog;a=self.arm
  if d.get('active') not in ('native','trained') or not all(d.get('gate',{}).get(k,False) for k in ('telemetry','stationary','posture')):
   raise ControlFault('FOLD_STANDING_REQUIRED','折叠前需要机器狗稳定站立；原生或 P4 模式均可')
  if d.get('joint_manager',{}).get('engaged') or d.get('stationary_reach',{}).get('engaged'):
   raise ControlFault('FOLD_BODY_BUSY','机体仍有控制任务，请先停止联合运动并保持站立')
  if a.get('debug') is not False or a['state']['servo']!=1 or a['state']['fsm_state']!='POSITION' or a.get('error') not in (0,None):
   raise ControlFault('FOLD_ARM_STATE','机械臂需要真实反馈、无故障且已使能的位置模式')
  if not moving and (a['active'] or a['state']['task']['exe_flag']):raise ValueError('机械臂正在执行其他任务，请先停止')
  if not np.allclose(a['tool_coordinate'],[0,0,0,0,0,0,1],atol=1e-8):raise ValueError('机械臂工具坐标已改变，不能使用已验证的折叠路径')
  s=self.state()
  if not np.isfinite(np.r_[s['pos'],s['rot'].ravel(),s['v'],s['gyro'],s['aq'],s['aqd']]).all():raise ValueError('折叠所需的实际状态不是有限数值')
  if np.linalg.norm(s['v'][:2])>.05 or np.linalg.norm(s['gyro'])>.15 or (not moving and max(abs(s['aqd']))>.05):
   raise ControlFault('FOLD_NOT_STILL','需要机体静止；开始折叠前机械臂也必须静止')
  self.fold_joint_limits(s['aq'],tolerance=.001)
  return s

 def fold_joint_limits(self,q,tolerance=0):
  for j,(angle,lo,hi) in enumerate(zip(q,self.g.arm_lo,self.g.arm_hi)):
   if angle<lo-tolerance or angle>hi+tolerance:
    message=f'J{j+1} 折叠角度 {np.rad2deg(angle):.2f}° 超出 [{np.rad2deg(lo):.2f}°, {np.rad2deg(hi):.2f}°]'
    raise ControlFault('JOINT_LIMIT',message,[{'joint':f'J{j+1}','code':'JOINT_LIMIT','message':message}])

 def fold_contacts(self,s,q):
  self.g.set(s['body'],q,s['q'],s['pos'][:2])
  for c in self.g.d.contact:
   x,y=int(c.geom1),int(c.geom2)
   if (self.g.m.geom_bodyid[x]==0 and y in self.g.feet) or (self.g.m.geom_bodyid[y]==0 and x in self.g.feet):continue
   if c.dist<0:
    names=[self.g.m.geom(i).name or f'geom_{i}' for i in (x,y)]
    message=f'折叠路径模型碰撞：{names[0]} ↔ {names[1]}，穿透 {-c.dist*1000:.2f} mm'
    raise ControlFault('FOLD_COLLISION',message,[{'code':'FOLD_COLLISION','links':names,'message':message}])

 def fold_check_cancel(self):
  self.require_remote_idle()
  if self.cancel or time.monotonic()-self.beats.get(self.owner,0)>2:raise ControlFault('FOLD_CANCELLED','折叠已取消或页面连接中断')

 def fold_check_carrier(self,s,p):
  if self.dog['active']!=p['dog_mode'] or np.linalg.norm(s['pos']-np.array(p['start_pos']))>.015 or Rotation.from_matrix(s['rot'].T@np.array(p['start_rot'])).magnitude()>.04:
   raise ControlFault('FOLD_BODY_MOVED','机体位置、姿态或运动模式改变，停止机械臂折叠')

 async def fold_path(self,s,target):
  self.fold_joint_limits(target)
  samples=max(401,int(np.ceil(max(abs(target-s['aq']))/.005))+1)
  try:
   for i,t in enumerate(np.linspace(0,1,samples)):
    if i%32==0:
     await asyncio.sleep(0);self.fold_check_cancel()
    self.fold_contacts(s,(1-t)*s['aq']+t*target)
  finally:self.g.set(s['body'],s['aq'],s['q'],s['pos'][:2])
  return samples

 async def run_fold(self,p):
  started=False;trace=[];target=FOLD_TARGET.copy();samples=0
  try:
   self.fold_check_cancel();await self.refresh();s=self.fold_ready();self.fold_check_carrier(s,p)
   if max(abs(s['aq']-np.array(p['start_aq'])))>.005:raise ValueError('机械臂姿态已改变，请重新点击折叠')
   samples=await self.fold_path(s,target)
   await self.refresh();s=self.fold_ready();self.fold_check_cancel();self.fold_check_carrier(s,p)
   if max(abs(s['aq']-np.array(p['start_aq'])))>.005:raise ValueError('路径检查期间机械臂姿态改变，未发送折叠动作')
   await self.armcmd('heartbeat');self.fold_check_cancel()
   started=True;await self.armcmd('move',kind='joint',joints_deg=np.rad2deg(target).tolist(),speed=5)
   begin=time.monotonic();stable=None
   while time.monotonic()-begin<40:
    self.fold_check_cancel();await self.refresh();self.fold_check_cancel()
    s=self.fold_ready(moving=True);self.fold_check_carrier(s,p);self.fold_contacts(s,s['aq'])
    await self.armcmd('heartbeat');now=time.monotonic();error=float(max(abs(s['aq']-target)))
    settled=error<.008 and max(abs(s['aqd']))<.05
    stable=(stable if stable is not None else now) if settled else None
    trace.append({'t':now-begin,'q':s['aq'].tolist(),'error_deg':float(np.rad2deg(error))})
    self.result={'phase':'folding','operation':'fold','fold_id':p.get('fold_id'),'message':'机械臂以 5% 速度折叠，机器狗保持当前运动模式','joint_error_deg':float(np.rad2deg(error))}
    if stable is not None and now-stable>=.5:
     self.result.update(phase='folded',message='机械臂已到折叠姿态；机器狗模式和夹爪开合未改变');break
    await asyncio.sleep(.08)
   else:raise ControlFault('FOLD_TIMEOUT','机械臂折叠 40 秒内未到位，已停止')
  except Exception as exc:
   self.result={'phase':'stopped','operation':'fold','fold_id':p.get('fold_id'),'message':str(exc),**getattr(exc,'detail',{})}
  finally:
   if started:
    try:await self.armcmd('stop')
    except Exception as exc:self.result.update(phase='stopped',stop_error=str(exc),message='机械臂停止未确认：'+str(exc))
   self.plan=None;self.standby_valid=False;self.logdir.mkdir(exist_ok=True,parents=True)
   (self.logdir/(time.strftime('%Y%m%d_%H%M%S')+'_fold.json')).write_text(json.dumps({'start':p,'target_rad':target.tolist(),'speed_percent':5,'samples':samples,'result':self.result,'trace':trace},ensure_ascii=False))
