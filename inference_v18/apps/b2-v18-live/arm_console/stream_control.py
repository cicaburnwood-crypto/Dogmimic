"""Bounded, short-lease access to CVTE SDK trajectoryTrackingTasks.
No automatic enable, mode change, retry or trajectory queue.
"""
import time,math,asyncio
class StreamControl:
 def __init__(self,arm):self.arm=arm;self.owner=None;self.seq=-1;self.deadline=0;self.target=None;self.velocity=[0.]*6;self.epoch=-1
 def engaged(self):return bool(self.arm.active and self.arm.active.get('stream') and self.epoch==self.arm.epoch)
 def status(self):return dict(engaged=self.engaged(),client=self.owner if self.engaged() else None,seq=self.seq,deadline_remaining=max(0,self.deadline-time.monotonic()))
 async def action(self,b):
  op=b['op'];client=b['client'];a=self.arm;a.fresh();a.healthy()
  if op=='stream_begin':
   a.idle()
   if a.state.get('servo')!=1 or a.state.get('fsm_state')!='POSITION':raise ValueError('需要已使能的位置控制模式')
   if max(abs(v) for v in a.state['reality']['vel'])>.05:raise ValueError('机械臂尚未静止')
   a.heartbeats[client]=time.monotonic();self.owner=client;self.seq=-1;self.epoch=a.epoch;self.target=list(a.state['reality']['pose']);self.velocity=[0.]*6;self.deadline=time.monotonic()+.5
   a.active={'stream':True,'client':client,'start':time.monotonic(),'seen':False,'target':self.target}
   return {**self.status(),'target_rad':list(self.target),'velocity_rad_s':list(self.velocity)}
  if not self.engaged() or client!=self.owner:raise ValueError('机械臂联合跟随会话已失效')
  if op=='stream_hold':await a.stop();return self.status()
  if op!='stream_frame':raise ValueError('未知跟随操作')
  if time.monotonic()>self.deadline:await a.stop();raise ValueError('跟随帧超时，不能自动恢复')
  if time.monotonic()-a.stamp>.15:await a.stop();raise ValueError('机械臂传感器反馈过期')
  if a.state.get('servo')!=1 or a.state.get('fsm_state')!='POSITION':await a.stop();raise ValueError('机械臂已退出位置控制')
  seq=b.get('seq');q=a.joints(b.get('joints_rad'))
  if type(seq)!=int or seq<=self.seq:raise ValueError('拒绝重复 / 倒序跟随帧')
  vel=[(x-y)/.02 for x,y in zip(q,self.target)]
  for j,(v,o) in enumerate(zip(vel,self.velocity)):
   if abs(v)>.6001:raise ValueError(f'J{j+1} 关节帧超过 V18 速度上限：{v:.6f} rad/s，上限0.6 rad/s，seq={seq}')
   if abs(v-o)>.0241 and abs(v)>1e-7:raise ValueError(f'J{j+1} 关节帧超过 V18 加速度上限：{(v-o)/.02:.6f} rad/s²，上限1.2 rad/s²，前速度{o:.6f}，新速度{v:.6f}，seq={seq}')
  # Exact Python SDK track_joint packet; it is fire-and-forget, not MOVJ.
  if hasattr(a,'require_remote_idle'):a.require_remote_idle()
  await a.ws.send_json({'command':'trajectoryTrackingTasks','task_id':'TASK_TRACKING','arm_index':0,'point_type':{'space':0},'data':{'way_point':q}})
  self.seq=seq;self.target=q;self.velocity=vel;self.deadline=time.monotonic()+.15;a.active['target']=q;a.heartbeats[client]=time.monotonic()
  return {'seq':seq,'fresh':True,'target_rad':list(self.target),'velocity_rad_s':list(self.velocity)}
 async def watchdog(self):
  while True:
   await asyncio.sleep(.02)
   if self.engaged() and (time.monotonic()>self.deadline or time.monotonic()-self.arm.stamp>.15):
    try:await self.arm.stop();self.arm.log('watchdog','V18 跟随租约 / 传感器过期，已请求停止')
    except Exception as e:self.arm.log('error','V18 停止未确认：'+str(e))
