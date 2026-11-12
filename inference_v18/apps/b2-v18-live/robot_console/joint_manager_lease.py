"""Expiring V17 body-command lease. Original P4 handles command dynamics."""
import math
class JointManagerLease:
 def __init__(self,e):self.e=e;self.engaged=False;self.client=None;self.seq=-1;self.until=0.;self.fault=None;self.cmd=[0.]*6
 def context(self):
  b=self.e.backend
  if not hasattr(b,"lock") or not hasattr(b,"controllers"):return None
  with b.lock:
   c=b.controllers.get('locomotion');x=getattr(c,'context',None)
   return x.tolist() if x is not None and len(x)==17 else None
 def freeze(self,reason):
  self.fault=reason;x=self.context()
  self.cmd=[0.,0.,0.,*(x[3:5] if x else self.cmd[3:5]),x[13] if x else self.cmd[5]]
 def release(self):
  if self.engaged:self.e.tilt_request=self.cmd[3:5];self.e.height_request=self.cmd[5]
  self.engaged=False;self.client=None
 def validate(self,c):
  if not isinstance(c,list) or len(c)!=6 or not all(type(v) in (int,float) and math.isfinite(v) for v in c):raise ValueError('V17 需要有限六维机体命令')
  x,y,w,r,p,h=c
  if abs(x)>2.00001 or abs(y)>1.00001 or math.hypot(x,y)>2.00001 or abs(w)>.50001 or not -.200001<=h<=.050001 or max(abs(r),abs(p))>self.e.stationary_reach.limit(h)+1e-6:raise ValueError('V17 命令超出训练包络')
  return list(c)
 def request(self,v):
  if not isinstance(v,dict) or set(v)-{'op','client','seq','command'}:raise ValueError('未知 V17 字段')
  op=v.get('op');client=v.get('client')
  if not isinstance(client,str) or not 8<=len(client)<=100:raise ValueError('无效 V17 会话')
  if op=='begin':
   if self.engaged:raise ValueError('已有 V17 会话，请先停止并释放')
   if self.e.stationary_reach.engaged:raise ValueError('原地选点仍占用控制桥')
   if not self.e.height_config or not self.e._tilt_stable(allow_tilt=self.e.active=='trained'):raise ValueError('仅稳定站立允许开始 V17')
   self.e.stationary_reach.pose()
   if self.e.telemetry.remote['buttons'] or any(abs(x)>.05 for x in self.e.telemetry.remote['axes'].values()):raise ValueError('遥控器需要回中并松开按键')
   if self.e.active=='native':self.e.selected='native';self.e.toggle_locomotion()
   if self.e.active!='trained' or self.e.tilt_returning or self.e.tilt_handover:raise ValueError('P4 接管未完成')
   self.engaged=True;self.client=client;self.seq=-1;self.fault=None;self.until=self.e.clock()+.5
   x=self.context();self.cmd=[0.,0.,0.,*x[3:5],x[13]]
  elif op in ('frame','hold','release'):
   if not self.engaged or self.client!=client:raise ValueError('V17 会话已失效或已被遥控器接管')
   if op=='hold':self.freeze('已停止联合策略，平移与转向归零，保持机体姿态')
   elif op=='release':self.release()
   else:
    if self.fault:raise ValueError(self.fault)
    seq=v.get('seq')
    if type(seq)!=int or seq<=self.seq:raise ValueError('拒绝重复 / 倒序 V17 帧')
    self.cmd=self.validate(v.get('command'));self.seq=seq;self.until=self.e.clock()+.15
  else:raise ValueError('未知 V17 操作')
  return self.status()
 def command(self,now):
  if not self.engaged:return None
  try:self.e.stationary_reach.pose()
  except ValueError as e:self.freeze(str(e))
  if now>self.until and not self.fault:self.freeze('V17 帧超时，停止平移和转向')
  return list(self.cmd)
 def status(self):return dict(engaged=self.engaged,client=self.client,seq=self.seq,fault=self.fault,command=self.cmd,translation_allowed=True,lease_seconds=.15)
