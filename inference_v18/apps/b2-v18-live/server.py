"""V18 real-sensor joint manager. Startup and preview never send motion."""
import asyncio,contextlib,json,secrets,time,uuid
from pathlib import Path
import aiohttp
from aiohttp import web
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from policy import Manager,PhysicalGeometry,observe,commands,SHA,NOMINAL
from teleop import KeyboardGoal,ControlFault
from folding import FoldArmMixin
ROOT=Path(__file__).resolve().parent

def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def clean(x):return json.loads(json.dumps(x,default=serial))

class Portal(FoldArmMixin):
 def __init__(self):
  self.g=PhysicalGeometry();self.model=Manager();self.token=secrets.token_urlsafe(32);self.client='v18-'+uuid.uuid4().hex
  self.http=None;self.dog=None;self.arm=None;self.ref=None;self.error='等待真实反馈';self.stamp=0;self.plan=None;self.job=None;self.owner=None;self.beats={};self.cancel=False;self.seq=0;self.result={'phase':'idle','message':'预览不运动，确认执行后才启动联合策略'};self.arm_token=None;self.dog_token=None;self.latency=[];self.keyboard=KeyboardGoal();self.control_epoch=0;self.logdir=ROOT/'logs'
  self.standby_valid=False;self.remote_generation=None;self.remote_blocked=False
 async def req(self,url,body=None,headers=None,timeout=.7):
  async with self.http.request('GET' if body is None else 'POST',url,json=body,headers=headers,timeout=aiohttp.ClientTimeout(total=timeout)) as r:
   d=await r.json()
   if r.status>=400 or d.get('ok') is False:raise ValueError(d.get('error',str(r.status)))
   return d
 async def refresh(self):
  begin=time.monotonic()
  d,a=await asyncio.gather(self.req('http://127.0.0.1:18865/api/status',headers={'Host':'127.0.0.1:8765'}),self.req('http://127.0.0.1:8879/api/status'))
  self.dog_token=d.pop('csrf');self.dog=d;self.arm=a
  self.observe_remote()
  if not a['fresh'] or a['age_ms']>150:raise ValueError('机械臂反馈过期')
  if a.get('debug') is not False:raise ValueError('机械臂反馈不是实时硬件模式')
  p=self.g.measured(d)
  if not self.ref:self.ref={'position':p['position'],'yaw':p['rpy'][2],'created':time.time()}
  if len(d.get('policy_context') or [])!=17:raise ValueError('P4 实际 17 维状态上下文未就绪')
  self.g.state(d,a,self.ref)
  self.stamp=time.monotonic();self.error='';self.latency=(self.latency+[self.stamp-begin])[-100:]
 async def poll(self):
  while True:
   if not self.busy():
    try:await self.refresh()
    except Exception as e:self.error=str(e)
    if self.dog and (self.dog.get('active')!='trained' or (self.dog.get('joint_manager',{}).get('engaged') and self.dog['joint_manager'].get('client')!=self.client)):self.standby_valid=False
    if self.keyboard.enabled and time.monotonic()-self.keyboard.stamp>.35:
     self.keyboard.disable();self.result={'phase':'stopped','message':'按键连接中断，请重新启用键盘','code':'KEYBOARD_STALE'}
   await asyncio.sleep(.05)
 def remote_status(self):
  value=(self.dog or {}).get('remote_priority')
  if not isinstance(value,dict):return {'blocked':True,'reason':'等待狗端遥控器优先状态','generation':None}
  return value
 def observe_remote(self):
  value=self.remote_status();generation=value.get('generation');blocked=value.get('blocked',True)
  changed=self.remote_generation is not None and generation!=self.remote_generation
  if changed or (blocked and not self.remote_blocked):
   self.control_epoch+=1;self.cancel=True;self.plan=None;self.keyboard.disable()
   if not self.busy():self.result={'phase':'paused','code':'REMOTE_PRIORITY','message':value.get('reason') or '遥控器已接管，旧目标已取消'}
  self.remote_generation=generation;self.remote_blocked=blocked
 def require_remote_idle(self):
  value=self.remote_status()
  if value.get('blocked',True):raise ControlFault('REMOTE_PRIORITY',value.get('reason') or '遥控器优先，其他操作暂时暂停')
 def joint_available(self):
  jm=(self.dog or {}).get('joint_manager',{})
  return bool(self.standby_valid and self.dog and self.dog.get('active')=='trained' and (not jm.get('engaged') or jm.get('client')==self.client))
 def busy(self):return bool(self.job and not self.job.done())
 def fresh(self):
  if self.error or time.monotonic()-self.stamp>.15:raise ValueError(self.error or '反馈过期')
 def state(self):return self.g.state(self.dog,self.arm,self.ref)
 def joint_engaged(self):
  jm=(self.dog or {}).get('joint_manager',{})
  return bool(self.dog and self.dog.get('active')=='trained' and jm.get('engaged') and jm.get('client')==self.client)
 def require_joint(self):
  if not self.joint_available():raise ControlFault('JOINT_STANDBY_REQUIRED','请先点击“进入联合待命”，完成机械臂准备和 P4 站立切换')
 def preparation_ready(self,mode="native"):
  self.require_remote_idle();self.fresh();a=self.arm;d=self.dog
  if d['active']!=mode or not d['gate']['switch_allowed']:raise ValueError('进入联合待命前，需要'+('P4 零速' if mode=='trained' else '原生策略')+'稳定站立、遥控器回中')
  if mode=='trained':self.require_p4_zero()
  remote=d.get('remote',{})
  if remote.get('buttons') or any(abs(v)>.05 for v in remote.get('axes',{}).values()):raise ValueError('进入联合待命前，请松开遥控器按键并将摇杆回中')
  if d['model_info']['models']['locomotion']['weights_sha256']!='e7be320b2e2486f454dcfdb7a292d9349d4e1671c02a240efb4be40754f9462b':raise ValueError('狗端不是配套 B2-P4')
  jm=d.get('joint_manager',{})
  if (jm.get('engaged') and not (mode=='trained' and jm.get('client')==self.client)) or d.get('stationary_reach',{}).get('engaged'):raise ValueError('其他控制会话正在占用机体')
  if a['active'] or a['state']['task']['exe_flag'] or a.get('error') not in (0,None):raise ValueError('机械臂正在执行或存在错误')
  if a.get('debug') is not False or a['state']['servo']!=1 or a['state']['fsm_state']!='POSITION':raise ValueError('机械臂需要真实反馈与已使能的位置模式')
  if not np.allclose(a['tool_coordinate'],[0,0,0,0,0,0,1],atol=1e-8):raise ValueError('机械臂工具坐标已改变')
  s=self.state()
  if np.linalg.norm(s['v'][:2])>.05 or max(abs(s['aqd']))>.05:raise ValueError('准备前需要机体与机械臂静止')
  if not np.isfinite(s['aq']).all() or np.any(s['aq']<self.g.arm_lo) or np.any(s['aq']>self.g.arm_hi):raise ValueError('当前机械臂反馈超出关节范围')
  return s
 def require_p4_zero(self):
  s=self.state();d=self.dog;c=d.get('command',[])
  if d.get('active')!='trained' or len(c)<3 or not np.isfinite(c).all() or max(abs(float(v)) for v in c[:3])>1e-4:raise ValueError('需要 P4 零平移、零转向指令')
  if np.linalg.norm(s['v'][:2])>.05 or np.linalg.norm(s['gyro'])>.15:raise ValueError('P4 尚未零速稳定，请等待机体静止')
 def check_cancel(self):
  self.require_remote_idle()
  if self.cancel or time.monotonic()-self.beats.get(self.owner,0)>2:raise ControlFault('PREPARATION_CANCELLED','准备已取消或页面连接中断')
 async def dogcmd(self,op,**kw):
  return await self.req('http://127.0.0.1:18865/api/command',{'id':uuid.uuid4().hex,'action':'reach_v17','value':{'op':op,'client':self.client,**kw}},headers={'Host':'127.0.0.1:8765','Origin':'http://127.0.0.1:8765','X-B2-CSRF':self.dog_token},timeout=5 if op=='begin' else .7)
 async def armcmd(self,op,**kw):
  epoch=self.control_epoch
  if self.arm_token is None:self.arm_token=(await self.req('http://127.0.0.1:8879/api/session'))['token']
  if op not in ('stop','heartbeat','stream_hold'):
   self.require_remote_idle()
   if epoch!=self.control_epoch:raise ControlFault('REMOTE_PRIORITY','控制权已变化，待发机械臂命令已取消')
  return await self.req('http://127.0.0.1:8879/api/action',{'op':op,'client':self.client,**kw},headers={'X-Arm-Token':self.arm_token},timeout=5 if op in ('move','gripper_set') else .7)
 def ready(self):
  self.require_remote_idle();self.fresh();a=self.arm
  if a.get('debug') is not False:raise ValueError('机械臂未处于真实反馈模式')
  if self.dog['model_info']['models']['locomotion']['weights_sha256']!='e7be320b2e2486f454dcfdb7a292d9349d4e1671c02a240efb4be40754f9462b':raise ValueError('狗端不是配套 B2-P4')
  if a['active'] or a['state']['task']['exe_flag']:raise ValueError('机械臂正在被其他任务占用')
  if a['state']['servo']!=1 or a['state']['fsm_state']!='POSITION':raise ValueError('机械臂需要已使能 POSITION 模式')
  if a.get('error') not in (0,None):raise ValueError('机械臂存在控制器错误')
  if not np.allclose(a['tool_coordinate'],[0,0,0,0,0,0,1],atol=1e-8):raise ValueError('机械臂工具坐标已改变')
  if self.dog.get('stationary_reach',{}).get('engaged'):raise ValueError('原地选点仍在占用狗端桥')
  jm=self.dog.get('joint_manager',{})
  if jm.get('engaged') and jm.get('client')!=self.client:raise ValueError('其他联合策略会话正在执行')
  s=self.state();valid,clear,ground=self.g.details(s,s['aq'])
  if not valid:
   issues=self.g.violations(s,s['aq'],clear,ground);raise ControlFault('PHYSICAL_CONSTRAINT','当前姿态不满足策略起始条件：'+issues[0]['message'],issues)
  if np.linalg.norm(s['v'][:2])>.08 or max(abs(s['aqd']))>.05:raise ValueError('开始前需要机体与机械臂静止')
  return s
 def snapshot(self):
  current=None
  with contextlib.suppress(Exception):
   s=self.state();current={'ee':s['ee'],'quaternion_xyzw':Rot.from_matrix(s['eerot']).as_quat(),'rpy_deg':Rot.from_matrix(s['eerot']).as_euler('xyz',degrees=True),'body_xyz':s['pos'],'body_rpy_deg':Rot.from_matrix(s['rot']).as_euler('xyz',degrees=True),'arm_deg':np.rad2deg(s['aq']),'joint_limits_deg':np.rad2deg(np.stack([self.g.arm_lo,self.g.arm_hi],axis=1)),'speed':np.linalg.norm(s['v'][:2])}
  start_error=None;start_fault=None
  if not self.busy():
   try:self.ready()
   except Exception as e:
    start_error=str(e);start_fault=getattr(e,'detail',None)
  standby_error=None
  fold_error=None
  p4_standby_error=None
  if not self.busy():
   try:self.preparation_ready()
   except Exception as e:standby_error=str(e)
   try:self.preparation_ready('trained')
   except Exception as e:p4_standby_error=str(e)
   try:self.fold_ready()
   except Exception as e:fold_error=str(e)
  return clean({'error':self.error,'start_error':start_error,'start_fault':start_fault,'joint_engaged':self.joint_engaged(),'joint_ready':self.joint_available(),'remote_priority':self.remote_status(),'standby_error':standby_error,'p4_standby_error':p4_standby_error,'fold_error':fold_error,'keyboard':self.keyboard.snapshot(),'gripper':(self.arm or {}).get('gripper_control'),'current':current,'busy':self.busy(),'job':self.result,'plan':self.plan,'reference':self.ref,'dog_mode':self.dog.get('active') if self.dog else None,'model':'V18 best24500 + frozen B2-P4','checkpoint_sha256':SHA,'mount':'当前 B02/B03/B10/B11；v18 已按当前安装位置训练','sensor_source':'独立里程计 + IMU + 关节编码器 + 当前安装几何；不是 GT','read_latency_ms':round(np.mean(self.latency)*1000,2) if self.latency else None})
 def preparation(self,s,mode="native"):
  self.preparation_ready(mode)
  home=np.array(self.g.meta['arm_home'])
  for t in np.linspace(0,1,max(241,int(np.max(abs(home-s['aq']))/.005)+1)):
   self.g.set(s['body'],s['aq']*(1-t)+home*t,s['q'],s['pos'][:2])
   for c in self.g.d.contact:
    x,y=int(c.geom1),int(c.geom2)
    if (self.g.m.geom_bodyid[x]==0 and y in self.g.feet) or (self.g.m.geom_bodyid[y]==0 and x in self.g.feet):continue
    if c.dist<0:raise ValueError('准备路径存在模型碰撞，未发送动作')
  if not self.g.details(s,home)[0]:raise ValueError('准备终点不满足训练几何条件')
  return home
 async def preview_prepare(self):
  self.plan=None
  await self.refresh();s=self.state();home=self.preparation(s)
  self.plan=clean({'kind':'prepare','id':uuid.uuid4().hex,'created':time.monotonic(),'start_aq':s['aq'],'start_pos':s['pos'],'start_rot':s['rot'],'home':home,'notice':'准备机械臂至训练初姿；狗保持原生站立。5%速度，已采样检查模型碰撞；不含外部障碍物。'})
  return {'plan':self.plan}
 async def run_prepare(self,p):
  started=False
  try:
   mode=p.get("start_mode","native")
   await self.refresh();s=self.state();home=self.preparation(s,mode)
   self.check_cancel()
   await self.armcmd('heartbeat')
   self.check_cancel()
   started=True
   await self.armcmd('move',kind='joint',joints_deg=np.rad2deg(home).tolist(),speed=5)
   t0=time.monotonic();stable=None
   while time.monotonic()-t0<40:
    self.check_cancel()
    await self.refresh();s=self.state()
    remote=self.dog.get('remote',{})
    if remote.get('buttons') or any(abs(v)>.05 for v in remote.get('axes',{}).values()):raise ValueError('检测到遥控器输入，停止机械臂准备')
    if mode=='trained':self.require_p4_zero()
    if self.dog['active']!=mode or np.linalg.norm(s['pos'][:2]-np.array(p['start_pos'])[:2])>.015 or Rot.from_matrix(s['rot'].T@np.array(p['start_rot'])).magnitude()>.04:raise ValueError('机体移动，停止机械臂准备')
    if self.arm.get('error') not in (0,None) or self.arm['state']['servo']!=1:raise ValueError('机械臂控制器状态异常')
    await self.armcmd('heartbeat');e=float(np.max(abs(s['aq']-home)))
    stable=(stable or time.monotonic()) if e<.008 and max(abs(s['aqd']))<.03 else None
    self.result={'phase':'preparing','operation':p.get('kind'),'message':'1/2 机械臂低速进入待命，机器狗保持'+('P4 零速站立' if mode=='trained' else '原生站立'),'joint_error_deg':float(np.rad2deg(e))}
    if stable and time.monotonic()-stable>.5:
     self.result={'phase':'prepared','message':'机械臂准备完成；现在可预览联合目标'};break
    await asyncio.sleep(.08)
   else:raise ValueError('机械臂准备超时')
  except Exception as e:self.result={'phase':'stopped','message':str(e)}
  finally:
   if started:
    try:await self.armcmd('stop')
    except Exception as e:self.result['stop_error']=str(e)
   self.plan=None
 async def run_standby(self,p):
  dog_started=False;home=np.array(self.g.meta['arm_home']);trace=[];self.standby_valid=False
  try:
   mode=p.get('start_mode','native')
   if mode=='trained':
    await self.refresh();self.preparation_ready(mode);self.check_cancel()
    if self.joint_engaged():await self.dogcmd('hold');await self.dogcmd('release')
   # Preserve the selected standing controller during arm preparation.
   await self.run_prepare(p)
   if self.result.get('phase')!='prepared' or self.result.get('stop_error'):
    raise ControlFault('ARM_PREPARATION_FAILED',self.result.get('stop_error') or self.result.get('message','机械臂准备未完成'))
   self.check_cancel();await self.refresh();s=self.ready();self.check_cancel()
   if self.dog['active']!=mode or not self.dog['gate']['switch_allowed']:raise ValueError('机械臂准备后机体模式或稳定状态改变，未接管')
   if mode=='trained':self.require_p4_zero()
   if max(abs(s['aq']-home))>.008:raise ValueError('机械臂离开待命位置，未切换 P4')
   self.result={'phase':'switching','operation':'standby','message':('2/2 保持 P4，建立联合控制会话并确认稳定' if mode=='trained' else '2/2 切换 P4，零平移 / 零转向站立，等待稳定')}
   dog_started=True;await self.dogcmd('begin');self.check_cancel()
   tick=time.monotonic();stable=None;seq=0
   while time.monotonic()-tick<12:
    t0=time.monotonic();self.check_cancel()
    await self.refresh();self.check_cancel();s=self.state();a=self.arm;jm=self.dog.get('joint_manager',{})
    if not self.joint_engaged() or jm.get('fault'):raise ControlFault('DOG_CONTROL_EXIT',jm.get('fault') or 'P4 未接管或已被遥控器接管')
    if a['active'] or a['state']['task']['exe_flag'] or a['state']['servo']!=1 or a['state']['fsm_state']!='POSITION' or a.get('error') not in (0,None):raise ValueError('机械臂未保持使能待命状态')
    delta=abs(s['aq']-home);j=int(np.argmax(delta))
    if delta[j]>.04:raise ControlFault('ARM_STANDBY_DRIFT',f'J{j+1} 离开待命位置 {np.rad2deg(delta[j]):.2f}°', [{'joint':f'J{j+1}','code':'ARM_STANDBY_DRIFT','message':f'J{j+1} 待命位置偏差 {np.rad2deg(delta[j]):.2f}°'}])
    valid,clear,ground=self.g.details(s,s['aq'])
    if not valid:
     issues=self.g.violations(s,s['aq'],clear,ground);raise ControlFault('PHYSICAL_CONSTRAINT','P4 切换期间几何约束不满足：'+issues[0]['message'],issues)
    if s['pos'][2]<.25 or max(np.abs(s['body'][:2]))>np.deg2rad(55):raise ValueError('机体姿态触发原物理终止条件')
    # P4 handles its own original posture blending. No V18 goal/Actor inference here.
    await self.dogcmd('frame',seq=seq,command=[0.]*6);seq+=1
    settled=np.linalg.norm(s['v'][:2])<.08 and np.linalg.norm(s['gyro'])<.15 and max(np.abs(s['body'][:2]))<np.deg2rad(10) and abs(s['pos'][2]-NOMINAL)<.035 and max(abs(s['aqd']))<.05 and max(delta)<.008
    stable=(stable or t0) if settled else None
    self.result={'phase':'settling','operation':'standby','message':'2/2 P4 已接管，等待机体和机械臂稳定','stable_s':t0-stable if stable else 0,'height_m':float(s['pos'][2]),'speed_m_s':float(np.linalg.norm(s['v'][:2]))}
    trace.append(clean({'t':t0-tick,'pos':s['pos'],'aq':s['aq'],'settled':bool(settled)}))
    if stable and t0-stable>=1:
     self.result={'phase':'standby','operation':'standby','message':'联合待命就绪：P4 零速站立，机械臂已在待命位置；现在可启用 EE 键盘控制'};break
    if time.monotonic()-t0>.1:raise ValueError('待命控制帧超过 100 ms，停止切换流程')
    await asyncio.sleep(max(0,.02-(time.monotonic()-t0)))
   else:raise ControlFault('STANDBY_NOT_SETTLED','P4 已接管，但 12 秒内未确认稳定待命；已停止，未开放 EE 控制')
  except Exception as exc:self.result={'phase':'stopped','operation':'standby','message':str(exc),**getattr(exc,'detail',{})}
  finally:
   if dog_started:
    try:await self.dogcmd('hold')
    except Exception as e:self.result.update(phase='stopped',stop_error=str(e),message='P4 保持指令未确认：'+str(e))
   self.standby_valid=self.result.get('phase')=='standby' and not self.cancel
   self.plan=None;self.logdir.mkdir(exist_ok=True,parents=True)
   (self.logdir/(time.strftime('%Y%m%d_%H%M%S')+'_standby.json')).write_text(json.dumps({'goal':p,'result':self.result,'trace':trace},ensure_ascii=False))
 async def preview(self,b):
  self.plan=None
  await self.refresh();s=self.ready();xyz=np.asarray(b.get('xyz'),float);rpy=np.asarray(b.get('rpy_deg'),float)
  if xyz.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(np.r_[xyz,rpy]).all():raise ValueError('需要有限 XYZ 与三个朝向角')
  if np.linalg.norm(xyz[:2])>3 or not .1<=xyz[2]<=1.6:raise ValueError('目标限定参考原点水平 3 m 内，高度 0.1–1.6 m')
  gr=Rot.from_euler('xyz',rpy,degrees=True).as_matrix();u=self.model(observe(s,self.dog['policy_context'],xyz,gr,np.zeros(12),s['aq'],np.zeros(6)))
  c=commands(u);t,v,allowed=self.g.project(s,u,s['aq'],np.zeros(6),c)
  self.plan=clean({'id':uuid.uuid4().hex,'created':time.monotonic(),'xyz':xyz,'rpy_deg':rpy,'start_aq':s['aq'],'start_pos':s['pos'],'start_rot':s['rot'],'first_body_command':c,'first_arm_delta_deg':np.rad2deg(t-s['aq']),'arm_projection_allowed':allowed,'notice':'仅校验输入、当前几何和策略首帧；不保证全轨迹可达。确认后会真实移动。'})
  return {'plan':self.plan}
 async def stop(self):
  self.cancel=True;self.control_epoch+=1;self.keyboard.disable()
  results=await asyncio.gather(self.armcmd('stop'),self.dogcmd('hold'),return_exceptions=True)
  return [str(e) for e in results if isinstance(e,Exception)]
 async def run(self,p):
  arm_started=False;dog_started=False;trace=[];tick=None;hold=None;follow=None;durations=[];warnings=[];e=None;ang=None
  keyboard=p.get('kind')=='keyboard'
  try:
   await self.refresh();s=self.ready();last=np.zeros(12);target=s['aq'].copy();velocity=np.zeros(6);goal=np.array(p['xyz']);gr=Rot.from_quat(p['quaternion_xyzw']).as_matrix() if keyboard else Rot.from_euler('xyz',p['rpy_deg'],degrees=True).as_matrix()
   if self.cancel or time.monotonic()-self.beats.get(self.owner,0)>2:raise ValueError('已取消或页面已断开')
   if keyboard:self.keyboard.advance(s)
   jm=self.dog.get('joint_manager',{})
   if jm.get('engaged'):await self.dogcmd('release')
   if self.cancel:raise ValueError('用户已停止')
   dog_started=True;await self.dogcmd('begin')
   if self.cancel:raise ValueError('用户已停止')
   if keyboard:self.keyboard.advance(s)
   arm_started=True;begin=await self.armcmd('stream_begin')
   # Use the receiver's actual session origin, not a preceding telemetry sample.
   if isinstance(begin,dict) and 'target_rad' in begin:
    target=np.array(begin['target_rad']);velocity=np.array(begin['velocity_rad_s'])
   tick=time.monotonic();last_change=tick;self.seq=0
   while True:
    t0=time.monotonic()
    if self.cancel:raise ValueError('用户已停止')
    if t0-self.beats.get(self.owner,0)>2:raise ControlFault('PAGE_STALE','页面心跳中断，停止联合动作')
    await self.refresh();self.require_remote_idle();s=self.state();a=self.arm;d=self.dog;jm=d.get('joint_manager',{})
    if d.get('active')!='trained' or not jm.get('engaged') or jm.get('client')!=self.client or jm.get('fault'):raise ControlFault('DOG_CONTROL_EXIT',jm.get('fault') or '狗端已退出联合策略接管')
    if not a.get('stream',{}).get('engaged') or a['stream']['client']!=self.client or a.get('error') not in (0,None):raise ControlFault('ARM_CONTROL_EXIT','机械臂跟随已停止或控制器错误')
    if s['pos'][2]<.25 or max(np.abs(s['body'][:2]))>np.deg2rad(55):raise ControlFault('BODY_LIMIT','机体高度或倾角触发原物理终止条件')
    valid,clear,ground=self.g.details(s,s['aq'])
    if not valid:
     issues=self.g.violations(s,s['aq'],clear,ground);raise ControlFault('PHYSICAL_CONSTRAINT','实际几何不满足策略条件：'+issues[0]['message'],issues)
    if np.linalg.norm(s['pos'][:2])>4:raise ControlFault('BODY_WORKSPACE','机体已离开当前目标工作区域')
    if keyboard:
     changed=self.keyboard.advance(s)
     goal=self.keyboard.position.copy();gr=self.keyboard.rotation.copy()
     if changed:last_change=t0;hold=None
    e=float(np.linalg.norm(goal-s['ee']));ang=float(np.rad2deg(Rot.from_matrix(s['eerot'].T@gr).magnitude()))
    hold=(hold or t0) if e<=.01 and ang<=10 else None
    if hold and t0-hold>=1 and (not keyboard or not self.keyboard.keys):
     self.result={'phase':'complete','message':'传感器估计位置≤1 cm、朝向≤10°，同时保持1秒；尚非外部实测精度认证','error_m':e,'orientation_deg':ang};break
    if t0-(last_change if keyboard else tick)>=12:
     detail='；'.join(x['message'] for x in warnings[:3]) if warnings else '未检测到可归因的具体关节限位，不能据此证明目标无解'
     raise ControlFault('POLICY_NOT_CONVERGED',f'目标连续12秒未收敛：位置误差 {e*100:.2f} cm，朝向误差 {ang:.2f}°。'+detail,warnings)
    delta=abs(target-s['aq']);j=int(np.argmax(delta));following=float(delta[j]);follow=(follow or t0) if following>.12 else None
    if follow and t0-follow>.3:
     issue={'code':'JOINT_FOLLOWING_ERROR','joint':f'J{j+1}','actual_deg':float(np.rad2deg(s['aq'][j])),'target_deg':float(np.rad2deg(target[j])),'error_deg':float(np.rad2deg(following)),'message':f'J{j+1} 关节反馈未跟上指令，误差 {np.rad2deg(following):.2f}°（持续超出 6.88° 达 0.3 s）'}
     raise ControlFault(issue['code'],issue['message'],[issue])
    u=self.model(observe(s,d['policy_context'],goal,gr,last,target,velocity));c=commands(u);nt,nv,projected=self.g.project(s,u,target,velocity,c)
    warnings=self.g.last_projection_issues+self.g.saturation(s,u,nt)
    if self.cancel:raise ValueError('用户已停止')
    try:
     await asyncio.gather(self.dogcmd('frame',seq=self.seq,command=c.tolist()),self.armcmd('stream_frame',seq=self.seq,joints_rad=nt.tolist()))
    except Exception as frame_error:
     trace.append(clean({'t':t0-tick,'rejected':True,'seq':self.seq,'aq':s['aq'],'previous_target':target,'previous_velocity':velocity,'requested_target':nt,'requested_velocity':nv,'command':c,'error':str(frame_error)}))
     raise
    last=u;target=nt;velocity=nv;self.seq+=1;dt=time.monotonic()-t0;durations.append(dt)
    self.result={'phase':'running','message':'键盘 EE 目标 · 联合策略追踪中' if keyboard else 'V18 联合控制执行中','elapsed_s':time.monotonic()-tick,'error_m':e,'orientation_deg':ang,'frame':self.seq,'frame_compute_ms':dt*1000,'arm_projected':projected,'issues':warnings}
    trace.append(clean({'t':t0-tick,'goal':goal,'goal_quaternion_xyzw':Rot.from_matrix(gr).as_quat(),'pos':s['pos'],'ee':s['ee'],'aq':s['aq'],'target':target,'command':c,'error_m':e,'orientation_deg':ang,'duration':dt,'issues':warnings}))
    # Bounded in-memory trace for an intentionally long keyboard session.
    if len(trace)>3000:trace=trace[-2500:]
    durations=durations[-10:]
    if dt>.1:raise ControlFault('CONTROL_LATENCY','控制链路单帧超过100ms，停止联合运动')
    if len(durations)>=10 and np.mean(durations)>.03:raise ControlFault('CONTROL_RATE','控制链路不能维持策略频率，停止联合运动')
    await asyncio.sleep(max(0,.02-(time.monotonic()-t0)))
  except Exception as exc:
   self.result={'phase':'timeout' if getattr(exc,'detail',{}).get('code')=='POLICY_NOT_CONVERGED' else 'stopped','message':str(exc),**getattr(exc,'detail',{})}
   if e is not None:self.result.update(error_m=e,orientation_deg=ang)
   if keyboard:self.keyboard.disable()
  finally:
   outcomes=await asyncio.gather(*([self.armcmd('stop')] if arm_started else []),*([self.dogcmd('hold')] if dog_started else []),return_exceptions=True)
   errors=[str(x) for x in outcomes if isinstance(x,Exception)]
   if errors:self.result['stop_errors']=errors
   self.plan=None;folder=self.logdir;folder.mkdir(exist_ok=True,parents=True);(folder/(time.strftime('%Y%m%d_%H%M%S')+'.json')).write_text(json.dumps({'goal':p,'result':self.result,'trace':trace},ensure_ascii=False))
 async def action(self,b):
  op=b.get('op');browser=b.get('client')
  if not isinstance(browser,str) or not 8<=len(browser)<=100:raise ValueError('页面会话无效')
  if op=='heartbeat':self.beats={k:v for k,v in self.beats.items() if time.monotonic()-v<30};self.beats[browser]=time.monotonic();return {}
  if op=='stop':return {'stop_errors':await self.stop()}
  if op not in ('keyboard_disable','release'):self.require_remote_idle()
  if op=='gripper_set':
   if (self.busy() and self.owner!=browser) or (self.keyboard.enabled and self.keyboard.owner!=browser):raise ValueError('联合控制属于另一页面')
   if self.busy() and self.result.get('operation') in ('standby','fold'):raise ValueError('机械臂准备 / 折叠中，请等待完成再操作夹爪')
   if time.monotonic()-self.beats.get(browser,0)>2:raise ValueError('页面心跳过期，请重新操作夹爪')
   return await self.armcmd('gripper_set',id=b.get('id'),target=b.get('target'))
  if op=='keyboard_input':
   self.keyboard.input(browser,b.get('seq'),b.get('keys'));self.beats[browser]=time.monotonic()
   if self.keyboard.keys and not self.busy():
    self.require_joint()
    self.owner=browser;self.cancel=False;self.plan=None
    self.job=asyncio.create_task(self.run({'kind':'keyboard','xyz':self.keyboard.position.tolist(),'quaternion_xyzw':Rot.from_matrix(self.keyboard.rotation).as_quat().tolist()}))
   return {}
  if op=='keyboard_disable':
   if self.keyboard.owner not in (None,browser):raise ValueError('键盘控制属于另一个页面')
   active=self.keyboard.enabled;self.keyboard.disable();self.control_epoch+=1
   if active and self.busy():return {'stop_errors':await self.stop()}
   return {}
  if self.busy():raise ValueError('正在执行，请先停止')
  if op=='fold_arm':
   if self.keyboard.enabled:raise ValueError('请先关闭键盘控制，再折叠机械臂')
   epoch=self.control_epoch;await self.refresh();s=self.fold_ready()
   if epoch!=self.control_epoch or self.busy() or self.keyboard.enabled:raise ValueError('控制状态已改变，请重新点击折叠')
   if time.monotonic()-self.beats.get(browser,0)>2:raise ValueError('页面心跳过期')
   p=clean({'kind':'fold','fold_id':uuid.uuid4().hex,'start_pos':s['pos'],'start_rot':s['rot'],'start_aq':s['aq'],'dog_mode':self.dog['active']})
   self.owner=browser;self.cancel=False;self.plan=None;self.standby_valid=False
   self.result={'phase':'checking','operation':'fold','fold_id':p['fold_id'],'message':'正在检查到折叠姿态的机械臂路径；通过后以 5% 速度执行'}
   self.job=asyncio.create_task(self.run_fold(p));return {'fold_id':p['fold_id']}
  if op in ('enter_standby','enter_p4_standby'):
   if self.keyboard.enabled:raise ValueError('请先关闭键盘控制')
   mode='trained' if op=='enter_p4_standby' else 'native'
   epoch=self.control_epoch;await self.refresh();s=self.preparation_ready(mode)
   if epoch!=self.control_epoch or self.busy():raise ValueError('状态已改变，请重新进入联合待命')
   if time.monotonic()-self.beats.get(browser,0)>2:raise ValueError('页面心跳过期')
   p=clean({'kind':'standby','start_mode':mode,'start_pos':s['pos'],'start_rot':s['rot'],'start_aq':s['aq']})
   self.owner=browser;self.cancel=False;self.plan=None;self.standby_valid=False
   self.result={'phase':'preparing','operation':'standby','message':'正在检查机械臂待命路径；'+('保持 P4 零速站立' if mode=='trained' else '随后切换 P4 零速站立')}
   self.job=asyncio.create_task(self.run_standby(p));return {}
  if op=='keyboard_enable':
   epoch=self.control_epoch
   await self.refresh();s=self.ready();self.require_joint()
   if epoch!=self.control_epoch or self.busy():raise ValueError('状态已改变，请重新启用键盘')
   if self.keyboard.enabled and self.keyboard.owner!=browser:raise ValueError('另一页面已启用键盘')
   self.keyboard.enable(browser,s);self.owner=browser;self.cancel=False;self.plan=None;self.beats[browser]=time.monotonic()
   self.result={'phase':'keyboard_ready','message':'键盘已启用；首次按下映射键时启动联合运动。'}
   return {'keyboard':self.keyboard.snapshot()}
  if self.keyboard.enabled:raise ValueError('请先关闭键盘控制，再使用数值目标、准备或重置')
  if op=='release':
   await self.refresh()
   if self.dog.get('joint_manager',{}).get('client')==self.client:
    await self.dogcmd('hold');await self.dogcmd('release')
   self.plan=None;self.standby_valid=False;return {}
  if op=='preview_prepare':return await self.preview_prepare()
  if op=='preview':return await self.preview(b)
  if op=='reset':
   await self.refresh();p=self.g.measured(self.dog);self.ref={'position':p['position'],'yaw':p['rpy'][2],'created':time.time()};self.plan=None;return {}
  if op=='execute':
   epoch=self.control_epoch
   p=self.plan
   if not p or b.get('id')!=p['id'] or time.monotonic()-p['created']>120:raise ValueError('请重新预览目标')
   if p.get('kind')!='prepare':self.require_joint()
   await self.refresh()
   if epoch!=self.control_epoch:raise ControlFault('REMOTE_PRIORITY','控制权已变化，旧确认已取消，请重新预览')
   self.require_remote_idle();s=self.state() if p.get('kind')=='prepare' else self.ready()
   if np.max(abs(s['aq']-p['start_aq']))>.02 or np.linalg.norm(s['pos']-p['start_pos'])>.015 or Rot.from_matrix(s['rot'].T@np.array(p['start_rot'])).magnitude()>.03:raise ValueError('机器人姿态已改变，请重新预览')
   if time.monotonic()-self.beats.get(browser,0)>2:raise ValueError('页面心跳过期')
   self.owner=browser;self.cancel=False;self.job=asyncio.create_task(self.run_prepare(p) if p.get('kind')=='prepare' else self.run(p));return {}
  raise ValueError('未知操作')

def make_app(p):
 @web.middleware
 async def security(r,handler):
  if r.host not in ('127.0.0.1:8881','localhost:8881'):return web.json_response({'error':'仅限本机访问'},status=403)
  if r.method=='POST' and (r.headers.get('Origin') not in ('http://127.0.0.1:8881','http://localhost:8881') or not secrets.compare_digest(r.headers.get('X-V18-Token',''),p.token)):return web.json_response({'error':'会话失效，请刷新'},status=403)
  try:return await handler(r)
  except (ValueError,TypeError,KeyError) as e:return web.json_response({'error':str(e),**getattr(e,'detail',{})},status=409)
 app=web.Application(middlewares=[security],client_max_size=8192)
 async def index(r):return web.Response(text=(ROOT/'web/index.html').read_text().replace('__TOKEN__',p.token),content_type='text/html',headers={'Cache-Control':'no-store'})
 async def status(r):return web.json_response(p.snapshot())
 async def action(r):return web.json_response({'ok':True,**await p.action(await r.json())})
 app.router.add_get('/',index);app.router.add_get('/api/status',status);app.router.add_post('/api/action',action);app.router.add_static('/static/',ROOT/'web')
 async def lifecycle(app):
  async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=.7)) as h:
   p.http=h;poll=asyncio.create_task(p.poll());yield
   if p.busy():await p.stop();await asyncio.wait([p.job],timeout=2)
   poll.cancel();await asyncio.gather(poll,return_exceptions=True)
 app.cleanup_ctx.append(lifecycle);return app
if __name__=='__main__':web.run_app(make_app(Portal()),host='127.0.0.1',port=8881)
