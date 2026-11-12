import asyncio
import copy
import math
import sys
import time
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import Arm, vector, quat_from_rpy, rpy_from_quat, pose_error, make_app, load_token
from aiohttp.test_utils import TestServer, TestClient

class Fake(Arm):
 def __init__(self):
  super().__init__();self.connected=True;self.stamp=time.monotonic();self.sent=[];self.no_ik=False;self.bad_fk=False;self.hook=None
  self.raw={'error':0,'arm':[{'arm_dof':6,'arm_name':'A3_DM86_C','servo':1,'state':0,'fsm_state':'POSITION','tool':1,'task':{'exe_flag':False},'reality':{'pose':[0]*6,'vel':[0]*6}}]}
  self.limits={'limit_lower':[-1]*6,'limit_upper':[1]*6}
  self.heartbeats['test-client']=time.monotonic()
 def log(self,*args): pass
 async def request(self,command,**kw):
  self.sent.append(dict(command=command,**kw))
  if self.hook: await self.hook(command,kw)
  if command=='setServoEnable':self.state['servo']=int(kw['enable'])
  if command=='setControlMode':
   if kw.get('params')!={'level':100}: raise ValueError('Task_Reject: missing params.level')
   self.state['fsm_state']='POSITION'
  if command=='getKinematics':
   return {'recv':'Task_Recieve','data':({'point1':[.4 if self.bad_fk else .1,0,.2,0,0,0,1]} if kw['task_id']=='forward' else ({} if self.no_ik else {'joint1':[.1]*6}))}
  return {'recv':'Task_Recieve'}

class Tests(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self): self.a=Fake()
 async def call(self,op='move',**kw):return await self.a.action({'client':'test-client','op':op,'kind':'joint','joints_deg':[10]*6,**kw})
 async def test_joint_limit_rejects_no_motion(self):
  with self.assertRaises(ValueError):await self.call(joints_deg=[1000]*6)
  self.assertFalse(self.a.sent)
 async def test_nonfinite_rejected(self):
  for x in (float('nan'),float('inf'),True):
   with self.assertRaises(ValueError):await self.call(joints_deg=[x]*6)
  self.assertFalse(self.a.sent)
 async def test_stale_rejects(self):
  self.a.stamp-=1
  with self.assertRaises(ValueError):await self.call()
  self.assertFalse(self.a.sent)
 async def test_disabled_rejects(self):
  self.a.state['servo']=0
  with self.assertRaises(ValueError):await self.call()
  self.assertFalse(self.a.sent)
 async def test_single_axis_uses_live_other_axes(self):
  self.a.state['reality']['pose']=[.2]*6
  await self.call(joint_index=2)
  q=self.a.sent[-1]['data']['target_pos']
  self.assertEqual(q[:2]+q[3:],[.2]*5);self.assertAlmostEqual(q[2],math.radians(10))
 async def test_preview_only_kinematics(self):
  await self.call('preview')
  self.assertTrue(all(x['command']=='getKinematics' for x in self.a.sent));self.assertIsNone(self.a.active)
 async def test_ik_verifies_fk(self):
  await self.call(kind='ee',xyz_mm=[100,0,200],rpy_deg=[0,0,0])
  self.assertEqual(self.a.sent[-1]['task_id'],'TASK_MOVJ')
  self.assertEqual(self.a.sent[-1]['data']['tool'],1)
 async def test_ik_no_solution_rejects(self):
  self.a.no_ik=True
  with self.assertRaisesRegex(ValueError,'未找到'):await self.call(kind='ee',xyz_mm=[150,0,200],rpy_deg=[0,0,0])
  self.assertFalse(any(x['command']=='webRecieveTasks' for x in self.a.sent))
 async def test_exact_singular_current_pose(self):
  self.a.no_ik=True
  r=await self.call('preview',kind='ee',xyz_mm=[100,0,200],rpy_deg=[0,0,0])
  self.assertEqual(r['joints_rad'],[0]*6)
 async def test_ik_residual_rejects(self):
  self.a.bad_fk=True
  with self.assertRaisesRegex(ValueError,'残差'):await self.call(kind='ee',xyz_mm=[100,0,200],rpy_deg=[0,0,0])
  self.assertFalse(any(x['command']=='webRecieveTasks' for x in self.a.sent))
 async def test_stop_cancels_pending_motion(self):
  async def hook(cmd,kw):
   if cmd=='getKinematics':self.a.epoch+=1
  self.a.hook=hook
  with self.assertRaisesRegex(ValueError,'取消'):await self.call()
  self.assertFalse(any(x['command']=='webRecieveTasks' for x in self.a.sent))
 async def test_stop_bypasses_lock(self):
  async with self.a.lock:await self.call('stop')
  self.assertEqual(self.a.sent[-1]['stop_id'],'SIG_TASK_STOP')
 async def test_disable_bypasses_lock(self):
  async with self.a.lock:await self.call('disable')
  self.assertEqual(self.a.sent[-1],{'command':'setServoEnable','enable':False})
 async def test_heartbeat_stale_blocks_motion(self):
  self.a.heartbeats['test-client']=0
  with self.assertRaisesRegex(ValueError,'心跳'):await self.call()
  self.assertFalse(any(x['command']=='webRecieveTasks' for x in self.a.sent))
 async def test_watchdog_stops(self):
  await self.call();self.a.heartbeats['test-client']=0
  task=asyncio.create_task(self.a.watchdog());await asyncio.sleep(.15);task.cancel()
  await asyncio.gather(task,return_exceptions=True)
  self.assertEqual(self.a.sent[-1]['stop_id'],'SIG_TASK_STOP');self.assertIsNone(self.a.active)
 async def test_busy_refuses_queue(self):
  await self.call()
  with self.assertRaises(ValueError):await self.call()
  self.assertEqual(sum(x['command']=='webRecieveTasks' for x in self.a.sent),1)
 async def test_reset_in_error_not_enable(self):
  self.a.state['state']=-1;self.a.raw['error']=1
  await self.call('reset')
  self.assertEqual([x['command'] for x in self.a.sent],['setControllerErrorReset','stopSignals'])
 async def test_enable_no_motion(self):
  self.a.state['servo']=0;self.a.state['fsm_state']='IDLE'
  await self.call('enable')
  self.assertEqual([x['command'] for x in self.a.sent],['setControllerErrorReset','setServoEnable','setControlMode'])
  self.assertEqual(self.a.sent[-1]['params'],{'level':100})
 async def test_enable_canceled_after_prepare(self):
  async def hook(cmd,kw):
   if cmd=='setControllerErrorReset':self.a.epoch+=1
  self.a.hook=hook
  with self.assertRaisesRegex(ValueError,'取消'):await self.call('enable')
  self.assertEqual([x['command'] for x in self.a.sent],['setControllerErrorReset'])
 async def test_enable_prepare_rejection_stops_sequence(self):
  async def hook(cmd,kw):
   if cmd=='setControllerErrorReset':raise ValueError('controller rejected reset')
  self.a.hook=hook
  with self.assertRaises(ValueError):await self.call('enable')
  self.assertEqual([x['command'] for x in self.a.sent],['setControllerErrorReset'])
 async def test_already_enabled_does_not_repeat_servo(self):
  self.a.state['fsm_state']='IDLE'
  await self.call('enable')
  self.assertEqual([x['command'] for x in self.a.sent],['setControllerErrorReset','setControlMode'])
 async def test_rpy_roundtrip(self):
  for rpy in ([20,30,40],[-100,-89.3,-82],[180,0,0]):
   q=quat_from_rpy(rpy);q2=quat_from_rpy(rpy_from_quat(q));self.assertLess(pose_error([0,0,0]+q,[0,0,0]+q2)[1],.00001)
 async def test_token_survives_restart(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'token'
   self.assertEqual(load_token(p),load_token(p))
   self.assertEqual(p.stat().st_mode & 0o777,0o600)
 async def test_expired_session_not_executed(self):
  app=make_app(self.a);app.cleanup_ctx.clear()
  async with TestClient(TestServer(app)) as c:
   h={'Host':'127.0.0.1:8879','X-Arm-Token':'stale-token'}
   r=await c.post('/api/action',json={'op':'enable','client':'test-client'},headers=h)
   self.assertEqual(r.status,403);self.assertEqual((await r.json())['code'],'SESSION_EXPIRED')
   self.assertFalse(self.a.sent)
   r=await c.get('/api/session',headers={'Host':'127.0.0.1:8879'})
   self.assertEqual((await r.json())['token'],self.a.token)
   self.assertFalse(self.a.sent)  # Refresh never replays the rejected action.
 async def test_web_security(self):
  app=make_app(self.a);app.cleanup_ctx.clear()
  async with TestClient(TestServer(app)) as c:
   r=await c.get('/',headers={'Host':'evil.example'});self.assertEqual(r.status,403)
   r=await c.post('/api/action',json={'op':'enable'},headers={'Host':'127.0.0.1:8879'});self.assertEqual(r.status,403)
   r=await c.post('/api/action',json={'op':'enable'},headers={'Host':'127.0.0.1:8879','X-Arm-Token':self.a.token,'Origin':'http://evil.example'});self.assertEqual(r.status,403)
   self.assertFalse(self.a.sent)
if __name__=='__main__':unittest.main()
