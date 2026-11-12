import asyncio
import copy
import time
import unittest
from unittest.mock import AsyncMock
from test_control import Fake

PARAMS={'dof':1,'eeff_name':'gripper_can','eeff_type':'gripper','eeff_lower':[0.], 'eeff_upper':[.077], 'eeff_tau':[100.]}
class GripperTests(unittest.IsolatedAsyncioTestCase):
 def arm(self):
  a=Fake();a.raw['debug']=False
  a.state['eeff']={'eeff_name':'gripper_can','eeff_type':'gripper','eeff_dof':1,'eeff_state':1,'is_connect':True,'eeff_pos':[0.], 'eeff_plan_pos':[0.]}
  a.gripper.params=copy.deepcopy(PARAMS)
  return a
 async def call(self,a,target='open',ident='command-001'):
  return await a.action({'op':'gripper_set','client':'test-client','target':target,'id':ident})
 async def test_only_effector_packet_and_fixed_limits(self):
  a=self.arm();await self.call(a)
  self.assertEqual(a.sent,[{'command':'setEffectorCtr','pos':[.04],'vel':[0.],'tau':[5.],'control_motor':False}])
  self.assertIsNone(a.active);self.assertEqual(a.epoch,0)
  self.assertFalse(a.gripper.snapshot()['next_open'])
  a.state['eeff']['eeff_pos']=[.04];self.assertEqual(a.gripper.snapshot()['phase'],'at_target')
  await self.call(a,'closed','command-002');self.assertEqual(a.sent[-1]['pos'],[0.]);self.assertTrue(a.gripper.snapshot()['next_open'])
 async def test_duplicate_and_changed_id_payload(self):
  a=self.arm();r=await self.call(a);self.assertEqual(await self.call(a),r);self.assertEqual(len(a.sent),1)
  with self.assertRaisesRegex(ValueError,'另一个目标'):await self.call(a,'closed')
  self.assertEqual(len(a.sent),1)
 async def test_stale_disconnected_debug_disabled_rejected(self):
  for key in ['stale','disconnect','debug','disabled','badpos','error','name']:
   a=self.arm()
   if key=='stale':a.stamp-=1
   if key=='disconnect':a.state['eeff']['is_connect']=False
   if key=='debug':a.raw['debug']=True
   if key=='disabled':a.state['eeff']['eeff_state']=0
   if key=='badpos':a.state['eeff']['eeff_pos']=[float('nan')]
   if key=='error':a.raw['error']=1
   if key=='name':a.state['eeff']['eeff_name']='another'
   with self.subTest(key=key):
    with self.assertRaises(ValueError):await self.call(a)
    self.assertFalse(a.sent)
 async def test_active_stream_same_owner_allowed_unchanged(self):
  a=self.arm();active={'stream':True,'client':'test-client','target':[.1]*6};a.active=active
  a.state['task']['exe_flag']=True
  await self.call(a);self.assertIs(a.active,active);self.assertEqual(a.epoch,0);self.assertEqual(a.stream.seq,-1)
 async def test_foreign_stream_and_joint_preparation_rejected(self):
  for active in [{'stream':True,'client':'another-owner'},{'client':'test-client','target':[0]*6}]:
   a=self.arm();a.active=active
   with self.assertRaises(ValueError):await self.call(a)
   self.assertFalse(a.sent)
 async def test_unknown_target_rejected(self):
  a=self.arm()
  for target in ['toggle',.09,None,True]:
   with self.assertRaises(ValueError):await self.call(a,target)
  self.assertFalse(a.sent)
 async def test_negative_ack_no_retry(self):
  a=self.arm();a.request=AsyncMock(return_value={'recv':'Task_Recieve','ret':False})
  for _ in range(2):
   with self.assertRaisesRegex(ValueError,'拒绝'):await self.call(a)
  a.request.assert_awaited_once();self.assertEqual(a.gripper.snapshot()['phase'],'unconfirmed')
 async def test_timeout_no_retry(self):
  a=self.arm();a.request=AsyncMock(side_effect=TimeoutError())
  for _ in range(2):
   with self.assertRaisesRegex(ValueError,'结果未知'):await self.call(a)
  a.request.assert_awaited_once()
 async def test_concurrent_command_not_queued(self):
  a=self.arm();started=asyncio.Event();release=asyncio.Event()
  async def request(*args,**kw):started.set();await release.wait();return {'recv':'Task_Recieve'}
  a.request=request;t=asyncio.create_task(self.call(a));await started.wait()
  with self.assertRaisesRegex(ValueError,'等待应答'):await self.call(a,'closed','command-002')
  release.set();await t
 async def test_real_params_checked(self):
  a=self.arm();a.request=AsyncMock(return_value={'recv':'Task_Recieve','params':PARAMS})
  await a.gripper.read_params();a.request.assert_awaited_once_with('getEeffParams')
  for field,bad in [('eeff_upper',[.03]),('eeff_lower',[.001]),('eeff_tau',[4.]),('eeff_upper',[float('nan')]),('dof',2)]:
   a.request=AsyncMock(return_value={'params':{**PARAMS,field:bad}})
   with self.assertRaises(ValueError):await a.gripper.read_params()
   self.assertIsNone(a.gripper.params)
 async def test_obstructed_close_next_opens_without_claim_grasp(self):
  a=self.arm();a.state['eeff']['eeff_pos']=[.03];a.state['eeff']['eeff_plan_pos']=[0.]
  self.assertTrue(a.gripper.snapshot()['next_open'])
  await self.call(a,'closed');a.gripper.last['time']-=6
  self.assertEqual(a.gripper.snapshot()['phase'],'not_reached')
if __name__=='__main__':unittest.main()
