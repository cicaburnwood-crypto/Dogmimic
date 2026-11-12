import unittest,time,asyncio
from unittest.mock import AsyncMock
import test_gripper
from remote_monitor import RemoteMonitor
class RemoteTests(unittest.IsolatedAsyncioTestCase):
 def arm(self):
  a=test_gripper.GripperTests.arm(self);a.remote_priority=RemoteMonitor(a,'unused');return a
 def dog(self,blocked=False,generation=0):return {'connected':True,'remote_priority':{'blocked':blocked,'fresh':True,'reason':'遥控器优先' if blocked else '', 'generation':generation}}
 async def test_every_motion_entry_is_blocked_even_before_await(self):
  a=self.arm();await a.remote_priority.update(self.dog(True,1))
  for op in ['move','enable','disable','reset','stream_begin','stream_frame','gripper_set']:
   with self.assertRaisesRegex(ValueError,'遥控器'):await a.action({'op':op,'client':'test-client','target':'open','id':'command-001'})
  self.assertFalse(a.sent)
 async def test_active_arm_and_gripper_stopped_once_no_auto_resume(self):
  a=self.arm();await a.remote_priority.update(self.dog());a.active={'stream':True,'client':'test-client'};a.gripper.last={'target_m':.04,'phase':'accepted','time':time.monotonic()};a.state['eeff']['eeff_pos']=[.01]
  a.stop=AsyncMock();await a.remote_priority.update(self.dog(True,1));a.stop.assert_awaited_once()
  self.assertEqual(a.sent,[{'command':'setEffectorCtr','pos':[.01],'vel':[0.],'tau':[5.],'control_motor':False,'_priority_stop':True}]);self.assertEqual(a.gripper.last['phase'],'paused')
  await a.remote_priority.update(self.dog(True,1));await a.remote_priority.update(self.dog(False,1));a.stop.assert_awaited_once();self.assertEqual(len(a.sent),1);a.remote_priority.require()
 async def test_missed_short_tap_cancels_old_epoch(self):
  a=self.arm();await a.remote_priority.update(self.dog());epoch=a.epoch
  await a.remote_priority.update(self.dog(False,2));self.assertGreater(a.epoch,epoch)
 async def test_stale_read_rejects_and_stop_always_available(self):
  a=self.arm();await a.remote_priority.update(self.dog());a.remote_priority.stamp-=1
  with self.assertRaisesRegex(ValueError,'中断'):a.require_remote_idle()
  await a.action({'op':'stop','client':'test-client'});self.assertEqual(a.sent[-1]['command'],'stopSignals')
 async def test_gripper_pending_ack_cannot_erase_interrupt(self):
  a=self.arm();await a.remote_priority.update(self.dog());started=asyncio.Event();release=asyncio.Event()
  original=a.request
  async def req(command,**kw):
   if command=='setEffectorCtr' and not kw.get('_priority_stop'):started.set();await release.wait()
   return await original(command,**kw)
  a.request=req;task=asyncio.create_task(a.action({'op':'gripper_set','client':'test-client','target':'open','id':'command-001'}));await started.wait()
  await a.remote_priority.update(self.dog(True,1));release.set();await task;self.assertEqual(a.gripper.last['phase'],'paused')
if __name__=='__main__':unittest.main()
