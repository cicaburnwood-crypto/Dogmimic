import sys,unittest,asyncio,time,threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'robot_console'),str(ROOT/'arm_console')]
from joint_manager_lease import JointManagerLease
from stream_control import StreamControl
class DogTests(unittest.TestCase):
 def setUp(self):
  self.clock=10.;ctx=np.zeros(17);ctx[13]=-.1
  self.e=SimpleNamespace(clock=lambda:self.clock,backend=SimpleNamespace(lock=threading.RLock(),controllers={'locomotion':SimpleNamespace(context=ctx)}),height_config={},active='trained',tilt_returning=False,tilt_handover=False,telemetry=SimpleNamespace(remote={'buttons':[],'axes':{'x':0.}}),stationary_reach=SimpleNamespace(engaged=False,pose=lambda:None,limit=lambda h:.35),_tilt_stable=lambda **kw:True)
  self.e.height_config={'enabled':True};self.l=JointManagerLease(self.e)
 def begin(self):self.l.request({'op':'begin','client':'test-manager'})
 def test_expiry_holds_actual_context(self):
  self.begin();self.l.request({'op':'frame','client':'test-manager','seq':1,'command':[1,0,.2,.1,.1,.05]});self.clock+=.16;c=self.l.command(self.clock);self.assertEqual(c,[0,0,0,0,0,-.1]);self.assertTrue(self.l.fault)
 def test_remote_override_and_duplicate(self):
  self.begin();v={'op':'frame','client':'test-manager','seq':0,'command':[0.]*6};self.l.request(v)
  with self.assertRaises(ValueError):self.l.request(v)
  self.e.telemetry.remote['axes']['x']=.3;self.assertEqual(self.l.command(self.clock)[:3],[0,0,0])
 def test_range_and_ownership(self):
  self.begin()
  for command in ([3,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,.2]):
   with self.assertRaises(ValueError):self.l.request({'op':'frame','client':'test-manager','seq':1,'command':command})
  with self.assertRaises(ValueError):self.l.request({'op':'hold','client':'other-client'})
class ArmTests(unittest.IsolatedAsyncioTestCase):
 def arm(self):
  a=SimpleNamespace(epoch=0,active=None,stamp=time.monotonic(),state={'servo':1,'fsm_state':'POSITION','reality':{'pose':[0.]*6,'vel':[0.]*6}},heartbeats={},ws=SimpleNamespace(send_json=AsyncMock()),fresh=lambda:None,healthy=lambda:None,idle=lambda:None,joints=lambda q:list(q),stop=AsyncMock());return a
 async def test_begin_sends_nothing_and_small_frame(self):
  a=self.arm();s=StreamControl(a);await s.action({'op':'stream_begin','client':'manager-test'});a.ws.send_json.assert_not_called();await s.action({'op':'stream_frame','client':'manager-test','seq':0,'joints_rad':[.00048]*6});packet=a.ws.send_json.call_args.args[0];self.assertEqual(packet['command'],'trajectoryTrackingTasks');self.assertNotIn('grp_point',packet['data'])
 async def test_big_step_never_sent(self):
  a=self.arm();s=StreamControl(a);await s.action({'op':'stream_begin','client':'manager-test'})
  with self.assertRaises(ValueError):await s.action({'op':'stream_frame','client':'manager-test','seq':0,'joints_rad':[.1]*6})
  a.ws.send_json.assert_not_called()
 async def test_stop_invalidates_epoch(self):
  a=self.arm();s=StreamControl(a);await s.action({'op':'stream_begin','client':'manager-test'});a.epoch+=1
  with self.assertRaises(ValueError):await s.action({'op':'stream_frame','client':'manager-test','seq':0,'joints_rad':[0.]*6})
 async def test_expiry_cannot_auto_resume(self):
  a=self.arm();s=StreamControl(a);await s.action({'op':'stream_begin','client':'manager-test'});s.deadline=time.monotonic()-1
  with self.assertRaises(ValueError):await s.action({'op':'stream_frame','client':'manager-test','seq':0,'joints_rad':[0.]*6})
  a.stop.assert_awaited_once();a.ws.send_json.assert_not_called()
if __name__=='__main__':unittest.main()
