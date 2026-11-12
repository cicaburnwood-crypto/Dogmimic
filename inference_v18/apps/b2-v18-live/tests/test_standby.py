import asyncio,sys,time,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server,test_portal
from teleop import ControlFault
class StandbyTests(unittest.IsolatedAsyncioTestCase):
 def setup_portal(self):
  p,s=test_portal.PortalTests.setup_portal(self);p.owner='browser-test';p.beats[p.owner]=time.monotonic();return p,s
 def plan(self,s):return {'kind':'standby','start_pos':s['pos'].tolist(),'start_rot':s['rot'].tolist(),'start_aq':s['aq'].tolist()}
 def mocks(self,p,s):
  events=[]
  async def prepare(plan):
   events.append('arm_prepared');s['aq']=np.array(p.g.meta['arm_home']);p.result={'phase':'prepared','message':'prepared'}
  async def dog(op,**kw):
   events.append(op)
   if op=='begin':
    self.assertEqual(events[0],'arm_prepared');p.dog['active']='trained';p.dog['joint_manager']={'engaged':True,'client':p.client,'fault':None}
   if op=='frame':self.assertEqual(kw['command'],[0.]*6)
  p.run_prepare=AsyncMock(side_effect=prepare);p.dogcmd=AsyncMock(side_effect=dog)
  s['pos'][2]=server.NOMINAL;s['v'][:]=0;s['gyro'][:]=0;s['aqd'][:]=0
  return events
 async def test_native_mode_cannot_enable_keyboard(self):
  p,s=self.setup_portal()
  with self.assertRaisesRegex(ControlFault,'联合待命'):await p.action({'op':'keyboard_enable','client':p.owner})
  p.dogcmd.assert_not_called();p.armcmd.assert_not_called()
 async def test_begin_standby_schedules_only_on_explicit_action(self):
  p,s=self.setup_portal();p.run_standby=AsyncMock()
  snap=p.snapshot();self.assertFalse(snap['joint_ready']);self.assertIsNone(snap['standby_error']);self.assertIsNone(p.job)
  await p.action({'op':'enter_standby','client':p.owner});await p.job
  self.assertEqual(p.run_standby.await_args.args[0]['kind'],'standby');p.dogcmd.assert_not_called();p.armcmd.assert_not_called()
 async def test_success_orders_arm_before_dog_and_only_zero_commands(self):
  p,s=self.setup_portal();events=self.mocks(p,s);p.model=lambda obs:(_ for _ in ()).throw(AssertionError('Standby must not run reach Actor'))
  await p.run_standby(self.plan(s))
  self.assertEqual(p.result['phase'],'standby',p.result);self.assertTrue(p.standby_valid)
  self.assertEqual(events[:2],['arm_prepared','begin']);self.assertEqual(events[-1],'hold');self.assertGreater(events.count('frame'),1)
  p.armcmd.assert_not_called();self.assertFalse(p.keyboard.enabled)
 async def test_horizontal_offset_over_ten_cm_is_not_a_standby_failure(self):
  p,s=self.setup_portal();self.mocks(p,s);original=p.dogcmd.side_effect
  async def offset(op,**kw):
   await original(op,**kw)
   if op=='begin':s['pos'][0]+=.15
  p.dogcmd.side_effect=offset;await p.run_standby(self.plan(s))
  self.assertEqual(p.result['phase'],'standby',p.result);self.assertTrue(p.standby_valid)
 async def test_arm_failure_never_switches_dog(self):
  p,s=self.setup_portal()
  async def fail(plan):p.result={'phase':'stopped','message':'J3 preparation error'}
  p.run_prepare=AsyncMock(side_effect=fail);await p.run_standby(self.plan(s))
  p.dogcmd.assert_not_called();self.assertFalse(p.standby_valid);self.assertIn('J3',p.result['message'])
 async def test_cancel_after_arm_preparation_never_switches_dog(self):
  p,s=self.setup_portal();self.mocks(p,s);p.cancel=True
  await p.run_standby(self.plan(s));p.dogcmd.assert_not_called();self.assertFalse(p.standby_valid)
 async def test_cancel_after_begin_sends_hold_without_frames(self):
  p,s=self.setup_portal();self.mocks(p,s)
  async def cancel(op,**kw):
   if op=='begin':p.cancel=True
  p.dogcmd=AsyncMock(side_effect=cancel);await p.run_standby(self.plan(s))
  self.assertEqual([c.args[0] for c in p.dogcmd.await_args_list],['begin','hold']);self.assertFalse(p.standby_valid)
 async def test_arm_drift_names_joint_and_holds_dog(self):
  p,s=self.setup_portal();self.mocks(p,s);original=p.dogcmd.side_effect
  async def drift(op,**kw):
   await original(op,**kw)
   if op=='begin':s['aq'][3]+=.1
  p.dogcmd.side_effect=drift;await p.run_standby(self.plan(s))
  self.assertEqual(p.result['code'],'ARM_STANDBY_DRIFT');self.assertEqual(p.result['issues'][0]['joint'],'J4');self.assertFalse(p.standby_valid);self.assertEqual(p.dogcmd.await_args.args[0],'hold')
 async def test_wrong_weights_or_tool_never_moves(self):
  p,s=self.setup_portal();p.dog['model_info']['models']['locomotion']['weights_sha256']='wrong'
  with self.assertRaisesRegex(ValueError,'P4'):await p.action({'op':'enter_standby','client':p.owner})
  p.dogcmd.assert_not_called();p.armcmd.assert_not_called();self.assertIsNone(p.job)
 async def test_remote_input_prevents_arm_and_dog_entry(self):
  p,s=self.setup_portal();p.dog['remote']={'buttons':['R2'],'axes':{'ly':0}}
  with self.assertRaisesRegex(ValueError,'遥控器'):await p.action({'op':'enter_standby','client':p.owner})
  p.armcmd.assert_not_called();p.dogcmd.assert_not_called();self.assertIsNone(p.job)
 async def test_unsettled_timeout_does_not_open_keyboard(self):
  p,s=self.setup_portal();self.mocks(p,s);s['gyro'][:]=.3
  value=[0.]
  def clock():value[0]+=.01;return value[0]
  fake=SimpleNamespace(monotonic=clock,strftime=time.strftime,time=time.time);p.beats[p.owner]=1000
  with patch.object(server,'time',fake):await p.run_standby(self.plan(s))
  self.assertEqual(p.result.get('code'),'STANDBY_NOT_SETTLED',p.result);self.assertFalse(p.standby_valid)
  with self.assertRaises(ControlFault):p.require_joint()
if __name__=='__main__':unittest.main()
