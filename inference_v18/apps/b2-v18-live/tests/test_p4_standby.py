import unittest,time
from unittest.mock import AsyncMock
import test_standby

class P4StandbyTests(unittest.IsolatedAsyncioTestCase):
 def setup(self):
  p,s=test_standby.StandbyTests.setup_portal(self)
  p.dog['active']='trained';p.dog['command']=[0.]*6
  s['v'][:]=0;s['gyro'][:]=0;s['aqd'][:]=0
  return p,s
 async def test_explicit_p4_button_only_schedules(self):
  p,s=self.setup();p.run_standby=AsyncMock()
  self.assertIsNone(p.snapshot()['p4_standby_error'])
  await p.action({'op':'enter_p4_standby','client':p.owner});await p.job
  self.assertEqual(p.run_standby.await_args.args[0]['start_mode'],'trained')
  p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_native_rejected_by_p4_button(self):
  p,s=self.setup();p.dog['active']='native'
  with self.assertRaisesRegex(ValueError,'P4'):p.preparation_ready('trained')
 async def test_nonzero_command_rejected_even_if_stationary(self):
  for axis in range(3):
   p,s=self.setup();p.dog['command'][axis]=.1
   with self.assertRaisesRegex(ValueError,'零平移'):p.preparation_ready('trained')
 async def test_actual_motion_rejected(self):
  p,s=self.setup();s['gyro'][2]=.3
  with self.assertRaisesRegex(ValueError,'静止'):p.preparation_ready('trained')
 async def test_remote_priority_still_blocks(self):
  p,s=self.setup();p.dog['remote_priority']['blocked']=True
  with self.assertRaises(ValueError):p.preparation_ready('trained')
 async def test_foreign_owner_rejected(self):
  p,s=self.setup();p.dog['joint_manager']={'engaged':True,'client':'someone-else'}
  with self.assertRaisesRegex(ValueError,'占用'):p.preparation_ready('trained')
 async def test_complete_without_native_switch_or_actor(self):
  p,s=self.setup();events=test_standby.StandbyTests.mocks(self,p,s)
  original=p.dogcmd.side_effect
  async def dog(op,**kw):
   self.assertEqual(p.dog['active'],'trained')
   self.assertIn(op,('begin','frame','hold'))
   await original(op,**kw)
  p.dogcmd.side_effect=dog
  p.model=lambda obs:(_ for _ in ()).throw(AssertionError('No EE Actor in standby'))
  plan=test_standby.StandbyTests.plan(self,s);plan['start_mode']='trained'
  await p.run_standby(plan)
  self.assertTrue(p.standby_valid,p.result);self.assertEqual(p.dog['active'],'trained')
 async def test_mode_change_during_prepare_does_not_take_control(self):
  p,s=self.setup()
  async def changed(plan):p.dog['active']='native';p.result={'phase':'prepared'}
  p.run_prepare=AsyncMock(side_effect=changed)
  plan=test_standby.StandbyTests.plan(self,s);plan['start_mode']='trained'
  await p.run_standby(plan)
  self.assertFalse(p.standby_valid);p.dogcmd.assert_not_called()

if __name__=='__main__':unittest.main()
