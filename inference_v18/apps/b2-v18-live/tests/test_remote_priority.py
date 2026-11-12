import unittest,asyncio,time
import test_portal
class RemotePortalTests(unittest.IsolatedAsyncioTestCase):
 def setup(self):return test_portal.PortalTests.setup_portal(self)
 async def test_remote_blocks_all_new_goals_and_gripper(self):
  p,s=self.setup();p.dog['remote_priority'].update(blocked=True,reason='遥控器优先');p.observe_remote()
  for op in ['enter_standby','keyboard_enable','keyboard_input','gripper_set','preview','preview_prepare','execute','reset']:
   with self.assertRaisesRegex(ValueError,'遥控器'):await p.action({'op':op,'client':'browser-test'})
  p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_interrupt_invalidates_goal_keys_and_pending_prepare(self):
  p,s=self.setup();p.observe_remote();p.keyboard.enable('browser-test',s);p.plan={'id':'old'};p.dog['remote_priority'].update(blocked=True,generation=1)
  p.observe_remote();self.assertFalse(p.keyboard.enabled);self.assertTrue(p.cancel);self.assertIsNone(p.plan)
 async def test_after_silence_controls_open_without_old_motion(self):
  p,s=self.setup();p.dog['active']='trained';p.standby_valid=True;p.dog['joint_manager']={};p.dog['remote_priority'].update(blocked=True,generation=1);p.observe_remote()
  p.dog['remote_priority']['blocked']=False;p.observe_remote()
  self.assertTrue(p.joint_available());await p.action({'op':'keyboard_enable','client':'browser-test'})
  self.assertTrue(p.keyboard.enabled);self.assertIsNone(p.job);self.assertEqual(p.keyboard.keys,set());p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_native_handover_never_auto_switches_back(self):
  p,s=self.setup();p.standby_valid=True;p.dog['active']='native';self.assertFalse(p.joint_available())
if __name__=='__main__':unittest.main()
