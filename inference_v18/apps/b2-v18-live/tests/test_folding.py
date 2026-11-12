import asyncio,time,unittest
from unittest.mock import AsyncMock,patch
from types import SimpleNamespace
import numpy as np
import test_portal
import folding
from folding import FOLD_TARGET
from teleop import ControlFault

class FoldingTests(unittest.IsolatedAsyncioTestCase):
 def setup(self):
  p,s=test_portal.PortalTests.setup_portal(self);p.owner='browser-test';p.beats[p.owner]=time.monotonic()
  p.dog['gate'].update(telemetry=True,stationary=True,posture=True)
  s['v'][:]=0;s['gyro'][:]=0;s['aqd'][:]=0;p.standby_valid=True
  return p,s
 def plan(self,p,s):return {'kind':'fold','start_aq':s['aq'].tolist(),'start_pos':s['pos'].tolist(),'start_rot':s['rot'].tolist(),'dog_mode':p.dog['active']}
 def mock_path(self,p):p.fold_path=AsyncMock(return_value=401);p.fold_contacts=lambda *args:None
 async def test_explicit_button_schedules_without_dog_command(self):
  p,s=self.setup();p.dog['active']='trained';p.run_fold=AsyncMock()
  self.assertIsNone(p.snapshot()['fold_error']);self.assertIsNone(p.job)
  await p.action({'op':'fold_arm','client':p.owner});await p.job
  self.assertFalse(p.standby_valid);p.run_fold.assert_awaited_once();p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_fold_pose_is_verified_storage_pose_not_training_home(self):
  p,s=self.setup();self.assertFalse(np.allclose(FOLD_TARGET,p.g.meta['arm_home']));p.fold_joint_limits(FOLD_TARGET)
 async def test_input_and_ownership_guards_send_no_motion(self):
  for scenario in ['remote','moving','arm_busy','body_busy','keyboard','stale','no_posture','no_heartbeat']:
   with self.subTest(scenario=scenario):
    p,s=self.setup()
    if scenario=='remote':p.dog['remote_priority']['blocked']=True
    elif scenario=='moving':s['v'][0]=.1
    elif scenario=='arm_busy':p.arm['active']=True
    elif scenario=='body_busy':p.dog['joint_manager']={'engaged':True}
    elif scenario=='keyboard':p.keyboard.enable(p.owner,s)
    elif scenario=='stale':p.error='反馈过期'
    elif scenario=='no_posture':p.dog['gate']['posture']=False
    else:p.beats[p.owner]-=3
    with self.assertRaises(ValueError):await p.action({'op':'fold_arm','client':p.owner})
    p.armcmd.assert_not_called();p.dogcmd.assert_not_called();self.assertIsNone(p.job)
 async def test_collision_prevents_move_and_names_links(self):
  p,s=self.setup();p.fold_contacts=lambda *args:(_ for _ in ()).throw(ControlFault('FOLD_COLLISION','arm_link_2 ↔ trunk'))
  await p.run_fold(self.plan(p,s))
  self.assertEqual(p.result['code'],'FOLD_COLLISION');self.assertIn('trunk',p.result['message']);p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_cancel_before_and_during_path_never_moves(self):
  for during in [False,True]:
   p,s=self.setup()
   if during:
    async def cancel(*args):p.cancel=True;return 401
    p.fold_path=AsyncMock(side_effect=cancel)
   else:p.cancel=True
   await p.run_fold(self.plan(p,s));self.assertEqual(p.result['phase'],'stopped');p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_body_drift_during_path_never_moves(self):
  p,s=self.setup()
  async def move(*args):s['pos'][0]+=.03;return 401
  p.fold_path=AsyncMock(side_effect=move);await p.run_fold(self.plan(p,s))
  self.assertEqual(p.result['code'],'FOLD_BODY_MOVED');p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_remote_input_after_move_stops_arm_only(self):
  p,s=self.setup();self.mock_path(p)
  async def arm(op,**kw):
   if op=='move':p.dog['remote_priority']['blocked']=True
  p.armcmd=AsyncMock(side_effect=arm);await p.run_fold(self.plan(p,s))
  self.assertEqual(p.result['phase'],'stopped');self.assertEqual(p.armcmd.await_args.args[0],'stop');p.dogcmd.assert_not_called();self.assertFalse(p.standby_valid)
 async def test_success_preserves_dog_mode_and_never_operates_gripper(self):
  for mode in ['native','trained']:
   p,s=self.setup();p.dog['active']=mode;self.mock_path(p)
   async def arm(op,**kw):
    if op=='move':
     self.assertEqual(kw['kind'],'joint');self.assertEqual(kw['speed'],5);np.testing.assert_allclose(np.deg2rad(kw['joints_deg']),FOLD_TARGET);s['aq']=FOLD_TARGET.copy()
   p.armcmd=AsyncMock(side_effect=arm);await p.run_fold(self.plan(p,s))
   self.assertEqual(p.result['phase'],'folded',p.result);self.assertEqual(p.dog['active'],mode);p.dogcmd.assert_not_called()
   self.assertEqual(set(c.args[0] for c in p.armcmd.await_args_list),{'heartbeat','move','stop'});self.assertFalse(p.standby_valid)
 async def test_timeout_and_stop_failure_are_not_success(self):
  p,s=self.setup();self.mock_path(p);value=[0]
  def clock():value[0]+=.01;return value[0]
  async def arm(op,**kw):
   if op=='move':value[0]+=41
   if op=='heartbeat' and value[0]>41:value[0]+=41
   if op=='stop':raise ValueError('stop failed')
  p.armcmd=AsyncMock(side_effect=arm);p.beats[p.owner]=1e9
  with patch.object(folding,'time',SimpleNamespace(monotonic=clock,strftime=time.strftime)):
   await p.run_fold(self.plan(p,s))
  self.assertEqual(p.result['phase'],'stopped');self.assertIn('停止未确认',p.result['message']);p.dogcmd.assert_not_called()
 async def test_changed_mode_during_motion_stops(self):
  p,s=self.setup();self.mock_path(p)
  async def arm(op,**kw):
   if op=='move':p.dog['active']='trained'
  p.armcmd=AsyncMock(side_effect=arm);await p.run_fold(self.plan(p,s))
  self.assertEqual(p.result['code'],'FOLD_BODY_MOVED');self.assertEqual(p.armcmd.await_args.args[0],'stop');p.dogcmd.assert_not_called()

if __name__=='__main__':unittest.main()
