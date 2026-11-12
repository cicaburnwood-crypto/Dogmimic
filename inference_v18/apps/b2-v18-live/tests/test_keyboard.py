import asyncio,json,sys,time,unittest
from pathlib import Path
from unittest.mock import AsyncMock
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from teleop import KeyboardGoal,ControlFault
import test_portal

class IntegrationTests(unittest.TestCase):
 def setUp(self):
  self.now=10.;self.k=KeyboardGoal(lambda:self.now);self.s={'ee':np.array([.5,0,.8]),'eerot':R.from_euler('y',90,degrees=True).as_matrix()};self.k.enable('browser-test',self.s)
 def step(self,keys,dt=.02,seq=0):
  self.k.input('browser-test',seq,keys);self.now+=dt;return self.k.advance(self.s)
 def test_all_translation_keys(self):
  for key,axis,sign in [('KeyW',0,1),('KeyS',0,-1),('KeyA',1,1),('KeyD',1,-1),('KeyR',2,1),('KeyF',2,-1)]:
   self.k.enable('browser-test',self.s);self.step([key]);expected=self.s['ee'].copy();expected[axis]+=sign*.001;np.testing.assert_allclose(self.k.position,expected)
 def test_diagonal_speed_and_keyup(self):
  self.step(['KeyW','KeyA','KeyR']);self.assertAlmostEqual(np.linalg.norm(self.k.position-self.s['ee']),.001)
  before=self.k.position.copy();self.step([],seq=1);np.testing.assert_array_equal(before,self.k.position)
 def test_orientation_semantics(self):
  for key,world_axis in [('ArrowUp',-self.s['eerot'][:,0]),('ArrowDown',self.s['eerot'][:,0]),('ArrowLeft',np.array([0,0,1])),('ArrowRight',np.array([0,0,-1])),('KeyO',self.s['eerot'][:,2]),('KeyP',-self.s['eerot'][:,2])]:
   self.k.enable('browser-test',self.s);self.step([key]);delta=R.from_matrix(self.k.rotation@self.s['eerot'].T).as_rotvec();np.testing.assert_allclose(delta,world_axis*np.deg2rad(.2),atol=1e-12)
   if key in ('KeyO','KeyP'):np.testing.assert_allclose(self.k.rotation[:,2],self.s['eerot'][:,2],atol=1e-12)
 def test_opposites_and_rotation_stability(self):
  self.assertFalse(self.step(['KeyW','KeyS','KeyO','KeyP']))
  for i in range(200):self.step(['ArrowLeft','KeyO','ArrowUp'],seq=i+1);self.s['eerot']=self.k.rotation.copy()
  np.testing.assert_allclose(self.k.rotation.T@self.k.rotation,np.eye(3),atol=1e-12)
 def test_stale_sequence_owner_and_disable(self):
  self.step(['KeyW'])
  for who,seq,keys in [('browser-test',0,[]),('other-person',1,[]),('browser-test',True,[]),('browser-test',1,['KeyQ'])]:
   with self.assertRaises(ControlFault):self.k.input(who,seq,keys)
  self.now+=.36
  with self.assertRaisesRegex(ControlFault,'连接超时'):self.k.advance(self.s)
  self.k.disable()
  with self.assertRaises(ControlFault):self.k.input('browser-test',2,[])
 def test_lag_and_workspace_are_not_joint_failures(self):
  self.k.position=self.s['ee']+np.array([.1,0,0]);self.assertFalse(self.step(['KeyW']));self.assertEqual(self.k.notice['code'],'TARGET_TRACKING_LAG')
  self.s['ee']=np.array([3.,0,.8]);self.k.enable('browser-test',self.s);self.assertFalse(self.step(['KeyW']));self.assertEqual(self.k.notice['code'],'TARGET_WORKSPACE')
 def test_changed_key_does_not_integrate_old_elapsed_time(self):
  self.now+=.3;self.k.input('browser-test',0,['KeyW']);self.assertFalse(self.k.advance(self.s));np.testing.assert_array_equal(self.k.position,self.s['ee'])
 def test_dt_is_bounded(self):
  self.step(['KeyW'],dt=.3);self.assertAlmostEqual(self.k.position[0]-.5,.002)

class KeyboardPortalTests(unittest.IsolatedAsyncioTestCase):
 def setup_portal(self):
  p,s=test_portal.PortalTests.setup_portal(self);p.dog['active']='trained';p.dog['joint_manager']={'engaged':True,'client':p.client};p.standby_valid=True;return p,s
 async def test_enable_and_empty_frames_do_not_move(self):
  p,s=self.setup_portal();d=await p.action({'op':'keyboard_enable','client':'browser-test'})
  np.testing.assert_array_equal(d['keyboard']['xyz'],s['ee']);self.assertIsNone(p.job)
  await p.action({'op':'keyboard_input','client':'browser-test','seq':0,'keys':[]});self.assertIsNone(p.job)
  p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_first_key_only_schedules_joint_policy(self):
  p,s=self.setup_portal();p.run=AsyncMock();await p.action({'op':'keyboard_enable','client':'browser-test'})
  await p.action({'op':'keyboard_input','client':'browser-test','seq':0,'keys':['KeyW']});await p.job
  plan=p.run.await_args.args[0];self.assertEqual(plan['kind'],'keyboard');np.testing.assert_array_equal(plan['xyz'],s['ee'])
  p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_disable_during_startup_cannot_send_motion(self):
  p,s=self.setup_portal();await p.action({'op':'keyboard_enable','client':'browser-test'})
  await p.action({'op':'keyboard_input','client':'browser-test','seq':0,'keys':['KeyW']})
  await p.action({'op':'keyboard_disable','client':'browser-test'});await p.job
  self.assertFalse(p.keyboard.enabled);self.assertNotIn('begin',[c.args[0] for c in p.dogcmd.await_args_list]);self.assertNotIn('stream_begin',[c.args[0] for c in p.armcmd.await_args_list])
 async def test_running_keyboard_frames_then_stale_stops_both(self):
  p,s=self.setup_portal();await p.action({'op':'keyboard_enable','client':'browser-test'})
  p.keyboard.input('browser-test',0,['KeyW']);frames=[]
  async def dog(op,**kw):
   if op=='begin':p.dog.update(active='trained',joint_manager={'engaged':True,'client':p.client})
   if op=='frame':frames.append(kw['command']);p.keyboard.stamp=time.monotonic()-1
  async def arm(op,**kw):
   if op=='stream_begin':p.arm['stream']={'engaged':True,'client':p.client}
  p.dogcmd=AsyncMock(side_effect=dog);p.armcmd=AsyncMock(side_effect=arm)
  await p.run({'kind':'keyboard','xyz':s['ee'].tolist(),'quaternion_xyzw':R.from_matrix(s['eerot']).as_quat().tolist()})
  self.assertEqual(len(frames),1,p.result);self.assertEqual(len(frames[0]),6);self.assertEqual(p.result['code'],'KEYBOARD_STALE');self.assertFalse(p.keyboard.enabled)
  self.assertEqual(p.dogcmd.await_args_list[-1].args[0],'hold');self.assertEqual(p.armcmd.await_args_list[-1].args[0],'stop')
 async def test_actual_joint_limit_names_joint_and_range(self):
  p,s=self.setup_portal();s['aq'][2]=p.g.arm_hi[2]+.01
  with self.assertRaises(ControlFault) as e:p.ready()
  issues=e.exception.detail['issues'];j=next(x for x in issues if x.get('joint')=='J3');self.assertEqual(j['code'],'JOINT_LIMIT');self.assertGreater(j['value_deg'],j['upper_deg'])
 async def test_collision_names_links_without_inventing_joint(self):
  p,s=self.setup_portal();s['aq']=np.array([0,0,0,-.207,0,.241])
  with self.assertRaises(ControlFault) as e:p.ready()
  collisions=[x for x in e.exception.detail['issues'] if x['code']=='SELF_COLLISION_CLEARANCE'];self.assertTrue(collisions);self.assertEqual(len(collisions[0]['links']),2);self.assertNotIn('joint',collisions[0])
 async def test_enabled_keyboard_excludes_manual_goal(self):
  p,s=self.setup_portal();await p.action({'op':'keyboard_enable','client':'browser-test'})
  with self.assertRaisesRegex(ValueError,'关闭键盘'):await p.action({'op':'reset','client':'browser-test'})
  with self.assertRaises(ValueError):await p.action({'op':'keyboard_input','client':'other-person','seq':0,'keys':['KeyW']})
 async def test_stop_cannot_rearm_from_old_key_frame(self):
  p,s=self.setup_portal();await p.action({'op':'keyboard_enable','client':'browser-test'});await p.stop()
  with self.assertRaises(ControlFault):await p.action({'op':'keyboard_input','client':'browser-test','seq':2,'keys':['KeyW']})
  self.assertIsNone(p.job)
if __name__=='__main__':unittest.main()
