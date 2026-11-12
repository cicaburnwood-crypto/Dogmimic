import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from remote_priority import RemotePriority
from control import Engine,Rejected
from test_control import Clock
from test_height_control import HeightFake
class PriorityTests(unittest.TestCase):
 def test_any_axis_and_any_button_exact_two_seconds(self):
  p=RemotePriority();neutral={'buttons':[],'axes':{'lx':0.,'ly':0.,'rx':0.,'ry':0.}}
  self.assertFalse(p.update(neutral,True,0)['blocked'])
  for inp in [{'buttons':['F1'],'axes':{}},{'buttons':[],'axes':{'ry':.00001}}]:
   self.assertTrue(p.update(inp,True,1)['blocked']);self.assertTrue(p.status(9)['blocked'])
   self.assertEqual(p.update(neutral,True,10)['quiet_remaining_s'],2.)
   self.assertTrue(p.update(neutral,True,11.999)['blocked']);self.assertFalse(p.update(neutral,True,12)['blocked'])
 def test_repeated_input_restarts_quiet_window_and_generation(self):
  p=RemotePriority();z={'buttons':[],'axes':{}}
  p.update({'buttons':['UP']},True,0);p.update(z,True,1);n=p.generation
  p.update({'buttons':['DOWN']},True,2.9);self.assertEqual(p.generation,n+1)
  p.update(z,True,3);self.assertTrue(p.status(4.9)['blocked']);self.assertFalse(p.status(5)['blocked'])
 def test_lost_feedback_blocks_until_two_seconds_fresh(self):
  p=RemotePriority();z={'buttons':[],'axes':{}}
  p.update(z,True,0);self.assertTrue(p.update(z,False,1)['blocked']);self.assertTrue(p.update(z,True,10)['blocked']);self.assertFalse(p.update(z,True,12)['blocked'])
 def engine(self):
  c=Clock();b=HeightFake(c);e=Engine(b,c);e.tick();b.active=e.active='trained';e.selected='trained';return c,b,e
 def test_sticks_release_lease_and_drive_p4_same_tick(self):
  c,b,e=self.engine();e.joint_manager.engaged=True;e.joint_manager.client='test-client';e.joint_manager.cmd=[0,0,0,.1,.05,-.1]
  b.t.remote['axes']['ly']=.5;c.value+=.02;e.tick()
  self.assertFalse(e.joint_manager.engaged);self.assertGreater(e.last_command[0],0);self.assertTrue(e.remote_priority.status(c())['blocked']);self.assertEqual(e.active,'trained')
  with self.assertRaises(Rejected):e.request('reach_v17',{'op':'begin','client':'test-client'})
  before=list(e.last_command)
  with self.assertRaises(Rejected):e.request('reach_v17',{'op':'hold','client':'test-client'})
  self.assertEqual(e.last_command,before)
 def test_unmapped_button_preempts_stationary_overlay(self):
  c,b,e=self.engine();e.stationary_reach.engaged=True;e.stationary_reach.hold=[.1,.1,0,-.1];b.t.remote['buttons']=['F1'];c.value+=.02;e.tick()
  self.assertFalse(e.stationary_reach.engaged)
  for action in ['select','hindstand','attitude']:
   with self.assertRaises(Rejected):e.request(action,'trained')
 def test_damping_chord_bypasses_other_source_block(self):
  c,b,e=self.engine();b.t.remote['buttons']=['L2','B'];c.value+=.02;e.tick();self.assertEqual(e.phase,'damping');self.assertIn('damp',b.calls)
 def test_input_recovery_does_not_recreate_old_lease(self):
  c,b,e=self.engine();e.joint_manager.engaged=True;e.joint_manager.client='test-client';b.t.remote['buttons']=['F1'];c.value+=.02;e.tick()
  b.t.remote['buttons']=[];c.value+=.02;e.tick();c.value+=2.;e.tick()
  self.assertFalse(e.status()['remote_priority']['blocked']);self.assertFalse(e.joint_manager.engaged)
if __name__=='__main__':unittest.main()
