import sys,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runner import Backend

class Transport:
 def __init__(self):
  self.commands=[];self.acquired=False;self.restored=False;self.fail=False
 def snapshot(self):
  if self.fail and self.acquired:raise RuntimeError('lost odometry')
  return dict(q=[0.]*29,dq=[0.]*29,gyro=[0.]*3,quaternion=[1.,0.,0.,0.],position=[0.,0.,.76],velocity=[0.]*3)
 def acquire(self,meta=None):self.acquired=True;return {'owner':'native'}
 def command(self,q,meta,s):self.commands.append(q)
 def release(self,r):self.restored=True
class FakeController:
 def __init__(self,*a):self.meta={'reference_frames':4,'policy_dt':.02};self.frame=0
 def reset(self):self.frame=0
 def step(self,*a):self.frame+=1;return np.zeros(29),{'progress':self.frame/3}
class Tests(unittest.TestCase):
 def make(self):
  import tempfile
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  t=Transport()
  with patch('runner.Controller',FakeController):b=Backend(t,log_dir=self.tmp.name)
  return b,t
 def test_start_is_readonly_complete_restores(self):
  b,t=self.make();self.assertEqual(t.commands,[]);self.assertFalse(t.acquired)
  b.execute('jump_60cm');b.worker.join(2)
  self.assertEqual(len(t.commands),3);self.assertTrue(t.restored);self.assertEqual(b.progress,1.)
 def test_failure_is_reported_and_restores(self):
  b,t=self.make();t.fail=True;b.execute('funky_chicken');b.worker.join(2)
  self.assertTrue(t.restored);self.assertEqual(t.commands,[]);self.assertIn('lost odometry',b.last_error)
 def test_unstable_never_takes_control(self):
  b,t=self.make()
  with patch.object(b,'stable',return_value=False):
   with self.assertRaises(ValueError):b.execute('jump_60cm')
  self.assertFalse(t.acquired)
 def test_zero_torque_is_not_stable_standing(self):
  b,t=self.make();s=t.snapshot();s['native_fsm']=0;self.assertFalse(b.stable(s))
  s['native_fsm']=500;self.assertTrue(b.stable(s))
 def test_duplicate_is_not_a_restart(self):
  b,t=self.make();b.execute('jump_60cm');result=b.execute('funky_chicken');b.worker.join(2)
  self.assertTrue(result['already_running']);self.assertEqual(len(t.commands),3)

if __name__=='__main__':unittest.main()
