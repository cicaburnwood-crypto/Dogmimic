import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy import arm_action

class AccelerationTests(unittest.TestCase):
 def step(self,old=.4,speed=2):
  q=np.zeros(6);u=np.ones(12)
  return arm_action(q,q,np.zeros(3),np.zeros(3),np.array([speed,0,0,0,0,0]),u,q,np.full(6,old),np.full(6,-3.),np.full(6,3.))
 def test_sudden_carrier_acceleration_brakes_without_jump(self):
  q,v=self.step()
  np.testing.assert_allclose(v,.376,atol=1e-12)
  # Old final clamp would produce .12, or -14rad/s².
  self.assertLessEqual(np.max(abs((v-.4)/.02)),1.2+1e-10)
 def test_slowdown_converges_to_speed_dependent_demand(self):
  old=.4
  for _ in range(12):
   _,v=self.step(old);self.assertLessEqual(abs(v[0]-old),.024+1e-10);old=v[0]
  self.assertAlmostEqual(old,.12)
 def test_negative_direction(self):
  q=np.zeros(6);u=-np.ones(12)
  _,v=arm_action(q,q,q[:3],q[:3],np.array([2,0,0,0,0,0]),u,q,np.full(6,-.4),np.full(6,-3.),np.full(6,3.))
  np.testing.assert_allclose(v,-.376,atol=1e-12)
 def test_zero_start_unchanged(self):
  _,v=self.step(0);np.testing.assert_allclose(v,.024,atol=1e-12)
 def test_infeasible_joint_bound_stops(self):
  q=np.full(6,2.999)
  with self.assertRaisesRegex(ValueError,'J1'):
   arm_action(q,np.zeros(6),np.zeros(3),np.zeros(3),np.zeros(6),np.ones(12),q,np.full(6,.4),np.full(6,-3.),np.full(6,3.))
 def test_varying_envelope_preserves_hard_speed_and_acceleration(self):
  rng=np.random.default_rng(113);q=np.zeros(6);v=np.zeros(6)
  for _ in range(1000):
   u=rng.uniform(-1,1,12);c=np.array([rng.uniform(0,2),0,0,0,0,0])
   nxt,nv=arm_action(q,np.zeros(6),np.zeros(3),np.zeros(3),c,u,q,v,np.full(6,-3.),np.full(6,3.))
   self.assertLessEqual(abs(nv).max(),.6+1e-10)
   self.assertLessEqual(abs(nv-v).max(),.024+1e-10)
   q,v=nxt,nv

if __name__=='__main__':unittest.main()
