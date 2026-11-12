import math,unittest
from test_height_control import HeightControls
from control import Rejected

class StationaryReachTests(HeightControls):
    def begin(self,target=None):
        self.tick(5);self.b.t.position_source='odom';self.b.t.position_quaternion=[1.,0.,0.,0.];self.tick()
        self.e.request('stationary_reach',{'op':'begin','client':'test-reach-client','target':target or [0,0,0,0]})
    def test_remote_translation_releases_stationary_overlay(self):
        self.begin([0,0,.3,0]);self.b.t.remote['axes']['ly']=1.;self.tick(4)
        self.assertGreater(self.e.last_command[0],0.);self.assertFalse(self.e.stationary_reach.engaged)
    def test_yaw_and_lease(self):
        self.begin([0,0,.3,0]);self.tick(3)
        self.assertEqual(self.e.last_command[:2],[0.,0.]);self.assertGreater(self.e.last_command[2],0)
        self.tick(45);self.assertEqual(self.e.last_command[:3],[0.,0.,0.])
    def test_reject_translation(self):
        with self.assertRaises(Rejected):self.e.request('stationary_reach',dict(op='begin',client='test-client',target=[0,0,0,0],vx=1))
    def test_low_height_envelope(self):
        with self.assertRaises(Rejected):self.begin([.3,0,0,-.2])
    def test_drift(self):
        self.begin();self.b.t.position[0]+=.06;self.tick();self.assertIn('漂移',self.e.stationary_reach.fault)
    def test_original_remote_priority(self):
        self.begin();self.e.handle_remote('reset_posture');self.assertFalse(self.e.stationary_reach.engaged)
    def test_stale_odom(self):
        self.begin();self.b.position_stale=True;self.tick();self.assertEqual(self.e.last_command[:3],[0.,0.,0.]);self.assertTrue(self.e.stationary_reach.fault)
    def test_height_tilt_rate(self):
        self.begin();s=self.e.stationary_reach;s.hold=[.349,0,0,-.1];s.target=[0,0,0,-.2]
        old=s.hold.copy();self.tick();self.assertLessEqual(abs(s.hold[0]-old[0]),math.radians(2)*.021);self.assertLessEqual(abs(s.hold[0]),s.limit(s.hold[3])+1e-8)

if __name__=='__main__':unittest.main()
