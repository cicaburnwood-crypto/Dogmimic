"""Real frozen Actor, without DDS: command transition regression tests."""
import math, os, sys, unittest
from pathlib import Path
import numpy as np

BUNDLE=Path(os.environ.get('B2_TEST_BUNDLE',Path(__file__).resolve().parents[4]/'dog_bundle'))
sys.path.insert(0,str(BUNDLE))
from tilt_runtime import TiltLocomotionController, JointTargetNotch, policy_goal, constrain_request


class ContinuousGait(unittest.TestCase):
    def setUp(self):
        self.c=TiltLocomotionController(BUNDLE/'models/tilt_locomotion')
        self.q=np.empty(12,np.float32);self.q[self.c.perm]=self.c.q0
    def step(self,request):
        return self.c.step(self.q,np.zeros(12),[0.,0.,0.],[1.,0.,0.,0.],request)
    def test_first_deadband_crossing_has_no_full_swing_jump(self):
        previous=np.ones(4);crossed=False
        for _ in range(5):
            _,d=self.step([.4,0,0,0,0]);clocks=np.array(d['goal'][12:])
            if d['gait_blend']>0 and not crossed:
                self.assertLess(float(np.max(np.abs(clocks-previous))),.02)
                self.assertLess(d['gait_blend'],.01);crossed=True
            previous=clocks
        self.assertTrue(crossed)
    def test_low_speed_threshold_chatter_stays_continuous(self):
        self.c.context[12]=.73;previous=np.ones(4)
        for i in range(100):
            self.c.context[0]=.051 if i%2==0 else .049
            _,d=self.step([.05,0,0,0,0]);clocks=np.array(d['goal'][12:])
            self.assertLess(float(np.max(np.abs(clocks-previous))),.02);previous=clocks
    def test_stop_fades_and_resets_phase_only_when_amplitude_zero(self):
        for _ in range(50):self.step([.4,0,0,0,0])
        self.c.context[:3]=0;self.c.context[12]=.61
        _,d=self.step([0.]*5)
        self.assertGreater(d['gait_blend'],.99);self.assertNotAlmostEqual(float(self.c.context[12]),.25)
        for _ in range(40):_,d=self.step([0.]*5)
        np.testing.assert_allclose(d['goal'][12:],[1.]*4,atol=1e-6)
        self.assertEqual(self.c.gait_progress,0.);self.assertEqual(float(self.c.context[12]),.25)
    def test_steady_walking_uses_original_goal_at_same_phase(self):
        for _ in range(80):self.step([.4,.2,.2,.1,-.1])
        context=self.c.context.copy();context[7:12]=[.4,.2,.2,.1,-.1]
        expected=policy_goal(context)
        _,d=self.step([.4,.2,.2,.1,-.1])
        self.assertEqual(d['gait_blend'],1.)
        np.testing.assert_allclose(d['goal'],expected,atol=2e-6)
    def test_restarting_mid_fade_does_not_restart_full_amplitude(self):
        for _ in range(50):self.step([.4,0,0,0,0])
        self.c.context[:3]=0
        for _ in range(12):_,d=self.step([0.]*5)
        prior=d['gait_blend'];self.c.context[0]=.1
        _,d=self.step([.4,0,0,0,0])
        self.assertGreater(d['gait_blend'],prior);self.assertLess(d['gait_blend'],1.)
    def test_reset_drops_previous_phase_history_and_transition(self):
        for _ in range(60):self.step([.4,0,0,0,0])
        self.c.reset();_,d=self.step([0.]*5)
        self.assertEqual(self.c.frame,1);self.assertEqual(d['gait_blend'],0.)
        np.testing.assert_allclose(d['goal'][12:],[1.]*4,atol=1e-6)
    def test_preserves_requested_limits_and_frozen_model_identity(self):
        self.assertEqual(self.c.meta['checkpoint_sha256'],'96dbb589f9fafb39e7d25dea3e4566d9a13e32788ef39ab685489d1617b9ecb4')
        r=constrain_request([4,1,2,math.radians(20),-math.radians(20)])
        self.assertAlmostEqual(float(np.linalg.norm(r[:2])),4.,places=5)
        self.assertLessEqual(abs(r[1]),1.);self.assertEqual(r[2],2.)
        with self.assertRaises(ValueError):constrain_request([0,0,0,math.radians(21),0])
    def test_target_is_actor_output_not_a_hidden_joint_slew_filter(self):
        self.c.target_filter=None
        for req in ([.4,0,0,0,0],[0,.3,.5,.2,-.2],[0.]*5):
            out,d=self.step(req)
            raw=self.c.q0+self.c.scale*np.asarray(d['action'],np.float32)
            target=np.clip(raw,self.c.limits[:,0]+.02,self.c.limits[:,1]-.02)
            np.testing.assert_allclose(out[self.c.perm],target,atol=1e-7)


class TargetNotchTests(unittest.TestCase):
    def test_constant_target_and_reset_have_no_startup_impulse(self):
        f=JointTargetNotch();x=np.linspace(-2.,1.,12,dtype=np.float32)
        for _ in range(100):np.testing.assert_allclose(f(x),x,atol=1e-7)
        f.reset();np.testing.assert_allclose(f(-x),-x,atol=1e-7)

    def test_suppresses_measured_band_and_preserves_gait_band(self):
        for hz,max_gain,min_gain in [(2.,1.01,.98),(3.4,1.01,.96),(10.,.36,0),(11.,.001,0),(12.,.36,0)]:
            f=JointTargetNotch();x=np.sin(2*np.pi*hz*np.arange(1500)/50)
            y=np.array([f([v])[0] for v in x])
            gain=np.linalg.norm(y[500:])/np.linalg.norm(x[500:])
            self.assertLess(gain,max_gain);self.assertGreaterEqual(gain,min_gain)

    def test_impulse_decays_and_channels_are_independent(self):
        f=JointTargetNotch();f(np.zeros(12));x=np.zeros(12);x[3]=1
        y=f(x);self.assertEqual(np.count_nonzero(y),1)
        for _ in range(120):y=f(np.zeros(12))
        self.assertLess(float(np.max(np.abs(y))),1e-12)

    def test_invalid_filter_parameters_fail_before_inference(self):
        for args in [(25,2,50),(11,0,50),(11,.1,50),(float('nan'),2,50)]:
            with self.assertRaises(ValueError):JointTargetNotch(*args)

    def test_runtime_filter_limits_and_reset(self):
        c=TiltLocomotionController(BUNDLE/'models/tilt_locomotion')
        c.target_filter=JointTargetNotch();q=np.empty(12);q[c.perm]=c.q0
        for i in range(50):
            out,d=c.step(q,np.zeros(12),np.zeros(3),[1,0,0,0],[.4,.1,.2,.3,-.3])
            target=out[c.perm]
            self.assertTrue(np.all(target>=c.limits[:,0]+.02))
            self.assertTrue(np.all(target<=c.limits[:,1]-.02))
            np.testing.assert_allclose(c.previous_action,d['action'],atol=0)
        c.reset();self.assertIsNone(c.target_filter.z1)


if __name__=='__main__':unittest.main()
