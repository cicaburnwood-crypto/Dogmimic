"""Hardware-free coverage of height buttons and native return from crouch."""
import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from control import Engine,RemoteEvents
from standing_geometry import tilted_support
from test_control import Clock
from test_tilt_toggle import TiltFake,tilt_sdk

CONFIG=dict(min_m=-.20,max_m=.05,button_rate_m_s=.05,
    tilt_knots=[[-.20,5.],[-.15,10.],[-.10,20.],[.05,20.]],
    xy_max_m_s=.5,lateral_max_m_s=.3,yaw_max_rad_s=.5)

class HeightFake(TiltFake):
    def height_config(self):return dict(CONFIG)

class HeightControls(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.b=HeightFake(self.clock);self.e=Engine(self.b,self.clock)
    def tick(self,n=1):
        for _ in range(n):self.clock.value+=.02;self.e.tick()
    def keys(self,keys):self.b.t.remote['buttons']=keys;self.tick()
    def enter(self):self.tick();self.keys(['R2','Y']);self.keys([])
    def test_native_boot_ignores_height_keys(self):
        self.keys(['R2','DOWN']);self.tick(300)
        self.assertEqual(self.e.active,'native');self.assertEqual(self.e.height_request,0.)
        self.assertFalse(self.b.calls)
    def test_height_hold_five_cm_per_second_no_pitch(self):
        self.enter();self.b.t.remote['buttons']=['R2','DOWN'];self.tick(100)
        self.assertAlmostEqual(self.e.height_request,-.10)
        self.assertEqual(self.e.tilt_request,[0.,0.]);self.assertEqual(len(self.e.last_command),6)
        self.keys([]);self.tick(40);self.assertAlmostEqual(self.e.height_request,-.10)
        self.b.t.remote['buttons']=['R2','UP'];self.tick(100)
        self.assertAlmostEqual(self.e.height_request,0.)
    def test_range_and_opposite_buttons(self):
        self.enter();self.keys(['R2','DOWN']);self.tick(500);self.assertEqual(self.e.height_request,-.2)
        self.keys(['R2','UP','DOWN']);self.tick(50);self.assertEqual(self.e.height_request,-.2)
        self.keys(['R2','UP']);self.tick(500);self.assertEqual(self.e.height_request,.05)
    def test_plain_dpad_keeps_angle_control_and_low_height_clamps_both_axes(self):
        self.enter();self.b.t.remote['buttons']=['UP','RIGHT'];self.tick(250)
        np.testing.assert_allclose(np.degrees(self.e.tilt_request),[20,20])
        self.b.t.remote['buttons']=['R2','DOWN'];self.tick(150)
        self.assertAlmostEqual(self.e.height_request,-.15)
        np.testing.assert_allclose(np.degrees(self.e.tilt_request),[10,10],atol=1e-6)
        self.tick(100);np.testing.assert_allclose(np.degrees(self.e.tilt_request),[5,5],atol=1e-6)
    def test_gap_does_not_accumulate_height(self):
        self.enter();self.keys(['R2','DOWN']);before=self.e.height_request
        self.clock.value+=5.;self.e.tick();self.assertEqual(self.e.height_request,before)
    def test_high_speed_button_cannot_exceed_current_model_envelope(self):
        self.enter();self.e.speed_level='high';self.b.t.remote['axes'].update(lx=1.,ly=1.,rx=1.)
        self.tick();cmd=self.e.last_command
        self.assertLessEqual(math.hypot(*cmd[:2]),.500001)
        self.assertLessEqual(abs(cmd[1]),.3);self.assertLessEqual(abs(cmd[2]),.5)
    def test_crouched_custom_mode_can_request_return_but_waits_for_neutral_height(self):
        self.enter();self.e.height_request=-.2;self.b.neutral=False
        thigh=math.acos(.27/.7);self.b.t.q=[0.,thigh,-2*thigh]*4
        self.assertFalse(tilted_support(self.b.t.q,self.b.t.quaternion))
        self.assertTrue(tilted_support(self.b.t.q,self.b.t.quaternion,minimum_height=.24))
        self.keys(['R2','Y']);self.assertTrue(self.e.tilt_returning)
        self.assertEqual(self.e.height_request,0.);self.assertEqual(self.e.last_command,[0.]*6)
        self.assertNotIn('begin_return',self.b.calls)
    def test_damping_and_native_actions_remain_available(self):
        self.enter();self.e.height_request=-.2;self.keys(['R2','DOWN','L2','B'])
        self.assertEqual(self.e.phase,'damping')
        self.e.handle_remote('stand_toggle');self.assertIn(('return_native','stand_toggle',False),self.b.calls)
    def test_runtime_height_and_gait_must_be_neutral_before_handover(self):
        b=tilt_sdk();c=b.controllers['locomotion'];c.meta['skill']='height_locomotion'
        c.meta['height_control']=dict(CONFIG);c.context=np.zeros(17)
        self.assertTrue(b.tilt_neutral())
        for index,value in [(13,-.2),(14,.02),(16,.5)]:
            c.context[:]=0.;c.context[index]=value;self.assertFalse(b.tilt_neutral())
    def test_reset_chord_both_orders_edge_only(self):
        for first in ('R2','A'):
            r=RemoteEvents();r.feed({'buttons':[]},100.)
            self.assertEqual(r.feed({'buttons':[first]},100.02),[])
            self.assertEqual(r.feed({'buttons':['R2','A']},100.04),['reset_posture'])
            self.assertEqual(r.feed({'buttons':['R2','A']},100.06),[])
    def test_reset_requests_during_motion_preserves_velocity_and_mode(self):
        self.enter();self.e.tilt_request=[.08,-.06];self.e.height_request=-.15
        self.b.t.remote['axes'].update(ly=.6,rx=.4);self.b.t.dq[0]=1.
        self.tick();before=self.e.last_command[:3];starts=self.b.calls.count('start_trained')
        self.keys(['R2','A'])
        self.assertEqual(self.e.last_command[:3],before)
        self.assertEqual(self.e.last_command[3:],[0.,0.,0.])
        self.assertEqual((self.e.active,self.e.selected,self.e.phase),('trained','trained','classic'))
        self.assertFalse(self.e.tilt_returning);self.assertEqual(self.b.calls.count('start_trained'),starts)
        self.assertNotIn('begin_return',self.b.calls)
    def test_reset_held_blocks_accidental_height_arrow_and_runs_once(self):
        self.enter();self.e.height_request=-.2;self.e.tilt_request=[.08,.08]
        self.keys(['R2','A','UP','RIGHT']);event_id=self.e.event_id;self.tick(50)
        self.assertEqual(self.e.last_command[3:],[0.,0.,0.]);self.assertEqual(self.e.event_id,event_id)
        self.keys(['R2','UP']);self.assertGreater(self.e.height_request,0.)
    def test_reset_ignored_in_native_and_damping_and_pending_handover(self):
        self.tick();self.e.height_request=-.1;self.keys(['R2','A'])
        self.assertEqual(self.e.height_request,-.1);self.assertFalse(self.b.calls)
        self.keys([]);self.enter();self.keys(['L2','B']);self.e.height_request=-.1
        self.keys(['R2','A']);self.assertEqual(self.e.phase,'damping');self.assertEqual(self.e.height_request,-.1)
        self.setUp();self.enter();self.b.neutral=False;self.keys(['R2','Y']);self.keys([])
        self.keys(['R2','A']);self.assertTrue(self.e.tilt_returning)
    def test_reset_never_overrides_native_system_chords(self):
        for extra,event in [(['L2','B'],'damp'),(['L2'],'stand_toggle')]:
            r=RemoteEvents();r.feed({'buttons':[]},100.)
            self.assertEqual(r.feed({'buttons':['R2','A',*extra]},100.02),[event])

if __name__=='__main__':unittest.main()
