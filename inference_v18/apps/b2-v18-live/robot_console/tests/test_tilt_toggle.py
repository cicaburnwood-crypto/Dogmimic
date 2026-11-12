"""Robot-free tests of the installed remote control and ownership contracts."""
import math
import sys
import threading
import time
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from control import Engine,RemoteEvents,Rejected,NOMINAL
from standing_geometry import tilted_support
from test_control import Clock,FakeBackend,sdk_fixture


class TiltFake(FakeBackend):
    def __init__(self,clock):
        super().__init__(clock);self.result={};self.neutral=True
    def tilt_enabled(self):return True
    def tilt_neutral(self):return self.neutral
    def begin_locomotion_return(self):self.calls.append('begin_return');self.result={'status':'running'}
    def locomotion_return_status(self):return self.result
    def cancel_locomotion_return(self):self.calls.append('cancel_return');self.result={'status':'cancelled'}


class TiltControls(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.b=TiltFake(self.clock);self.e=Engine(self.b,self.clock)
    def tick(self,n=1):
        for _ in range(n):self.clock.value+=.02;self.e.tick()
    def keys(self,keys):self.b.t.remote['buttons']=keys;self.tick()
    def enter(self):self.tick();self.keys(['R2','Y']);self.keys([])
    def test_native_on_boot_no_output_even_chord_held(self):
        self.b.t.remote['buttons']=['R2','Y'];self.tick(100)
        self.assertEqual(self.e.active,'native');self.assertFalse(self.b.calls)
        self.assertEqual(self.e.selected,'native')
    def test_chord_both_orders_once_and_new_edge_to_leave(self):
        for first in ('R2','Y'):
            self.setUp();self.tick();self.keys([first]);self.keys(['R2','Y']);self.tick(100)
            self.assertEqual(self.b.calls.count('start_trained'),1);self.assertEqual(self.e.active,'trained')
            self.assertFalse(self.e.tilt_returning)
            self.keys([]);self.keys(['R2','Y']);self.assertTrue(self.e.tilt_returning)
    def test_failed_press_cannot_auto_activate_after_becoming_stable(self):
        self.tick();self.b.t.dq[0]=1.;self.keys(['R2','Y'])
        self.assertEqual(self.e.active,'native');self.b.t.dq[0]=0.;self.tick(50)
        self.assertEqual(self.e.active,'native')
        self.keys([]);self.keys(['R2','Y']);self.assertEqual(self.e.active,'trained')
    def test_both_directions_require_stable_stance(self):
        self.enter();self.b.t.velocity[0]=.2;self.keys(['R2','Y'])
        self.assertFalse(self.e.tilt_returning);self.assertEqual(self.e.active,'trained')
    def test_web_cannot_preselect_or_activate_tilt(self):
        self.tick()
        with self.assertRaises(Rejected):self.e.request('select','trained')
        self.assertEqual(self.e.selected,'native');self.assertFalse(self.b.calls)
    def test_independent_five_degrees_per_second_and_hold(self):
        self.enter();self.b.t.remote['buttons']=['UP','LEFT'];self.tick(50)
        np.testing.assert_allclose(np.degrees(self.e.tilt_request),[-5.,5.],atol=1e-6)
        self.keys([]);self.tick(40)
        np.testing.assert_allclose(np.degrees(self.e.tilt_request),[-5.,5.],atol=1e-6)
        self.b.t.remote['buttons']=['DOWN','RIGHT'];self.tick(50)
        np.testing.assert_allclose(self.e.tilt_request,[0,0],atol=1e-7)
    def test_independent_clamp_and_opposite_keys_cancel(self):
        self.enter();self.b.t.remote['buttons']=['UP','RIGHT'];self.tick(300)
        np.testing.assert_allclose(np.degrees(self.e.tilt_request),[20.,20.])
        self.b.t.remote['buttons']=['UP','DOWN','LEFT','RIGHT'];self.tick(100)
        np.testing.assert_allclose(np.degrees(self.e.tilt_request),[20.,20.])
    def test_native_dpad_and_transport_gap_do_not_accumulate(self):
        self.tick();self.keys(['UP']);self.tick(100);self.assertEqual(self.e.tilt_request,[0,0])
        self.enter();self.keys(['UP']);before=list(self.e.tilt_request)
        self.clock.value+=5.;self.e.tick();self.assertEqual(self.e.tilt_request,before)
        self.b.offline=True;self.tick();self.b.offline=False;self.tick()
        self.assertEqual(self.e.tilt_request,before)
    def test_movement_envelope_and_yaw_training_range(self):
        self.enter();self.e.speed_level='high';self.b.t.remote['axes'].update(lx=1.,ly=1.,rx=1.)
        self.tick();cmd=self.e.last_command
        self.assertEqual(len(cmd),5);self.assertAlmostEqual(math.hypot(*cmd[:2]),4.)
        self.assertLessEqual(abs(cmd[1]),1.);self.assertAlmostEqual(cmd[2],-.5)
    def test_damping_wins_over_toggle(self):
        self.enter();self.keys(['R2','Y','L2','B']);self.tick(10)
        self.assertEqual(self.e.phase,'damping');self.assertFalse(self.e.tilt_returning)
        self.assertEqual(self.b.calls.count('start_trained'),1)
    def test_system_functions_work_from_custom_without_stability_gate(self):
        for event in ('stand_toggle','recovery','running','free_walk','pose','vision'):
            self.setUp();self.enter();self.b.t.q=[0,1.2,-2.8]*4;self.tick()
            self.e.handle_remote(event)
            self.assertIn(('return_native',event,False),self.b.calls);self.assertEqual(self.e.active,'native')
    def test_return_neutralizes_then_waits_for_actual_stability_and_ack(self):
        self.enter();self.b.neutral=False;self.keys(['R2','Y']);self.keys([])
        self.assertEqual(self.e.phase,'neutralizing');self.assertNotIn('begin_return',self.b.calls)
        self.b.neutral=True;self.b.t.dq[0]=1.;self.tick();self.assertNotIn('begin_return',self.b.calls)
        self.b.t.dq[0]=0.;self.tick();self.assertEqual(self.b.calls.count('begin_return'),1)
        self.tick(10);self.assertEqual(self.e.active,'trained');self.assertEqual(self.e.phase,'handover')
        self.b.active='native';self.b.result={'status':'succeeded'};self.tick()
        self.assertEqual((self.e.selected,self.e.active,self.e.phase),('native','native','classic'))
    def test_handover_failure_not_false_success(self):
        self.enter();self.keys(['R2','Y']);self.b.result={'status':'failed','error':'RPC timeout'};self.tick()
        self.assertEqual(self.e.last_error,'RPC timeout');self.assertEqual(self.e.phase,'handover_failed')
    def test_system_action_cancels_pending_handover(self):
        self.enter();self.keys(['R2','Y']);self.e.handle_remote('stand_toggle')
        self.assertIn('cancel_return',self.b.calls);self.assertFalse(self.e.tilt_handover)
        self.assertEqual(self.e.active,'native')
    def test_reboot_discards_custom_selection_and_angles(self):
        self.enter();self.b.t.remote['buttons']=['UP'];self.tick(100)
        new=Engine(self.b,self.clock)
        self.assertEqual(new.selected,'native');self.assertEqual(new.tilt_request,[0.,0.])
    def test_system_chords_work_with_modifier_pressed_second(self):
        for key,event in [('A','stand_toggle'),('X','recovery'),('START','running')]:
            remote=RemoteEvents();remote.feed({'buttons':[]},100.)
            remote.feed({'buttons':[key]},100.02)
            self.assertEqual(remote.feed({'buttons':[key,'L2']},100.04),[event])
    def test_measured_simulation_stance_can_return_without_forcing_old_pose(self):
        self.enter();self.b.t.q=RECOVERED_Q.copy()
        self.keys(['R2','Y'])
        self.assertIn('begin_return',self.b.calls)


# Captured from the exported Actor's 30 s CPU cycle, not the nominal q0.
RECOVERED_Q=[.0495421243,1.178781999,-1.502942330,.0970019833,.743215547,-1.602488901,
    .107154288,.583958427,-1.649394988,-.379966848,.827321127,-1.709767311]


def tilt_sdk():
    b=sdk_fixture();c=b.controllers['locomotion']
    c.meta=dict(skill='tilt_locomotion',policy_to_sdk=[3,4,5,0,1,2,9,10,11,6,7,8],torque_limits=[200,200,300]*4,kp=250.,kd=8.)
    c.perm=np.array(c.meta['policy_to_sdk']);c.q0=np.array([0,.85,-1.7]*4);c.context=np.zeros(13)
    b.locomotion_return={};b.locomotion_attempt=None
    b.sport_client.SetTimeout=lambda value:None;b.sport_client.GetServerApiVersion=lambda:(0,'1.0')
    return b


class TiltSDK(unittest.TestCase):
    def test_recovered_support_geometry_not_old_joint_template(self):
        b=tilt_sdk();snapshot=b.telemetry
        def recovered():t=snapshot();t.q=RECOVERED_Q.copy();return t
        b.telemetry=recovered
        self.assertEqual(b._locomotion_return_posture().q,RECOVERED_Q)
        b.telemetry=lambda:None
    def test_new_motion_during_transport_preparation_prevents_transfer(self):
        b=tilt_sdk();b.start_trained(b.telemetry());b.attitude_attempt=None;snapshot=b.telemetry
        def moving():t=snapshot();t.dq[0]=.4;return t
        b._refresh_sport_client=lambda:setattr(b,'telemetry',moving)
        b.begin_locomotion_return();b.handover_thread.join(2)
        self.assertTrue(b.owned);self.assertNotIn(('select','ai'),b.calls)
        self.assertEqual(b.locomotion_return_status()['status'],'failed')
    def test_moving_before_release_never_releases_native(self):
        b=tilt_sdk();original=b.telemetry
        def moving():t=original();t.dq[0]=.3;return t
        b.telemetry=moving
        with self.assertRaises(Rejected):b.start_trained(b.telemetry())
        self.assertEqual(b.calls,[])
    def test_moving_after_release_restores_native_without_publisher(self):
        b=tilt_sdk();original=b.telemetry;release=b.switch.ReleaseMode
        def moving():t=original();t.dq[0]=.3;return t
        def raced():result=release();b.telemetry=moving;return result
        b.switch.ReleaseMode=raced
        with self.assertRaises(Rejected):b.start_trained(b.telemetry())
        self.assertEqual(b.mode,'ai');self.assertIsNone(b.pub)
    def test_native_return_no_crouch_and_no_concurrent_output(self):
        b=tilt_sdk();b.start_trained(b.telemetry());b.attitude_attempt=None
        select=b.switch.SelectMode
        def guarded(mode):
            self.assertFalse(b.owned);b._publish_once();self.assertEqual(b.packets,[]);return select(mode)
        b.switch.SelectMode=guarded
        b.begin_locomotion_return();b.handover_thread.join(2)
        self.assertEqual(b.locomotion_return_status()['status'],'succeeded')
        self.assertEqual(b.calls.count(('ClassicWalk',True)),1)
        for name in ('StandDown','StandUp','BalanceStand','RecoveryStand'):self.assertNotIn((name,),b.calls)
    def test_late_damping_cancels_native_switch(self):
        b=tilt_sdk();b.start_trained(b.telemetry());b.attitude_attempt=None
        b._refresh_sport_client=b.abort_immediate
        b.begin_locomotion_return();b.handover_thread.join(2)
        self.assertTrue(b.damping);self.assertNotIn(('select','ai'),b.calls)
    def test_cancel_before_worker_acquires_operation_lock(self):
        b=tilt_sdk();b.start_trained(b.telemetry());b.attitude_attempt=None
        with b.operation_lock:
            b.begin_locomotion_return();b.cancel_locomotion_return()
        b.handover_thread.join(2)
        self.assertNotIn(('select','ai'),b.calls);self.assertEqual(b.locomotion_return_status()['status'],'cancelled')
    def test_pd_torque_caps_in_policy_to_sdk_order(self):
        b=tilt_sdk();b.start_trained(b.telemetry());b.target=[5.]*12
        t=b.telemetry();b.low=(time.monotonic(),dict(q=t.q,dq=t.dq))
        b._publish_once();m=b.packets[-1].motor_cmd
        tau=np.array([x.kp*(x.q-q)-x.kd*dq for x,q,dq in zip(m,t.q,t.dq)])
        np.testing.assert_allclose(tau,[200,200,300]*4,atol=1e-6)
    def test_tilt_gate_rejects_lying_and_inconsistent_tilt(self):
        self.assertTrue(tilted_support([0,.85,-1.7]*4,[1,0,0,0]))
        self.assertFalse(tilted_support([0,1.2,-2.8]*4,[1,0,0,0]))
        self.assertFalse(tilted_support([0,.85,-1.7]*4,[math.cos(.3),math.sin(.3),0,0]))


if __name__=='__main__':unittest.main()
