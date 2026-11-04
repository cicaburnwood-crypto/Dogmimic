"""Sequence and transport tests. Never connect to DDS or the real robot."""
import copy
import math
import sys
import unittest
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'deployments/b2_192_168_123_164_20261101'))
from attitude_plan import AttitudePlan,validate_plan
from attitude_runtime import advance_command
from control import Engine,Rejected,posture_checks
from test_control import Clock,FakeBackend,HttpControls,sdk_fixture

PLAN=dict(mode='return_each',actions=[dict(axis='roll',degrees=-12.5),dict(axis='pitch',degrees=6.25)])

class PlanTests(unittest.TestCase):
    def test_plan_is_copied_before_execution(self):
        source=copy.deepcopy(PLAN);p=AttitudePlan(source);source['actions'][0]['degrees']=30
        self.assertEqual(p.plan['actions'][0]['degrees'],-12.5)

    def test_validation_precedes_any_output(self):
        for p in [None,{},dict(mode='stream',actions=PLAN['actions']),dict(actions=[]),
                  dict(actions=[dict(axis='yaw',degrees=1)]),dict(actions=[dict(axis='roll',degrees=True)]),
                  dict(actions=[dict(axis='roll',degrees=float('nan'))]),dict(actions=[dict(axis='roll',degrees=30.01)])]:
            with self.subTest(p=p),self.assertRaises(ValueError):validate_plan(p)
        for angle in [-30,0,.001,22.123456,30]:
            self.assertEqual(validate_plan(dict(actions=[dict(axis='roll',degrees=angle)]))['actions'][0]['degrees'],angle)

    def test_return_each_waits_for_actual_stable_recovery(self):
        p=AttitudePlan(PLAN);context=np.zeros(6,np.float32)
        for _ in range(100):p.advance(True,context)
        self.assertEqual(p.target(),[math.radians(-12.5),0.])
        for _ in range(200):
            context[4:]=p.target();context=advance_command(context);p.advance(True,context)
        self.assertEqual(p.phase,'recover');self.assertEqual(p.index,0)
        for _ in range(300):
            context[4:]=p.target();context=advance_command(context);p.advance(False,context)
        self.assertEqual(p.index,0)
        p.advance(True,context);self.assertEqual(p.index,1);self.assertEqual(p.phase,'execute')
        for _ in range(200):
            context[4:]=p.target();context=advance_command(context);p.advance(True,context)
        for _ in range(300):
            context[4:]=p.target();context=advance_command(context)
            if p.advance(True,context):break
        self.assertEqual(p.phase,'complete');self.assertEqual(p.target(),[0.,0.])

    def test_continuous_switches_without_reset_or_intermediate_standing_gate(self):
        p=AttitudePlan(dict(PLAN,mode='continuous'));context=np.zeros(6,np.float32)
        for _ in range(300):
            context[4:]=p.target();context=advance_command(context);p.advance(False,context)
        self.assertEqual(p.index,1);self.assertEqual(p.phase,'execute')
        self.assertGreater(abs(context[0]),.2);self.assertEqual(p.target(),[0.,math.radians(6.25)])


class AttitudeFake(FakeBackend):
    def start_attitude(self,t,plan):
        self.calls.append(('start_attitude',copy.deepcopy(plan)));self.active='attitude';self.frames=0;self.damping=False
    def step_attitude(self,t):self.frames+=1;return self.frames>=10
    def execution_status(self):return dict(skill='attitude',plan_phase='complete' if self.frames>=10 else 'execute')
    def begin_attitude_return(self):
        self.calls.append('begin_native_return');self.native_return=dict(status='succeeded')
    def attitude_return_status(self):return self.native_return


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.backend=AttitudeFake(self.clock);self.engine=Engine(self.backend,self.clock);self.engine.tick()
    def test_status_and_time_never_start_plan(self):
        for _ in range(20):self.clock.value+=.02;self.engine.tick();self.engine.status()
        self.assertFalse(self.backend.calls)
    def test_confirm_starts_plan_once_then_returns_native(self):
        self.engine.request('attitude',PLAN)
        for _ in range(12):self.clock.value+=.02;self.engine.tick()
        self.assertEqual(self.engine.active,'native')
        self.assertEqual(self.engine.phase,'classic')
        self.assertEqual(self.engine.selected,'native')
        self.assertEqual(sum(isinstance(c,tuple) and c[0]=='start_attitude' for c in self.backend.calls),1)
    def test_stable_standing_is_the_start_condition(self):
        self.backend.t.dq[0]=.4;self.engine.tick()
        with self.assertRaises(Rejected):self.engine.request('attitude',PLAN)
        self.assertFalse(self.backend.calls)
        self.backend.t.dq[0]=0;self.backend.t.remote['axes']['ly']=1;self.engine.tick()
        self.engine.request('attitude',PLAN);self.assertEqual(self.engine.active,'attitude')
    def test_remote_recovery_still_interrupts(self):
        self.engine.request('attitude',PLAN);self.backend.t.q=[0,1.2,-2.8]*4
        self.engine.handle_remote('recovery');self.engine.tick();self.assertEqual(self.engine.active,'native')
    def test_recovered_body_is_not_misreported_as_native_success(self):
        self.engine.request('attitude',PLAN)
        self.backend.begin_attitude_return=lambda:setattr(self.backend,'native_return',dict(status='running'))
        for _ in range(11):self.clock.value+=.02;self.engine.tick()
        self.assertEqual(self.engine.phase,'handover')
        self.assertTrue(self.engine.attitude_return_pending)
        self.backend.native_return=dict(status='failed',error='RPC timeout')
        self.clock.value+=.02;self.engine.tick()
        self.assertEqual(self.engine.last_error,'RPC timeout')
        self.assertFalse(any('可使用遥控器移动' in e['message'] for e in self.engine.events))
        frames=self.backend.frames
        for _ in range(5):self.clock.value+=.02;self.engine.tick()
        self.assertEqual(self.backend.frames,frames+5)
        self.assertEqual(self.engine.last_error,'RPC timeout')
        self.assertNotIn('suspend_output',self.backend.calls)
    def test_attitude_neutral_pose_is_recognized_only_for_selected_reference(self):
        t=self.backend.telemetry();t.q=[0,.85,-1.7]*4
        self.assertFalse(posture_checks(t,self.clock())['posture'])
        self.assertTrue(posture_checks(t,self.clock(),[0,.85,-1.7]*4)['posture'])


class AttitudeHTTP(HttpControls):
    def setUp(self):
        super().setUp()
        self.backend.start_attitude=lambda t,p:self.backend.calls.append(('attitude',copy.deepcopy(p)))
    def test_confirm_is_idempotent_and_copies_the_plan(self):
        body=dict(id='confirmed_attitude_1',action='attitude',value=PLAN)
        self.assertEqual(self.post(body)[0],200);self.assertEqual(self.post(body)[0],200)
        self.assertEqual(sum(isinstance(c,tuple) and c[0]=='attitude' for c in self.backend.calls),1)
        changed=copy.deepcopy(body);changed['value']['actions'][0]['degrees']=30
        self.assertEqual(self.post(changed)[0],409)
    def test_invalid_plan_does_not_start(self):
        self.assertEqual(self.post(dict(action='attitude',value=dict(actions=[])))[0],409)
        self.assertFalse(self.backend.calls)


class PublisherTests(unittest.TestCase):
    def test_attitude_uses_its_own_training_torque_limits(self):
        b=sdk_fixture();b.owned=True;b.active_skill='attitude';b.pub=b.pub_factory()
        b.controllers['attitude']=SimpleNamespace(meta=dict(policy_to_sdk=[3,4,5,0,1,2,9,10,11,6,7,8],torque_limits=[200,200,300]*4,kp=250.,kd=8.))
        b.target=[3.]*12;b.low=(0,dict(q=[0.]*12,dq=[0.]*12))
        b._publish_once();self.assertEqual(len(b.packets),1)
        for i,m in enumerate(b.packets[0].motor_cmd[:12]):self.assertAlmostEqual(250*m.q,[200,200,300][i%3])
        b.abort_immediate();b._publish_once();self.assertEqual(b.packets[-1].motor_cmd[0].kp,0.)


class NativeReturnTests(unittest.TestCase):
    def test_actual_recovered_pose_is_recognized_without_a_joint_pose_snap(self):
        # Actual pitch +30 attempt: upright, but a thigh stays 0.304 rad from q0.
        q=[.0636946,.99302268,-1.77266347,-.06162381,.86051202,-1.76051974,
           .06297910,.54550898,-1.65611315,.02724484,.64560103,-1.66277647]
        b=sdk_fixture();t=b.telemetry();t.q=q
        checks=posture_checks(t,time.monotonic(),[0,.85,-1.7]*4)
        self.assertGreater(checks['joint_error_rad'],.25)
        self.assertTrue(checks['posture']);self.assertTrue(checks['stationary'])
        self.assertEqual(checks['posture_basis'],'foot_geometry')
        self.assertFalse(posture_checks(t,time.monotonic())['posture'])

    def test_geometry_does_not_accept_crouching_tilting_or_motion(self):
        b=sdk_fixture();t=b.telemetry();ref=[0,.85,-1.7]*4
        for knee in (-2.4,-2.8):
            t.q=[0,1.2,knee]*4
            self.assertFalse(posture_checks(t,time.monotonic(),ref)['posture'])
            t.q=ref.copy();t.q[1:3]=[1.2,knee]
            self.assertFalse(posture_checks(t,time.monotonic(),ref)['posture'])
        t.q=ref.copy();t.quaternion=[math.cos(.1),math.sin(.1),0,0]
        self.assertFalse(posture_checks(t,time.monotonic(),ref)['posture'])
        t.quaternion=[1,0,0,0];t.dq[0]=.251
        self.assertFalse(posture_checks(t,time.monotonic(),ref)['stationary'])

    def test_kinematic_bounds_include_original_joint_tolerance(self):
        from standing_geometry import standing_feet
        rng=np.random.default_rng(12)
        for ref in ([0,.67,-1.3]*4,[0,.85,-1.7]*4):
            for _ in range(100):
                q=np.array(ref)+rng.uniform(-.25,.25,12)
                self.assertTrue(standing_feet(q,[ref])[0])

    def fixture(self):
        b=sdk_fixture();b.start_trained(b.telemetry());b.attitude_attempt='test'
        b.last_execution=dict(skill='attitude',complete=True);b.native_return={}
        b.sport_client.SetTimeout=lambda value:b.calls.append(('timeout',value))
        b.sport_client.GetServerApiVersion=lambda:(0,'1.0.0.1')
        return b

    def test_return_enters_classic_without_crouch_recovery_or_duplicate_publishers(self):
        b=self.fixture()
        def refresh():
            self.assertTrue(b.owned);self.assertIsNotNone(b.target)
            b.calls.append('fresh_transport')
        b._refresh_sport_client=refresh
        select=b.switch.SelectMode
        def select_with_writer_check(mode):
            self.assertFalse(b.owned);b._publish_once();self.assertFalse(b.packets)
            return select(mode)
        b.switch.SelectMode=select_with_writer_check
        b.return_native(b.telemetry(),'attitude_complete',True)
        self.assertEqual(b.calls.count(('ClassicWalk',True)),1)
        for name in ['BalanceStand','StandDown','StandUp','RecoveryStand']:
            self.assertNotIn((name,),b.calls)
        self.assertFalse(b.owned)

    def test_readonly_handshake_retries_lost_reply_before_classic(self):
        b=self.fixture();replies=iter([(3104,None),(3102,None),(0,'1.0.0.1')])
        b.sport_client.GetServerApiVersion=lambda:next(replies)
        b.return_native(b.telemetry(),'attitude_complete',True)
        self.assertEqual(b.calls.count(('ClassicWalk',True)),1)

    def test_slow_native_rpc_does_not_block_status_or_claim_completion(self):
        b=self.fixture();entered=threading.Event();release=threading.Event()
        def classic(flag):entered.set();release.wait(1);return 0
        b.sport_client.ClassicWalk=classic
        b.begin_attitude_return()
        self.assertTrue(entered.wait(1))
        self.assertEqual(b.attitude_return_status()['status'],'running')
        self.assertFalse(b.execution_status().get('native_restored',False))
        release.set();b.handover_thread.join(2)
        self.assertEqual(b.attitude_return_status()['status'],'succeeded')
        self.assertTrue(b.execution_status()['native_restored'])

    def test_motion_timeout_is_not_replayed_or_reported_as_success(self):
        b=self.fixture();calls=[]
        def classic(flag):calls.append(flag);return 3104
        b.sport_client.ClassicWalk=classic
        b.begin_attitude_return();b.handover_thread.join(2)
        self.assertEqual(calls,[True])
        self.assertEqual(b.attitude_return_status()['status'],'failed')
        self.assertFalse(b.execution_status()['native_restored'])

    def test_damping_during_transport_setup_cancels_handover(self):
        b=self.fixture();b._refresh_sport_client=b.abort_immediate
        with self.assertRaises(Rejected):b.return_native(b.telemetry(),'attitude_complete',True)
        self.assertNotIn(('select','ai'),b.calls);self.assertTrue(b.damping)

    def test_completed_plan_time_is_frozen(self):
        p=AttitudePlan(PLAN);p.phase='complete';before=p.status()
        for _ in range(50):self.assertTrue(p.advance(True,[0]*6))
        self.assertEqual(p.status(),before)


if __name__=='__main__':unittest.main()
