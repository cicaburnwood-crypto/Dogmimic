"""Regression tests use only fakes; no DDS connection or robot output."""
import copy
import json
import math
import sys
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from collections import deque
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from control import Engine,Telemetry,NOMINAL,Rejected,RemoteEvents
from backend import SDKBackend
from server import Application,make_handler

def quaternion_rotation(q):
    w,x,y,z=np.array(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


class Clock:
    def __init__(self):self.value=100.
    def __call__(self):return self.value


class FakeBackend:
    kind='test_fixture'
    def __init__(self,clock):
        self.clock=clock;self.calls=[];self.ready=True;self.frames=0;self.active='native';self.damping=False
        self.t=Telemetry(q=NOMINAL.copy(),dq=[0.]*12,quaternion=[1.,0,0,0],gyro=[0.]*3,
                         position=[0.,0.,.59],velocity=[0.]*3,battery=86,source='test_fixture')
        self.offline=False;self.position_stale=False;self.failure=False
    def telemetry(self):
        t=copy.deepcopy(self.t);now=self.clock()
        t.received=t.tick_changed=now if not self.offline else now-1
        t.position_received=t.position_changed=now if not self.position_stale else now-1
        return t
    def capabilities(self):return {k:dict(ready=self.ready,reason='historical validation flag') for k in ('native','trained','hindstand')}
    def controller_state(self):return self.active
    def output_mode(self):return 'damping' if self.damping else 'policy'
    def start_trained(self,t):
        self.calls.append('start_trained')
        if self.failure:raise RuntimeError('handover failed')
        self.active='trained';self.damping=False
    def start_hindstand(self,t):self.calls.append('start_hindstand');self.frames=0;self.active='hindstand';self.damping=False
    def step_trained(self,t,cmd):self.calls.append(('command',cmd))
    def step_hindstand(self,t):
        if self.position_stale:raise RuntimeError('model input unavailable')
        self.frames+=1;return self.frames>=775
    def return_native(self,t,event,require_standing=False):
        self.calls.append(('return_native',event,require_standing));self.active='native';self.damping=False
    def native_classic(self,replay=False):self.calls.append(('native_classic',replay))
    def stop(self,active):self.calls.append(('stop',active))
    def damp(self):self.calls.append('damp');self.damping=True
    def suspend_output(self):self.calls.append('suspend_output')
    def execution_status(self):return dict(reference_frame=self.frames,policy_time_s=self.frames*.02,progress=self.frames/775)
    def close(self):pass


class Controls(unittest.TestCase):
    def setUp(self):self.clock=Clock();self.backend=FakeBackend(self.clock);self.engine=Engine(self.backend,self.clock)
    def tick(self,n=1):
        for _ in range(n):self.clock.value+=.02;self.engine.tick()
    def ready(self):self.tick();self.assertTrue(self.engine.status()['can_select'])
    def buttons(self,keys):self.backend.t.remote['buttons']=keys;self.tick()
    def double(self,key):
        self.buttons([]);self.buttons([key]);self.buttons([]);self.tick(5);self.buttons([key]);self.buttons([])
    def start(self):self.ready();self.engine.request('select','trained');self.double('R1')
    def test_no_dwell_or_stick_neutral_requirement(self):
        self.backend.t.remote['axes']['ly']=1.;self.ready()
        self.engine.request('select','trained');self.assertEqual(self.engine.selected,'trained')
        self.assertEqual(self.backend.calls,[])
    def test_stable_stance_with_non_neutral_stick_can_start_stunt(self):
        self.backend.t.remote['axes']['lx']=.5;self.ready();self.engine.request('hindstand')
        self.assertIn('start_hindstand',self.backend.calls)
    def test_only_stunt_requires_stability(self):
        self.backend.t.dq[0]=.4;self.backend.t.velocity[0]=.2;self.ready()
        self.engine.request('select','trained')
        with self.assertRaises(Rejected):self.engine.request('hindstand')
    def test_prone_rejects_both_requested_actions(self):
        self.backend.t.q=[0,1.2,-2.8]*4;self.tick()
        for action,value in [('select','trained'),('hindstand',None)]:
            with self.assertRaises(Rejected):self.engine.request(action,value)
        self.assertFalse(self.backend.calls)
    def test_missing_root_does_not_prevent_locomotion_entry(self):
        self.backend.position_stale=True;self.start();self.assertEqual(self.engine.active,'trained')
    def test_no_commissioning_approval_gate(self):
        self.backend.ready=False;self.start();self.assertEqual(self.engine.active,'trained')
        self.engine.request('hindstand');self.assertEqual(self.engine.active,'hindstand')
    def test_stale_revision_is_not_a_motion_gate(self):
        self.ready();self.engine.request('select','trained',-100);self.assertEqual(self.engine.selected,'trained')
    def test_single_press_or_hold_does_not_activate(self):
        self.ready();self.engine.request('select','trained');self.buttons([]);self.buttons(['R1']);self.tick(80)
        self.assertNotIn('start_trained',self.backend.calls)
    def test_double_r1_activates_once(self):
        self.start();self.assertEqual(self.backend.calls.count('start_trained'),1)
        self.assertEqual(self.engine.active,'trained')
    def test_click_before_selection_not_reused(self):
        self.ready();self.buttons(['R1']);self.buttons([]);self.engine.request('select','trained')
        self.buttons([]);self.tick(5);self.buttons(['R1']);self.assertNotIn('start_trained',self.backend.calls)
    def test_buttons_held_at_connect_are_not_presses(self):
        self.backend.t.remote['buttons']=['R1'];self.ready();self.assertFalse(self.backend.calls)
    def test_native_keys_are_not_duplicated(self):
        self.ready();self.buttons(['START']);self.buttons([]);self.buttons(['L2','A']);self.buttons([])
        self.buttons(['L2','B']);self.assertFalse(self.backend.calls)
        self.double('R1');self.assertIn(('native_classic',False),self.backend.calls)
    def test_damping_can_be_left_by_native_recovery_without_standing(self):
        self.start();self.buttons(['L2','B']);self.assertEqual(self.engine.phase,'damping')
        self.backend.t.q=[0,1.2,-2.8]*4;self.buttons([]);self.buttons(['L2','X'])
        self.assertIn(('return_native','recovery',False),self.backend.calls)
        self.assertEqual(self.engine.active,'native');self.assertIsNone(self.engine.last_error)
    def test_damping_can_be_left_by_double_r1_from_standing(self):
        self.start();self.buttons(['L2','B']);self.double('R1')
        self.assertEqual(self.engine.phase,'classic');self.assertFalse(self.backend.damping)
        self.assertEqual(self.backend.calls.count('start_trained'),2)
    def test_no_extra_deadman_key(self):
        self.start();self.backend.t.remote['axes']['ly']=.8;self.buttons([])
        self.assertGreater(self.engine.last_command[0],0)
    def test_combo_does_not_double_l2(self):
        r=RemoteEvents();r.feed(dict(buttons=[]),1);r.feed(dict(buttons=['L2']),1.02)
        self.assertEqual(r.feed(dict(buttons=['L2','A']),1.1),['stand_toggle'])
        r.feed(dict(buttons=[]),1.12);self.assertEqual(r.feed(dict(buttons=['L2']),1.2),[])
    def test_switching_requires_actual_standing(self):
        self.start();self.backend.t.q=[0,1.2,-2.8]*4;self.tick()
        with self.assertRaises(Rejected):self.engine.request('select','native')
        self.backend.t.q=NOMINAL.copy();self.tick();self.engine.request('select','native')
        self.assertIn(('return_native',None,True),self.backend.calls)
    def test_original_keys_not_blocked_by_action_phase_or_moving_state(self):
        for event in ('recovery','free_walk','stand_toggle','pose','running','vision'):
            with self.subTest(event=event):
                self.setUp();self.ready();self.engine.request('hindstand')
                self.backend.t.q=[0,1.2,-2.8]*4;self.backend.t.velocity[0]=.5;self.tick()
                self.engine.handle_remote(event)
                self.assertIn(('return_native',event,False),self.backend.calls)
    def test_stunt_has_no_phase_lock_when_physically_standing(self):
        self.ready();self.engine.request('hindstand');self.engine.request('hindstand')
        self.assertEqual(self.backend.calls.count('start_hindstand'),2)
        self.engine.request('select','trained');self.assertEqual(self.engine.selected,'trained')
    def test_full_stunt_returns_to_previous_controller(self):
        self.ready();self.engine.request('hindstand');self.tick(780)
        self.assertEqual(self.engine.active,'native')
        self.assertEqual(self.backend.calls.count('start_hindstand'),1)
    def test_stale_data_recovers_without_unlock(self):
        self.start();self.backend.offline=True;self.tick()
        self.assertIn('suspend_output',self.backend.calls);self.assertEqual(self.engine.phase,'waiting_input')
        self.backend.offline=False;self.tick();self.assertEqual(self.engine.phase,'classic')
        self.assertIsNone(self.engine.last_error);self.assertEqual(self.backend.calls.count('start_trained'),1)
    def test_missing_stunt_input_recovers_without_latch(self):
        self.ready();self.engine.request('hindstand');self.backend.position_stale=True;self.tick()
        self.assertEqual(self.engine.phase,'waiting_input')
        self.backend.position_stale=False;self.tick();self.assertEqual(self.engine.phase,'hindstand')
    def test_stunt_progress_is_executed_frames_not_wall_time(self):
        self.ready();self.engine.request('hindstand');self.tick(5)
        progress=self.engine.status()['stunt_progress'];self.assertGreater(progress,0)
        self.backend.position_stale=True;self.clock.value+=100.;self.tick()
        self.assertEqual(self.engine.status()['stunt_progress'],progress);self.assertLess(progress,.01)
    def test_first_step_reads_sensors_again_after_blocking_handover(self):
        start=self.backend.start_trained
        def slow_start(t):start(t);self.clock.value+=.3
        self.backend.start_trained=slow_start;ages=[]
        self.backend.step_trained=lambda t,cmd:ages.append(self.clock()-t.received)
        self.start();self.assertTrue(ages);self.assertLess(max(ages),.1)
    def test_failed_handover_can_be_retried(self):
        self.backend.failure=True;self.start();self.assertEqual(self.engine.active,'native')
        self.backend.failure=False;self.double('R1');self.assertEqual(self.engine.phase,'classic')
    def test_stop_does_not_require_r1_rearm(self):
        self.start();self.backend.t.remote['axes']['ly']=1;self.engine.request('stop')
        self.tick();self.assertEqual(self.engine.last_command,[0.,0.,0.])
        self.tick();self.assertGreater(self.engine.last_command[0],0)
    def test_user_velocity_constraints(self):
        self.start();self.double('L1');self.backend.t.remote['axes'].update(lx=1,ly=1,rx=1);self.tick()
        self.assertLessEqual(math.hypot(*self.engine.last_command[:2]),2.000001)
        self.assertLessEqual(abs(self.engine.last_command[1]),.6)
        self.backend.t.remote['axes'].update(lx=1.05,ly=0);self.tick()
        self.assertLessEqual(abs(self.engine.last_command[1]),.6)
    def test_invalid_pose_is_not_reported_as_standing(self):
        self.backend.t.q[0]=float('nan');self.tick();self.assertFalse(self.engine.status()['can_select'])


class HttpControls(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.backend=FakeBackend(self.clock);self.temp=TemporaryDirectory()
        self.app=Application(self.backend,self.temp.name);self.app.stop_event.set();self.app.thread.join(1)
        self.app.engine=Engine(self.backend,self.clock);self.app.engine.tick()
        self.http=ThreadingHTTPServer(('127.0.0.1',0),lambda *a:None)
        self.port=self.http.server_address[1];self.http.RequestHandlerClass=make_handler(self.app,self.port)
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
    def tearDown(self):self.http.shutdown();self.http.server_close();self.app.close();self.temp.cleanup()
    def post(self,body,origin=None,csrf=None,host=None):
        c=HTTPConnection('127.0.0.1',self.port)
        headers={'Origin':origin or f'http://127.0.0.1:{self.port}','X-B2-CSRF':self.app.csrf if csrf is None else csrf,'Content-Type':'application/json'}
        if host:headers['Host']=host
        c.request('POST','/api/command',json.dumps(body),headers)
        r=c.getresponse();v=json.loads(r.read());c.close();return r.status,v
    def test_transport_origin_checks_do_not_change_robot(self):
        body=dict(action='select',value='trained')
        for kwargs in ({'csrf':'wrong'},{'origin':'https://evil.example'},{'host':'evil.example'}):
            self.assertEqual(self.post(body,**kwargs)[0],403)
        self.assertEqual(self.app.engine.selected,'native')
    def test_revision_and_request_id_are_optional(self):
        self.assertEqual(self.post(dict(action='select',value='trained',revision=-999))[0],200)
        self.assertEqual(self.post(dict(action='hindstand'))[0],200)
    def test_same_request_retransmission_not_executed_twice(self):
        body=dict(id='request_02',action='hindstand')
        self.assertEqual(self.post(body)[0],200);self.assertEqual(self.post(body)[0],200)
        self.assertEqual(self.backend.calls.count('start_hindstand'),1)
    def test_post_reads_actual_current_posture(self):
        self.backend.t.q=[0,1.2,-2.8]*4
        self.assertEqual(self.post(dict(action='select',value='trained'))[0],409)
    def test_get_never_executes_action(self):
        c=HTTPConnection('127.0.0.1',self.port);c.request('GET','/api/command?action=hindstand')
        r=c.getresponse();r.read();self.assertEqual(r.status,404);c.close();self.assertFalse(self.backend.calls)


def sdk_fixture():
    b=SDKBackend.__new__(SDKBackend);b.np=np;b.lock=threading.RLock();b.operation_lock=threading.RLock()
    b.closed=False;b.owned=False;b.damping=False;b.command_epoch=0;b.target=None;b.target_at=0
    b.active_skill=None;b.owner_name='ai';b.native_owner='ai';b.owner_error=None;b.owner_checked=0
    b.output_error=None;b.writer_packets=0;b.initial_target=np.array(NOMINAL);b.blend_at=time.monotonic()-1
    b.rotation=quaternion_rotation;b.hind_alignment=None;b.hind_attempt=None;b.last_execution={};b.control_trace=deque(maxlen=1000)
    b.pub=None;b.mode='ai';b.calls=[];b.packets=[]
    def snapshot():
        n=time.monotonic();return Telemetry(received=n,tick_changed=n,position_received=n,position_changed=n,
            q=NOMINAL.copy(),dq=[0.]*12,quaternion=[1.,0,0,0],gyro=[0.]*3,position=[0,0,.59],velocity=[0.]*3,
            position_source='odom',position_quaternion=[1.,0,0,0])
    b.telemetry=snapshot
    def release():b.calls.append('release');b.mode='';return 0,None
    def select(mode):b.calls.append(('select',mode));b.mode=mode;return 0,None
    b.switch=SimpleNamespace(CheckMode=lambda:(0,{'name':b.mode}),ReleaseMode=release,SelectMode=select)
    def sport(name):
        def call(*args):b.calls.append((name,*args));return 0,None
        return call
    b.sport_client=SimpleNamespace(**{n:sport(n) for n in ['Damp','BalanceStand','ClassicWalk','StandDown','StandUp','RecoveryStand','FastWalk','FreeWalk','FreeEuler','VisionWalk','StopMove']})
    b.sport_client.SetTimeout=lambda value:None
    b.sport_client.GetServerApiVersion=lambda:(0,'1.0.0.1')
    def write(cmd):b.packets.append(copy.deepcopy(cmd));return True
    b.pub_factory=lambda:SimpleNamespace(Init=lambda:b.calls.append('publisher_init'),Write=write)
    b.cmd=SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(20)])
    b.crc=SimpleNamespace(Crc=lambda cmd:123)
    def controller(reference=None):
        return SimpleNamespace(reset=lambda:None,frame=0,reference=reference,
            step=lambda *args,**kwargs:(np.array(NOMINAL)+.5,dict(target_clipped=True)))
    b.controllers={'locomotion':controller(),'hindstand':controller({'pos':list(range(776))})}
    return b


class SDKBehavior(unittest.TestCase):
    def test_takeover_without_commissioning_file(self):
        b=sdk_fixture();b.start_trained(b.telemetry());self.assertTrue(b.owned)
        self.assertTrue(b.capabilities()['trained']['ready']);self.assertEqual(b.active_skill,'locomotion')
    def test_other_native_controller_identity_is_supported(self):
        b=sdk_fixture();b.mode='native_variant';b.start_trained(b.telemetry())
        b.return_native(b.telemetry(),None,True);self.assertIn(('select','native_variant'),b.calls)
    def test_native_released_then_posture_changed_restores_owner(self):
        b=sdk_fixture();original=b._posture_now;count=0
        def posture(stable=False):
            nonlocal count
            count+=1
            if count==2:raise Rejected('posture changed')
            return original(stable)
        b._posture_now=posture
        with self.assertRaises(Rejected):b.start_trained(b.telemetry())
        self.assertEqual(b.mode,'ai');self.assertFalse(b.owned);self.assertIsNone(b.pub)
    def test_operator_damping_cancels_old_rpc_but_not_future_command(self):
        b=sdk_fixture();release=b.switch.ReleaseMode
        def late_damp():result=release();b.abort_immediate();return result
        b.switch.ReleaseMode=late_damp
        with self.assertRaises(Rejected):b.start_trained(b.telemetry())
        self.assertEqual(b.mode,'ai');self.assertIn(('Damp',),b.calls)
        b.switch.ReleaseMode=release;b.start_trained(b.telemetry());self.assertTrue(b.owned);self.assertFalse(b.damping)
    def test_required_root_input_is_not_fabricated(self):
        b=sdk_fixture();snapshot=b.telemetry
        def no_root():t=snapshot();t.position_received=0;return t
        b.telemetry=no_root
        with self.assertRaises(RuntimeError):b.start_hindstand(b.telemetry())
        self.assertNotIn('release',b.calls)
        b.start_trained(b.telemetry());self.assertTrue(b.owned)
    def test_independent_odometry_survives_native_controller_release(self):
        b=sdk_fixture();t=b.telemetry();now=time.monotonic()
        b.low=(now,dict(q=t.q,dq=t.dq,quaternion=t.quaternion,gyro=t.gyro));b.tick_changed=now
        b.sport={'odom':(now,dict(position=[-2.8,-.6,-.01],velocity=[0,0,0],position_quaternion=[1,0,0,0],position_source='odom',body_height=0,native_mode=0),now,100)}
        b.telemetry=lambda:SDKBackend.telemetry(b)
        b.start_hindstand(b.telemetry());self.assertEqual(b.mode,'')
        b.step_hindstand(b.telemetry());self.assertIsNotNone(b.target)
        self.assertEqual(b.execution_status()['position_source'],'odom')
        self.assertEqual(len(b.drain_trace()),1)
    def test_odom_heading_is_aligned_with_lowstate_imu(self):
        b=sdk_fixture();t=b.telemetry();t.quaternion=[math.cos(math.pi/4),0,0,math.sin(math.pi/4)]
        t.position=[3.,4.,-.01];p0=b._root_position(t)
        t.position=[4.,4.,.49];p1=b._root_position(t)
        np.testing.assert_allclose(p1-p0,[0,1,.5],atol=1e-6)
    def test_native_position_origin_not_mixed_into_hindstand(self):
        b=sdk_fixture();t=b.telemetry();t.position_source='sport'
        with self.assertRaises(RuntimeError):b._root_position(t)
    def test_odometry_z_not_mislabelled_as_height_above_ground(self):
        from control import posture_checks
        b=sdk_fixture();t=b.telemetry();t.position[2]=-.01;t.body_height=0
        self.assertIsNone(posture_checks(t,time.monotonic())['height_m'])
    def test_clipped_target_big_step_and_tilt_are_not_extra_cutoffs(self):
        b=sdk_fixture();b.start_trained(b.telemetry());b.blend_at=time.monotonic()-1
        t=b.telemetry();t.quaternion=[math.cos(.5),math.sin(.5),0,0]
        b.step_trained(t,[0,0,0]);self.assertIsNotNone(b.target)
        np.testing.assert_allclose(b.target,np.array(NOMINAL)+.5)
    def test_unavailable_input_can_resume_without_unlock(self):
        b=sdk_fixture();b.start_trained(b.telemetry());t=b.telemetry();t.received-=1
        with self.assertRaises(RuntimeError):b.step_trained(t,[0,0,0])
        b.suspend_output();b._publish_once();self.assertFalse(b.packets)
        b.step_trained(b.telemetry(),[0,0,0]);b._publish_once();self.assertTrue(b.packets)
    def test_only_explicit_damping_sends_damping_packets(self):
        b=sdk_fixture();b.start_trained(b.telemetry());b.target_at=0;b.owner_checked=0
        b._publish_once();self.assertEqual(b.packets[-1].motor_cmd[0].kp,250.)
        b.abort_immediate();b._publish_once();self.assertEqual(b.packets[-1].motor_cmd[0].kp,0.)
        b.start_trained(b.telemetry());b._publish_once();self.assertEqual(b.packets[-1].motor_cmd[0].kp,250.)
    def test_publish_failure_is_retried_without_mode_lock(self):
        b=sdk_fixture();b.start_trained(b.telemetry());write=b.pub.Write;b.pub.Write=lambda cmd:False
        b._publish_once();self.assertIsNotNone(b.output_error);self.assertFalse(b.damping)
        b.pub.Write=write;b._publish_once();self.assertIsNone(b.output_error);self.assertTrue(b.packets)
    def test_original_recovery_works_when_not_standing(self):
        b=sdk_fixture();b.start_trained(b.telemetry());t=b.telemetry();t.q=[0,1.2,-2.8]*4
        b.return_native(t,'recovery');self.assertIn(('RecoveryStand',),b.calls)
        self.assertNotIn(('BalanceStand',),b.calls)
    def test_native_stand_toggle_uses_current_pose(self):
        for standing,method in [(True,'StandDown'),(False,'StandUp')]:
            with self.subTest(standing=standing):
                b=sdk_fixture();b.start_trained(b.telemetry());t=b.telemetry()
                if not standing:t.q=[0,1.2,-2.8]*4
                b.return_native(t,'stand_toggle');self.assertIn((method,),b.calls)
    def test_native_classic_replayed_only_after_handover(self):
        b=sdk_fixture();b.native_classic(False);self.assertFalse(b.calls)
        b.native_classic(True);self.assertIn(('ClassicWalk',True),b.calls)
    def test_newer_damping_cancels_older_native_function(self):
        b=sdk_fixture();b.start_trained(b.telemetry());select=b.switch.SelectMode
        def late_damp(mode):result=select(mode);b.abort_immediate();return result
        b.switch.SelectMode=late_damp
        with self.assertRaises(Rejected):b.return_native(b.telemetry(),'running')
        self.assertIn(('Damp',),b.calls);self.assertNotIn(('FastWalk',True),b.calls)
        b.switch.SelectMode=select;b.return_native(b.telemetry(),'free_walk')
        self.assertIn(('FreeWalk',),b.calls)
    def test_native_send_failure_retries_until_server_discovery(self):
        b=sdk_fixture();replies=iter([(3102,None),(0,None)]);calls=[]
        def rpc():calls.append(1);return next(replies)
        b._native_call(rpc,'test');self.assertEqual(len(calls),2)
    def test_native_response_timeout_is_not_replayed(self):
        b=sdk_fixture();calls=[]
        def rpc():calls.append(1);return 3104,None
        with self.assertRaises(RuntimeError):b._native_call(rpc,'test')
        self.assertEqual(len(calls),1)

    def test_recovery_refreshes_transport_after_releasing_custom_output(self):
        b=sdk_fixture();b.start_trained(b.telemetry());order=[]
        def refresh():
            self.assertFalse(b.owned);self.assertEqual(b.mode,'ai')
            b._publish_once();self.assertFalse(b.packets)
            order.append('refresh')
        b._refresh_sport_client=refresh
        b.sport_client.GetServerApiVersion=lambda:(order.append('ready') or (0,'1.0.0.1'))
        b.sport_client.RecoveryStand=lambda:(order.append('recovery') or 0)
        b.return_native(b.telemetry(),'recovery')
        self.assertEqual(order,['refresh','ready','recovery'])

    def test_standup_queries_readiness_before_one_motion_request(self):
        b=sdk_fixture();b.start_trained(b.telemetry());t=b.telemetry();t.q=[0,1.2,-2.8]*4
        replies=iter([(3102,None),(3104,None),(0,'1.0.0.1')])
        b.sport_client.GetServerApiVersion=lambda:next(replies)
        b.return_native(t,'stand_toggle')
        self.assertEqual(b.calls.count(('StandUp',)),1)

    def test_damping_during_recovery_handshake_prevents_standup(self):
        b=sdk_fixture();b.start_trained(b.telemetry());t=b.telemetry();t.q=[0,1.2,-2.8]*4
        def ready():b.abort_immediate();return 0,'1.0.0.1'
        b.sport_client.GetServerApiVersion=ready
        with self.assertRaises(Rejected):b.return_native(t,'stand_toggle')
        self.assertNotIn(('StandUp',),b.calls)

    def test_standup_reply_timeout_is_never_replayed(self):
        b=sdk_fixture();b.start_trained(b.telemetry());t=b.telemetry();t.q=[0,1.2,-2.8]*4;calls=[]
        b.sport_client.StandUp=lambda:(calls.append('up') or 3104)
        with self.assertRaises(RuntimeError):b.return_native(t,'stand_toggle')
        self.assertEqual(calls,['up']);self.assertFalse(b.owned)


if __name__=='__main__':unittest.main()
