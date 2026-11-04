from pd_adapter import saturated_pd_target
"""SDK2 transport. Motion entry uses only the operator's standing requirements."""
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from control import Telemetry, decode_remote, Rejected, posture_checks
from attitude_plan import AttitudePlan


def model_info(controllers):
    models={}
    for skill,controller in controllers.items():
        meta=controller.meta
        models[skill]={key:meta.get(key) for key in ('label','iteration','checkpoint_iteration',
            'checkpoint_sha256','weights_sha256','payload_kg','reference_seconds','physical_eval_passed','hardware_validated',
            'simulation_strict_passed','simulation_strict_total')}
    hind=models.get('hindstand',{});loco=models.get('locomotion',{})
    return dict(models=models,models_loaded=bool(controllers),hindstand_iteration=hind.get('iteration'),
        locomotion_iteration=loco.get('iteration'),attitude_iteration=models.get('attitude',{}).get('iteration'),hardware_validated=False)


def load_controllers(bundle):
    from runtime import Controller
    controllers={k:Controller(bundle/'models'/k) for k in ('locomotion','hindstand')}
    if (bundle/'models/attitude/policy.json').exists():
        from attitude_runtime import AttitudeController
        controllers['attitude']=AttitudeController(bundle/'models/attitude')
    if (bundle/'models/tilt_locomotion/policy.json').exists():
        from tilt_runtime import TiltLocomotionController
        controllers['locomotion']=TiltLocomotionController(bundle/'models/tilt_locomotion')
    return controllers


class OfflineBackend:
    kind='offline'
    def tilt_enabled(self):return getattr(self.controllers.get('locomotion'),'meta',{}).get('skill')=='tilt_locomotion'
    def __init__(self,reason='未连接机器人',bundle=None):
        self.reason=reason
        self.bundle=Path(bundle or Path(__file__).resolve().parents[2]/'deployments/b2_192_168_123_164_20261101').resolve()
        sys.path.insert(0,str(self.bundle))
        from runtime import Controller
        self.controllers=load_controllers(self.bundle)
    def telemetry(self):return Telemetry()
    def model_info(self):return model_info(self.controllers)
    def capabilities(self):return {k:dict(ready=False,reason=self.reason) for k in ('native','trained','hindstand','attitude')}
    def suspend_output(self):pass
    def close(self):pass


class SDKBackend:
    kind='robot'
    def tilt_enabled(self):return getattr(self.controllers.get('locomotion'),'meta',{}).get('skill')=='tilt_locomotion'
    def model_info(self):return model_info(self.controllers)
    def __init__(self,bundle,interface='eth0'):
        self.bundle=Path(bundle).resolve()
        sys.path.insert(0,str(self.bundle))
        sys.path.insert(0,str(self.bundle/'vendor/unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206'))
        from runtime import Controller,np,sha256,rotation
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber,ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_,SportModeState_,LowCmd_
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.b2.sport.sport_client import SportClient
        self.np=np;self.rotation=rotation
        manifest=json.loads((self.bundle/'manifest.json').read_text())
        for name,digest in manifest['files'].items():
            if sha256(self.bundle/name)!=digest:raise RuntimeError('部署文件完整性校验失败：'+name)
        self.controllers=load_controllers(self.bundle)
        self.lock=threading.RLock();self.operation_lock=threading.RLock()
        self.low=None;self.sport={};self.latest_tick=None;self.tick_changed=0
        self.command_epoch=0;self.remote_buttons=set()
        self.subscribers=[];self.closed=False;self.owned=False;self.pub=None
        self.target=None;self.target_at=0.;self.damping=False;self.active_skill=None;self.blend_at=0.
        self.initial_target=None;self.inference_seconds=0;self.writer_packets=0;self.output_error=None
        self.hind_alignment=None;self.hind_attempt=None;self.control_trace=deque(maxlen=1000)
        self.last_execution={}
        self.attitude_plan=None;self.attitude_attempt=None
        self.native_return={};self.handover_thread=None
        self.locomotion_return={};self.locomotion_attempt=None
        self.handover_trace_at=0.
        self.owner_name=None;self.native_owner='ai';self.owner_checked=0.;self.owner_error=None
        self.pub_factory=lambda:ChannelPublisher('rt/lowcmd',LowCmd_)
        self.cmd=unitree_go_msg_dds__LowCmd_();self.crc=CRC()
        self.cmd.head=[0xFE,0xEF];self.cmd.level_flag=0xFF;self.cmd.gpio=0
        for motor in self.cmd.motor_cmd:
            motor.mode=0x0A;motor.q=2.146e9;motor.dq=16000.;motor.kp=0.;motor.kd=0.;motor.tau=0.
        ChannelFactoryInitialize(0,interface)
        def low_callback(msg):
            now=time.monotonic()
            try:
                remote=decode_remote(msg.wireless_remote)
                data=dict(q=[float(m.q) for m in msg.motor_state[:12]],dq=[float(m.dq) for m in msg.motor_state[:12]],
                    gyro=list(msg.imu_state.gyroscope),quaternion=list(msg.imu_state.quaternion),remote=remote,
                    battery=float(msg.bms_state.soc),foot_force=list(msg.foot_force),
                    bit_flag=int(msg.bit_flag),tau_est=[float(m.tau_est) for m in msg.motor_state[:12]],
                    motor_lost=[int(m.lost) for m in msg.motor_state[:12]],power_v=float(msg.power_v),power_a=float(msg.power_a))
                with self.lock:
                    buttons=set(remote['buttons'])
                    if {'L2','B'}<=buttons and not {'L2','B'}<=self.remote_buttons:
                        self.abort_immediate()
                    self.remote_buttons=buttons
                    if int(msg.tick)!=self.latest_tick:self.latest_tick=int(msg.tick);self.tick_changed=now
                    self.low=(now,data)
                    if (self.native_return.get('status')=='running' or self.locomotion_return.get('status')=='running') and now-self.handover_trace_at>=.02:
                        self.handover_trace_at=now
                        self.control_trace.append(dict(skill='tilt_locomotion' if self.locomotion_return.get('status')=='running' else 'attitude',attempt=self.locomotion_attempt if self.locomotion_return.get('status')=='running' else self.attitude_attempt,
                            at=time.time(),record_type='native_handover_sample',owned=self.owned,
                            owner=self.owner_name,q_sdk=data['q'],dq_sdk=data['dq'],
                            imu_quaternion=data['quaternion'],gyro=data['gyro'],
                            foot_force=data['foot_force'],remote=data['remote']))
            except Exception:
                with self.lock:self.low=None
        def sport_callback(key):
            def callback(msg):
                now=time.monotonic();stamp=int(getattr(getattr(msg,'sample_info',None),'source_timestamp',0))
                data=dict(position=list(msg.position),velocity=list(msg.velocity),native_mode=int(msg.mode),
                          position_quaternion=list(msg.imu_state.quaternion),position_source=key,
                          body_height=float(msg.body_height))
                with self.lock:
                    old=self.sport.get(key)
                    changed=now if old is None or stamp!=old[3] else old[2]
                    self.sport[key]=(now,data,changed,stamp)
            return callback
        for topic,kind,callback in [('rt/lowstate',LowState_,low_callback),
                                    ('rt/odommodestate',SportModeState_,sport_callback('odom')),
                                    ('rt/lf/odommodestate',SportModeState_,sport_callback('lf_odom')),
                                    ('rt/sportmodestate',SportModeState_,sport_callback('sport')),
                                    ('rt/lf/sportmodestate',SportModeState_,sport_callback('lf'))]:
            sub=ChannelSubscriber(topic,kind);sub.Init(callback,1);self.subscribers.append(sub)
        self.switch=MotionSwitcherClient();self.switch.SetTimeout(1.);self.switch.Init()
        def sport_factory():
            client=SportClient();client.SetTimeout(1.);client.Init();return client
        self.sport_factory=sport_factory
        self.sport_client=sport_factory()
        self.writer=threading.Thread(target=self._writer,daemon=True,name='b2-writer');self.writer.start()
        self.monitor=threading.Thread(target=self._owner_monitor,daemon=True,name='b2-owner');self.monitor.start()

    def _owner_monitor(self):
        while not self.closed:
            try:
                with self.operation_lock:
                    code,data=self.switch.CheckMode()
                    with self.lock:
                        self.owner_checked=time.monotonic()
                        if code==0 and isinstance(data,dict):
                            self.owner_name=data.get('name');self.owner_error=None
                            if self.owner_name:
                                self.native_owner=self.owner_name
                                # This is an actual transfer of control, not a fault latch.
                                self.owned=False;self.target=None;self.active_skill=None;self.damping=False
                        else:self.owner_error=f'控制权查询失败，SDK 返回 {code}'
            except Exception as e:
                with self.lock:self.owner_error=str(e)
            time.sleep(.2)

    def telemetry(self):
        now=time.monotonic()
        with self.lock:
            if self.low is None:return Telemetry()
            received,low=self.low
            t=Telemetry(received=received,tick_changed=self.tick_changed,source='DDS',**low)
            # The odometry estimator continues after ReleaseMode; sport state
            # is published by the controller being released. Never mix its
            # coordinate origin into an ongoing hindstand trajectory.
            sources=('odom','lf_odom') if self.active_skill=='hindstand' else ('odom','lf_odom','sport','lf')
            for key in sources:
                record=self.sport.get(key)
                if record and record[3]>0 and now-record[0]<.2 and now-record[2]<.2:
                    t.position_received=record[0];t.position_changed=record[2]
                    for k,v in record[1].items():setattr(t,k,v)
                    break
            # The odometry topic carries a placeholder mode=0. It is a valid
            # pose source, but only the sport controller reports its actual mode.
            t.native_mode=None
            for key in ('sport','lf'):
                record=self.sport.get(key)
                if record and record[3]>0 and now-record[0]<.2 and now-record[2]<.2:
                    t.native_mode=record[1]['native_mode']
                    t.body_height=record[1]['body_height']
                    break
            return t

    def controller_state(self):
        with self.lock:
            if self.owned:return self.active_skill if self.active_skill in ('hindstand','attitude') else 'trained'
            if self.owner_name:return 'native'
            return 'unowned' if self.owner_name=='' and not self.owner_error else 'unknown'

    def output_mode(self):
        with self.lock:return 'damping' if self.owned and self.damping else 'policy'

    def capabilities(self):
        return dict(native=dict(ready=True,reason=''),
                    trained=dict(ready='locomotion' in self.controllers,reason=''),
                    hindstand=dict(ready='hindstand' in self.controllers,reason=''),
                    attitude=dict(ready='attitude' in self.controllers,reason=''))

    def standing_reference(self):
        skill='attitude' if self.active_skill=='attitude' else 'locomotion' if self.tilt_enabled() else None
        if skill is None:return None
        c=self.controllers[skill];q=self.np.empty(12,dtype=self.np.float32);q[c.perm]=c.q0
        return q.tolist()

    def _checked(self,result,label):
        code=result[0] if isinstance(result,tuple) else result
        if code!=0:raise RuntimeError(f'{label}失败，SDK 返回 {code}')

    def _posture_now(self,stable=False):
        t=self.telemetry();c=posture_checks(t,time.monotonic(),self.standing_reference())
        if not c['telemetry'] or not c['posture']:raise Rejected('切换需要机器人处于站立姿态')
        if stable and not c['stationary']:raise Rejected('启动动作需要稳定站立姿态')
        return t

    def _root_position(self,t):
        if not t.position_fresh(time.monotonic()) or len(t.position)!=3 or not all(math.isfinite(v) for v in t.position):
            raise RuntimeError('后腿模型所需的当前根部位置数据不可用')
        if t.position_source not in ('odom','lf_odom'):
            raise RuntimeError('独立里程计数据尚未收到，后腿模型没有持续的位置输入')
        if self.hind_alignment is None:
            # Odometry and lowstate IMUs have a small, measured heading offset.
            # Rotate the measured trajectory into the policy IMU world frame;
            # the Controller anchors translation at the initial measured pose.
            local=self.rotation(t.quaternion);odometry=self.rotation(t.position_quaternion)
            yaw=math.atan2(local[1,0],local[0,0])-math.atan2(odometry[1,0],odometry[0,0])
            c,s=math.cos(yaw),math.sin(yaw)
            self.hind_alignment=self.np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=self.np.float32)
        return self.hind_alignment@self.np.asarray(t.position,dtype=self.np.float32)

    def _locomotion_return_posture(self):
        from standing_geometry import tilted_support
        t=self.telemetry();checks=posture_checks(t,time.monotonic(),self.standing_reference())
        if not (checks['telemetry'] and checks['stationary'] and checks['tilt_deg'] is not None
                and checks['tilt_deg']<=3. and tilted_support(t.q,t.quaternion)):
            raise Rejected('交回原生前需要四脚支撑、机身回正并稳定站立')
        return t

    def _takeover(self,skill):
        with self.operation_lock:
            with self.lock:epoch=self.command_epoch;was_owned=self.owned
            if skill not in self.controllers:raise Rejected('所选模型尚未加载')
            stable_required=skill in ('hindstand','attitude') or (skill=='locomotion' and self.tilt_enabled())
            t=self._posture_now(stable=stable_required)
            if skill=='hindstand':
                self.hind_alignment=None;self._root_position(t)
            released=False
            try:
                if not was_owned:
                    code,data=self.switch.CheckMode();self._checked(code,'查询控制权')
                    if not isinstance(data,dict):raise RuntimeError('控制权查询未返回控制器身份')
                    if data.get('name'):
                        self.native_owner=data['name']
                        self._checked(self.switch.ReleaseMode(),'释放原生控制');released=True
                    code,data=self.switch.CheckMode();self._checked(code,'确认低层控制权')
                    if not isinstance(data,dict) or data.get('name'):raise RuntimeError('原生控制器尚未释放控制权')
                    # Re-read posture after RPC; this is the same requested condition.
                    t=self._posture_now(stable=stable_required)
                    if skill=='hindstand':self._root_position(t)
                    if self.pub is None:self.pub=self.pub_factory();self.pub.Init()
                with self.lock:
                    if epoch!=self.command_epoch:raise Rejected('已按后到达的阻尼指令取消本次交接')
                    self.controllers[skill].reset();self.active_skill=skill
                    self.last_execution={}
                    if skill=='locomotion':
                        self.locomotion_attempt=str(time.time_ns());self.locomotion_return={}
                    if skill=='hindstand':self.hind_attempt=str(time.time_ns())
                    self.blend_at=time.monotonic();self.initial_target=self.np.asarray(t.q,dtype=self.np.float32)
                    self.target=list(t.q);self.target_at=time.monotonic();self.owned=True
                    self.damping=False;self.output_error=None;self.owner_name='';self.owner_checked=time.monotonic()
            except Exception:
                if released:
                    self._checked(self.switch.SelectMode(self.native_owner),'交接未完成后恢复原生控制')
                    with self.lock:self.owned=False;self.target=None;self.active_skill=None;self.owner_name=self.native_owner
                    if epoch!=self.command_epoch:self._checked(self.sport_client.Damp(),'执行后到达的阻尼指令')
                raise

    def start_trained(self,t):self._takeover('locomotion')
    def start_hindstand(self,t):self._takeover('hindstand')

    def start_attitude(self,t,plan):
        sequence=AttitudePlan(plan)
        self._takeover('attitude')
        with self.lock:
            self.attitude_plan=sequence;self.attitude_attempt=str(time.time_ns())
            self.native_return={}
            self.last_execution=dict(skill='attitude',attempt=self.attitude_attempt,**sequence.status())

    def _step(self,t,command,advance):
        now=time.monotonic()
        with self.lock:
            if not self.owned:raise RuntimeError('当前控制权不属于自训练控制器')
            if self.damping:return False
            skill=self.active_skill;epoch=self.command_epoch;c=self.controllers[skill]
        if not t.fresh(now):raise RuntimeError('当前本体数据不可用')
        data=[*t.q,*t.dq,*t.gyro,*t.quaternion,*command]
        if len(t.q)!=12 or len(t.dq)!=12 or len(t.gyro)!=3 or len(t.quaternion)!=4 or not all(math.isfinite(v) for v in data):
            raise RuntimeError('模型输入维度或数值无效')
        if sum(v*v for v in t.quaternion)==0:raise RuntimeError('IMU 四元数无效')
        position=self._root_position(t) if skill=='hindstand' else None
        start=time.perf_counter()
        target,debug=c.step(t.q,t.dq,t.gyro,t.quaternion,requested=command,
                            root_position=position,advance_reference=advance)
        if self.np.shape(target)!=(12,) or not self.np.all(self.np.isfinite(target)):raise RuntimeError('模型未输出有效的 12 维关节目标')
        alpha=min(1.,max(0.,(time.monotonic()-self.blend_at)/.5))
        target=self.initial_target*(1-alpha)+target*alpha
        self.inference_seconds=time.perf_counter()-start
        with self.lock:
            if epoch!=self.command_epoch or not self.owned or self.damping:return False
            self.target=target.tolist();self.target_at=time.monotonic();self.output_error=None
            if skill=='hindstand':
                self.last_execution=dict(at=time.time(),attempt=self.hind_attempt,reference_frame=c.frame,
                    total_frames=len(c.reference['pos'])-2,policy_time_s=c.frame*.02,
                    progress=c.frame/(len(c.reference['pos'])-2),position_source=t.position_source,
                    root_position=position.tolist(),inference_ms=self.inference_seconds*1000)
                self.control_trace.append(dict(**self.last_execution,q_sdk=list(t.q),target_sdk=list(self.target),
                    imu_quaternion=list(t.quaternion),foot_force=list(t.foot_force),
                    driver_state={k:self.low[1].get(k) for k in ('bit_flag','tau_est','motor_lost','power_v','power_a')} if self.low else {},
                    pd_output=dict(getattr(self,'last_pd_output',{}))))
            elif skill=='attitude':
                checks=posture_checks(t,now,self.standing_reference())
                stable=checks['telemetry'] and checks['posture'] and checks['stationary']
                complete=self.attitude_plan.advance(stable,c.context)
                self.last_execution=dict(skill='attitude',at=time.time(),attempt=self.attitude_attempt,
                    **self.attitude_plan.status(),**debug,inference_ms=self.inference_seconds*1000,
                    recovery_checks=checks,
                    native_restored=False,native_return=dict(self.native_return))
                self.control_trace.append(dict(**self.last_execution,q_sdk=list(t.q),dq_sdk=list(t.dq),
                    target_sdk=list(self.target),imu_quaternion=list(t.quaternion),foot_force=list(t.foot_force),
                    gyro=list(t.gyro),velocity=list(t.velocity),position_fresh=t.position_fresh(now),
                    pd_output=dict(getattr(self,'last_pd_output',{}))))
                return complete
            elif skill=='locomotion' and self.tilt_enabled():
                self.last_execution=dict(skill='tilt_locomotion',at=time.time(),attempt=self.locomotion_attempt,
                    **debug,inference_ms=self.inference_seconds*1000,native_return=dict(self.locomotion_return))
                self.control_trace.append(dict(**self.last_execution,q_sdk=list(t.q),dq_sdk=list(t.dq),
                    target_sdk=list(self.target),imu_quaternion=list(t.quaternion),foot_force=list(t.foot_force),
                    gyro=list(t.gyro),velocity=list(t.velocity),remote=t.remote,
                    driver_state={k:self.low[1].get(k) for k in ('bit_flag','tau_est','motor_lost','power_v','power_a')} if self.low else {},
                    pd_output=dict(getattr(self,'last_pd_output',{}))))
        return c.frame>=len(c.reference['pos'])-2 if c.reference is not None else False

    def execution_status(self):
        with self.lock:return dict(self.last_execution) if self.active_skill in ('hindstand','attitude','locomotion') or self.last_execution.get('skill') in ('attitude','tilt_locomotion') else {}

    def tilt_neutral(self):
        with self.lock:
            c=self.controllers['locomotion']
            return (self.tilt_enabled() and self.np.max(self.np.abs(c.context[:3]))<.01
                and self.np.max(self.np.abs(c.context[3:5]))<math.radians(1.)
                and self.np.max(self.np.abs(c.context[5:7]))<math.radians(1.))

    def begin_locomotion_return(self):
        with self.lock:
            if self.locomotion_return.get('status')=='running':return
            epoch=self.command_epoch;attempt=self.locomotion_attempt
            self.locomotion_return=dict(status='running',attempt=attempt)
        def work():
            try:
                self.return_native(self.telemetry(),'locomotion_complete',require_standing=True,expected_epoch=epoch)
                result=dict(status='succeeded',attempt=attempt)
            except Exception as e:result=dict(status='failed',attempt=attempt,error=str(e))
            with self.lock:
                if epoch==self.command_epoch and attempt==self.locomotion_attempt:
                    self.locomotion_return=result
                    self.last_execution['native_return']=dict(result)
                    self.last_execution['native_restored']=result['status']=='succeeded'
            self._return_trace('finished',result=result)
        self.handover_thread=threading.Thread(target=work,daemon=True,name='b2-locomotion-return')
        self.handover_thread.start()

    def locomotion_return_status(self):
        with self.lock:return dict(self.locomotion_return)

    def cancel_locomotion_return(self):
        with self.lock:
            self.command_epoch+=1
            self.locomotion_return=dict(status='cancelled',attempt=self.locomotion_attempt)

    def drain_trace(self):
        with self.lock:
            records=list(self.control_trace);self.control_trace.clear();return records

    def _return_trace(self,stage,**extra):
        t=self.telemetry()
        with self.lock:
            tilt=getattr(self,'locomotion_return',{}).get('status') in ('running','succeeded','failed','cancelled')
            self.control_trace.append(dict(skill='tilt_locomotion' if tilt else 'attitude',attempt=getattr(self,'locomotion_attempt',None) if tilt else self.attitude_attempt,
                at=time.time(),record_type='native_handover',stage=stage,
                owned=self.owned,owner=self.owner_name,q_sdk=list(t.q),dq_sdk=list(t.dq),
                imu_quaternion=list(t.quaternion),gyro=list(t.gyro),foot_force=list(t.foot_force),
                checks=posture_checks(t,time.monotonic(),self.standing_reference()),**extra))

    def begin_attitude_return(self):
        """Keep HTTP/telemetry responsive while the native service reconnects."""
        with self.lock:
            if self.native_return.get('status')=='running':return
            attempt=self.attitude_attempt
            self.native_return=dict(status='running',attempt=attempt)
            self.last_execution['native_return']=dict(self.native_return)
        def work():
            try:
                self.return_native(self.telemetry(),'attitude_complete',require_standing=True)
                result=dict(status='succeeded',attempt=attempt)
            except Exception as e:
                result=dict(status='failed',attempt=attempt,error=str(e))
            with self.lock:
                if self.attitude_attempt==attempt:
                    self.native_return=result
                    self.last_execution['native_return']=dict(result)
                    self.last_execution['native_restored']=result['status']=='succeeded'
            self._return_trace('finished',result=result)
        self.handover_thread=threading.Thread(target=work,daemon=True,name='b2-native-return')
        self.handover_thread.start()

    def attitude_return_status(self):
        with self.lock:return dict(self.native_return)

    def _refresh_sport_client(self):
        # ReleaseMode/SelectMode restarts the sport service. Recreate the DDS
        # endpoints, then use a read-only version request to confirm readiness.
        factory=getattr(self,'sport_factory',None)
        if factory is None:return
        old=self.sport_client
        self.sport_client=factory()
        # The pinned SDK has no public Client.Close(). Close both of its known
        # channels so repeated sequences do not leave callback threads behind.
        stub=old._ClientBase__stub
        stub._ClientStub__sendChannel.CloseWriter()
        stub._ClientStub__recvChannel.CloseReader()

    def _wait_native_service(self,epoch):
        client=self.sport_client
        deadline=time.monotonic()+10.
        client.SetTimeout(.5)
        try:
            while True:
                if epoch!=self.command_epoch:raise Rejected('后到达的阻尼指令已取代本次操作')
                code,version=client.GetServerApiVersion()
                if code==0:return version
                # A read-only query may be retried even if its reply was lost.
                if code not in (3102,3104) or time.monotonic()>=deadline:
                    self._checked(code,'等待原生运动服务')
                time.sleep(.05)
        finally:client.SetTimeout(25.)

    def _native_call(self,fn,label,epoch=None):
        # 3102 is CLIENT_SEND: no DDS server reader was matched. SelectMode can
        # return before the restarted sport server is discoverable. Retry only
        # unsent requests, never a dispatched request with an unknown result.
        deadline=time.monotonic()+5.
        while True:
            if epoch is not None and epoch!=self.command_epoch:raise Rejected('后到达的阻尼指令已取代本次操作')
            result=fn();code=result[0] if isinstance(result,tuple) else result
            if code!=3102 or time.monotonic()>=deadline:self._checked(result,label);return
            time.sleep(.05)

    def step_trained(self,t,command):self._step(t,command,False)
    def step_hindstand(self,t):return self._step(t,[0,0,0],True)
    def step_attitude(self,t):return self._step(t,self.attitude_plan.target(),False)

    def native_classic(self,replay=False):
        # The native firmware already receives the physical R1 packet.
        if replay:self._native_call(lambda:self.sport_client.ClassicWalk(True),'原生经典模式')

    def return_native(self,t,event,require_standing=False,expected_epoch=None):
        with self.operation_lock:
            with self.lock:epoch=self.command_epoch
            if expected_epoch is not None and epoch!=expected_epoch:raise Rejected('本次切换已被后到达的操作取消')
            if require_standing:t=self._locomotion_return_posture() if event=='locomotion_complete' else self._posture_now()
            standing=posture_checks(t,time.monotonic(),self.standing_reference())['posture']
            if self.active_skill=='locomotion' and self.tilt_enabled() and not standing:
                from standing_geometry import tilted_support
                standing=tilted_support(t.q,t.quaternion)
            attitude_return=event in ('attitude_complete','locomotion_complete')
            if attitude_return:
                self._return_trace('prepare_transport')
                # Build the fresh client while the 500 Hz writer still holds
                # the final standing target. Do not leave a gap for discovery.
                self._refresh_sport_client()
                t=self._locomotion_return_posture() if event=='locomotion_complete' else self._posture_now()
            with self.lock:
                if epoch!=self.command_epoch:raise Rejected('后到达的阻尼指令已取代本次操作')
                self.owned=False;self.target=None
            if attitude_return:self._return_trace('select_native')
            self._checked(self.switch.SelectMode(self.native_owner),'选择原生控制')
            code,data=self.switch.CheckMode();self._checked(code,'核对原生控制')
            if not isinstance(data,dict) or not data.get('name'):raise RuntimeError('原生控制恢复未确认')
            with self.lock:
                self.owner_name=data['name'];self.owner_error=None;self.active_skill=None;self.damping=False
            if not attitude_return:
                # A system-key handover restarts the same native service as an
                # automatic return. Its old DDS client can lose the server.
                # First relinquish custom output, then prepare a fresh client;
                # recovery/damping must not keep a failed policy driving motors
                # during endpoint discovery.
                self._refresh_sport_client()
            if epoch!=self.command_epoch:
                self._native_call(self.sport_client.Damp,'执行后到达的阻尼指令')
                raise Rejected('已执行后到达的阻尼指令，之前的交接操作已取消')
            if attitude_return:
                try:
                    version=self._wait_native_service(epoch)
                    self._return_trace('service_ready',api_version=version)
                    # Already standing: no StandDown/StandUp/RecoveryStand cycle.
                    # Restore the same native classic mode as the R1 double click.
                    self._native_call(lambda:self.sport_client.ClassicWalk(True),'恢复原生经典行走模式',epoch)
                    code,data=self.switch.CheckMode();self._checked(code,'确认原生遥控接管')
                    if not isinstance(data,dict) or data.get('name')!=self.native_owner:
                        raise RuntimeError('原生经典模式应答后，控制权未确认')
                    if epoch!=self.command_epoch:raise Rejected('后到达的阻尼指令已取代本次操作')
                    self._return_trace('classic_acknowledged',native_owner=data['name'])
                finally:self.sport_client.SetTimeout(1.)
                return
            try:
                # Query readiness without moving first. A motion timeout has
                # an unknown outcome and must never be automatically replayed.
                self._wait_native_service(epoch)
                mapping=dict(stand_toggle=self.sport_client.StandDown if standing else self.sport_client.StandUp,
                             recovery=self.sport_client.RecoveryStand,running=lambda:self.sport_client.FastWalk(True),
                             free_walk=self.sport_client.FreeWalk,pose=lambda:self.sport_client.FreeEuler(True),
                             vision=lambda:self.sport_client.VisionWalk(True))
                if event:self._native_call(mapping[event],'执行原生遥控功能',epoch)
                else:self._native_call(self.sport_client.BalanceStand,'原生站立',epoch)
            finally:self.sport_client.SetTimeout(1.)

    def stop(self,active):
        if active=='native':self._checked(self.sport_client.StopMove(),'停止原生运动')

    def damp(self):
        self.abort_immediate()
        with self.operation_lock:
            with self.lock:owned=self.owned
            if not owned:self._checked(self.sport_client.Damp(),'原生阻尼模式')

    def abort_immediate(self):
        # Only the operator's L2+B / damping command changes this mode. A later
        # normal command starts from the new epoch and needs no unlock flag.
        with self.lock:
            self.command_epoch+=1
            if self.owned:self.damping=True;self.target=None

    def suspend_output(self):
        with self.lock:self.target=None

    def _publish_once(self):
        with self.lock:
            if not self.owned or self.pub is None or (self.target is None and not self.damping):return
            wire_target=self.target
            if not self.damping and (self.active_skill in ('hindstand','attitude') or (self.active_skill=='locomotion' and self.tilt_enabled())):
                if self.low is None:
                    self.output_error='当前本体数据不可用，无法换算 PD 指令'
                    return
                sample=self.low[1];meta=self.controllers[self.active_skill].meta
                limits=[0.]*12
                for index,sdk_index in enumerate(meta['policy_to_sdk']):limits[sdk_index]=meta['torque_limits'][index]
                wire_target,requested,limited=saturated_pd_target(self.target,sample['q'],sample['dq'],
                    meta.get('kp',250.),meta.get('kd',8.),limits,meta.get('torque_envelope'))
                self.last_pd_output=dict(at=time.time(),sample_age_s=time.monotonic()-self.low[0],
                    wire_target=wire_target,requested_torque=requested,limited_torque=limited)
            for i in range(12):
                motor=self.cmd.motor_cmd[i];motor.mode=0x0A;motor.dq=0.;motor.tau=0.
                motor.q=2.146e9 if self.damping else float(wire_target[i])
                motor.kp=0. if self.damping else 250.;motor.kd=8.
            try:
                self.cmd.crc=self.crc.Crc(self.cmd)
                if self.pub.Write(self.cmd):self.writer_packets+=1;self.output_error=None
                else:self.output_error='电机命令发布失败'
            except Exception as e:self.output_error='电机命令发布失败：'+str(e)

    def _writer(self):
        deadline=time.monotonic()
        while not self.closed:
            self._publish_once()
            deadline+=.002;time.sleep(max(0,deadline-time.monotonic()))
            if time.monotonic()-deadline>.02:deadline=time.monotonic()

    def close(self):
        with self.lock:self.owned=False;self.target=None;self.closed=True
        self.writer.join(1);self.monitor.join(2)
        for sub in self.subscribers:sub.Close()
        if self.pub:self.pub.Close()
