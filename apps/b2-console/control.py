"""B2 operator state: standing to switch; stable standing to start the stunt."""
from dataclasses import dataclass, field
from collections import deque
import math
import struct
import time

NOMINAL=[0.,.67,-1.3]*4
BUTTONS=['R1','L1','START','SELECT','R2','L2','F1','F3','A','B','X','Y','UP','RIGHT','DOWN','LEFT']


class Rejected(RuntimeError):
    pass


def decode_remote(payload):
    if len(payload)!=40:raise ValueError('遥控器数据长度不正确')
    lx,rx,ry,_,ly=struct.unpack_from('<5f',bytes(payload),4)
    if not all(math.isfinite(v) and abs(v)<=1.05 for v in (lx,ly,rx,ry)):
        raise ValueError('遥控器数据异常')
    mask=int(payload[2]) | int(payload[3])<<8
    return dict(buttons=[name for i,name in enumerate(BUTTONS) if mask&(1<<i)],
                axes=dict(lx=lx,ly=ly,rx=rx,ry=ry))


class RemoteEvents:
    """Interpret independent button edges, including a released R1 double click."""
    def __init__(self):
        self.previous=set();self.last_single={};self.last_time=None

    def reset(self):self.__init__()

    def feed(self,remote,now):
        buttons=set(remote['buttons'])
        if self.last_time is None or now-self.last_time>.15:
            self.previous=buttons;self.last_single.clear();self.last_time=now
            return []
        self.last_time=now;rising=buttons-self.previous;events=[]
        if {'L2','B'}<=buttons and ({'L2','B'} & rising):events=['damp']
        elif {'L2','A'}<=buttons and ({'L2','A'} & rising):events=['stand_toggle']
        elif {'L2','X'}<=buttons and ({'L2','X'} & rising):events=['recovery']
        elif {'L2','START'}<=buttons and ({'L2','START'} & rising):events=['running']
        elif {'R2','Y'}<=buttons and ({'R2','Y'} & rising):events=['toggle_locomotion']
        else:
            for button,event in [('START','free_walk'),('SELECT','pose'),('X','vision')]:
                if button in rising:events.append(event)
            for button,event in [('R1','classic'),('L1','high_speed'),('L2','low_speed')]:
                if button in rising and buttons=={button}:
                    previous=self.last_single.pop(button,None)
                    if previous is not None and .08<=now-previous<=.5:events.append(event)
                    else:self.last_single[button]=now
        if any(e in ('damp','stand_toggle','recovery','running') for e in events):self.last_single.clear()
        self.previous=buttons
        return events


@dataclass
class Telemetry:
    received:float=0.
    tick_changed:float=0.
    position_received:float=0.
    position_changed:float=0.
    q:list=field(default_factory=list)
    dq:list=field(default_factory=list)
    quaternion:list=field(default_factory=list)
    gyro:list=field(default_factory=list)
    position:list=field(default_factory=list)
    position_quaternion:list=field(default_factory=list)
    position_source:str|None=None
    body_height:float|None=None
    velocity:list=field(default_factory=list)
    remote:dict=field(default_factory=lambda:dict(buttons=[],axes=dict(lx=0,ly=0,rx=0,ry=0)))
    battery:float|None=None
    native_mode:int|None=None
    foot_force:list=field(default_factory=list)
    bit_flag:int|None=None
    tau_est:list=field(default_factory=list)
    motor_lost:list=field(default_factory=list)
    power_v:float|None=None
    power_a:float|None=None
    source:str='unavailable'

    def fresh(self,now):
        return self.received>0 and 0<=now-self.received<.1 and self.tick_changed>0 and 0<=now-self.tick_changed<.1

    def position_fresh(self,now):
        return (self.position_received>0 and 0<=now-self.position_received<.2 and
                self.position_changed>0 and 0<=now-self.position_changed<.2)


def posture_checks(t,now,reference=None):
    """Classify posture without dwell, neutral-stick or approval conditions.

    Root position is not required to recognize standing. Available velocity is
    used to recognize stable standing; missing actor inputs are runtime errors.
    """
    result=dict(telemetry=t.fresh(now),posture=False,stationary=False,
                joint_error_rad=None,tilt_deg=None,speed_m_s=None,height_m=None,
                position_available=t.position_fresh(now))
    if t.position_fresh(now):
        if t.position_source not in ('odom','lf_odom'):
            if len(t.position)==3 and all(math.isfinite(v) for v in t.position):result['height_m']=t.position[2]
        elif t.body_height is not None and math.isfinite(t.body_height) and t.body_height>0:
            result['height_m']=t.body_height
        if len(t.velocity)==3 and all(math.isfinite(v) for v in t.velocity):
            result['speed_m_s']=math.sqrt(sum(v*v for v in t.velocity))
    if len(t.q)!=12 or len(t.quaternion)!=4 or not all(math.isfinite(v) for v in [*t.q,*t.quaternion]):return result
    norm=math.sqrt(sum(v*v for v in t.quaternion))
    if not .9<norm<1.1:return result
    w,x,y,z=[v/norm for v in t.quaternion]
    tilt=math.acos(max(-1,min(1,1-2*(x*x+y*y))))
    error=max(abs(a-b) for a,b in zip(t.q,NOMINAL))
    if reference is not None:
        error=min(error,max(abs(a-b) for a,b in zip(t.q,reference)))
    joint_posture=error<=.25
    geometry_posture=False
    if reference is not None:
        from standing_geometry import standing_feet
        geometry_posture,feet=standing_feet(t.q,[NOMINAL,reference])
        result.update(foot_positions_relative_hip_m=feet)
    valid_motion=(len(t.dq)==12 and len(t.gyro)==3 and all(math.isfinite(v) for v in [*t.dq,*t.gyro]))
    stable=valid_motion and max(abs(v) for v in t.dq)<=.25 and math.sqrt(sum(v*v for v in t.gyro))<=.1
    if result['speed_m_s'] is not None:stable=stable and result['speed_m_s']<=.04
    result.update(joint_error_rad=error,tilt_deg=math.degrees(tilt),
                  posture=(joint_posture or geometry_posture) and tilt<=.14,stationary=bool(stable),
                  posture_basis='joint_pose' if joint_posture else 'foot_geometry' if geometry_posture else None)
    return result


class StandingGate:
    def __init__(self):self.value={}

    def update(self,t,now,reference=None):
        checks=posture_checks(t,now,reference)
        standing=checks['telemetry'] and checks['posture']
        self.value=dict(**checks,switch_allowed=standing,
                        hindstand_allowed=standing and checks['stationary'],
                        attitude_allowed=standing and checks['stationary'])
        return self.value


class Engine:
    def __init__(self,backend,clock=time.monotonic):
        self.backend=backend;self.clock=clock;self.gate=StandingGate();self.remote=RemoteEvents()
        self.telemetry=Telemetry();self.selected='native';self.active='native';self.phase='idle'
        self.revision=0;self.last_error=None;self.speed_level='low';self.last_command=[0.,0.,0.]
        self.events=deque(maxlen=80);self.event_id=0;self.zero_next_command=False
        self.stunt_return='native';self.stunt_started=None
        self.attitude_return_pending=False;self.attitude_return_attempted=False
        self.tilt_enabled=getattr(backend,'tilt_enabled',lambda:False)()
        self.tilt_request=[0.,0.];self.tilt_last_time=None
        self.tilt_returning=False;self.tilt_handover=False
        self.changed('控制台已启动，等待操作')

    def changed(self,message,kind='info'):
        self.revision+=1;self.event_id+=1
        self.events.appendleft(dict(id=self.event_id,time=time.time(),message=message,kind=kind))

    def require_standing(self,stable=False):
        value=self.gate.update(self.telemetry,self.clock(),self.standing_reference())
        if not value['switch_allowed']:raise Rejected('切换需要机器人处于站立姿态')
        if stable and not value['hindstand_allowed']:raise Rejected('启动动作需要稳定站立姿态')

    def standing_reference(self):
        return getattr(self.backend,'standing_reference',lambda:None)()

    def request(self,action,value=None,expected_revision=None):
        # State versions are informational. Evaluate the current physical state.
        if action in ('attitude','hindstand') and (self.tilt_returning or self.tilt_handover):
            raise Rejected('运动系统正在切换；仍可使用趴下、恢复或阻尼按键')
        if action=='select':
            if self.tilt_enabled:raise Rejected('请在稳定站立时按 R2 + Y 切换运动系统')
            if value not in ('native','trained'):raise Rejected('未知 locomotion 策略')
            self.require_standing()
            if value==self.selected:return
            if self.active!='native':self._return_native(None,standing=True)
            self.selected=value;self.phase='await_r1';self.last_error=None
            self.remote.reset()
            self.changed('已选择'+('原生 locomotion' if value=='native' else '自训练 locomotion')+'；双击 R1 进入经典模式')
        elif action=='hindstand':
            self.require_standing(stable=True)
            if self.active!='hindstand':self.stunt_return=self.active
            self.phase='handover';self.changed('启动后腿站立')
            try:
                self.backend.start_hindstand(self.telemetry)
                self.active='hindstand';self.phase='hindstand';self.stunt_started=self.clock();self.last_error=None
                self.changed('后腿站立：起立 → 保持 → 回落 → 四足恢复')
            except Exception as e:self._failure(e);raise Rejected(str(e))
        elif action=='attitude':
            from attitude_plan import validate_plan
            try:plan=validate_plan(value)
            except ValueError as e:raise Rejected(str(e))
            self.require_standing(stable=True)
            self.phase='handover';self.changed('已确认姿态编排，共 '+str(len(plan['actions']))+' 步')
            try:
                self.backend.start_attitude(self.telemetry,plan)
                self.attitude_return_pending=False
                self.attitude_return_attempted=False
                self.active='attitude';self.phase='attitude';self.last_error=None
                self.changed('姿态编排开始：准备站立 → 按顺序执行 → 回正站立')
            except Exception as e:self._failure(e);raise Rejected(str(e))
        elif action=='stop':
            self.backend.stop(self.active)
            self.zero_next_command=True;self.last_error=None
            self.changed('已发送停止运动指令；不要求重新解锁')
        elif action=='damp':
            self.backend.damp();self.phase='damping';self.last_error=None
            self.attitude_return_pending=False
            self.tilt_returning=False;self.tilt_handover=False
            self.changed('已进入阻尼模式；原生恢复按键仍可使用')
        else:raise Rejected('未知操作')

    def _failure(self,error):
        message=str(error)
        try:self.backend.suspend_output()
        except Exception:pass
        if self.last_error!=message:self.changed(message,'error')
        self.last_error=message
        actual=getattr(self.backend,'controller_state',lambda:self.active)()
        if actual:self.active=actual
        self.phase='waiting_input' if self.active in ('trained','hindstand','attitude') else 'idle'

    def _return_native(self,event,standing=False):
        if self.tilt_returning or self.tilt_handover:
            getattr(self.backend,'cancel_locomotion_return',lambda:None)()
        self.tilt_returning=False;self.tilt_handover=False
        self.phase='handover';self.changed('正在恢复原生控制')
        try:
            self.backend.return_native(self.telemetry,event,require_standing=standing)
            self.attitude_return_pending=False
            self.active='native';self.phase='idle';self.last_error=None
            self.changed('原生控制已恢复'+('，执行对应遥控功能' if event else ''))
        except Exception as e:self._failure(e);raise Rejected(str(e))

    def handle_remote(self,event):
        if event=='toggle_locomotion':
            if self.tilt_enabled:self.toggle_locomotion()
            return
        if event=='damp':
            if self.active!='native':self.request('damp')
            return
        if event=='classic':
            if self.tilt_returning or self.tilt_handover:return
            if self.selected=='trained':
                if self.active=='trained' and self.phase=='classic':return
                self.require_standing(stable=self.tilt_enabled);self.phase='handover'
                try:self.backend.start_trained(self.telemetry)
                except Exception as e:self._failure(e);raise Rejected(str(e))
                self.active='trained';self.phase='classic';self.last_error=None
                self.tilt_request=[0.,0.];self.tilt_last_time=self.clock()
                self.tilt_returning=False;self.tilt_handover=False
                self.attitude_return_pending=False
                self.changed('双击 R1：自训练经典模式')
            else:
                replay=self.active!='native'
                if replay:self.require_standing();self._return_native(None,standing=True)
                self.backend.native_classic(replay=replay)
                self.active='native';self.phase='classic';self.last_error=None
                self.changed('双击 R1：原生经典模式')
        elif event in ('high_speed','low_speed'):
            self.speed_level='high' if event=='high_speed' else 'low'
            self.changed('遥控器：'+('高速' if self.speed_level=='high' else '低速'))
        elif event in ('stand_toggle','recovery','running','free_walk','pose','vision'):
            # Original remote functions are not a website policy selection.
            # Do not suppress them with a stunt or fault latch.
            if self.active!='native':self._return_native(event)

    def _tilt_stable(self,allow_tilt=False):
        value=posture_checks(self.telemetry,self.clock(),self.standing_reference())
        posture=value['posture']
        if allow_tilt and not posture:
            from standing_geometry import tilted_support
            posture=tilted_support(self.telemetry.q,self.telemetry.quaternion)
        return value['telemetry'] and posture and value['stationary']

    def toggle_locomotion(self):
        if self.tilt_returning or self.tilt_handover:return
        if self.active not in ('native','trained') or self.phase=='damping':
            raise Rejected('请先恢复四足稳定站立，再按 R2 + Y')
        if not self._tilt_stable(allow_tilt=self.active=='trained'):
            raise Rejected('R2 + Y 仅在四足稳定站立时切换')
        if self.active=='trained':
            self.tilt_request=[0.,0.];self.tilt_returning=True
            self.phase='neutralizing';self.last_error=None
            self.changed('R2 + Y：回正站稳后交回原生经典模式')
        elif self.selected=='trained':
            self.selected='native';self.changed('R2 + Y：已选择原生运动系统')
        else:
            try:self.backend.start_trained(self.telemetry)
            except Exception as e:self._failure(e);raise Rejected(str(e))
            self.selected='trained';self.active='trained';self.phase='classic';self.last_error=None
            self.tilt_request=[0.,0.];self.tilt_last_time=self.clock()
            self.changed('R2 + Y：自训练倾斜行走已接管，方向键调整 roll/pitch')

    def _tilt_command(self,now,cmd):
        elapsed=0. if self.tilt_last_time is None else now-self.tilt_last_time
        self.tilt_last_time=now
        # Never catch up a held button across a transport gap or blocking RPC.
        dt=elapsed if 0.<=elapsed<=.1 else 0.
        buttons=set(self.telemetry.remote['buttons'])
        rate=math.radians(5.)*dt;limit=math.radians(20.)
        for i,direction in enumerate((int('RIGHT' in buttons)-int('LEFT' in buttons),int('UP' in buttons)-int('DOWN' in buttons))):
            self.tilt_request[i]=max(-limit,min(limit,self.tilt_request[i]+rate*direction))
        return [*cmd,*self.tilt_request]

    def _tick_tilt_return(self):
        if self.tilt_handover:
            result=self.backend.locomotion_return_status()
            if result.get('status')=='running':
                if self.backend.controller_state()=='trained':
                    try:self.backend.step_trained(self.telemetry,[0.]*5)
                    except Exception as e:
                        if self.backend.controller_state()=='trained':self._failure(e)
                self.phase='handover';return
            self.tilt_handover=False;self.tilt_returning=False
            self.active=self.backend.controller_state()
            if result.get('status')=='succeeded':
                self.active='native';self.selected='native';self.phase='classic';self.last_error=None
                self.remote.reset();self.changed('原生经典模式已恢复，可直接使用遥控器移动')
            else:
                self.phase='handover_failed';self.last_error=result.get('error','原生交接未确认')
                self.changed(self.last_error,'error')
            return
        try:
            self.last_command=[0.]*5
            self.backend.step_trained(self.telemetry,self.last_command)
            self.phase='neutralizing'
            checks=posture_checks(self.telemetry,self.clock(),self.standing_reference())
            if self._tilt_stable(allow_tilt=True) and checks['tilt_deg']<=3. and getattr(self.backend,'tilt_neutral',lambda:False)():
                self.backend.begin_locomotion_return();self.tilt_handover=True;self.phase='handover'
        except Exception as e:self._failure(e)

    def tick(self):
        now=self.clock();self.telemetry=self.backend.telemetry();self.gate.update(self.telemetry,now,self.standing_reference())
        if not self.telemetry.fresh(now):
            self.remote.reset()
            self.tilt_last_time=None
            if self.active!='native':self._failure(RuntimeError('未收到当前本体数据，无法计算新的控制输出'))
            return
        actual=getattr(self.backend,'controller_state',lambda:self.active)()
        if actual in ('native','unowned') and self.active!=actual and not self.attitude_return_pending and not self.tilt_handover:
            self.active=actual;self.phase='idle'
            self.tilt_returning=False
            self.changed('控制权当前属于原生控制器' if actual=='native' else '当前没有运动控制器接管')
        if getattr(self.backend,'output_mode',lambda:'policy')()=='damping':self.phase='damping'
        for event in self.remote.feed(self.telemetry.remote,now):
            try:self.handle_remote(event)
            except Rejected as e:self.changed(str(e),'warning')
        # A mode RPC can take longer than one control tick. Do not pass the
        # pre-handover snapshot into the first policy step.
        self.telemetry=self.backend.telemetry();self.gate.update(self.telemetry,self.clock(),self.standing_reference())
        if self.phase=='damping':return
        if self.tilt_returning or self.tilt_handover:
            self._tick_tilt_return();return
        if self.attitude_return_pending:
            result=self.backend.attitude_return_status()
            if result.get('status')=='running':
                if getattr(self.backend,'controller_state',lambda:self.active)()=='attitude':
                    try:self.backend.step_attitude(self.telemetry)
                    except Exception as e:
                        # Ownership may have moved between the check and step.
                        if self.backend.controller_state()=='attitude':self._failure(e)
                self.phase='handover';return
            self.attitude_return_pending=False
            if result.get('status')=='succeeded':
                self.active='native';self.selected='native';self.phase='classic';self.last_error=None
                self.remote.reset()
                self.changed('姿态编排完成，已恢复原生经典行走模式，可使用遥控器移动')
            else:
                # A service error is not a bad Actor input. If ownership has
                # not moved yet, keep the model balancing at zero angle.
                actual=getattr(self.backend,'controller_state',lambda:self.active)()
                if actual in ('native','unowned'):self.active=actual
                self.last_error=result.get('error','原生控制交接结果未知')
                self.phase='handover_failed' if self.active=='attitude' else 'idle'
                self.changed(self.last_error,'error')
            return
        if self.active=='trained':
            axes=self.telemetry.remote['axes'];level=(4. if self.tilt_enabled else 2.) if self.speed_level=='high' else .5
            def dead(v):return 0 if abs(v)<.05 else math.copysign((min(1.,abs(v))-.05)/.95,v)
            lateral=(1. if self.tilt_enabled else .6) if self.speed_level=='high' else .3
            cmd=[level*dead(axes['ly']),-lateral*dead(axes['lx']),-2.*dead(axes['rx'])]
            norm=math.hypot(*cmd[:2])
            maximum=4. if self.tilt_enabled else 2.
            if norm>maximum:cmd[0]*=maximum/norm;cmd[1]*=maximum/norm
            if self.zero_next_command:cmd=[0.,0.,0.];self.zero_next_command=False
            if self.tilt_enabled:
                cmd[2]*=min(1.,1./max(1.,math.hypot(*cmd[:2])))
                cmd=self._tilt_command(self.clock(),cmd)
            self.last_command=cmd
            try:
                self.backend.step_trained(self.telemetry,cmd);self.last_error=None;self.phase='classic'
            except Exception as e:self._failure(e)
        elif self.active=='attitude':
            try:
                complete=self.backend.step_attitude(self.telemetry)
                if not self.attitude_return_attempted:self.last_error=None
                execution=self.backend.execution_status()
                self.phase='handover_failed' if self.attitude_return_attempted else 'recovering' if execution.get('plan_phase')=='recover' else 'attitude'
                if complete and not self.attitude_return_attempted:
                    self.phase='handover';self.attitude_return_pending=True
                    self.attitude_return_attempted=True
                    self.changed('已回正并站稳，正在交回原生遥控经典模式')
                    self.backend.begin_attitude_return()
            except Exception as e:self._failure(e)
        elif self.active=='hindstand':
            try:
                complete=self.backend.step_hindstand(self.telemetry)
                self.last_error=None;self.phase='recovering' if complete else 'hindstand'
                if complete and self.gate.value['switch_allowed']:
                    if self.stunt_return=='trained':
                        self.backend.start_trained(self.telemetry);self.active='trained';self.phase='classic'
                        self.changed('后腿动作结束，已恢复自训练 locomotion')
                    else:self._return_native(None,standing=True)
            except Exception as e:self._failure(e)

    def status(self):
        now=self.clock();t=self.telemetry;gate=self.gate.update(t,now,self.standing_reference())
        execution=getattr(self.backend,'execution_status',lambda:{})()
        return dict(revision=self.revision,connected=t.fresh(now),backend=self.backend.kind,
                    target='unitree@192.168.123.164',selected=self.selected,
                    active=self.active if t.fresh(now) else 'unknown',phase=self.phase,
                    error=self.last_error or getattr(self.backend,'output_error',None),
                    gate=gate,capabilities=self.backend.capabilities(),can_select=gate['switch_allowed'] and not self.tilt_enabled,
                    can_hindstand=gate['hindstand_allowed'],can_attitude=gate['attitude_allowed'],speed_level=self.speed_level,
                    command=self.last_command,remote=t.remote,battery=t.battery if t.fresh(now) else None,
                    telemetry_age_ms=round((now-t.received)*1000,1) if t.received else None,
                    native_mode=t.native_mode,events=list(self.events)[:20],
                    stunt_progress=execution.get('progress',0.) if self.active=='hindstand' else 0.,
                    execution=execution,position_source=t.position_source,
                    requirements=dict(switch='stable_standing' if self.tilt_enabled else 'standing',hindstand='stable_standing',attitude='stable_standing_then_confirm_plan',entry='R2+Y' if self.tilt_enabled else 'double_R1'),
                    tilt_control=dict(enabled=self.tilt_enabled,requested_degrees=[math.degrees(v) for v in self.tilt_request],
                        rate_deg_s=5.,limit_deg=20.,returning=self.tilt_returning,
                        switch_allowed=self._tilt_stable(allow_tilt=self.active=='trained') if self.tilt_enabled else False),
                    model_info=getattr(self.backend,'model_info',lambda:{})())
