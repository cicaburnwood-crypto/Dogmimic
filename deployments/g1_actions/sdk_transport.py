"""Unitree SDK2 G1 29DOF PR-mode transport. Startup is subscription-only."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import os,json,time,threading
from pathlib import Path
import numpy as np
from runtime import rotation

class SDKTransport:
    def __init__(self, interface, odom_topic='rt/odommodestate', domain=0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        ChannelFactoryInitialize(domain,interface)
        self.lock=threading.RLock();self.low=None;self.odom=None
        self.low_time=0.;self.odom_time=0.;self.low_tick=None;self.odom_tick=None
        self.publisher=None;self.target=None;self.publishing=False;self.pubthread=None;self.publish_error=None
        self.alignment=None;self.mode_machine=None;self.raw={};self.sensor_errors={};self.packet=None;self.packets_sent=0;self.handover_events=[]
        self.low_type=LowState_;self.odom_type=SportModeState_
        self.low_sub=self.latest_reader('rt/lowstate',LowState_,'low')
        self.odom_sub=self.latest_reader(odom_topic,SportModeState_,'odom')
        self.switch=MotionSwitcherClient();self.switch.SetTimeout(5.);self.switch.Init()
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        self.loco=LocoClient();self.loco.SetTimeout(2.);self.loco.Init()
        self.fsm=None;self.fsm_time=0.
        threading.Thread(target=self.poll_fsm,daemon=True).start()

    def latest_reader(self,topic,kind,key):
        from unitree_sdk2py.core.channel import ChannelFactory
        from cyclonedds.sub import DataReader,ddspy_take
        from cyclonedds.topic import Topic
        from cyclonedds.core import Listener
        from cyclonedds.qos import Qos,Policy
        def receive(reader):
            if reader is None:return
            try:
                batch=ddspy_take(reader._ref,64)
                if isinstance(batch,int):raise RuntimeError('DDS take %s'%batch)
                valid=[(raw,info) for raw,info in batch if info.valid_data]
                if valid:
                    raw,info=valid[-1];stamp=info.source_timestamp
                    with self.lock:
                        old=self.raw.get(key)
                        if old is None or stamp!=old[2]:
                            self.raw[key]=(raw,time.monotonic(),stamp)
            except Exception as exc:
                with self.lock:self.sensor_errors[key]=str(exc)
        topic_obj=Topic(ChannelFactory._ChannelFactory__participant,topic,kind)
        reader=DataReader(ChannelFactory._ChannelFactory__participant,topic_obj,
            Qos(Policy.History.KeepLast(1),Policy.Reliability.BestEffort),listener=Listener(on_data_available=receive))
        return topic_obj,reader

    def sensor_age(self,key,now):
        item=self.raw.get(key) if hasattr(self,'raw') else None
        return now-item[1] if item else now-(self.low_time if key=='low' else self.odom_time)

    def poll_fsm(self):
        while True:
            try:
                code,value=self.loco.GetFsmId()
                with self.lock:
                    self.fsm=int(value) if code==0 and value is not None else None
                    self.fsm_time=time.monotonic()
            except Exception:
                with self.lock:self.fsm=None
            time.sleep(.5)

    @classmethod
    def from_environment(cls):
        interface=os.environ.get('G1_INTERFACE')
        if not interface:raise ValueError('Set G1_INTERFACE to the robot-facing network interface')
        return cls(interface,os.environ.get('G1_ODOM_TOPIC','rt/odommodestate'),int(os.environ.get('G1_DOMAIN','0')))

    def on_low(self,msg):
        with self.lock:
            if msg.tick!=self.low_tick:
                self.low_time=time.monotonic();self.low_tick=msg.tick
            self.low=msg

    def on_odom(self,msg):
        with self.lock:
            stamp=(int(msg.stamp.sec),int(msg.stamp.nanosec))
            if stamp!=self.odom_tick:
                self.odom_time=time.monotonic();self.odom_tick=stamp
            self.odom=msg

    def snapshot(self):
        if hasattr(self,'raw'):
            with self.lock:low_raw=self.raw.get('low');odom_raw=self.raw.get('odom')
            # Decode only the newest sample, never a backlog of 1 kHz messages.
            low=self.low_type.deserialize(low_raw[0]) if low_raw else None
            odom=self.odom_type.deserialize(odom_raw[0]) if odom_raw else None
            with self.lock:
                if low is not None:self.low=low;self.low_time=low_raw[1]
                if odom is not None:self.odom=odom;self.odom_time=odom_raw[1]
        with self.lock:
            now=time.monotonic()
            if self.low is None or self.odom is None or now-self.low_time>.1 or now-self.odom_time>.1:
                raise RuntimeError('G1 lowstate / root odometry unavailable or stale')
            low,odom=self.low,self.odom
            R=rotation(low.imu_state.quaternion)
            if self.alignment is None:
                O=rotation(odom.imu_state.quaternion)
                yaw=np.arctan2(-R[0,1],R[1,1])-np.arctan2(-O[0,1],O[1,1])
                c,s=np.cos(yaw),np.sin(yaw);self.alignment=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            result=dict(q=[float(m.q) for m in low.motor_state[:29]],
                        dq=[float(m.dq) for m in low.motor_state[:29]],
                        tau_est=[float(m.tau_est) for m in low.motor_state[:29]],
                        gyro=list(low.imu_state.gyroscope),quaternion=list(low.imu_state.quaternion),
                        position=(self.alignment@np.asarray(odom.position)).tolist(),
                        velocity=(self.alignment@np.asarray(odom.velocity)).tolist(),
                        mode_machine=int(low.mode_machine),low_tick=int(low.tick),
                        odom_stamp=[int(odom.stamp.sec),int(odom.stamp.nanosec)],low_age_s=now-self.low_time,odom_age_s=now-self.odom_time,
                        native_fsm=self.fsm if now-self.fsm_time<3 else None,
                        height=float(odom.body_height) if odom.body_height>0 else None,battery=None)
            for key in ('q','dq','gyro','quaternion','position','velocity'):
                if not np.isfinite(result[key]).all():raise RuntimeError('Nonfinite '+key)
            return result

    def prepare(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        if self.publisher is None:
            publisher=ChannelPublisher('rt/lowcmd',LowCmd_)
            publisher.Init()  # SDK discovery sleep happens while native still owns control.
            self.publisher=publisher

    def acquire(self,meta=None):
        self.publish_error=None;self.handover_events=[]
        code,owner=self.switch.CheckMode()
        if code:raise RuntimeError(f'CheckMode failed: {code}')
        receipt={'owner':owner.get('name','')}
        self.prepare()
        if receipt['owner']:
            code,_=self.switch.ReleaseMode()
            self.handover_events.append({'event':'release_native','time':time.time(),'code':code})
            if code:raise RuntimeError(f'ReleaseMode failed: {code}')
        try:
            state=self.snapshot()
            if meta is not None:
                # Immediately hold measured posture during the ownership verification RPC.
                self.command(state['q'],meta,state)
            code,current=self.switch.CheckMode()
            if code or current.get('name'):raise RuntimeError('Low-level ownership was not transferred')
        except Exception as exc:
            try:self.release(receipt)
            except Exception as recovery:
                raise RuntimeError(f'{exc}; native recovery also failed: {recovery}') from exc
            raise
        return receipt

    def command(self,target,meta,state):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        if self.publish_error:raise RuntimeError(self.publish_error)
        target=np.asarray(target)
        if target.shape!=(29,) or not np.isfinite(target).all():raise ValueError('Invalid G1 target')
        perm=np.asarray(meta['policy_to_sdk']);kp=np.empty(29);kd=np.empty(29)
        kp[perm]=meta['kp'];kd[perm]=meta['kd']
        msg=unitree_hg_msg_dds__LowCmd_();msg.mode_pr=0;msg.mode_machine=int(state['mode_machine'])
        for i in range(29):
            m=msg.motor_cmd[i];m.mode=1;m.q=float(target[i]);m.dq=0.;m.tau=0.;m.kp=float(kp[i]);m.kd=float(kd[i])
        msg.crc=CRC().Crc(msg)  # Once per target update, not 500 times per second.
        with self.lock:
            self.packet=msg
            if self.publisher is None:raise RuntimeError('Publisher not prepared before takeover')
            if not self.publishing:
                self.publishing=True
                self.pubthread=threading.Thread(target=self.publish_loop,daemon=True);self.pubthread.start()

    def publish_loop(self):
        deadline=time.monotonic()
        try:
            while self.publishing:
                with self.lock:
                    now=time.monotonic();ages={k:self.sensor_age(k,now) for k in ('low','odom')}
                    if max(ages.values())>.1:raise RuntimeError('Sensor stream stale: '+json.dumps(ages))
                    msg=self.packet
                if self.publisher.Write(msg) is False:raise RuntimeError('DDS LowCmd write failed')
                self.packets_sent+=1
                deadline+=.002;time.sleep(max(0.,deadline-time.monotonic()))
                if time.monotonic()-deadline>.002:deadline=time.monotonic()
        except Exception as exc:self.publish_error=str(exc)
        finally:self.publishing=False

    def release(self,receipt):
        self.publishing=False
        if self.pubthread:
            self.pubthread.join(timeout=1.)
            if self.pubthread.is_alive():raise RuntimeError('Command publisher did not stop; no overlapping native takeover')
        desired=receipt['owner'] or 'ai'
        deadline=time.monotonic()+5.;last=None;accepted=False
        while True:
            code,current=self.switch.CheckMode()
            if code==0 and current.get('name')==desired:
                self.packet=None
                self.handover_events.append({'event':'native_restored','time':time.time(),'owner':desired})
                return
            if code==0 and current.get('name') not in ('',desired):
                raise RuntimeError('Control owner changed externally: '+current['name'])
            if not accepted:
                code,_=self.switch.SelectMode(desired);last=code
                self.handover_events.append({'event':'select_native','time':time.time(),'code':code,'owner':desired})
                if code not in (0,7002,3104):raise RuntimeError(f'SelectMode failed: {code}')
                accepted=code in (0,3104)  # Timeout is ambiguous: verify owner without resending.
            if time.monotonic()>=deadline:raise RuntimeError(f'Native restore not confirmed, last SelectMode code: {last}')
            time.sleep(.1)
