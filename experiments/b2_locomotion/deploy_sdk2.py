"""SDK2 transport for our actor. Defaults to receive-only remote/IMU inspection.

No SportClient, StandUp/Move service, official model, or reference policy is used.
Motor output requires --enable-motors, a passed simulation evaluation, low-level
control availability, and a local L1+Start arm from an upright standard stance.
"""
import argparse,time,threading,signal,json
from pathlib import Path
import numpy as np
import torch
from controller import Controller,decode_remote,gravity_from_quaternion

def main():
    p=argparse.ArgumentParser();p.add_argument('--interface',required=True);p.add_argument('--policy-dir',default=str(Path(__file__).parent))
    p.add_argument('--domain',type=int,default=0);p.add_argument('--yaw-gain',type=float,default=2.)
    p.add_argument('--enable-motors',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);c=Controller(a.policy_dir)
    if a.enable_motors and not c.meta['physical_eval_passed']:
        raise RuntimeError('This checkpoint has not passed the complete locomotion simulation evaluation')
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelPublisher,ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_,LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    ChannelFactoryInitialize(a.domain,a.interface)
    shared={'state':None,'received':0.,'target':None,'target_at':0.,'armed':False,'fault':None,'quit':False}
    lock=threading.Lock()
    def receive(msg):
        with lock:shared.update(state=msg,received=time.monotonic())
    sub=ChannelSubscriber('rt/lowstate',LowState_);sub.Init(receive,10)
    pub=None
    if a.enable_motors:
        switch=MotionSwitcherClient();switch.SetTimeout(5.);switch.Init()
        code,mode=switch.CheckMode()
        if code!=0 or not isinstance(mode,dict):raise RuntimeError('Cannot verify the B2 control mode')
        if mode.get('name'):
            raise RuntimeError('Another motion controller is active. Enter low-level control locally before starting this process.')
        pub=ChannelPublisher('rt/lowcmd',LowCmd_);pub.Init()
    def stop(*_):
        with lock:shared.update(fault='operator exit',quit=True)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def writer():
        cmd=unitree_go_msg_dds__LowCmd_();cmd.head=[0xFE,0xEF];cmd.level_flag=0xFF;cmd.gpio=0;crc=CRC()
        for motor in cmd.motor_cmd:
            motor.mode=1;motor.q=2.146e9;motor.dq=16000.;motor.kp=0.;motor.kd=0.;motor.tau=0.
        deadline=time.monotonic()
        while True:
            now=time.monotonic()
            with lock:snap=shared.copy()
            fresh=now-snap['received']<.1 and now-snap['target_at']<.1
            if snap['armed'] and not fresh:
                with lock:shared['fault']='state or inference watchdog timeout'
                snap['fault']='watchdog'
            if snap['armed']:
                healthy=fresh and not snap['fault'] and not snap['quit']
                for i in range(12):
                    motor=cmd.motor_cmd[i];motor.mode=1;motor.dq=0.;motor.tau=0.
                    motor.q=float(snap['target'][i]) if healthy else 2.146e9
                    motor.kp=c.meta['kp'] if healthy else 0.;motor.kd=c.meta['kd']
                cmd.crc=crc.Crc(cmd);pub.Write(cmd)
            if snap['quit']:
                # Repeated damping packets before stopping the output thread.
                if snap['armed']:
                    for _ in range(25):pub.Write(cmd);time.sleep(.002)
                return
            deadline+=.002;time.sleep(max(0.,deadline-time.monotonic()))
            if time.monotonic()-deadline>.02:deadline=time.monotonic()
    thread=threading.Thread(target=writer,daemon=True) if pub else None
    if thread:thread.start()
    print('Receive-only' if not pub else 'Awaiting L1 + Start in standard stance; L1 held = move; release L1 = stand; B = latched damping',flush=True)
    previous_start=False;last_print=0.;deadline=time.monotonic()
    try:
        while not shared['quit']:
            now=time.monotonic()
            with lock:snap=shared.copy()
            state=snap['state']
            if state is not None and now-snap['received']<.1:
                remote=decode_remote(state.wireless_remote,a.yaw_gain)
                q=np.array([m.q for m in state.motor_state[:12]],dtype=np.float32)
                dq=np.array([m.dq for m in state.motor_state[:12]],dtype=np.float32)
                quat=state.imu_state.quaternion;gyro=state.imu_state.gyroscope
                tilt=float(np.arccos(np.clip(-gravity_from_quaternion(quat)[2],-1,1)))
                if remote['estop'] or tilt>.75:
                    with lock:shared['fault']='B emergency stop' if remote['estop'] else 'excessive tilt'
                if pub and not snap['armed'] and not shared['fault'] and remote['start'] and not previous_start and remote['deadman']:
                    if tilt>.25 or np.max(np.abs(q[c.perm]-c.q0))>.30 or np.max(np.abs(dq))>.5:
                        print('Arm rejected: begin at rest in standard upright stance',flush=True)
                    else:
                        c.reset();target,debug=c.step(q,dq,gyro,quat,[0.,0.,0.])
                        with lock:shared.update(armed=True,target=target,target_at=time.monotonic())
                elif snap['armed'] and not shared['fault']:
                    target,debug=c.step(q,dq,gyro,quat,remote['command'] if remote['deadman'] else [0.,0.,0.])
                    with lock:shared.update(target=target,target_at=time.monotonic())
                previous_start=remote['start']
                if now-last_print>1.:
                    print(json.dumps(dict(command=remote['command'],deadman=remote['deadman'],armed=shared['armed'],fault=shared['fault'],tilt=tilt)),flush=True);last_print=now
            deadline+=c.meta['policy_dt'];time.sleep(max(0.,deadline-time.monotonic()))
            if time.monotonic()-deadline>.02:deadline=time.monotonic()
    finally:
        with lock:shared.update(fault=shared['fault'] or 'controller exit',quit=True)
        if thread:thread.join(timeout=1.)

if __name__=='__main__':main()
