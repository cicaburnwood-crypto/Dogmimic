"""Read B2 telemetry and run actors. No publisher, RPC, or motor command exists here."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from runtime import Controller, np, remote_command

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206'))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface',default='eth0')
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--skill',choices=['both','hindstand','locomotion'],default='both')
    parser.add_argument('--remote-input',action='store_true',help='Use remote commands for shadow inference only')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if not 1<=args.seconds<=600: raise ValueError('Duration must be 1..600 seconds')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock();state={};counts={};changed={};tokens={}
    def handler(topic):
        def callback(msg):
            now=time.monotonic()
            if topic=='low':
                token=int(msg.tick)
                sample=dict(q=[float(m.q) for m in msg.motor_state[:12]],
                            dq=[float(m.dq) for m in msg.motor_state[:12]],
                            gyro=list(msg.imu_state.gyroscope),quaternion=list(msg.imu_state.quaternion),
                            motor_modes=[int(m.mode) for m in msg.motor_state[:12]],
                            foot_force=list(msg.foot_force),wireless=list(msg.wireless_remote),tick=token)
            else:
                # B2 firmware observed here leaves the application stamp at zero.
                # Use advancing DDS source timestamps plus local receipt age. The
                # publisher and PC4 wall clocks are not synchronized, so absolute
                # wall-clock subtraction would give a misleading freshness test.
                source_stamp=int(getattr(getattr(msg,'sample_info',None),'source_timestamp',0))
                token=(int(msg.stamp.sec),int(msg.stamp.nanosec),source_stamp)
                sample=dict(position=list(msg.position),velocity=list(msg.velocity),
                            body_height=float(msg.body_height),mode=int(msg.mode),
                            application_stamp=token[:2],dds_source_timestamp_ns=source_stamp)
            with lock:
                if token!=tokens.get(topic): changed[topic]=now;tokens[topic]=token
                state[topic]=(now,sample);counts[topic]=counts.get(topic,0)+1
        return callback
    ChannelFactoryInitialize(0,args.interface)
    subscribers=[]
    for name,topic,kind in [('low','rt/lowstate',LowState_),
                             ('sport','rt/sportmodestate',SportModeState_),
                             ('lf_sport','rt/lf/sportmodestate',SportModeState_)]:
        sub=ChannelSubscriber(topic,kind);sub.Init(handler(name),1);subscribers.append(sub)
    skills=['hindstand','locomotion'] if args.skill=='both' else [args.skill]
    controllers={name:Controller(ROOT/'models'/name) for name in skills}
    stats={name:dict(steps=0,clipped_steps=0,latency_ms=[],max_joint_delta_rad=0.) for name in skills}
    errors={};snapshot=None;last_output=-1
    start=time.monotonic();deadline=start
    try:
        with out.with_suffix('.jsonl').open('w') as log:
            while time.monotonic()-start<args.seconds:
                now=time.monotonic()
                with lock:
                    samples=dict(state);updates=dict(changed)
                if 'low' not in samples or now-samples['low'][0]>.1 or now-updates.get('low',-1e9)>.1:
                    errors['lowstate']='No fresh lowstate/tick (<100 ms)'
                else:
                    errors.pop('lowstate',None)
                    low=samples['low'][1]
                    root=None;position_source=None
                    for key in ['sport','lf_sport']:
                        if (key in samples and samples[key][1]['dds_source_timestamp_ns']>0
                            and now-samples[key][0]<.2 and now-updates.get(key,-1e9)<.2):
                            root=samples[key][1]['position'];position_source=key;break
                    remote=remote_command(low['wireless'])
                    command=remote['command'] if args.remote_input and remote['deadman'] and not remote['stop'] else [0.,0.,0.]
                    outputs={}
                    for name,controller in controllers.items():
                        try:
                            begin=time.perf_counter_ns()
                            target,debug=controller.step(low['q'],low['dq'],low['gyro'],low['quaternion'],
                                                        requested=command,root_position=root,
                                                        advance_reference=False)
                            dt=(time.perf_counter_ns()-begin)/1e6
                            stats[name]['steps']+=1;stats[name]['latency_ms'].append(dt)
                            stats[name]['clipped_steps']+=int(debug['target_clipped'])
                            stats[name]['max_joint_delta_rad']=max(stats[name]['max_joint_delta_rad'],debug['joint_delta_max_rad'])
                            outputs[name]=dict(inference_ms=dt,target_sdk=target.tolist(),
                                initial_pose_max_error_rad=float(np.max(np.abs(np.asarray(low['q'])[controller.perm]-controller.q0))),
                                **debug)
                            errors.pop(name,None)
                        except (ValueError,FloatingPointError) as e:
                            errors[name]=str(e)
                    snapshot=dict(elapsed_seconds=now-start,lowstate_age_ms=(now-samples['low'][0])*1000,
                        q_sdk=low['q'],dq_sdk=low['dq'],imu_quaternion=low['quaternion'],gyro=low['gyro'],
                        motor_modes=low['motor_modes'],foot_force=low['foot_force'],remote=remote,
                        root_position=root,position_source=position_source,shadow_outputs=outputs,
                        sport={k:v[1] for k,v in samples.items() if k!='low'},errors=dict(errors),
                        no_motor_commands=True,reference_is_frozen=True)
                    if int(now-start)!=last_output:
                        last_output=int(now-start)
                        log.write(json.dumps(snapshot)+'\n');log.flush()
                        print(json.dumps(dict(elapsed=round(now-start,1),steps={k:v['steps'] for k,v in stats.items()},
                                              errors=errors,no_motor_commands=True)),flush=True)
                deadline+=.02
                time.sleep(max(0.,deadline-time.monotonic()))
    finally:
        for sub in subscribers:sub.Close()
    for item in stats.values():
        latency=item.pop('latency_ms')
        if latency:item['latency']=dict(p50_ms=float(np.percentile(latency,50)),p99_ms=float(np.percentile(latency,99)),
                                        max_ms=max(latency),over_20ms=sum(v>=20 for v in latency))
    result=dict(no_motor_commands=True,no_mode_change=True,reference_is_frozen=True,
                elapsed_seconds=time.monotonic()-start,topic_message_counts=counts,
                skills=stats,errors=errors,last_snapshot=snapshot,
                hardware_motion_validated=False,
                position_freshness='Advancing DDS source timestamp and local receipt age; estimator accuracy and low-level availability remain unvalidated',
                passed=all(v['steps']>=50 for v in stats.values()) and not errors)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='last_snapshot'},indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':
    main()
