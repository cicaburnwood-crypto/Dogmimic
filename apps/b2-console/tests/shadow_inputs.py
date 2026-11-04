"""Read live sensors and run frozen-phase inference; never take over or publish."""
import argparse,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import SDKBackend

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',default='/home/unitree/supermimic_b2/current')
p.add_argument('--seconds',type=float,default=16.)
p.add_argument('--out',required=True)
a=p.parse_args()
b=SDKBackend(a.bundle,'eth0');controller=b.controllers['hindstand']
counts=Counter();failures=Counter();ages=[];latencies=[];last=None
start=time.monotonic();deadline=start
try:
    while time.monotonic()-start<a.seconds:
        t=b.telemetry()
        try:
            if not t.fresh(time.monotonic()):raise RuntimeError('lowstate unavailable')
            position=b._root_position(t);before=time.perf_counter()
            target,debug=controller.step(t.q,t.dq,t.gyro,t.quaternion,root_position=position,advance_reference=False)
            latencies.append((time.perf_counter()-before)*1000)
            ages.append((time.monotonic()-t.position_received)*1000)
            counts[t.position_source]+=1
            last=dict(position_source=t.position_source,odom_position=list(t.position),aligned_root=position.tolist(),
                      imu_quaternion=list(t.quaternion),odom_quaternion=list(t.position_quaternion),reference_frame=debug['reference_frame'])
        except Exception as e:failures[str(e)]+=1
        assert not b.owned and b.pub is None and b.writer_packets==0,'Read-only invariant violated'
        deadline+=.02;time.sleep(max(0.,deadline-time.monotonic()))
    result=dict(elapsed_seconds=time.monotonic()-start,samples=sum(counts.values()),sources=dict(counts),errors=dict(failures),
                max_position_age_ms=max(ages) if ages else None,
                inference_p99_ms=float(b.np.percentile(latencies,99)) if latencies else None,
                motor_packets_sent=b.writer_packets,motor_publisher_created=b.pub is not None,
                controller_owned=b.owned,mode_change_requested=False,reference_frozen=True,last=last,
                physical_motion_tested=False,passed=sum(counts.values())>=int(a.seconds*45) and not failures)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
finally:b.close()
if not result['passed']:raise SystemExit(1)
