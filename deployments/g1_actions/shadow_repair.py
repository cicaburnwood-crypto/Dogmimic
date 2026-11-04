"""Read real sensors, compute targets and exercise a fake command sink. No control RPC or DDS writes."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json
from pathlib import Path
import numpy as np
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'vendor/unitree_sdk2_python-814556d15970dd2ecf1c9984e845ca02ab07e206'))
sys.path.insert(0,str(base/'repair_v2'))
from runtime import Controller
from sdk_transport import SDKTransport
cs=[Controller(base/'models'/s) for s in ('jump_60cm','funky_chicken')]
t=SDKTransport('eth0')
class FakeSink:
 def __init__(self):self.n=0
 def Write(self,msg):self.n+=1;return True
sink=FakeSink();t.publisher=sink # No ChannelPublisher is constructed.
time.sleep(1)
rows=[];errors=[];started=time.monotonic();cpu=time.process_time();deadline=started
try:
 for i in range(1000):
  try:
   ts=time.perf_counter();s=t.snapshot();read_ms=(time.perf_counter()-ts)*1000
   c=cs[(i//250)%2]
   if c.frame==len(c.ref['pos'])-1:c.reset()
   ts=time.perf_counter();target,_=c.step(s['q'],s['dq'],s['gyro'],s['quaternion'],s['position']);inf_ms=(time.perf_counter()-ts)*1000
   ts=time.perf_counter();t.command(target,c.meta,s);cmd_ms=(time.perf_counter()-ts)*1000
   rows.append([s['low_age_s'],s['odom_age_s'],read_ms,inf_ms,cmd_ms])
  except Exception as e:errors.append(str(e))
  deadline+=.02;time.sleep(max(0.,deadline-time.monotonic()))
finally:
 t.publishing=False
 if t.pubthread:t.pubthread.join(1)
a=np.asarray(rows);result=dict(samples=len(rows),seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
 max_sensor_age_s=a[:,:2].max(0).tolist() if len(a) else None,
 p99_read_infer_command_ms=np.percentile(a[:,2:],99,axis=0).tolist() if len(a) else None,
 errors=errors,sensor_errors=t.sensor_errors,publish_error=t.publish_error,fake_sink_writes=sink.n,
 hardware_commands_sent=0,ownership_changes=0)
(base/'results/repair_v2_shadow.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if errors or t.publish_error:raise SystemExit(1)
