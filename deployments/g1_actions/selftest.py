"""Local export integrity, parity, model timing and state tests; never imports DDS."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import json,time
import numpy as np
from runtime import Actor,Controller,rotation
from runner import Backend
base=Path(__file__).parent
results=[]
for skill in ('jump_60cm','funky_chicken'):
 folder=base/'models'/skill;a=Actor(folder);c=Controller(folder)
 with np.load(folder/'parity_inputs.npz') as data:
  error=max(float(np.max(abs(a(o[None],g[None])[0]-expected))) for o,g,expected in zip(data['obs'],data['goal'],data['expected_action']))
 assert error<3e-5,error
 q=np.empty(29);q[c.perm]=c.ref['q'][0]
 args=(q,np.zeros(29),np.zeros(3),c.ref['quat'][0],c.ref['pos'][0])
 target,info=c.step(*args);assert c.frame==1 and np.isfinite(target).all()
 assert np.allclose(c.history[0],c.history[-1])
 previous=c.previous_action.copy();c.step(*args);assert np.allclose(c.history[-1,-29:],previous)
 c.reset();same,_=c.step(*args);assert np.allclose(target,same)
 s=dict(q=q.tolist(),dq=[0.]*29,gyro=[0.]*3,quaternion=[1.,0.,0.,0.],velocity=[0.]*3)
 assert Backend.stable(s)
 s['quaternion']=[.70710678,.70710678,0.,0.];assert not Backend.stable(s)
 times=[]
 for _ in range(30):
  tick=time.perf_counter();c.step(*args);times.append((time.perf_counter()-tick)*1000)
 results.append(dict(skill=skill,actor_max_abs_error=error,inference_p99_ms=float(np.percentile(times,99)),passed=True))
print(json.dumps(results,indent=2))
(base/'results'/'selftest.json').write_text(json.dumps(results,indent=2)+'\n')
