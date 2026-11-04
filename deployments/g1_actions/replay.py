"""Independent CPU MuJoCo execution of the deployment runtime, no DDS or GPU."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
import mujoco
from runtime import Controller,rotation
base=Path(__file__).parent
results=[]
for skill in ['jump_60cm','funky_chicken']:
 c=Controller(base/'models'/skill);meta=c.meta
 m=mujoco.MjModel.from_binary_path(str(base/'models'/skill/'training_model.mjb'))
 d=mujoco.MjData(m);qids=[m.joint(n).id for n in meta['policy_joint_names']]
 qa=m.jnt_qposadr[qids];va=m.jnt_dofadr[qids]
 act_to_policy=[qids.index(int(j)) for j in m.actuator_trnid[:,0]]
 root=m.body('pelvis').id;ra=m.jnt_qposadr[m.body_jntadr[root]];rv=m.jnt_dofadr[m.body_jntadr[root]]
 d.qpos[ra:ra+3]=c.ref['pos'][0];d.qpos[ra+3:ra+7]=c.ref['quat'][0];d.qpos[qa]=c.ref['q'][0]
 d.qvel[:]=0;mujoco.mj_forward(m,d)
 record=[];timings=[];start=time.monotonic();perm=c.perm
 for i in range(meta['reference_frames']-1):
  q=np.empty(29,dtype='f');dq=q.copy();q[perm]=d.qpos[qa];dq[perm]=d.qvel[va]
  tick=time.perf_counter()
  target,info=c.step(q,dq,d.qvel[rv+3:rv+6],d.qpos[ra+3:ra+7],d.qpos[ra:ra+3])
  timings.append((time.perf_counter()-tick)*1000)
  d.ctrl[:]=target[perm][act_to_policy]
  for _ in range(4):mujoco.mj_step(m,d)
  record.append(np.r_[d.qpos[ra:ra+7],d.qpos[qa]])
 r=np.asarray(record);tilt=np.arccos(np.clip(np.array([rotation(q)[2,2] for q in r[:,3:7]]),-1,1))
 result=dict(skill=skill,frames=len(r),duration_s=len(r)*.02,all_finite=bool(np.isfinite(r).all()),
  min_root_height=float(r[:,2].min()),final_root_displacement=(r[-1,:3]-c.ref['pos'][0]).tolist(),
  final_tilt_deg=float(np.rad2deg(tilt[-1])),max_tilt_deg=float(np.rad2deg(tilt.max())),
  joint_reference_rmse=float(np.sqrt(np.mean((r[:,7:]-c.ref['q'][1:len(r)+1])**2))),
  inference_p99_ms=float(np.percentile(timings,99)),wall_seconds=time.monotonic()-start,
  scope='One nominal CPU simulation; deployment runtime and original training PD, no sensor noise/delay')
 assert result['all_finite'] and result['min_root_height']>.4,result
 results.append(result)
 np.savez_compressed(base/'results'/f'{skill}_replay.npz',qpos=r)
 print(json.dumps(result),flush=True)
(base/'results'/'replay.json').write_text(json.dumps(results,indent=2)+'\n')
