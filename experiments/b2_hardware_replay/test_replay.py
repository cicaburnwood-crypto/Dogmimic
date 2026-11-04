"""CPU-only closed-loop replay comparisons. No hardware/DDS imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,argparse
from pathlib import Path
import numpy as np
import mujoco
p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--trace',required=True);p.add_argument('--out',required=True);a=p.parse_args()
project=Path(a.project);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
bundle=project/'deployments/b2_192_168_123_164_20261101';sys.path.insert(0,str(bundle))
from runtime import Controller,rotation,sha256
rows=[json.loads(l) for l in Path(a.trace).read_text().splitlines()];rows=[r for r in rows if r['attempt']==rows[-1]['attempt']];real=rows[0]
model_path=project/'experiments/three_skills/b2_train.xml'
results=[]
for pose in ['nominal','hardware']:
 for variant in ['training','deployment_no_blend','deployment','blend_applied_history']:
  m=mujoco.MjModel.from_xml_path(str(model_path));m.opt.timestep=.005;m.opt.iterations=10;m.opt.ls_iterations=20
  d=mujoco.MjData(m);c=Controller(bundle/'models/hindstand');perm=c.perm
  feet=[m.geom(n+'_foot_collision').id for n in ['FL','FR','RL','RR']]
  actmap=m.actuator_trnid[:,0]-1
  d.qpos[:3]=c.reference['pos'][0];d.qpos[3:7]=[1,0,0,0] if pose=='nominal' else real['imu_quaternion']
  d.qpos[7:]=c.reference['q'][0] if pose=='nominal' else np.array(real['q_sdk'])[perm]
  if pose=='hardware':
   mujoco.mj_forward(m,d);d.qpos[2]+=.032-d.geom_xpos[feet,2].min()
  mujoco.mj_forward(m,d);initial=d.qpos.copy();initial_sdk=np.empty(12);initial_sdk[perm]=d.qpos[7:]
  if variant=='training':c.anchor_position=c.reference['pos'][0].copy();c.anchor_rotation=np.eye(3,dtype=np.float32)
  data=[];target_prev=None
  for i in range(775):
   q_sdk=np.empty(12);dq_sdk=np.empty(12);q_sdk[perm]=d.qpos[7:];dq_sdk[perm]=d.qvel[6:]
   target,debug=c.step(q_sdk,dq_sdk,d.qvel[3:6],d.qpos[3:7],root_position=d.qpos[:3].copy(),advance_reference=True)
   if variant=='training':target=np.empty(12);target[perm]=c.q0+c.scale*np.array(debug['action'])
   if variant in ('deployment','blend_applied_history'):
    alpha=min(1.,i*.02/.5);target=initial_sdk*(1-alpha)+target*alpha
    if variant=='blend_applied_history':c.previous_action=((target[perm]-c.q0)/c.scale).astype(np.float32)
   d.ctrl[:]=target[perm][actmap]
   for _ in range(4):mujoco.mj_step(m,d)
   R=rotation(d.qpos[3:7]);pitch=float(np.degrees(np.arcsin(np.clip(-R[2,0],-1,1))))
   frame=min(i+1,len(c.reference['pos'])-1);height_error=abs(d.qpos[2]-c.reference['pos'][frame,2]);tilt_error=abs(R[2,2]-c.reference['rot'][frame,2,2])
   f=d.geom_xpos[feet].copy()
   data.append(dict(t=(i+1)*.02,pitch=pitch,pos=d.qpos[:3].copy(),q=d.qpos[7:].copy(),feet=f,
    tracking_error=float(np.max(abs(target[perm]-d.qpos[7:]))),target_step=float(np.max(abs(target-target_prev))) if target_prev is not None else 0,
    terminated=bool(height_error>.25 or tilt_error>.8),torque_max=float(abs(d.actuator_force).max())))
   target_prev=target.copy()
  t=np.array([x['t'] for x in data]);pitch=np.array([x['pitch'] for x in data]);fp=np.array([x['feet'] for x in data]);term=np.array([x['terminated'] for x in data])
  hold=(t>=4)&(t<=9.3);final=t>=13
  hind=((fp[:,2:,2]>=.020)&(fp[:,2:,2]<=.045)).all(1);allfeet=((fp[:,:,2]>=.020)&(fp[:,:,2]<=.045)).all(1)
  holdfraction=float(((pitch< -75)&(fp[:,:2,2].min(1)>.35)&hind)[hold].mean())
  finalfraction=float(((abs(pitch)<8)&allfeet)[final].mean())
  result=dict(pose=pose,variant=variant,initial_q=initial[7:].tolist(),initial_height=float(initial[2]),max_nose_up_deg=float(-pitch.min()),
   hold_pose_fraction=holdfraction,return_pose_fraction=finalfraction,first_training_termination_s=float(t[np.flatnonzero(term)[0]]) if term.any() else None,
   first_error_over_1rad_s=next((x['t'] for x in data if x['tracking_error']>1),None),max_target_step_rad=max(x['target_step'] for x in data),
   pose_cycle_pass=holdfraction>=.95 and finalfraction>=.95 and not term.any(),full_original_strict_metrics_evaluated=False)
  np.savez(out/(pose+'_'+variant+'.npz'),time=t,pitch=pitch,position=np.array([x['pos'] for x in data]),q=np.array([x['q'] for x in data]),feet=fp,tracking_error=np.array([x['tracking_error'] for x in data]))
  results.append(result);print(json.dumps(result),flush=True)
report=dict(cpu_only=True,robot_commands_sent=0,hardware_initial_velocities='zero; not recorded in hardware trace',ground_height='inferred from lowest foot geometry',
 cases=results,weights_sha256=sha256(bundle/'models/hindstand/weights.npz'),model=str(model_path),note='CPU MuJoCo reproduction, not original GPU randomized strict evaluation; no sensor noise, delay, payload or pushes')
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
