"""Extend our original B2 rise path into a full stand / return cycle."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

HERE=Path(__file__).resolve().parent
src=HERE.parent/'three_skills/targets/b2_hindstand.npz'
old=np.load(src);qold=old['qpos'].astype(float);dt=1/float(old['fps'])
t=np.arange(776)*dt
u=np.where(t<3.5,np.clip((t-1.)/2.5,0,1),np.where(t<9.5,1.,1-np.clip((t-9.5)/3.,0,1)))
source_time=.8+2.5*u;old_time=np.arange(len(qold))*dt
q=np.stack([np.interp(source_time,old_time,qold[:,j]) for j in range(qold.shape[1])],-1)
q[:,3:7]=Slerp(old_time,Rotation.from_quat(qold[:,[4,5,6,3]]))(source_time).as_quat()[:,[3,0,1,2]]
v=np.zeros((len(q),18));v[:,:3]=np.gradient(q[:,:3],dt,axis=0);v[:,6:]=np.gradient(q[:,7:],dt,axis=0)
r=Rotation.from_quat(q[:,[4,5,6,3]])
v[1:-1,3:6]=(r[2:]*r[:-2].inv()).as_rotvec()/(2*dt)
phase=np.select([t<1.,t<3.5,t<9.5,t<12.5],[0,1,2,3],default=4).astype('int64')
hold=np.isin(phase,[0,2,4]).astype('float32')
# Strong hold penalties fade in over 0.3 s after a transition, while a small
# acceleration penalty remains present throughout all movement phases.
for start,end in [(0,1),(3.5,9.5),(12.5,15.5)]:
    sel=(t>=start)&(t<=end)
    hold[sel]*=np.clip((t[sel]-start)/.3,0,1)
assert np.max(abs(q[0]-q[-1]))<1e-7
assert np.max(abs(v[[0,-1]]))<1e-6
np.savez_compressed(HERE/'b2_full_cycle.npz',qpos=q.astype('float32'),qvel=v.astype('float32'),starts=np.array([0]),lengths=np.array([len(q)]),names=np.array(['b2_stable_full_cycle']),fps=np.array(50.),phase=phase,hold_weight=hold)
meta=dict(source=str(src),source_kind='our original procedural reference',duration_s=15.5,
          phases=[dict(name='four_leg_stand',start=0,end=1),dict(name='rise',start=1,end=3.5),dict(name='hind_stand',start=3.5,end=9.5),dict(name='return_to_four',start=9.5,end=12.5),dict(name='four_leg_settle',start=12.5,end=15.5)],
          reference_initial_final_max_difference=float(abs(q[0]-q[-1]).max()),max_joint_velocity_rad_s=float(abs(v[:,6:]).max()))
(HERE/'reference.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
