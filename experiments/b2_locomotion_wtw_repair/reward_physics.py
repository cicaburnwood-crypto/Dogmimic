"""MuJoCo solver measurements required by the published WTW rewards.

Separate from gait.py, whose 200 Hz independent acceptance is unchanged.
"""
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
from supermimic.rotations import heading_mat,mtv

@wp.kernel
def _accumulate(nacon:wp.array(dtype=int),geom:wp.array(dtype=wp.vec2i),
                worlds:wp.array(dtype=int),force:wp.array(dtype=wp.spatial_vector),
                slots:wp.array(dtype=int),out:wp.array3d(dtype=float)):
    i=wp.tid()
    if i<nacon[0]:
        pair=geom[i]
        for side in range(2):
            slot=slots[pair[side]]
            if slot>=0:
                sign=float(2*side-1)
                for axis in range(3):
                    wp.atomic_add(out,worlds[i],slot,axis,sign*force[i][axis])

def geometry_slots(m):
    feet=[m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    limbs=[m.body(l+'_'+part).id for l in ('FL','FR','RL','RR') for part in ('thigh','calf')]
    slots=np.full(m.ngeom,-1,dtype=np.int32)
    for g in range(m.ngeom):
        if g in feet:slots[g]=feet.index(g)
        elif int(m.geom_bodyid[g]) in limbs:slots[g]=4+limbs.index(int(m.geom_bodyid[g]))
    return slots

class RewardContactRecorder:
    def __init__(self,env):
        self.env=env;self.sim=env.sim;d=self.sim.wp_data
        self.ids=wp.from_torch(torch.arange(d.naconmax,device=env.device,dtype=torch.int32))
        self.slots=wp.from_torch(torch.tensor(geometry_slots(env.mj_model),device=env.device))
        self.forces=wp.zeros(d.naconmax,dtype=wp.spatial_vector,device=env.device)
        self.total=torch.zeros(env.num_envs,12,3,device=env.device)
        self.total_wp=wp.from_torch(self.total)
    def sample(self,s):
        self.total.zero_();d=self.sim.wp_data
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.env.device))):
            mw.contact_force(self.sim.wp_model,d,self.ids,True,self.forces)
            wp.launch(_accumulate,dim=d.naconmax,inputs=[d.nacon,d.contact.geom,d.contact.worldid,self.forces,self.slots,self.total_wp])
        # Joint convention: solver actuator order differs from FL/FR/RL/RR.
        torque=torch.zeros(self.env.num_envs,12,device=self.env.device)
        torque[:,self.env.act_to_joint]=self.sim.data.actuator_force
        foot=mtv(heading_mat(s['rot'])[:,None],self.total[:,:4])
        collision=(self.total[:,4:].norm(dim=-1)>.1).float()
        return torch.cat([torque,foot.flatten(1),collision],-1)
