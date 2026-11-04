"""Gait measurements from solver contact forces at every 5 ms physics step.

No policy inputs or reference gait. Acceptance is independent of rewards.
"""
import numpy as np
import torch
import warp as wp
import mujoco_warp as mw
from contract import SIM_DT

GAIT_CRITERIA = dict(
    contact_force_n=5., geometry_clearance_m=.002,
    ignored_initial_seconds=.5,
    max_unloaded_fraction=.01, max_unloaded_interval_s=.025,
    max_airborne_fraction=.005, max_airborne_interval_s=.020,
    min_two_support_fraction=.90, min_front_and_rear_support_fraction=.90,
    max_vertical_velocity_rms_m_s=.20,
    max_nonfoot_contact_fraction=.01,
    moving_foot_contact_duty_min=.05,moving_foot_contact_duty_max=.98,
)

@wp.kernel
def _sum_foot_forces(nacon:wp.array(dtype=int), geom:wp.array(dtype=wp.vec2i),
                     worlds:wp.array(dtype=int), force:wp.array(dtype=wp.spatial_vector),
                     feet:wp.array(dtype=int), floor:int, out:wp.array2d(dtype=float)):
    i=wp.tid()
    if i<nacon[0]:
        pair=geom[i]
        matched=False
        for j in range(4):
            if (pair[0]==floor and pair[1]==feet[j]) or (pair[1]==floor and pair[0]==feet[j]):
                wp.atomic_add(out,worlds[i],j,wp.max(0.,force[i][0]))
                matched=True
        if not matched and (pair[0]==floor or pair[1]==floor):
            wp.atomic_add(out,worlds[i],4,wp.max(0.,force[i][0]))

class ContactRecorder:
    def __init__(self,env):
        self.env=env;self.sim=env.sim
        self.feet=[env.mj_model.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
        self.floor=env.mj_model.geom('floor').id
        self.ids=wp.from_torch(torch.arange(self.sim.wp_data.naconmax,device=env.device,dtype=torch.int32))
        self.foot_ids=wp.from_torch(torch.tensor(self.feet,device=env.device,dtype=torch.int32))
        self.forces=wp.zeros(self.sim.wp_data.naconmax,dtype=wp.spatial_vector,device=env.device)
        self.total=torch.zeros(env.num_envs,5,device=env.device)
        self.total_wp=wp.from_torch(self.total)
    def sample(self):
        self.total.zero_()
        d=self.sim.wp_data
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.env.device))):
            mw.contact_force(self.sim.wp_model,d,self.ids,False,self.forces)
            wp.launch(_sum_foot_forces,dim=d.naconmax,
                      inputs=[d.nacon,d.contact.geom,d.contact.worldid,self.forces,self.foot_ids,self.floor,self.total_wp])
        # geom_xpos and contact forces describe the same pre-integration contact solve.
        clearance=self.sim.data.geom_xpos[:,self.feet,2]-.032
        return self.total[:,:4].clone(),clearance.clone(),self.sim.data.qvel[:,2].clone(),self.total[:,4].clone()
