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

def longest_run(mask):
    current=np.zeros(mask.shape[1],dtype=np.int32);best=current.copy()
    for row in mask:
        current=np.where(row,current+1,0);best=np.maximum(best,current)
    return best*SIM_DT

def summarize_gait(force,clearance,vz,steady_steps,nonfoot_force,require_steps):
    """Inputs [control_time, decimation, environments, feet], vz lacks feet axis."""
    n=force.shape[2];support=force>GAIT_CRITERIA['contact_force_n']
    support=support.reshape(-1,n,4);clearance=clearance.reshape(-1,n,4);vz=vz.reshape(-1,n)
    first=round(GAIT_CRITERIA['ignored_initial_seconds']/SIM_DT)
    unloaded=~support[first:].any(-1)
    airborne=(clearance[first:]>GAIT_CRITERIA['geometry_clearance_m']).all(-1)
    nonfoot=nonfoot_force.reshape(-1,n)[first:]>GAIT_CRITERIA['contact_force_n']
    steady=np.concatenate([np.arange(i*4,(i+1)*4) for i in steady_steps])
    s=support[steady];count=s.sum(-1)
    duty=s.mean(0)
    stepping=(duty>=GAIT_CRITERIA['moving_foot_contact_duty_min']).all(-1)&(duty<=GAIT_CRITERIA['moving_foot_contact_duty_max']).all(-1)
    values=dict(
        unloaded_fraction=unloaded.mean(0),unloaded_seconds=unloaded.sum(0)*SIM_DT,
        longest_unloaded_s=longest_run(unloaded),
        airborne_fraction=airborne.mean(0),airborne_seconds=airborne.sum(0)*SIM_DT,
        longest_airborne_s=longest_run(airborne),
        two_support_fraction=(count>=2).mean(0),
        front_and_rear_support_fraction=(s[:,:,:2].any(-1)&s[:,:,2:].any(-1)).mean(0),
        vertical_velocity_rms_m_s=np.sqrt((vz[steady]**2).mean(0)),
        foot_contact_duty=duty,feet_alternate=(~require_steps)|stepping,
        nonfoot_contact_fraction=nonfoot.mean(0),
    )
    passed=(values['unloaded_fraction']<=GAIT_CRITERIA['max_unloaded_fraction'])
    passed&=values['longest_unloaded_s']<=GAIT_CRITERIA['max_unloaded_interval_s']+1e-9
    passed&=values['airborne_fraction']<=GAIT_CRITERIA['max_airborne_fraction']
    passed&=values['longest_airborne_s']<=GAIT_CRITERIA['max_airborne_interval_s']+1e-9
    passed&=values['two_support_fraction']>=GAIT_CRITERIA['min_two_support_fraction']
    passed&=values['front_and_rear_support_fraction']>=GAIT_CRITERIA['min_front_and_rear_support_fraction']
    passed&=values['vertical_velocity_rms_m_s']<=GAIT_CRITERIA['max_vertical_velocity_rms_m_s']
    passed&=values['nonfoot_contact_fraction']<=GAIT_CRITERIA['max_nonfoot_contact_fraction']
    # Every foot must both support and release during nonzero motion commands;
    # continuously dragging all four planted feet cannot count as walking.
    passed&=values['feet_alternate']
    return values,passed
