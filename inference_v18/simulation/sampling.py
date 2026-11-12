"""Training samples only; fixed independent test cases/seeds are not imported."""
import torch
from contract import constrain_targets,tilt_limit,yaw_sampling_bound
from legacy_sampling import positive_huber
MIXTURE=dict(p2_low_speed=.35,speed_curriculum=.50,target_2m_s=.15)
STAGES=[.5,1.,2.]

def stage_for_iteration(it,warmup=1000):
    # Exposure is not acceptance; target data never depend on a perfect dev score.
    return 0 if it<warmup+4000 else 1 if it<warmup+10000 else 2

def sample_targets(k,device,stage=0,return_labels=False):
    category=torch.rand(k,device=device)
    baseline=category<MIXTURE['p2_low_speed'];target=category>=1-MIXTURE['target_2m_s']
    vmax=torch.where(baseline,.5,torch.where(target,2.,STAGES[stage]))
    h=-.20+.25*torch.rand(k,device=device);choice=torch.rand(k,device=device)
    h=torch.where(choice<.20,0.,torch.where(choice<.40,.05,torch.where(choice<.55,-.15,torch.where(choice<.70,-.20,h))))
    speed=vmax*torch.rand(k,device=device);speed=torch.where(torch.rand(k,device=device)<.30,vmax,speed)
    vy=(2*torch.rand(k,device=device)-1)*speed.clamp(max=1.)
    sign=torch.where(torch.rand(k,device=device)<.5,-1.,1.)
    vx=(speed.square()-vy.square()).clamp_min(0).sqrt()*sign
    mode=torch.rand(k,device=device)
    vx=torch.where(mode>.75,torch.zeros_like(vx),vx);vy=torch.where(mode<.5,torch.zeros_like(vy),vy);vy=torch.where(mode>.90,torch.zeros_like(vy),vy)
    yaw_max=torch.minimum(torch.full_like(speed,1.),vmax)
    yaw=(2*torch.rand(k,device=device)-1)*torch.minimum(yaw_sampling_bound(speed),yaw_max)
    yaw=torch.where(mode<.65,torch.zeros_like(yaw),yaw)
    rp=(2*torch.rand(k,2,device=device)-1)*tilt_limit(h)[:,None]
    edge=torch.rand(k,device=device)<.30
    rp=torch.where(edge[:,None],torch.sign(rp)*tilt_limit(h)[:,None],rp)
    rp=torch.where((torch.rand(k,device=device)<.2)[:,None],torch.zeros_like(rp),rp)
    values=torch.cat([vx[:,None],vy[:,None],yaw[:,None],rp,h[:,None]],-1)
    values[torch.rand(k,device=device)<.15,:3]=0.
    result=constrain_targets(values)
    return (result,baseline) if return_labels else result
