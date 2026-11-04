"""Boundary replay mixture; retain continuous ranges and independent reset noise."""
import torch
HARD_CASES=[
 dict(name='mu015',friction=.15),dict(name='mu020',friction=.2),dict(name='mu030',friction=.3),
 dict(name='back_push',force=[-25,0,0]),dict(name='native_stress',measured=True),
 dict(name='combined_low',friction=.2,measured=True,delay_ms=20,mass=1.15,effort=.85,force=[0,15,0]),
 dict(name='combined_high',friction=1.5,measured=True,delay_ms=40,mass=1.25,kp=.8,kd=1.3),
 dict(name='forward_push',force=[25,0,0])]
def sample_domain(k,device,friction):
 x=torch.zeros(k,9,device=device)
 x[:,0]=.9+.35*torch.rand(k,device=device);x[:,1]=.8+.4*torch.rand(k,device=device)
 x[:,2]=.7+.6*torch.rand(k,device=device);x[:,3]=.7+.3*torch.rand(k,device=device)
 x[:,4]=torch.randint(0,9,(k,),device=device)/8;x[:,5]=torch.rand(k,device=device)*.6-.3
 x[:,6:8]=(torch.rand(k,2,device=device)*2-1)*25;x[:,8]=(torch.rand(k,device=device)*2-1)*10
 u=torch.rand(k,device=device);nominal=u<.2;continuous=(u>=.2)&(u<.5)
 isolated=(u>=.5)&(u<.55);crossed=(u>=.55)&(u<.6);hard=u>=.6
 mu=friction.clone();base=nominal|isolated
 x[base]=0;x[base,:4]=1;mu[base]=.8
 weak=isolated|crossed;x[weak,1]=.8;x[weak,2]=.7;x[weak,3]=.7
 hard_id=torch.where(hard,((u-.6)*20).long().clamp(0,7),-1)
 measured=torch.zeros(k,device=device,dtype=torch.bool)
 for idx,c in enumerate(HARD_CASES):
  m=hard_id==idx;x[m]=0;x[m,:4]=1;mu[m]=c.get('friction',.8)
  for j,key in enumerate(['mass','kp','kd','effort']):x[m,j]=c.get(key,1.)
  x[m,4]=c.get('delay_ms',0)/40.;x[m,6:]=torch.tensor(c.get('force',[0,0,0]),device=device,dtype=x.dtype)
  if c.get('measured'):measured|=m
 return x,mu,dict(nominal=nominal,continuous=continuous,isolated_weak=isolated,crossed_weak=crossed,hard_id=hard_id,measured_start=measured)
