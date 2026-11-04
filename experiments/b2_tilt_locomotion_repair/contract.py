"""Ground-heading velocity and simultaneous body Euler roll/pitch commands.

No yaw-angle target/reset, no yaw-rate clipping in the command interface.
The finite training/evaluation angular-velocity envelope is disclosed separately.
"""
import math
import torch

DT=.02
SIM_DT=.005
HISTORY=10
OBS=42
CONTEXT_DIM=13  # velocity3, attitude2, attitude-rate2, request5, gait-phase1
GOAL_DIM=16  # first 12 context entries + four per-foot clocks
MAX_ANGLE=math.radians(20)
MAX_RATE=math.radians(30)
MAX_ACCEL=math.radians(90)
COMMAND_OMEGA=5.
MAX_FORWARD=4.
MAX_LATERAL=1.
MAX_TRANSLATION=4.
LINEAR_ACCELERATION=1.5
ANGULAR_ACCELERATION=3.
TRAIN_YAW_MAX=2.
TURN_ACCELERATION_BUDGET=2.  # sampling envelope, not an interface command cap
Q0=[0.,.85,-1.70]*4
ACTION_SCALE=[.70,1.1,1.1]*4
KP,KD=250.,8.
POLICY_JOINTS=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
POLICY_TO_SDK=[3,4,5,0,1,2,9,10,11,6,7,8]
CURRICULUM=[(.5,5.,.35),(1.,10.,.6),(2.,15.,1.),(3.,20.,1.5),(4.,20.,2.)]

def angles(rot):
    return torch.stack([torch.atan2(rot[...,2,1],rot[...,2,2]),
        torch.atan2(-rot[...,2,0],torch.sqrt(rot[...,2,1].square()+rot[...,2,2].square()+1e-12))],-1)

def angle_error(actual,target):
    return torch.atan2(torch.sin(actual-target),torch.cos(actual-target))

def heading_velocity(rot,velocity):
    yaw=torch.atan2(rot[...,1,0],rot[...,0,0]);c,s=yaw.cos(),yaw.sin()
    return torch.stack([c*velocity[...,0]+s*velocity[...,1],-s*velocity[...,0]+c*velocity[...,1],velocity[...,2]],-1)

def euler_rates(rot,world_omega):
    body=torch.matmul(rot.transpose(-1,-2),world_omega.unsqueeze(-1)).squeeze(-1)
    rp=angles(rot);r,p=rp.unbind(-1)
    # Away from the deliberately terminated near-vertical posture.
    cp=p.cos().clamp_min(.1)
    yd=(r.sin()*body[...,1]+r.cos()*body[...,2])/cp
    return torch.stack([body[...,0]+p.sin()*yd,r.cos()*body[...,1]-r.sin()*body[...,2],yd],-1)

def target_body_rate(context):
    r,p=context[...,3],context[...,4];rd,pd=context[...,5],context[...,6];yd=context[...,2]
    return torch.stack([rd-yd*p.sin(),pd*r.cos()+yd*r.sin()*p.cos(),-pd*r.sin()+yd*r.cos()*p.cos()],-1)

def constrain_targets(values):
    """[vx,vy,yaw-rate,roll,pitch]. Roll and pitch each allow +/-20 degrees."""
    vx=values[...,0:1].clamp(-MAX_FORWARD,MAX_FORWARD)
    vy=values[...,1:2].clamp(-MAX_LATERAL,MAX_LATERAL)
    xy=torch.cat([vx,vy],-1)
    xy=xy*(MAX_TRANSLATION/xy.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    return torch.cat([xy,values[...,2:3],values[...,3:5].clamp(-MAX_ANGLE,MAX_ANGLE)],-1)

def command(vx=0.,vy=0.,yaw_rate=0.,roll_degrees=0.,pitch_degrees=0.):
    values=[vx,vy,yaw_rate,roll_degrees,pitch_degrees]
    if not all(math.isfinite(x) for x in values):raise ValueError('commands must be finite')
    if max(abs(roll_degrees),abs(pitch_degrees))>20.:raise ValueError('roll/pitch each must be within +/-20 degrees')
    return constrain_targets(torch.tensor([vx,vy,yaw_rate,math.radians(roll_degrees),math.radians(pitch_degrees)],dtype=torch.float64)).tolist()

def moving(context):
    return (context[...,:2].norm(dim=-1)>.05)|(context[...,2].abs()>.05)

def frequency(context):
    return 2.+.35*context[...,:2].norm(dim=-1).clamp(max=4.)

def duty(context):
    return .65-.10*context[...,:2].norm(dim=-1).clamp(max=4.)/4.

def foot_phases(context):
    offset=context.new_tensor([.5,0.,0.,.5])
    return torch.where(moving(context)[...,None],torch.remainder(context[...,12:13]+offset,1.),duty(context)[...,None]*.5)

def gait(context):
    phase=foot_phases(context);d=duty(context)[...,None]
    warp=torch.where(phase<d,phase*(.5/d),.5+(phase-d)*(.5/(1-d)))
    def cdf(x):return .5*(1+torch.erf(x/(.07*math.sqrt(2))))
    contact=cdf(warp)*(1-cdf(warp-.5))+cdf(warp-1)*(1-cdf(warp-1.5))
    contact=torch.where(moving(context)[...,None],contact,torch.ones_like(contact))
    swing=((phase-d)/(1-d)).clamp(0,1)
    height=.032+.08*torch.sin(math.pi*swing)
    return contact,height,torch.sin(2*math.pi*warp)

def policy_goal(context):
    return torch.cat([context[...,:12],gait(context)[2]],-1)

def advance_command(context):
    dv=context[...,7:10]-context[...,:3]
    xy=context[...,:2]+dv[...,:2]*(LINEAR_ACCELERATION*DT/dv[...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    yaw=context[...,2:3]+dv[...,2:3].clamp(-ANGULAR_ACCELERATION*DT,ANGULAR_ACCELERATION*DT)
    q,v,target=context[...,3:5],context[...,5:7],context[...,10:12]
    acc=(COMMAND_OMEGA**2*(target-q)-2*COMMAND_OMEGA*v).clamp(-MAX_ACCEL,MAX_ACCEL)
    v=(v+DT*acc).clamp(-MAX_RATE,MAX_RATE);q=q+DT*v
    temp=torch.cat([xy,yaw,q,v,context[...,7:]],-1)
    phase=torch.where(moving(temp),torch.remainder(context[...,12]+DT*frequency(temp),1.),.25)
    return torch.cat([temp[...,:12],phase[...,None]],-1)

def yaw_sampling_bound(speed,max_yaw=TRAIN_YAW_MAX):
    return (TURN_ACCELERATION_BUDGET/speed.clamp_min(1.)).clamp(max=max_yaw)
