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
CONTEXT_DIM=17  # original13 + height_delta, height_rate, requested_height, gait_progress
GOAL_DIM=20  # original16 + height_delta, height_rate, requested_height, gait_progress
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

def constrain_legacy(values):
    """[vx,vy,yaw-rate,roll,pitch]. Roll and pitch each allow +/-20 degrees."""
    vx=values[...,0:1].clamp(-MAX_FORWARD,MAX_FORWARD)
    vy=values[...,1:2].clamp(-MAX_LATERAL,MAX_LATERAL)
    xy=torch.cat([vx,vy],-1)
    xy=xy*(MAX_TRANSLATION/xy.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    return torch.cat([xy,values[...,2:3],values[...,3:5].clamp(-MAX_ANGLE,MAX_ANGLE)],-1)

HEIGHT_MIN=-.20
HEIGHT_MAX=.05
HEIGHT_RATE=.06
HEIGHT_ACCEL=.15
P3_MAX_SPEED=2.

def tilt_limit(height):
    # Joint envelope, applied to roll AND pitch together. Continuous at knots.
    degrees=torch.where(height>=-.10,20.,torch.where(height>=-.15,
        10.+200.*(height+.15),5.+100.*(height+.20)))
    return degrees.clamp(5.,20.)*(math.pi/180)

def minimum_height_for_tilt(rp):
    degrees=rp.abs().amax(-1)*(180/math.pi)
    return torch.where(degrees<=5.,-.20,torch.where(degrees<=10.,
        -.20+(degrees-5.)/100.,-.15+(degrees-10.)/200.)).clamp(-.20,-.10)

def constrain_targets(values):
    if values.shape[-1]==5:
        return constrain_legacy(values)  # legacy case generator only
    height=values[...,5:6].clamp(HEIGHT_MIN,HEIGHT_MAX)
    old=constrain_legacy(values[...,:5]);limit=tilt_limit(height)
    return torch.cat([old[...,:3],old[...,3:5].clamp(-limit,limit),height],-1)

def command(vx=0.,vy=0.,yaw_rate=0.,roll_degrees=0.,pitch_degrees=0.,height=0.):
    if not all(math.isfinite(x) for x in [vx,vy,yaw_rate,roll_degrees,pitch_degrees,height]):
        raise ValueError('commands must be finite')
    if max(abs(roll_degrees),abs(pitch_degrees))>20. or not HEIGHT_MIN<=height<=HEIGHT_MAX:
        raise ValueError('request outside height/tilt task bounds')
    return constrain_targets(torch.tensor([vx,vy,yaw_rate,math.radians(roll_degrees),
        math.radians(pitch_degrees),height],dtype=torch.float64)).tolist()

def moving(context):
    return (context[...,:2].norm(dim=-1)>.05)|(context[...,2].abs()>.05)

def frequency(context):
    # Longer low-speed steps, same 3.4 Hz at 4 m/s. Spin needs finite cadence.
    speed=torch.maximum(context[...,:2].norm(dim=-1),.25*context[...,2].abs())
    return 1.4+.5*speed.clamp(max=4.)

def duty(context):
    return .65-.10*context[...,:2].norm(dim=-1).clamp(max=4.)/4.

def blend(context):
    x=context[...,16].clamp(0,1)
    return x.square()*(3.-2.*x)

def clearance(context):
    # Retain 4/6/8 cm at -20/-15/-10 cm; raise normal-height swing to 12 cm.
    return (.08+.4*(context[...,13]+.10)).clamp(.04,.12)

def foot_phases(context):
    return torch.remainder(context[...,12:13]+context.new_tensor([.5,0.,0.,.5]),1.)

def gait(context):
    phase=foot_phases(context);d=duty(context)[...,None];b=blend(context)[...,None]
    warp=torch.where(phase<d,phase*(.5/d),.5+(phase-d)*(.5/(1-d)))
    def cdf(x):return .5*(1+torch.erf(x/(.07*math.sqrt(2))))
    contact=cdf(warp)*(1-cdf(warp-.5))+cdf(warp-1)*(1-cdf(warp-1.5))
    contact=1.+b*(contact-1.)
    swing=((phase-d)/(1-d)).clamp(0,1)
    height=.032+clearance(context)[...,None]*torch.sin(math.pi*swing)*b
    clocks=1.+b*(torch.sin(2*math.pi*warp)-1.)
    return contact,height,clocks

def policy_goal(context):
    return torch.cat([context[...,:12],gait(context)[2],context[...,13:17]],-1)

def set_requested(context,target):
    if target.shape[-1]==5:target=torch.cat([target,torch.zeros_like(target[...,:1])],-1)
    target=constrain_targets(target).expand(*context.shape[:-1],6)
    return torch.cat([context[...,:7],target[...,:5],context[...,12:15],target[...,5:6],context[...,16:17]],-1)

def advance_command(context):
    dv=context[...,7:10]-context[...,:3]
    xy=context[...,:2]+dv[...,:2]*(LINEAR_ACCELERATION*DT/dv[...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    yaw=context[...,2:3]+dv[...,2:3].clamp(-ANGULAR_ACCELERATION*DT,ANGULAR_ACCELERATION*DT)
    q,v=context[...,3:5],context[...,5:7]
    # While raising, height must open the envelope before tilt grows; while
    # lowering, tilt must shrink before height falls. Neither may jump.
    current_limit=tilt_limit(context[...,13:14])
    limit=torch.minimum(tilt_limit(context[...,15:16]),current_limit)
    target=context[...,10:12].clamp(-limit,limit)
    acc=(COMMAND_OMEGA**2*(target-q)-2*COMMAND_OMEGA*v).clamp(-MAX_ACCEL,MAX_ACCEL)
    v=(v+DT*acc).clamp(-MAX_RATE,MAX_RATE)
    qn=(q+DT*v).clamp(-current_limit,current_limit)
    v=(qn-q)/DT;q=qn
    # Level the body before descending through a tighter tilt envelope.
    h,hv=context[...,13],context[...,14]
    floor=minimum_height_for_tilt(q)
    ht=torch.maximum(context[...,15],floor)
    ha=(16.*(ht-h)-8.*hv).clamp(-HEIGHT_ACCEL,HEIGHT_ACCEL)
    hv=(hv+DT*ha).clamp(-HEIGHT_RATE,HEIGHT_RATE)
    hn=(h+DT*hv).clamp(HEIGHT_MIN,HEIGHT_MAX)
    clipped=hn<floor;hn=torch.maximum(hn,floor);hv=torch.where(clipped,torch.zeros_like(hv),hv)
    temp=torch.cat([xy,yaw,q,v,context[...,7:13],hn[...,None],hv[...,None],context[...,15:17]],-1)
    gp=(context[...,16]+torch.where(moving(temp),DT/.4,-DT/.4)).clamp(0,1)
    gb=gp.square()*(3-2*gp)
    phase=torch.where(gb>0,torch.remainder(context[...,12]+DT*frequency(temp)*gb,1.),.25)
    return torch.cat([temp[...,:12],phase[...,None],temp[...,13:16],gp[...,None]],-1)

def yaw_sampling_bound(speed,max_yaw=TRAIN_YAW_MAX):
    return (TURN_ACCELERATION_BUDGET/speed.clamp_min(1.)).clamp(max=max_yaw)
