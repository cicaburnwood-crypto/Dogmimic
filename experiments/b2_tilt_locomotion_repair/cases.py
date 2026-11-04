"""Fixed independent test matrix, shared by neither reward nor curriculum sampler."""
import math
from contract import command,CURRICULUM

DURATION=30.
CRITERIA=dict(velocity_rmse_absolute_m_s=.12,velocity_rmse_relative=.15,
    yaw_rmse_absolute_rad_s=.15,yaw_rmse_relative=.20,
    attitude_rmse_deg=2.,attitude_peak_deg=5.,dynamic_attitude_rmse_deg=5.,
    stop_speed_rms_m_s=.10,stop_yaw_rms_rad_s=.15,stop_drift_m=.10,
    contact_slip_rms_m_s=.20,nonfoot_contact_fraction=.01,
    actual_planar_speed_m_s=4.,actual_lateral_speed_m_s=1.,numerical_tolerance=1e-4,
    joint_limit_tolerance_rad=.02,min_body_height_m=.30,
    per_condition_pass_fraction=.95,nominal_must_pass=True)

def make_cases(full=False,seed=73129):
    rows=[]
    def add(name,vx,vy,yaw,r,p,level=4,basic=False,domain=None,reverse=None):
        target=command(vx,vy,yaw,r,p)
        rows.append(dict(name=name,target=target,reverse=reverse or [-x for x in target],level=level,basic=basic,domain=domain or {}))
    add('neutral',0,0,0,0,0,0,True)
    for level,(speed,tilt,yaw) in enumerate(CURRICULUM):
        y=min(.6*speed,1.);x=math.sqrt(speed**2-y**2)
        turn=min(yaw,2/max(1.,speed))
        for sign in (-1,1):
            add(f'L{level}_forward_{sign}',sign*speed,0,0,sign*tilt,sign*tilt,level,True)
            add(f'L{level}_side_{sign}',0,sign*min(1.,speed),0,sign*tilt,-sign*tilt,level,True)
            add(f'L{level}_diagonal_turn_{sign}',sign*x,-sign*y,sign*turn,-sign*tilt,sign*tilt,level,True)
            add(f'L{level}_spin_{sign}',0,0,sign*yaw,sign*tilt,sign*tilt,level,True)
    if full:
        for r in (-20.,0.,20.):
          for p in (-20.,0.,20.):
            tag=f'r{r:+g}_p{p:+g}'
            add(tag+'_stand',0,0,0,r,p)
            for sign in (-1,1):
                for speed in (.5,2.,4.):add(tag+f'_x{sign*speed:+g}',sign*speed,0,0,r,p)
                add(tag+f'_y{sign}',0,sign,0,r,p)
                add(tag+f'_spin{sign}',0,0,sign*2,r,p)
                for side in (-1,1):
                    x=sign*math.sqrt(15)
                    add(tag+f'_diagonal{sign}_{side}',x,side,0,r,p)
                    add(tag+f'_curve{sign}_{side}',x,side,side*.5,r,p)
        # Explicit low/high friction and continuous interior commands.
        for mu in (.4,1.3):
            for sign in (-1,1):add(f'mu{mu}_{sign}',sign*math.sqrt(15),sign,sign*.5,sign*20,-sign*20,domain={'friction':mu})
        import random
        rng=random.Random(seed)
        for i in range(12):
            vx=rng.uniform(-4,4);vy=rng.uniform(-1,1)
            speed=min(4.,math.hypot(vx,vy));yaw=rng.uniform(-1,1)*min(2,2/max(1,speed))
            add(f'interior_{i}',vx,vy,yaw,rng.uniform(-20,20),rng.uniform(-20,20),
                reverse=command(rng.uniform(-4,4),rng.uniform(-1,1),-.35,rng.uniform(-20,20),rng.uniform(-20,20)))
    return rows
