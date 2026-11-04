"""Represent the training model's saturated PD using the same SDK PD gains.

This limits the torque requested at the sampled joint state, not an independently
measured future motor torque. Firmware protections remain authoritative.
"""
import math

def saturated_pd_target(target,q,dq,kp,kd,limits,envelope=None):
    if not (len(target)==len(q)==len(dq)==len(limits)==12):
        raise ValueError('Expected 12 joint values')
    if not math.isfinite(kp) or kp<=0 or not math.isfinite(kd) or kd<0:
        raise ValueError('Invalid PD gains')
    result=[];requested=[];limited=[]
    for desired,position,velocity,limit in zip(target,q,dq,limits):
        if not all(math.isfinite(v) for v in (desired,position,velocity,limit)) or limit<=0:
            raise ValueError('Invalid PD sample')
        tau=kp*(desired-position)-kd*velocity
        clipped=max(-limit,min(limit,tau))
        # Preserve the exact original target when not saturated.
        result.append(desired if clipped==tau else position+(clipped+kd*velocity)/kp)
        requested.append(tau);limited.append(clipped)
    if envelope is not None:
        start=float(envelope['motoring_full_speed_rad_s'])
        end=float(envelope['motoring_zero_speed_rad_s'])
        power=float(envelope['command_power_w'])
        if not all(math.isfinite(x) for x in (start,end,power)) or not 0<=start<end or power<=0:
            raise ValueError('Invalid training torque envelope')
        limited=[tau*max(0.,min(1.,(end-abs(v))/(end-start))) if tau*v>0 else tau
                 for tau,v in zip(limited,dq)]
        scale=min(1.,power/max(power,sum(abs(tau*v) for tau,v in zip(limited,dq))))
        limited=[tau*scale for tau in limited]
        result=[desired if raw==tau else pos+(tau+kd*vel)/kp
                for desired,pos,vel,raw,tau in zip(target,q,dq,requested,limited)]
    return result,requested,limited
