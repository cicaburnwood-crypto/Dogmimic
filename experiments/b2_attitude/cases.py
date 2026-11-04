"""Frozen physical acceptance cases; test angles include non-integer values."""
import math
import random

CRITERIA=dict(hold_rmse_deg=2.,small_angle_relative_rmse=.1,small_angle_rmse_floor_deg=.25,
    hold_peak_error_deg=5.,tracking_rmse_deg=4.,max_overshoot_deg=5.,
    neutral_peak_deg=2.,hold_angular_rate_rms=.15,neutral_speed_rms=.08,
    max_foot_drift_m=.05,max_body_drift_m=.08,min_four_support_fraction=.95,
    min_three_support_fraction=.99,max_nonfoot_contact_fraction=0.,
    joint_speed_rad_s=6.,mechanical_power_w=1501.,min_case_success=.95,
    initial_stable_seconds=.5,settling_window_starts_after_command_s=3.)

def event(time,axis=None,angle=0.):
    target=[0.,0.]
    if axis is not None:target[axis]=float(angle)
    return dict(time=float(time),target_degrees=target)

def make_cases(full=False,seed=73129):
    out=[]
    def add(name,events,domain=None,basic=True):
        out.append(dict(name=name,events=[event(0)]+events,domain=domain or {},basic=basic))
    add('neutral',[])
    for axis in range(2):
        for angle in [-30.,-22.5,-15.,-7.5,-1.,-.5,.5,1.,7.5,15.,22.5,30.]:
            add(f'{["roll","pitch"][axis]}_{angle:+g}',[event(2,axis,angle),event(6)])
    sequences=[('roll_reversal',[(0,30),(0,-30),(0,13.7),(0,-8.3)]),
               ('pitch_reversal',[(1,30),(1,-30),(1,18.6),(1,-4.2)]),
               ('cross_axis',[(0,30),(1,-30),(0,-30),(1,30)]),
               ('cross_axis_reverse',[(1,30),(0,-30),(1,-30),(0,30)])]
    for name,targets in sequences:
        add(name,[event(2+4*i,a,v) for i,(a,v) in enumerate(targets)]+[event(18)])
    add('queued_return_roll_pitch',[event(2,0,30),event(6),event(10,1,-30),event(14)])
    add('queued_return_pitch_roll',[event(2,1,30),event(6),event(10,0,-30),event(14)])
    rng=random.Random(seed+111)
    for i in range(2 if not full else 10):
        add(f'random_sequence_{i}',[event(2+4*j,rng.randrange(2),rng.uniform(-30,30)) for j in range(5)]+[event(22)],basic=i<2)
    if full:
        domains=[('mu04',dict(friction=.4)),('mu13',dict(friction=1.3)),('mass09',dict(mass_scale=.9)),
                 ('mass11',dict(mass_scale=1.1)),('weak_pd',dict(kp_scale=.85,kd_scale=.8)),
                 ('strong_pd',dict(kp_scale=1.15,kd_scale=1.2)),('delay10ms',dict(delay=2)),
                 ('combined',dict(friction=.4,mass_scale=1.1,kp_scale=.85,kd_scale=.8,delay=2))]
        for label,domain in domains:
            for axis in range(2):
                for angle in [-30.,30.]:
                    add(f'{label}_{axis}_{angle:+g}',[event(2,axis,angle),event(6)],domain,basic=False)
        directions=[(0,30),(0,-30),(1,30),(1,-30)]
        for i,first in enumerate(directions):
            for j,second in enumerate(directions):
                add(f'ordered_pair_{i}_{j}',[event(2,*first),event(6,*second),event(10)],basic=False)
    return out

DURATION=27.

