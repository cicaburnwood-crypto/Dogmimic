"""Boundary checks for stopping, coverage and immutable two-batch acceptance."""
from copy import deepcopy
from selection import regression,improved,validate_accepted

def g1():
    return dict(robot='G1',pack='DEV-A',passed=651,completed=672,conditions=[dict(case_id='nominal' if i==0 else str(i),passed=31,completed=32,nominal_success=True) for i in range(21)])

a=g1();b=deepcopy(a);b['passed']-=64
assert not regression(b,a)
b['passed']-=1;assert regression(b,a)
b=deepcopy(a);b['conditions'][0]['passed']-=4;assert regression(b,a)
b=deepcopy(a);b['conditions'][0]['passed']-=3;assert not regression(b,a)
a.update(checkpoint_sha256='fixed',complete_coverage=True,pack='HOLD-A')
b=deepcopy(a);b['pack']='HOLD-B'
assert validate_accepted('G1',[a,b],'fixed')
b['conditions'][0]['nominal_success']=False;assert not validate_accepted('G1',[a,b],'fixed')
b=deepcopy(a);b['pack']='HOLD-B';b['checkpoint_sha256']='changed'
assert not validate_accepted('G1',[a,b],'fixed')
b=deepcopy(a);b['pack']='HOLD-B';b['conditions'].pop()
assert not validate_accepted('G1',[a,b],'fixed')
print('selection boundaries passed')

def b2():
    return dict(robot='B2',pack='DEV-A',passed=120,completed=128,conditions=[dict(case_id=f'b2_0{i}',passed=60,completed=64,nominal_success=True,
        peak_speed=6.,peak_mechanical_power=1501.,trial_peak_speed=[6.]*64,trial_peak_power=[1501.]*64) for i in range(2)])
a=b2();b=deepcopy(a);b['conditions'][0]['passed']-=6;assert not regression(b,a)
b['conditions'][0]['passed']-=1;assert regression(b,a)
a=b2();b=deepcopy(a)
for row in b['conditions']:row['passed']=64
assert improved(b,a)  # the predeclared near-ceiling rule requires reaching64
b['conditions'][1]['passed']=63;assert not improved(b,a)
a.update(pack='HOLD-A',complete_coverage=True,checkpoint_sha256='fixed')
a['conditions']=[dict(a['conditions'][0],case_id=str(i),passed=61) for i in range(48)]
b=deepcopy(a);b['pack']='HOLD-B'
assert validate_accepted('B2',[a,b],'fixed')
b['conditions'][0]['peak_mechanical_power']=1501.0001;assert not validate_accepted('B2',[a,b],'fixed')
b=deepcopy(a);b['pack']='HOLD-B';b['conditions'][0]['peak_speed']=6.00001;assert not validate_accepted('B2',[a,b],'fixed')
print('B2 acceptance and regression boundaries passed')
from selection import RegressionGuard,score
baseline=g1();guard=RegressionGuard(baseline)
x=deepcopy(baseline);x['conditions'][1]['passed']-=4
assert not guard.update(x,baseline)
y=deepcopy(baseline);y['conditions'][2]['passed']-=4
assert not guard.update(y,baseline)
assert guard.update(y,baseline)
# A newly better total score must not move a reference within a bad-case streak.
guard=RegressionGuard(baseline);assert not guard.update(x,baseline)
new_best=deepcopy(baseline);new_best['conditions'][1]['passed']=28
assert guard.update(x,new_best)
low=deepcopy(baseline);low['conditions'][1]['passed']=0;low['passed']=671
assert score(baseline)>score(low)
print('per-case fixed-reference guards and G1 worst-case ranking passed')
