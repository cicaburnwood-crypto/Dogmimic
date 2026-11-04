"""Predeclared development comparisons; success remains in unchanged evaluators."""
import numpy as np
import copy


def cases(report):return {r['case_id']:r for r in report['conditions']}


def comparable(a,b):
    return a['robot']==b['robot'] and a['pack']==b['pack'] and set(cases(a))==set(cases(b))


def score(report):
    rr=report['conditions']
    if report['robot']=='G1':return (min(r['passed'] for r in rr),report['passed'],report['completed'])
    return (int(report['all_passed']),min(r['completed'] for r in rr),min(r['passed'] for r in rr),
            sum(r['passed'] for r in rr),-max(r['peak_speed'] for r in rr))


def regression(latest,best):
    assert comparable(latest,best)
    a,b=cases(latest),cases(best)
    if latest['robot']=='G1':
        return best['passed']-latest['passed']>64 or any(
            b[k]['passed']-a[k]['passed']>=4 or b[k]['completed']-a[k]['completed']>=4 for k in a)
    return any(b[k]['passed']-a[k]['passed']>=7 or b[k]['completed']-a[k]['completed']>=7 for k in a)


def b2_actuator(report):
    s=np.concatenate([r['trial_peak_speed'] for r in report['conditions']])
    p=np.concatenate([r['trial_peak_power'] for r in report['conditions']])
    return dict(violation_rate=float(((s>6)|(p>1501)).mean()),
                combined_tail=float(np.quantile(np.maximum(s/6-1,0)+np.maximum(p/1501-1,0),.95)),
                speed_tail=float(np.quantile(np.maximum(s/6-1,0),.95)),
                power_tail=float(np.quantile(np.maximum(p/1501-1,0),.95)))


def improved(new,baseline):
    assert comparable(new,baseline)
    a,b=cases(new),cases(baseline)
    if new['robot']=='G1':
        return new['passed']-baseline['passed']>=16 and baseline['completed']-new['completed']<=4 and \
               a['nominal']['passed']==32 and all(a[k]['passed']>=31 for k in b if b[k]['passed']>=31)
    nonreg=all(b[k]['passed']-a[k]['passed']<=3 and b[k]['completed']-a[k]['completed']<=2 for k in a)
    if not nonreg:return False
    worst=min(r['passed'] for r in b.values());current=min(r['passed'] for r in a.values())
    motion=worst<64 and current-worst>=min(7,64-worst)
    aa,bb=b2_actuator(new),b2_actuator(baseline)
    physical=any(bb[k]>0 and aa[k]<=.8*bb[k] for k in ['violation_rate','combined_tail']) and all(aa[k]<=1.1*bb[k]+1e-12 for k in bb)
    physical=physical and all(a[k]['completed']>=b[k]['completed'] for k in a)
    return motion or physical


class RegressionGuard:
    """Keep the reference fixed during a streak and require the same bad case."""
    def __init__(self,initial):
        self.initial=copy.deepcopy(initial);self.reference=None;self.previous_keys=set()

    def update(self,latest,best):
        if self.reference is None:
            self.reference=copy.deepcopy(self.initial if latest['robot']=='B2' else best)
        ref=self.reference;a,b=cases(latest),cases(ref)
        assert comparable(latest,ref)
        threshold=7 if latest['robot']=='B2' else 4
        keys={k for k in a if b[k]['passed']-a[k]['passed']>=threshold or b[k]['completed']-a[k]['completed']>=threshold}
        if latest['robot']=='G1' and ref['passed']-latest['passed']>64:keys.add('__total__')
        persistent=keys & self.previous_keys
        self.previous_keys=keys
        if not keys:self.reference=None
        return bool(persistent)


def validate_accepted(robot,reports,checkpoint_sha):
    required=48 if robot=='B2' else 21
    if len(reports)!=2 or {r['pack'] for r in reports}!={'HOLD-A','HOLD-B'}:return False
    for r in reports:
        if r['checkpoint_sha256']!=checkpoint_sha or not r['complete_coverage'] or len(r['conditions'])!=required:return False
        if robot=='B2':
            if not all(c['passed']>=61 and c['nominal_success'] and c['peak_speed']<=6 and c['peak_mechanical_power']<=1501 for c in r['conditions']):return False
        elif not all(c['passed']>=31 for c in r['conditions']) or not cases(r)['nominal']['nominal_success']:return False
    return True
