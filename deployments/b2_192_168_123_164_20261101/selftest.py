"""Verify exported actors against Torch fixtures and benchmark CPU inference."""
import argparse
import json
import platform
import time
from pathlib import Path
from runtime import Actor, Controller, np, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha256(root/name)==digest, f'Integrity failure: {name}'
    result={'hostname':platform.node(),'numpy':np.__version__, 'no_motor_commands':True,
            'file_integrity_passed':True,'skills':{}}
    for skill in ['hindstand','locomotion']:
        folder=root/'models'/skill
        actor=Actor(folder)
        fixture=np.load(folder/'parity_inputs.npz',allow_pickle=False)
        errors=[];timings=[]
        for o,g,expected in zip(fixture['obs'],fixture['goal'],fixture['expected_action']):
            start=time.perf_counter_ns()
            action=actor(o[None],g[None])[0]
            timings.append((time.perf_counter_ns()-start)/1e6)
            errors.append(float(np.max(np.abs(action-expected))))
        assert max(errors)<2e-5,(skill,max(errors))
        controller=Controller(folder)
        q=np.empty(12,dtype=np.float32)
        q[controller.perm]=controller.q0
        cycle=[]
        for i in range(1200):
            start=time.perf_counter_ns()
            controller.step(q,np.zeros(12),np.zeros(3),[1,0,0,0],
                            root_position=[0,0,.56738955] if skill=='hindstand' else None)
            cycle.append((time.perf_counter_ns()-start)/1e6)
        statistics=lambda t: dict(p50_ms=float(np.percentile(t,50)),p95_ms=float(np.percentile(t,95)),
                                  p99_ms=float(np.percentile(t,99)),max_ms=float(max(t)),
                                  over_20ms=sum(v>=20 for v in t))
        result['skills'][skill]={'checkpoint_iteration':actor.meta['iteration'],
            'checkpoint_sha256':actor.meta['checkpoint_sha256'],
            'parity_cases':len(errors),'max_abs_error':max(errors),'parity_passed':True,
            'actor_latency':statistics(timings),'controller_latency':statistics(cycle),
            'benchmark_is_synthetic_input':True}
        assert np.percentile(cycle,99)<20, (skill,statistics(cycle))
    result['passed']=True
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
