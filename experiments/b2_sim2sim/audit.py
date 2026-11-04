"""Check transfer math against frozen original outputs before a physical sweep."""
import argparse
import numpy as np
import mujoco
import run
from pathlib import Path


def main(out):
    run.verify_inputs();checks={}
    z=np.load(run.INPUTS/'hindstand/parity_inputs.npz')
    actor=run.Controller(run.INPUTS/'hindstand').actor
    checks['actor_max_abs_error']=float(abs(actor(z['obs'],z['goal'])-z['expected_action']).max())
    assert checks['actor_max_abs_error']<2e-5
    z=np.load(run.INPUTS/'hindstand/envelope_inputs.npz')
    vals=[run.saturated_pd_target(t,q,v,250,8,run.META['torque_limits'],run.META['torque_envelope'])[2]
          for t,q,v in zip(z['target'],z['q'],z['dq'])]
    checks['envelope_max_abs_error']=float(abs(np.array(vals)-z['expected_tau']).max())
    assert checks['envelope_max_abs_error']<1e-3
    m=run.model_for(0,{'friction':1},.002);a2j=np.array([int(m.actuator_trnid[i,0])-1 for i in range(12)])
    checks['actuator_to_policy_joint']=a2j.tolist()
    d=mujoco.MjData(m);q,v=run.initial_states(m,{'start':'reference'},1)
    d.qpos[:]=q[0];d.qvel[:]=v[0]
    target=d.qpos[7:]+np.linspace(-.02,.02,12)
    d.ctrl[:]=target[a2j];mujoco.mj_forward(m,d)
    expected=250*(target-d.qpos[7:])-8*d.qvel[6:]
    checks['mapped_pd_force_error']=float(abs(d.actuator_force[np.argsort(a2j)]-expected).max())
    assert checks['mapped_pd_force_error']<1e-8
    # Verify orientation sign and full-cycle scoring on the immutable original
    # 128 trajectories, not on newly generated expected values.
    source=run.ROOT/'runs/supermimic_core_v3_20261103_001524/B2/pair_1202601/C1/eval_002000'
    original=run.json.loads((source/'result.json').read_text());counts=[]
    for row in original['conditions']:
        arr=dict(np.load(source/(row['case_id']+'_rollout.npz')))
        arr['qpos']=arr['state'][:,:,:19]
        qw,qx,qy,qz=np.moveaxis(arr['state'][:,:,3:7],-1,0)
        arr['inclination']=np.degrees(np.arcsin(np.clip(2*(qx*qz-qw*qy),-1,1)))
        n=arr['qpos'].shape[1]
        synthetic={'time_s':np.array([0.]),'dq':np.zeros((1,n,12)),'power':np.zeros((1,n))}
        res=run.summarise(arr,synthetic,None)
        assert [r['task_pass'] for r in res['trials']]==[r['success'] for r in row['trials_detail']]
        counts.append(res['counts']['task_pass'])
    checks['original_scoring_parity_passed']=counts
    checks.update(passed=True,weights_sha256=run.META['weights_sha256'],
                  metric_source_sha256=run.digest(run.ROOT/'experiments/b2_cycle/evaluate_cycle.py'))
    run.save(out,checks);print(run.json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
