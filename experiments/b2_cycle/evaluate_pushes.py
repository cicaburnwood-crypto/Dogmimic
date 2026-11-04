"""Frozen-policy, paired control / random external-force evaluation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from supermimic import networks
from supermimic.networks import Policy
from supermimic.task import state_to_flat
from cycle_env import CycleLibrary,make_cycle_env
from evaluate_cycle import evaluate

HERE=Path(__file__).resolve().parent

@torch.no_grad()
def verify_force(lib,device,model_path=None):
    env=make_cycle_env(lib,device,8,evaluate=True,model_path=model_path);ids=torch.arange(8,device=device)
    env.reset();env.reset_envs(ids,torch.zeros_like(ids))
    env.sim.data.qpos[:,2]=3.;env.sim.data.qvel[:]=0.;env._finish_resets(ids,force=True)
    env.sim.data.xfrc_applied[:]=0.
    for axis in range(3):env.sim.data.xfrc_applied[2*axis+1,1,axis]=10.
    _,info=env.step(torch.zeros(8,12,device=device))
    v=info['com_velocity'].cpu().numpy();mass=float(env.mj_model.body_mass.sum())
    expected=10*.02/mass
    delta=np.stack([v[2*k+1]-v[2*k] for k in range(3)])
    error=float(abs(delta-np.eye(3)*expected).max())
    assert error<max(2e-5,.02*expected),(delta,expected,error)
    return dict(passed=True,mass_kg=mass,force_n=10.,duration_s=.02,expected_com_delta_velocity_m_s=expected,
                measured_xyz_delta_velocity=delta.tolist(),max_abs_error_m_s=error)

def schedule(n,steps,dt,seed):
    rng=np.random.default_rng(seed);force=np.zeros((steps,n,3),dtype=np.float32);events=[]
    windows=[(1.5,2.7,'rise'),(4.5,5.5,'hindstand_1'),(7.2,8.2,'hindstand_2'),(10.2,11.5,'return'),(13.1,13.8,'four_leg_settle')]
    for trial in range(n):
        trial_events=[]
        for lo,hi,phase in windows:
            start=int(round(rng.uniform(lo,hi)/dt));length=int(rng.integers(6,13))
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            magnitude=float(rng.uniform(8.,16.));vector=(magnitude*direction).astype(np.float32)
            force[start:start+length,trial]=vector
            trial_events.append(dict(phase=phase,start_step=start,end_step=start+length,start_s=start*dt,end_s=(start+length)*dt,
                                     duration_s=length*dt,magnitude_n=magnitude,force_xyz_n=vector.tolist()))
        events.append(trial_events)
    return force,events

def compact(result):
    return {k:v for k,v in result.items() if k!='trials_detail'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--trials',type=int,default=64);p.add_argument('--seed',type=int,default=20261109);p.add_argument('--push-seed',type=int,default=20261110)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);start=time.time()
    torch.set_num_threads(2);networks.AMP['enabled']=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    lib=CycleLibrary(str(HERE/'b2_full_cycle.npz'),'cuda')
    audit=verify_force(lib,'cuda');(out/'force_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('FORCE_AUDIT '+json.dumps(audit),flush=True)
    ck=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    arch=ck['policy_architecture']
    policy=Policy(ck['policy']['obs_norm.mean'].numel(),ck['policy']['goal_norm.mean'].numel(),enc_hidden=tuple(arch['encoder']),dec_hidden=tuple(arch['decoder'])).cuda()
    policy.load_state_dict(ck['policy']);env=make_cycle_env(lib,'cuda',args.trials,evaluate=True)
    env.sim.data.xfrc_applied[:]=0.
    control_initial=[];control_step=env.step
    def no_push_step(action):
        if not control_initial:control_initial.append(state_to_flat(env.state).cpu().numpy().copy())
        f=torch.zeros(args.trials,3,device='cuda')
        env.sim.data.xfrc_applied[:]=0.;env.sim.data.xfrc_applied[:,1,:3]=f
        f.cpu().numpy()  # Match the observation/synchronization path of the push group.
        return control_step(action)
    env.step=no_push_step
    control,control_arr=evaluate(policy,env,seed=args.seed)
    print('CONTROL '+json.dumps(compact(control)),flush=True)
    np.savez_compressed(out/'control_rollout.npz',**control_arr,dt=.02)
    (out/'control_evaluation.json').write_text(json.dumps(control,indent=2)+'\n')
    # Both groups use fresh simulators and identical initial states. GPU contact
    # reduction roundoff can still grow over time; record it, don't equate exact
    # trajectory replay with a matched-initial-condition experiment.
    env=make_cycle_env(lib,'cuda',args.trials,evaluate=True)
    planned,events=schedule(args.trials,lib.num_frames-1,.02,args.push_seed)
    force=torch.tensor(planned,device='cuda');applied=[];base_step=env.step;counter=0
    trial_live=torch.ones(args.trials,device='cuda',dtype=torch.bool);push_initial=[]
    def pushed_step(action):
        nonlocal counter,trial_live
        if not push_initial:push_initial.append(state_to_flat(env.state).cpu().numpy().copy())
        f=force[counter]*trial_live[:,None]
        env.sim.data.xfrc_applied[:]=0.
        env.sim.data.xfrc_applied[:,1,:3]=f
        applied.append(f.cpu().numpy().copy())
        state,info=base_step(action)
        trial_live&=~info['terminated'];counter+=1
        return state,info
    env.step=pushed_step
    perturbed,arr=evaluate(policy,env,seed=args.seed)
    env.sim.data.xfrc_applied[:]=0.;arr['force_world']=np.asarray(applied)
    assert counter==len(planned)
    earliest=min(e['start_step'] for tr in events for e in tr)
    prepush_error=float(abs(arr['state'][:earliest]-control_arr['state'][:earliest]).max())
    np.savez_compressed(out/'prepush_diagnostic.npz',control=control_arr['state'][:earliest],pushed=arr['state'][:earliest])
    print('PREPUSH_DIAGNOSTIC '+json.dumps(dict(max_error=prepush_error,first_step_error=float(abs(arr['state'][0]-control_arr['state'][0]).max()),trial0_error=float(abs(arr['state'][:earliest,0]-control_arr['state'][:earliest,0]).max()))),flush=True)
    initial_error=float(abs(control_initial[0]-push_initial[0]).max())
    assert initial_error==0.,initial_error
    assert all(torch.equal(v,ck['policy'][k]) for k,v in policy.state_dict().items()),'Policy changed during evaluation'
    for i,trial in enumerate(perturbed['trials_detail']):
        trial['first_failure_s']=float(arr['time_s'][np.flatnonzero(~arr['alive'][:,i])[0]]) if not arr['alive'][:,i].all() else None
    perturbed.update(it=int(ck['it']),source_iteration=int(ck['source_iteration']),checkpoint=args.checkpoint,
                     test_type='paired_random_force_evaluation',policy_frozen=True,seed=args.seed,push_seed=args.push_seed,
                     force_range_n=[8,16],duration_range_s=[.12,.24],pushes_per_trial=5,application='world XYZ force at trunk body COM; zero imposed torque',
                     model_mass_kg=audit['mass_kg'],force_fraction_of_weight=[8/(audit['mass_kg']*9.81),16/(audit['mass_kg']*9.81)],
                     pre_push_max_state_difference=prepush_error,render_trial=0,render_trial_selection='fixed before observing outcomes')
    perturbed.update(initial_state_max_difference=initial_error,inference_precision='bfloat16 MLP, float32 physics/state',
                     pairing='identical initial states and weights; independent GPU physics runs',
                     pre_push_max_root_position_difference_m=float(abs(arr['state'][:earliest,:,:3]-control_arr['state'][:earliest,:,:3]).max()))
    np.savez_compressed(out/'rollout.npz',**arr,dt=.02)
    (out/'evaluation.json').write_text(json.dumps(perturbed,indent=2)+'\n')
    (out/'push_schedule.json').write_text(json.dumps(events,indent=2)+'\n')
    c=np.array([t['success'] for t in control['trials_detail']]);q=np.array([t['success'] for t in perturbed['trials_detail']])
    summary=dict(checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                 trials=args.trials,control=compact(control),perturbed=compact(perturbed),
                 paired_lost_passes=np.flatnonzero(c&~q).tolist(),paired_gained_passes=np.flatnonzero(~c&q).tolist(),
                 elapsed_seconds=time.time()-start,force_audit=audit)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (HERE/'latest_push_evaluation.json').write_text(json.dumps(dict(out=str(out.resolve()),summary=compact(perturbed)),indent=2)+'\n')
    print('PERTURBED '+json.dumps(compact(perturbed)),flush=True)
    print('SAVED '+str(out),flush=True)

if __name__=='__main__':main()
