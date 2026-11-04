"""Frozen-model D0: complete-state replay, held-out dynamics and direction audit.

Finite differences only measure model credibility. They never update a policy.
Each stratum contains 16 fresh episodes; calibration/holdout split is by episode.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from common import ROOT, DESIGN, setup, read_json, write_json, tensor_digest


def diagnostic(robot,output,smoke=False,wm_checkpoint=None):
    Trainer,base,_=setup(robot)
    import torch
    from state import EnvironmentSnapshot,verify_replay
    from supermimic.task import state_to_flat,flat_to_state
    from supermimic.networks import run_mlp,apply_delta_c
    torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    n=2 if smoke else 16
    cfg=base.TrainCfg(motions='experiments/b2_cycle/b2_full_cycle.npz' if robot=='B2' else 'data/processed/g1_standing_jump_060m.npz',
            out=str(output),seed=4202601,num_envs=n,buffer_steps=64,wm_batch=n,pi_batch=n,wm_batches=1,
            k=8,horizon=32,use_critic=False,compile_wm=False,eval_every=0)
    t=Trainer(cfg,'cuda')
    wm_source_sha=t.source_hash
    if wm_checkpoint:
        from common import sha256
        ck=torch.load(wm_checkpoint,map_location=t.device,weights_only=False)
        assert all(torch.equal(v,t.policy.state_dict()[k]) for k,v in ck['policy'].items())
        assert ck['actor_updates']==0
        t.wm.load_state_dict(ck['wm'],strict=True);wm_source_sha=sha256(wm_checkpoint)
    # Fresh independent evaluation environment, exact same original physics.
    if robot=='B2':
        from robust_env import make_env
        e=make_env(t.lib,t.device,n,evaluate=True,kg=0.)
        strata=[dict(case=f'{kg}kg',kg=kg,frame=fr,phase=ph)
                for kg in (0.,5.) for ph,fr in [('rise',100),('hold',300),('descent',550),('recovery',680)]]
    else:
        from robust_env import RobustEnv
        e=RobustEnv(t.lib,t.device,n,evaluate=True)
        strata=[dict(case=c,frame=f,phase=ph) for c,ph,f in [
            ('nominal','takeoff',45),('nominal','landing',70),('mu_0.15','takeoff',45),('mu_0.15','landing',70),
            ('native_start','prepare',15),('combined_high','prepare',15),('combined_low','landing',70),('push_back','recovery',130)]]
    t.env=e;t.task=e.task;t.tcfg=e.tcfg
    task,lib,wm,policy=e.task,t.lib,t.wm,t.policy
    policy.eval();wm.eval()
    for mod in [policy,wm,t.critic]:mod.requires_grad_(False)
    original=dict(policy=tensor_digest(policy.state_dict()),wm=tensor_digest(wm.state_dict()))
    steps=32;all_rows=[];all_errors=[];replays=[];start=time.time()
    generator=torch.Generator(device=t.device);generator.manual_seed(4202611)
    directions=(torch.randint(0,2,(4,len(e.q0)),generator=generator,device=t.device)*2-1).float()
    manifest=read_json(DESIGN/'evaluation_manifest.json')
    # Read every actual physics substep before the environment can reset.
    substep=[]
    original_sim_step=e.sim.step
    def record_step():
        original_sim_step()
        qd=e.sim.data.qvel[:,e.qd_adr][:,e.act_to_joint].clone()
        force=e.sim.data.actuator_force.clone()
        substep.append((qd,force,e.sim.data.ctrl.clone()))
    e.sim.step=record_step

    def initialize(st,seed):
        torch.manual_seed(seed)
        if robot=='B2':e.fixed_loads.fill_(st['kg']);e.fixed_mass=1.;e.fixed_friction=1.
        else:e.case=next(c['parameters'] for c in manifest['cases']['G1'] if c['case_id']==st['case'])
        e.reset();ids=torch.arange(n,device=t.device);frame=torch.zeros(n,dtype=torch.long,device=t.device)
        e.reset_envs(ids,frame)
        if robot=='B2':
            e._finish_resets(ids,force=True)
            e.sim.data.qpos[:,e.q_adr]+=.006*torch.randn(n,len(e.q0),device=t.device)
        else:
            from supermimic.rotations import quat_from_euler_xyz,quat_mul
            ref=lib.frame(frame);pos,quat,q=ref['pos'].clone(),ref['quat'].clone(),ref['q'].clone()
            pos[:,:2]+=(torch.rand(n,2,device=t.device)*2-1)*.003
            q+=(torch.rand_like(q)*2-1)*.008
            angles=(torch.rand(n,3,device=t.device)*2-1)*.008
            quat=quat_mul(quat_from_euler_xyz(*angles.unbind(-1)),quat)
            pos,quat,q=e.prepare_eval(pos,quat,q)
            e.write_state(ids,pos,quat,q,torch.zeros(n,3,device=t.device),torch.zeros(n,3,device=t.device),torch.zeros_like(q))
        e._finish_resets(ids,force=True)
        ids0=e.ep_id.clone();alive=torch.ones(n,dtype=torch.bool,device=t.device)
        with torch.no_grad():
            for i in range(st['frame']):
                _,info=e.step(policy(*e.policy_inputs()));alive &= ~info['done']
        return alive & (e.ep_id==ids0)

    def imagined(initial,perturb,actions=None):
        state={k:v.clone() for k,v in initial['state'].items()}
        history=initial['history'].clone();context=initial['context'].clone();prev=initial['prev'].clone()
        frame=initial['frame'];alive=torch.ones(n,device=t.device);J=torch.zeros_like(alive)
        states=[];rewards=[];deaths=[];summed=[]
        for step in range(steps):
            idx=lib.clamp_to_clip(frame+step,frame);nxt=lib.clamp_to_clip(frame+step+1,frame)
            future=e.ref_future(idx);ref=lib.frame(idx)
            kin=e.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
            an=task.analytics_fast(state,ref,kin,future,prev,prev,context[:,1:1+len(e.q0)])
            action=policy(history.flatten(1),an['goal']) if actions is None else actions[step]
            if step==0:action=action+perturb/e.action_scale
            xt=torch.cat([t.context_at(initial['context'],step,e.tcfg.dt),context[:,21:]],-1) if robot=='B2' else context
            delta=run_mlp(wm.net,wm.in_norm(wm.inputs(state,action,xt,an['feats'])))*wm.out_norm.std+wm.out_norm.mean
            ns=apply_delta_c(state,delta)
            ref=lib.frame(nxt);kin=e.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
            if robot=='B2':out=task.transition_fast(state,ns,ref,kin,e.ref_future(nxt),action,prev,xt)
            else:out=task.analytics_fast(ns,ref,kin,e.ref_future(nxt),action,prev,xt[:,1:1+len(e.q0)])
            valid=(lib.remaining(idx)>0).float()
            J=J+cfg.gamma**step*alive*out['r']*valid
            alive=alive*(1-out['soft'])*valid
            if robot=='B2':
                from actuator_filter import advance_context
                context=task.advance_context(ns,nxt,xt)
                context=torch.cat([context[:,:31],advance_context(xt,action,task.q0,task.action_scale,task.qmin,task.qmax)[:,31:43]],-1)
            else:
                from supermimic.dynamic_context import advance
                context=advance(context,action,task.q0,task.action_scale)
            history=torch.cat([history[:,1:],out['obs'][:,None]],1)
            state,prev=ns,action
            states.append(state_to_flat(ns));rewards.append(out['r']);deaths.append(out['hard']);summed.append(J)
        return dict(states=torch.stack(states),rewards=torch.stack(rewards),deaths=torch.stack(deaths),returns=torch.stack(summed))

    def contact_signature(state):
        if robot=='B2':return task.foot_positions(task.body_kin(state))[...,2]<.045
        from evaluate_jump import geometry
        _,h,_=geometry(e,state)
        return torch.from_numpy(h).to(t.device)<.003

    def physical(snapshot,perturb):
        snapshot.restore()
        state={k:v.clone() for k,v in e.state.items()}
        states=[];rewards=[];deaths=[];summed=[];actions=[];modes=[];contexts=[];live=[];speed5=[];power5=[];consumed=[]
        alive=torch.ones(n,device=t.device);J=torch.zeros_like(alive);hard_live=alive.bool()
        with torch.no_grad():
            for step in range(steps):
                xt=e.xi.clone();prev=e.prev_action.clone();frame=e.frame.clone();action=policy(*e.policy_inputs())
                if step==0:action=action+perturb/e.action_scale
                substep.clear();ns,info=e.step(action)
                assert len(substep)==4
                speed5.append(torch.stack([x[0].abs().amax(-1) for x in substep]))
                power5.append(torch.stack([(x[0]*x[1]).abs().sum(-1) for x in substep]))
                consumed.append(torch.stack([x[2] for x in substep]))
                ref=lib.frame(frame+1);kin=e.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
                if robot=='B2':out=task.transition_fast(state,ns,ref,kin,e.ref_future(frame+1),action,prev,xt)
                else:out=task.analytics_fast(ns,ref,kin,e.ref_future(frame+1),action,prev,xt[:,1:1+len(e.q0)])
                valid=(lib.remaining(frame)>0).float()
                J=J+cfg.gamma**step*alive*info['reward']*valid*hard_live
                alive=alive*(1-out['soft'])*valid
                states.append(state_to_flat(ns));rewards.append(info['reward']);deaths.append(info['terminated']);summed.append(J.clone())
                actions.append(action);modes.append(contact_signature(ns));contexts.append(xt);live.append(hard_live.clone())
                hard_live &= ~info['done'];state=ns
        return dict(states=torch.stack(states),rewards=torch.stack(rewards),deaths=torch.stack(deaths),
                    returns=torch.stack(summed),actions=actions,modes=torch.stack(modes),live=torch.stack(live),contexts=torch.stack(contexts),
                    speed5=torch.stack(speed5),power5=torch.stack(power5),consumed=torch.stack(consumed))

    for stratum,st in enumerate(strata[:1] if smoke else strata):
        coverage=initialize(st,4202700+stratum)
        snap=EnvironmentSnapshot(e)
        replay=verify_replay(e,policy,steps=8);replays.append(replay)
        if not replay['passed']:
            write_json(output/'result.json',dict(robot=robot,passed=False,reason='complete_state_replay_failed',replay=replay));return
        initial=dict(state=copy.deepcopy(e.state),history=e.obs_hist.clone(),context=e.xi.clone(),prev=e.prev_action.clone(),frame=e.frame.clone())
        zero=torch.zeros(n,len(e.q0),device=t.device)
        real0=physical(snap,zero);real_repeat=physical(snap,zero)
        with torch.no_grad():
            pred=imagined(initial,zero,real0['actions']);closed=imagined(initial,zero)
        target=torch.zeros_like(zero,requires_grad=True)
        differentiable=imagined(initial,target)
        grads={}
        for horizon in (8,32):grads[horizon]=torch.autograd.grad(differentiable['returns'][horizon-1].sum(),target,retain_graph=horizon==8)[0].detach()
        sf=state_to_flat(initial['state'])
        for horizon in (1,8,32):
            for i in range(n):
                actual=real0['states'][horizon-1,i];pr=pred['states'][horizon-1,i];cl=closed['states'][horizon-1,i]
                all_errors.append(dict(stratum=stratum,episode=f'{stratum}:{i}',split='holdout' if i%2 else 'calibration',horizon=horizon,
                    covered=bool(coverage[i] and real0['live'][horizon-1,i]),state_rmse=float((pr-actual).square().mean().sqrt()),
                    persistence_rmse=float((sf[i]-actual).square().mean().sqrt()),closed_loop_rmse=float((cl-actual).square().mean().sqrt()),
                    joint_velocity_rmse=float((flat_to_state(pr)['qd']-flat_to_state(actual)['qd']).square().mean().sqrt()),
                    reward_error=float((pred['rewards'][horizon-1,i]-real0['rewards'][horizon-1,i]).abs()),
                    termination_match=bool(pred['deaths'][horizon-1,i]==real0['deaths'][horizon-1,i]),
                    actual_speed_5ms_peak=float(real0['speed5'][:horizon,:,i].max()),
                    actual_power_5ms_peak=float(real0['power5'][:horizon,:,i].max()),
                    predicted_endpoint_speed=float(flat_to_state(pr)['qd'].abs().max()),
                    speed_violation_missed=bool(robot=='B2' and real0['speed5'][:horizon,:,i].max()>6 and flat_to_state(pr)['qd'].abs().max()<=6)))
        for di,direction in enumerate(directions[:1] if smoke else directions):
            for magnitude in (.005,.010):
                minus=physical(snap,-magnitude*direction.expand(n,-1))
                plus=physical(snap,magnitude*direction.expand(n,-1))
                for horizon in (8,32):
                    delta=plus['returns'][horizon-1]-minus['returns'][horizon-1]
                    derivative=(grads[horizon]*direction).sum(-1)
                    noise=(real0['returns'][horizon-1]-real_repeat['returns'][horizon-1]).abs()
                    same=(minus['modes'][:horizon]==plus['modes'][:horizon]).all(0).all(-1)
                    for i in range(n):
                        identifiable=bool(delta[i].abs()>max(10*float(noise[i]),1e-5)) and bool(derivative[i].abs()>1e-7)
                        all_rows.append(dict(stratum=stratum,case=st['case'],phase=st['phase'],episode=f'{stratum}:{i}',
                            split='holdout' if i%2 else 'calibration',direction=di,magnitude=magnitude,horizon=horizon,
                            start_covered=bool(coverage[i]),same_contact_mode=bool(same[i]),identifiable=identifiable,
                            physical_difference=float(delta[i]),model_derivative=float(derivative[i]),replay_noise=float(noise[i]),
                            agreement=bool(delta[i]*derivative[i]>0)))
        snap.restore()
        np.savez_compressed(output/f'stratum_{stratum:02d}.npz',state=real0['states'].cpu().numpy(),
                    model_state=pred['states'].cpu().numpy(),closed_model_state=closed['states'].cpu().numpy(),
                    action=torch.stack(real0['actions']).cpu().numpy(),context=real0['contexts'].cpu().numpy(),
                    initial=sf.cpu().numpy(),frame=initial['frame'].cpu().numpy(),modes=real0['modes'].cpu().numpy(),live=real0['live'].cpu().numpy(),
                    actual_target_5ms=real0['consumed'].cpu().numpy(),speed_5ms=real0['speed5'].cpu().numpy(),power_5ms=real0['power5'].cpu().numpy())
        write_json(output/'progress.json',dict(robot=robot,strata_done=stratum+1,strata_total=len(strata),actor_updates=0,updated=time.time()))
        print(json.dumps(dict(robot=robot,stratum=stratum,covered=int(coverage.sum()),n=n,elapsed=time.time()-start)),flush=True)
        if time.time()-start>3600:break
    summaries=[];rng=np.random.default_rng(4202991)
    for horizon in (8,32):
        candidates=[r for r in all_rows if r['horizon']==horizon and r['split']=='holdout']
        eligible=[r for r in candidates if r['start_covered'] and r['same_contact_mode'] and r['identifiable']]
        episodes=sorted(set(r['episode'] for r in eligible));counts=[]
        for ep in episodes:
            group=[r for r in eligible if r['episode']==ep];counts.append((sum(r['agreement'] for r in group),len(group)))
        if counts:
            c=np.array(counts);boot=c[rng.integers(0,len(c),(2000,len(c)))].sum(1);interval=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()
            fraction=float(c[:,0].sum()/c[:,1].sum())
        else:fraction=None;interval=[None,None]
        strata_covered=len(set(r['stratum'] for r in eligible))
        good=bool(not smoke and strata_covered==8 and len(episodes)>=32 and fraction is not None and fraction>=.75 and interval[0]>.5)
        summaries.append(dict(horizon=horizon,all_holdout_directions=len(candidates),eligible=len(eligible),episodes=len(episodes),
            strata_covered=strata_covered,agreement=fraction,episode_bootstrap_95=interval,passed=good,
            contact_switches=sum(not r['same_contact_mode'] for r in candidates),unidentifiable=sum(not r['identifiable'] for r in candidates)))
    assert original['policy']==tensor_digest(policy.state_dict()) and original['wm']==tensor_digest(wm.state_dict())
    report=dict(robot=robot,passed=all(r['passed'] for r in summaries),smoke=smoke,actor_updates=0,
                source_sha256=t.source_hash,source_iteration=t.source_iteration,strata=strata,summaries=summaries,
                direction_rows=all_rows,model_errors=all_errors,replays=replays,elapsed=time.time()-start,
                scope='model credibility only; unchanged physical acceptance is separate',
                direction_definition='initial 20ms request target; subsequent frozen Actor feedback; original discounted soft-survival task return',
                proposed_coverage_rule='all8 strata represented, >=32 heldout episodes, agreement>=.75, episodeCI lower>.5',
                clone_state_count=n*(1 if smoke else len(replays)),max_local_perturbed_rollouts=2048,
                extra_WM_fit_used=bool(wm_checkpoint),wm_source_sha256=wm_source_sha)
    report['contact_mode_definition']='geometry support proxy: B2 sole center below .045m, G1 capsule bottom below .003m; NOT contact force validation'
    write_json(output/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['direction_rows','model_errors','replays','strata']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robot',choices=['B2','G1'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--wm-checkpoint');a=p.parse_args();diagnostic(a.robot,a.out,a.smoke,a.wm_checkpoint)
