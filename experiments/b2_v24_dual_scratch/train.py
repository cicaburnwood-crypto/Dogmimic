"""Fresh dual-task Actor; continuing WM; unmodified official actor-critic learner."""
from bootstrap import HERE
import os,time,json,hashlib,argparse,traceback,shutil,math
from pathlib import Path
import torch
from tensordict import TensorDict
import supermimic
from supermimic.learners.wm_learner import WorldModelLearner,WorldModelLearnerCfg
from supermimic.learners.policy_learner import ActorCriticPolicyLearner,ActorCriticPolicyLearnerCfg
import dual
from dual import *
from arm import ArmController

class WarmCriticLearner(ActorCriticPolicyLearner):
    """Official updates with a task-level bootstrap warm-in coefficient.
    At zero this is the official BPTT Actor objective; Critic fitting continues.
    """
    bootstrap_scale=1.
    def _bootstrap(self,obs,survival,term_prob):
        return self.bootstrap_scale*super()._bootstrap(obs,survival,term_prob)

def atomic(p,obj):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def digest(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def configure():
    os.environ.pop('B2_Q_ESTIMATOR',None)
    torch.set_num_threads(2)
    if not getattr(ArmController,'v24_configured',False):
        original=ArmController.envelope
        ArmController.envelope=lambda self,speed:(original(self,speed)[0],original(self,speed)[1]*1.25)
        ArmController.v24_configured=True
def make_env(n,randomize=True,device='cuda'):
    e=JointEnv(str(device),n,randomize=randomize,compile=False,autoreset=False)
    e.reset();e.eval_mode=True;e.cfg.obs_noise=False;e.max_ep_len=1000;e.track_stationarity=True
    e.z_min=.0;e.z_max=2.0 # Not used for global goals without workspace report below.
    bind(e.arm)
    return e
def act(actor,e,t,prev):
    x=observation(e.state,e.xi,t['goal'],t['cmd'],t['mode'],prev,t['anchor'])
    return actor(TensorDict({'actor':x},batch_size=[len(x)]))
def setup(a):
    configure();torch.manual_seed(a.seed);dev=getattr(a,'device','cuda')
    e=make_env(a.envs,device=dev)
    if a.workspace:
        w=json.loads(Path(a.workspace).read_text());e.z_min=w['z_min'];e.z_max=w['z_max']
    with torch.no_grad():
        for _ in range(50):e.upper.zero_();e.step(e.upper[:,:12])
    e.episode_origin=e.state['pos'].clone()
    t=sample_tasks(e,0);prev=torch.zeros(a.envs,18,device=dev)
    dim=observation(e.state,e.xi,t['goal'],t['cmd'],t['mode'],prev,t['anchor']).shape[-1]
    actor=Actor(dim,e.task).to(dev);critic=Critic(dim+(22 if getattr(a,'privileged_critic',False) else 0)).to(dev)
    rep=Representation(torch.cat([e.q0,e.arm.home]),e.arm,clearance_loss=getattr(a,'wm_clearance',False)).to(dev);wm=RobotDynamics(rep,e.task).to(dev)
    donor_meta = None
    if a.wm_donor:
        donor = torch.load(a.wm_donor, map_location=dev, weights_only=False)
        wm.load_state_dict(donor['wm'], strict=True)
        donor_meta = dict(path=a.wm_donor, sha256=hashlib.sha256(Path(a.wm_donor).read_bytes()).hexdigest(),
                          exact_weights_and_normalizers=True, continued_update=True, actor_loaded=False)
        del donor
    cfg=WorldModelLearnerCfg(num_batches=1,batch_size=a.wm_batch,horizon=getattr(a,'wm_horizon',8),lr=2e-5)
    # Units-based WM losses. Do not let 2mm/s root error overwhelm arm/position fitting.
    for k,v in dict(nominal_body_clearance=.003,nominal_root_linear_velocity=.05,nominal_root_angular_velocity=.08,nominal_leg_velocity=.2,
                    nominal_error_joint_pos=.02,nominal_error_root_pose_pos=.01,nominal_error_root_pose_ori=.02,
                    nominal_error_arm_pos=.01,nominal_error_arm_vel=.10,nominal_error_ee_pos=.01,
                    nominal_response_arm_pos=.0005,nominal_response_arm_vel=.01,nominal_response_ee=.0005,
                    nominal_response_root_velocity=.01,nominal_response_leg_velocity=.1).items():setattr(cfg,k,v)
    wl=WorldModelLearner(wm,rep,cfg,device=dev)
    im=ImaginedEnv(wm,rep,e)
    e.velocity_posture=getattr(a,'velocity_posture',False)
    e.finite_pair_barrier=getattr(a,'finite_pair_barrier',False)
    im.critic_privileged=getattr(a,'privileged_critic',False)
    pl=WarmCriticLearner(actor,critic,im,ActorCriticPolicyLearnerCfg(batch_size=a.batch,horizon=a.horizon,
                              lr=a.lr,gamma=.99,lam=.95,survival_masking=True),device=dev)
    # Start eager for verifiability. Official update equations/optimizers stay unchanged.
    wl._compiled_rollout=wl._rollout;wl._compiled_represent=rep.represent;wl._compiled_compute_losses=wl._compute_losses
    wl._captured_update=wl._update_from;pl._rollout_actor=actor;pl._captured_update=pl._update_from
    return e,t,prev,actor,critic,wm,rep,wl,im,pl,donor_meta

def _evaluate_batch(actor,n,seed,zmin,zmax,steps=600,full=False):
    # Save RNG: evaluation must not perturb training data/command sampling.
    dev=next(actor.parameters()).device
    with torch.random.fork_rng(devices=[dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type=='cuda' else []):
        torch.manual_seed(seed)
        names=['forward','backward','left','right','yaw','zero','hold','near','far','low','high']
        additional={'forward_max':[2,0,0],'backward_max':[-2,0,0],'left_max':[0,1,0],
                    'right_max':[0,-1,0],'diagonal_max':[3**.5,1,0],'yaw_max':[0,0,1],
                    'walking_turn':[.5,0,.5],'velocity_to_EE':[.5,0,0],'EE_to_velocity':[.5,0,0]}
        global_cases=['cylinder_mid','cylinder_low','cylinder_high','edge_low','edge_high','target_switch']
        if full:names+=list(additional)+global_cases
        count=len(names)*n;e=make_env(count,device=dev);e.z_min=zmin;e.z_max=zmax
        prev=torch.zeros(count,18,device=dev)
        with torch.no_grad():
            for _ in range(50):e.upper=prev;e.step(prev[:,:12])
            t=sample_tasks(e,0);t['mode'][:6*n]=0;t['mode'][6*n:]=1;t['cmd'].zero_()
            commands=[[.5,0,0],[-.5,0,0],[0,.3,0],[0,-.3,0],[0,0,.5],[0,0,0]]
            for i,c in enumerate(commands):t['cmd'][i*n:(i+1)*n]=torch.tensor(c,device=dev)
            if full:
                for i,name in enumerate(list(additional),11):
                    t['mode'][i*n:(i+1)*n]=0;t['cmd'][i*n:(i+1)*n]=torch.tensor(additional[name],device=dev)
                cancel=names.index('EE_to_velocity');t['mode'][cancel*n:(cancel+1)*n]=1;t['cmd'][cancel*n:(cancel+1)*n]=0
            tip=ee(e.arm,e.state)[0];t['goal']=tip.clone();t['known'].zero_()
            near=slice(7*n,8*n);st={k:v[near] for k,v in e.state.items()}
            q=st['aq'].clone();ok=torch.zeros(n,dtype=torch.bool,device=dev)
            for _ in range(64):
                proposal=(st['aq']+.65*torch.randn_like(q)).clamp(e.arm.lower+.03,e.arm.upper-.03)
                good=torch.ones_like(ok)
                for f in [0.,.25,.5,.75,1.]:good &= e.arm.safety_fast(st,st['aq']+f*(proposal-st['aq']))[0]
                d=(ee(e.arm,dict(st,aq=proposal))[0]-tip[near]).norm(dim=-1)
                good&=(d>.10)&(d<.35);q=torch.where((good&~ok)[:,None],proposal,q);ok|=good
                if ok.all():break
            if not ok.all():raise RuntimeError('Evaluation near-goal witness missing; do not replace it by an easy target')
            t['goal'][near]=ee(e.arm,dict(st,aq=q))[0];t['known'][near]=1
            t['goal'][8*n:9*n,0]=t['anchor'][8*n:9*n,0]+2.5
            t['goal'][8*n:9*n,1]=t['anchor'][8*n:9*n,1]
            for i,z in [(9,zmin+.05*(zmax-zmin)),(10,zmax-.05*(zmax-zmin))]:
                t['goal'][i*n:(i+1)*n,:2]=t['anchor'][i*n:(i+1)*n,:2]+torch.tensor([.8,0],device=dev)
                t['goal'][i*n:(i+1)*n,2]=z
            if full:
                for name in global_cases:
                    i=names.index(name);ix=slice(i*n,(i+1)*n)
                    radius=(2.5+.5*torch.rand(n,device=dev)) if name.startswith('edge') else 3*torch.sqrt(torch.rand(n,device=dev))
                    theta=2*torch.pi*torch.rand(n,device=dev)
                    fraction=torch.rand(n,device=dev)
                    if name.endswith('low'):fraction*=.15
                    elif name.endswith('high'):fraction=.85+.15*fraction
                    else:fraction=.15+.70*fraction
                    t['goal'][ix,:2]=t['anchor'][ix,:2]+radius[:,None]*torch.stack([theta.cos(),theta.sin()],-1)
                    t['goal'][ix,2]=zmin+fraction*(zmax-zmin)
                i=names.index('target_switch');ix=slice(i*n,(i+1)*n)
                switch_goal=t['goal'][ix].clone();t['goal'][ix]=tip[ix]
            initial=(tip-t['goal']).norm(dim=-1);failed=torch.zeros(count,dtype=torch.bool,device=dev)
            hold=torch.zeros(count,device=dev);ever=failed.clone();path=hold.clone();travel=torch.zeros(count,6,device=dev)
            verror=hold.clone();yerror=hold.clone();peaks=hold.clone();power=hold.clone();smooth=hold.clone()
            near_ground_speed=hold.clone();near_ground_samples=hold.clone();qvel_history=[]
            peak_arm=hold.clone();root_height_min=e.state['pos'][:,2].clone();root_height_max=root_height_min.clone()
            feet,_=e.task.foot_state(e.task.body_kin(e.state));air=feet[...,2]>.052
            liftoff=torch.zeros(count,4,device=dev);foot_peak=torch.zeros_like(liftoff)
            mean_v=torch.zeros(count,3,device=dev);mean_w=hold.clone()
            for step in range(steps):
                if full and step==steps//2:
                    i=names.index('velocity_to_EE');ix=slice(i*n,(i+1)*n)
                    t['mode'][ix]=1;t['cmd'][ix]=0;t['goal'][ix]=ee(e.arm,e.state)[0][ix]
                    i=names.index('EE_to_velocity');ix=slice(i*n,(i+1)*n)
                    t['mode'][ix]=0;t['cmd'][ix]=torch.tensor([.5,0,0],device=dev)
                    i=names.index('target_switch');ix=slice(i*n,(i+1)*n)
                    t['goal'][ix]=switch_goal;hold[ix]=0;ever[ix]=False
                    initial[ix]=(ee(e.arm,e.state)[0][ix]-switch_goal).norm(dim=-1)
                old={k:v.clone() for k,v in e.state.items()};u=act(actor,e,t,prev);e.upper=u;s,info=e.step(u[:,:12])
                failed|=info['terminated']|e.last_step_invalid
                d=(ee(e.arm,s)[0]-t['goal']).norm(dim=-1)
                substep_error=(info['tip_substeps']-t['goal'][:,None]).norm(dim=-1).amax(-1)
                hit=(substep_error<=POSITION_TOLERANCE)&still_mask(info['motion'])&~failed
                hold=torch.where(hit,hold+DT,0.);ever|=hold>=1-1e-6
                path+=(s['pos']-old['pos']).norm(dim=-1);travel+=(s['aq']-old['aq']).abs()
                if step>=steps//2:
                    vv=heading_velocity(s['rot'],s['v']);verror+=(vv[:,:2]-t['cmd'][:,:2]).square().sum(-1)
                    yerror+=(s['w'][:,2]-t['cmd'][:,2]).square()
                    mean_v+=vv;mean_w+=s['w'][:,2]
                smooth+=(u[:,:12]-prev[:,:12]).square().mean(-1)
                feet,fv=e.task.foot_state(e.task.body_kin(s));near_ground=(feet[...,2]<=.035).float()
                next_air=torch.where(air,feet[...,2]>.037,feet[...,2]>.052)
                liftoff+=(next_air&~air&~failed[:,None]).float();air=next_air
                foot_peak=torch.maximum(foot_peak,torch.where(~failed[:,None],(feet[...,2]-.032).clamp_min(0),0.))
                near_ground_speed+=(near_ground*fv[...,:2].square().sum(-1)).sum(-1);near_ground_samples+=near_ground.sum(-1)
                qvel_history.append(s['qd'].clone());peak_arm=torch.maximum(peak_arm,e.arm_peak_speed)
                root_height_min=torch.minimum(root_height_min,s['pos'][:,2]);root_height_max=torch.maximum(root_height_max,s['pos'][:,2])
                peaks=torch.maximum(peaks,info['peak_speed']);power=torch.maximum(power,info['peak_power']);prev=u
            spectrum=torch.fft.rfft(torch.stack(qvel_history),dim=0)
            freq=torch.fft.rfftfreq(steps,d=DT,device=dev);weight=torch.full_like(freq,2.)
            weight[0]=1.
            if steps%2==0:weight[-1]=1.
            hf=((spectrum.abs().square()*weight[:,None,None])[(freq>=8)&(freq<=25)].sum(0).mean(-1)/steps**2).sqrt()
            rows=[]
            for i,name in enumerate(names):
                ix=slice(i*n,(i+1)*n);alive=~failed[ix]
                ve=(verror[ix]/(steps-steps//2)).sqrt();ye=(yerror[ix]/(steps-steps//2)).sqrt()
                speed=t['cmd'][ix,:2].norm(dim=-1);yaw=t['cmd'][ix,2].abs()
                tracked=alive&(ve<=torch.maximum(.1+speed*0,speed*.15))&(ye<=torch.maximum(.1+yaw*0,yaw*.15))
                rows.append(dict(case=name,n=n,completed=int(alive.sum()),velocity_pass=int(tracked.sum()),
                    final_mode='EE' if bool(t['mode'][i*n]) else 'velocity',
                    velocity_rmse_m_s=float(ve.mean()),yaw_rmse_rad_s=float(ye.mean()),
                    mean_heading_velocity_m_s=(mean_v[ix]/(steps-steps//2)).mean(0).tolist(),
                    mean_yaw_rate_rad_s=float((mean_w[ix]/(steps-steps//2)).mean()),
                    liftoffs_per_foot=liftoff[ix].mean(0).tolist(),max_foot_clearance_m=float(foot_peak[ix].max()),
                    safe_end_10cm=int(((d[ix]<=.10)&alive).sum()),safe_end_5cm=int(((d[ix]<=.05)&alive).sum()),
                    strict_hold_end=int(((hold[ix]>=1-1e-6)&alive).sum()),ever_held=int((ever[ix]&alive).sum()),
                    initial_error_m=float(initial[ix].mean()),final_error_m=float(d[ix].mean()),
                    body_path_m=float(path[ix].mean()),arm_travel_deg=float(torch.rad2deg(travel[ix]).mean()),
                    peak_joint_speed=float(peaks[ix].max()),peak_power=float(power[ix].max()),action_change=float(smooth[ix].mean()/steps),
                    near_ground_foot_speed_rms_m_s=float((near_ground_speed[ix]/near_ground_samples[ix].clamp_min(1)).sqrt().mean()),
                    high_frequency_joint_velocity_rms_rad_s=float(hf[ix].mean()),high_frequency_band_hz=[8,25],
                    peak_arm_speed_rad_s=float(peak_arm[ix].max()),root_height_min_m=float(root_height_min[ix].min()),root_height_max_m=float(root_height_max[ix].max()),
                    goal_xyz=t['goal'][ix].tolist(),final_tip_xyz=ee(e.arm,s)[0][ix].tolist()))
    return rows

def evaluate(actor,n,seed,zmin,zmax,steps=600,full=False):
    """Bounded evaluation memory, with all requested trials kept in denominators."""
    import gc
    rows=None;done=0;chunk=0
    counts={'n','completed','velocity_pass','safe_end_10cm','safe_end_5cm','strict_hold_end','ever_held'}
    maximum={'peak_joint_speed','peak_power','peak_arm_speed_rad_s','root_height_max_m','max_foot_clearance_m'}
    minimum={'root_height_min_m'}
    concatenate={'goal_xyz','final_tip_xyz','sample_seeds'}
    while done<n:
        take=min(8,n-done);sample_seed=seed+chunk*1009
        part=_evaluate_batch(actor,take,sample_seed,zmin,zmax,steps,full)
        for r in part:r['sample_seeds']=[sample_seed]
        if rows is None:rows=part
        else:
            for total,now in zip(rows,part):
                assert total['case']==now['case'] and total['final_mode']==now['final_mode']
                old_n=total['n']
                for k,v in now.items():
                    if k in counts:total[k]+=v
                    elif k in maximum:total[k]=max(total[k],v)
                    elif k in minimum:total[k]=min(total[k],v)
                    elif k in concatenate:total[k]+=v
                    elif isinstance(v,(int,float)):total[k]=(total[k]*old_n+v*take)/(old_n+take)
                    elif k in ['mean_heading_velocity_m_s','liftoffs_per_foot']:
                        total[k]=[(a*old_n+b*take)/(old_n+take) for a,b in zip(total[k],v)]
        done+=take;chunk+=1
        # Evaluation environments contain self-referential compiled wrappers;
        # release their graph buffers before allocating the next batch.
        dual.bind(None);gc.collect()
        if next(actor.parameters()).is_cuda:torch.cuda.empty_cache()
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--wm-donor');p.add_argument('--out',required=True)
    p.add_argument('--workspace');p.add_argument('--envs',type=int,default=512);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--wm-batch',type=int,default=256);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--wm-horizon',type=int,default=8)
    p.add_argument('--wm-rehearsal-fraction',type=float,default=0.,help='Persistent old failure fraction in every WM batch, independent of fresh failures')
    p.add_argument('--wm-rehearsal',help='Factual calibration episodes; ONLY first3training blocks, never heldout block')
    p.add_argument('--wm-clearance',action='store_true',help='Supervised exact body-ground clearance from factual states, 3mm nominal scale')
    p.add_argument('--failure-sequences',action='store_true',help='Fit factual64-step first-failure precursors instead of only terminal single steps')
    p.add_argument('--finite-pair-barrier',action='store_true',help='Finite positive pair reserve; physical collision criteria unchanged')
    p.add_argument('--velocity-posture',action='store_true',help='Soft height/upright/pose preference only in velocity mode; no physical bounds changed')
    p.add_argument('--reset-critic',action='store_true',help='Refit value for a changed task reward; preserve Actor/WM')
    p.add_argument('--updates',type=int,default=10000);p.add_argument('--warmup',type=int,default=40)
    p.add_argument('--collect-steps',type=int,default=16);p.add_argument('--replay-steps',type=int,default=512)
    p.add_argument('--seed',type=int,default=171724);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--max-hours',type=float,default=12);p.add_argument('--eval-minutes',type=float,default=20)
    p.add_argument('--eval-n',type=int,default=8);p.add_argument('--eval-steps',type=int,default=600)
    p.add_argument('--final-n',type=int,default=64)
    p.add_argument('--critic-after-velocity',action='store_true',help='Fit critic during stage 0 but warm its Actor gradient in only after physical velocity qualification')
    p.add_argument('--ee-warm-scale',type=float,default=1.,help='EE objective scale in velocity-first stage; linearly restored over 30min of stage1')
    p.add_argument('--deadline-hk',help='Absolute campaign deadline, including pilots and final evaluations')
    p.add_argument('--privileged-critic',action='store_true',help='Expose reward witness, DR dynamics and task weight to training Critic only')
    p.add_argument('--velocity-curriculum',action='store_true',help='Qualify fore/aft, then lateral, then combined yaw before stage B')
    p.add_argument('--exploration-mixture',action='store_true',help='CPU-validated leg excitation strata .025/.06/.08 with 0.2s OU correlation; no gait targets')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume');a=p.parse_args();dev=a.device
    assert a.smoke or a.workspace,'A measured workspace report is required for a training run'
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source=out/'source_snapshot';source.mkdir(exist_ok=True)
    for f in HERE.iterdir():
        if f.is_file() and f.suffix in ['.py','.json','.txt']:shutil.copy2(f,source/f.name)
    atomic(out/'source_hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()})
    e,t,prev,actor,critic,wm,rep,wl,im,pl,donor=setup(a)
    assert 0<a.ee_warm_scale<=1
    e.ee_task_scale=a.ee_warm_scale
    e.velocity_level=0 if a.velocity_curriculum else 2
    origin=dict(actor=digest(actor),critic=digest(critic),wm=digest(wm));it_start=0;stage=0
    if a.resume:
        checkpoint=torch.load(a.resume,map_location=dev,weights_only=False)
        assert checkpoint['task']=='V24_dual_direct' and checkpoint['fresh_origin']
        actor.load_state_dict(checkpoint['actor']);wm.load_state_dict(checkpoint['wm'])
        critic_state=checkpoint['critic'];critic_migrated=False
        if critic_state['net.0.weight'].shape!=critic.net[0].weight.shape:
            assert a.privileged_critic and critic.net[0].weight.shape[1]==critic_state['net.0.weight'].shape[1]+22
            old=critic_state['net.0.weight'];critic_state['net.0.weight']=torch.cat([old,old.new_zeros(old.shape[0],22)],1)
            critic_migrated=True
        if not a.reset_critic:critic.load_state_dict(critic_state)
        for opts,key in [(pl.actor_optimizers,'actor_opt'),(pl.critic_optimizers,'critic_opt'),(wl.optimizers,'wm_opt')]:
            if (critic_migrated or a.reset_critic) and key=='critic_opt':continue
            for opt,sd in zip(opts,checkpoint[key]):opt.load_state_dict(sd)
        origin=checkpoint['fresh_origin'];it_start=checkpoint['iteration'];stage=checkpoint['stage']
        e.velocity_level=checkpoint.get('velocity_level',e.velocity_level)
    if a.velocity_curriculum:t=sample_tasks(e,stage)
    loaded_at_start=dict(actor=digest(actor),critic=digest(critic),wm=digest(wm))
    atomic(out/'loaded_weights.json',dict(**loaded_at_start,resume=a.resume,
        critic_input_migrated=bool(a.resume and critic_migrated)))
    atomic(out/'config.json',dict(**vars(a),donor=donor,actor_initialization='fresh_random' if not a.resume else 'same_fresh_run_resume',
        official_module=supermimic.__file__,learner=type(pl).__name__,GT=True,orientation_goal=False,
        no_pretrained_locomotion=True,physical_body_limits=True,position_tolerance_m=.015,hold_s=1,still_limits=STILL_LIMITS,
        cylinder=dict(radius_m=3,z_min=e.z_min,z_max=e.z_max),reward_version='dual_v1'))
    atomic(out/'fresh_initialization.json',dict(**origin,donor=donor,actor_loaded_from_donor=False,critic_fresh=not bool(a.resume)))
    rehearsal=None
    if a.wm_rehearsal:
        episodes=torch.load(a.wm_rehearsal,map_location=dev,weights_only=False)[:3];states=[];actions=[]
        for ds in episodes:
            for j in range(len(ds['states'])):
                hits=ds['done'][j].nonzero().flatten()
                if not len(hits):continue
                end=int(hits[0])+1
                if end<64 or ds['invalid'][j,:end].any():continue
                states.append(ds['states'][j,end-64:end+1]);u=ds['actions'][j,end-64:end];actions.append(torch.cat([u,u[-1:]],0))
        if not states:raise RuntimeError('WM rehearsal has no valid training-only failure sequences')
        ss=torch.stack(states);aa=torch.stack(actions)
        rehearsal=TensorDict(dict(state=ss,action=aa,sim_ctx=ss.new_zeros(len(ss),65,0)),batch_size=[len(ss),65])
        atomic(out/'wm_rehearsal.json',dict(source=a.wm_rehearsal,training_blocks=[0,1,2],heldout_excluded=True,n=len(ss),horizon=64))
        del episodes,states,actions
    assert 0<=a.wm_rehearsal_fraction<1
    if a.wm_rehearsal_fraction and (rehearsal is None or a.wm_horizon>64):raise ValueError('Persistent rehearsal requires factual archive and horizon<=64')
    def rehearsal_sample(n,h):
        ix=torch.randint(len(rehearsal),(n,),device=dev);start=torch.randint(65-h,(n,),device=dev)
        times=start[:,None]+torch.arange(h+1,device=dev)[None,:]
        return rehearsal[ix[:,None],times]
    replay=Replay(a.replay_steps,a.envs,dev);started=time.time();eval_seconds=0.;last_eval=started
    periodic_eval_cost=120. if dev=='cpu' else 60.
    if a.deadline_hk:
        from datetime import datetime
        campaign_deadline=datetime.fromisoformat(a.deadline_hk).timestamp()
    else:campaign_deadline=float('inf')
    stage_started=started;stage_pass=0;velocity_pass=0;best={};last_it=it_start;actor_updates=0
    task_left=torch.randint(300,501,(a.envs,),device=dev);task_switches=0
    exploration=torch.zeros(a.envs,18,device=dev);noise_scale=torch.full_like(exploration,.025)
    noise_scale[a.envs//2:3*a.envs//4,:12]=.06;noise_scale[3*a.envs//4:,:12]=.08
    feet,_=e.task.foot_state(e.task.body_kin(e.state));foot_air=feet[...,2]>.052
    log=(out/'log.jsonl').open('a');timings=[]
    def save(name,it):
        data=dict(task='V24_dual_direct',expanded_wm_contract=True,iteration=it,stage=stage,velocity_level=e.velocity_level,fresh_origin=origin,
            actor=actor.state_dict(),critic=critic.state_dict(),wm=wm.state_dict(),
            actor_opt=[o.state_dict() for o in pl.actor_optimizers],critic_opt=[o.state_dict() for o in pl.critic_optimizers],
            wm_opt=[o.state_dict() for o in wl.optimizers],config=vars(a),donor=donor)
        temp=out/(name+'.tmp');torch.save(data,temp);temp.replace(out/name)
    def ev(it,kind='periodic'):
        nonlocal eval_seconds,last_eval,stage_pass,stage,stage_started,velocity_pass,periodic_eval_cost
        start=time.time();rows=evaluate(actor,a.eval_n,17172411,e.z_min,e.z_max,a.eval_steps);bind(e.arm)
        duration=time.time()-start
        eval_seconds+=duration;last_eval=time.time();periodic_eval_cost=max(periodic_eval_cost,duration)
        report=dict(iteration=it,kind=kind,stage=stage,seed=17172411,rows=rows,accepted=False,evaluation_seconds=time.time()-start)
        atomic(out/f'eval_{it:06d}.json',report);atomic(out/'latest_eval.json',report)
        vel=sum(r['velocity_pass'] for r in rows[:6])/(6*a.eval_n)
        near=rows[7];hold=rows[6]
        metrics=dict(velocity=(min(r['completed']/r['n'] for r in rows[:6]),vel,
                              min(r['velocity_pass']/r['n'] for r in rows[:6]),-sum(r['velocity_rmse_m_s'] for r in rows[:6])),
                     arrival=(min(r['safe_end_10cm']/r['n'] for r in rows[8:]),sum(r['safe_end_10cm'] for r in rows[8:]),
                              min(r['completed']/r['n'] for r in rows[8:]),sum(r['completed'] for r in rows[8:]),
                              -sum(r['final_error_m'] for r in rows[8:])),
                     strict=(min(r['strict_hold_end']/r['n'] for r in rows[6:]),sum(r['strict_hold_end'] for r in rows[6:])))
        for key,rank in metrics.items():
            if key not in best or rank>best[key]:
                best[key]=rank;save('best_'+key+'.pt',it)
                sha=hashlib.sha256((out/f'best_{key}.pt').read_bytes()).hexdigest()
                atomic(out/f'best_{key}_eval.json',dict(**report,best_sha256=sha,selection_rank=rank))
        if a.velocity_curriculum and stage==0 and e.velocity_level<2:
            subset=rows[:2] if e.velocity_level==0 else rows[:4]
            passed=min(r['completed']/r['n'] for r in subset)>=.95 and min(r['velocity_pass']/r['n'] for r in subset)>=.75
            velocity_pass=velocity_pass+1 if passed else 0
            if velocity_pass>=2:
                e.velocity_level+=1;velocity_pass=0
                atomic(out/f'velocity_level_{e.velocity_level}.json',dict(iteration=it,level=e.velocity_level,report=report))
        qualified=vel>=.90 and min(r['completed']/r['n'] for r in rows[:6])>=.95
        if stage>=1:qualified &= near['safe_end_5cm']/a.eval_n>=.75 and near['arm_travel_deg']>1
        if stage>=2:qualified &= min(r['safe_end_10cm']/a.eval_n for r in rows[8:])>=.75
        stage_pass=stage_pass+1 if qualified else 0
        if stage<3 and stage_pass>=2 and time.time()-stage_started>=1800:
            stage+=1;stage_pass=0;stage_started=time.time();atomic(out/'stage_transition.json',dict(iteration=it,new_stage=stage,report=report))
        print(json.dumps(dict(event='evaluation',**report)),flush=True)
    save('initial.pt',it_start)
    if not a.smoke:ev(it_start,'initial')
    stop='update_budget'
    for it in range(it_start+1,it_start+a.updates+1):
        if (out/'STOP').exists():stop='requested_stop';break
        # Reserve using measured evaluation throughput, including both full
        # seeds and the final periodic matrix. CPU fallback cannot assume the
        # twenty-minute GPU estimate. Never reduce sample counts to fit time.
        full_ratio=2*(a.final_n/max(1,a.eval_n))*(26/11) if a.final_n else 0
        reserve=0 if a.smoke else 60+1.5*periodic_eval_cost*(1+full_ratio)
        if time.time()>=campaign_deadline-reserve:stop='campaign_walltime_budget';break
        if time.time()-started>=a.max_hours*3600-reserve:stop='walltime_budget';break
        tick=time.time();stats={}
        sampled_liftoffs=0;sampled_failures=0
        with torch.no_grad():
            excitation=.025*torch.randn(a.envs,18,device=dev)
            for sub in range(a.collect_steps):
                s={k:v.clone() for k,v in e.state.items()};xi=e.xi.clone();full=pack(s,xi);episode=e.ep_id.clone()
                if a.exploration_mixture:
                    rho=math.exp(-DT/.2);exploration=rho*exploration+(1-rho*rho)**.5*noise_scale*torch.randn_like(prev)
                    u=act(actor,e,t,prev)+exploration
                else:u=act(actor,e,t,prev)+excitation+.003*torch.randn_like(prev)
                u[:,:12]=torch.maximum(torch.minimum(u[:,:12],actor.positive),-actor.negative);u[:,12:].clamp_(-1,1)
                e.upper=u;sn,info=e.step(u[:,:12]);next_full=pack(sn,e.xi.clone())
                feet,_=e.task.foot_state(e.task.body_kin(sn));height=feet[...,2]
                next_air=torch.where(foot_air,height>.037,height>.052)
                support=(sn['rot'][:,2,2]>.866)&((height<=.037).sum(-1)>=2)&~info['done']
                sampled_liftoffs+=((next_air&~foot_air)&support[:,None]).sum();foot_air=next_air
                sampled_failures+=info['terminated'].sum()
                replay.add(state=full,next_state=next_full,action=wm_action(u),prev=prev,episode=episode,
                           tick=torch.full_like(episode,replay.count),valid=~e.last_step_invalid,
                           done=info['done'],terminated=info['terminated'],**t)
                prev=u.clone();ids=info['done'].nonzero().flatten()
                if len(ids):
                    e.reset_envs(ids);e._finish_resets(ids);prev[ids]=0
                    exploration[ids]=0;foot_air[ids]=False
                    e.episode_origin[ids]=e.state['pos'][ids]
                    for k,v in sample_tasks(e,stage,ids).items():t[k][ids]=v
                    task_left[ids]=torch.randint(300,501,(len(ids),),device=dev)
                # Switch task only, preserving physical/filter state and the episode's cylinder center.
                task_left-=1;switch=(task_left<=0).nonzero().flatten()
                if len(switch):
                    for k,v in sample_tasks(e,stage,switch).items():t[k][switch]=v
                    task_left[switch]=torch.randint(300,501,(len(switch),),device=dev);task_switches+=len(switch)
        stats['collection_supported_liftoffs']=sampled_liftoffs
        stats['collection_physical_failures']=sampled_failures
        td=replay.wm_batch(a.wm_batch,a.wm_horizon)
        if td is not None:
            wl.update_normalization(td['state'],td['action'])
            if a.wm_rehearsal_fraction:
                n=round(a.wm_batch*a.wm_rehearsal_fraction)
                td=torch.cat([td[:a.wm_batch-n],rehearsal_sample(n,a.wm_horizon)],0)
                stats['wm_rehearsal_count']=torch.tensor(float(n),device=dev)
            stats.update(wl.update([td]))
        if it%4==0:
            terminal=replay.wm_batch(min(a.wm_batch,64),64,precursor=True) if a.failure_sequences else replay.wm_batch(min(a.wm_batch,64),1,terminal=True)
            if terminal is None and a.failure_sequences and rehearsal is not None:terminal=rehearsal
            if terminal is None and a.failure_sequences:terminal=replay.wm_batch(min(a.wm_batch,64),1,terminal=True)
            if terminal is not None:
                st=wl.update([terminal]);stats.update({'terminal/'+k:v for k,v in st.items()})
        if it-it_start>=a.warmup:
            e.ee_task_scale=(a.ee_warm_scale if stage==0 else a.ee_warm_scale+(1-a.ee_warm_scale)*min(1.,(time.time()-stage_started)/1800)) if stage<2 else 1.
            pl.bootstrap_scale=(0. if stage==0 else min(1.,(time.time()-stage_started)/3600)) if a.critic_after_velocity else 1.
            starts = replay.sample_starts(a.batch, [.2, .5, .6, .6][stage])
            stats.update(pl.update([starts]))
            actor_updates += 1
            stats['critic_bootstrap_scale']=torch.tensor(pl.bootstrap_scale,device=dev)
            stats['ee_task_scale']=torch.tensor(e.ee_task_scale,device=dev)
            stats['body_head_gradient']=actor.net[-1].weight.grad[:12].norm()
            stats['arm_head_gradient']=actor.net[-1].weight.grad[12:].norm()
        if not all(torch.isfinite(v).all() for v in stats.values()):raise FloatingPointError('Nonfinite learner statistics')
        last_it=it;timings.append(time.time()-tick);timings=timings[-50:]
        row=dict(iteration=it,actor_updates=actor_updates,stage=stage,phase='WM_calibration' if it-it_start<a.warmup else 'joint_actor_critic',
            elapsed_seconds=time.time()-started,training_seconds=time.time()-started-eval_seconds,evaluation_seconds=eval_seconds,
            iteration_seconds=sum(timings)/len(timings),physical_transitions=replay.count*a.envs,task_switches=task_switches,updated_unix=time.time(),
            velocity_level=e.velocity_level,
            **{k:float(v) for k,v in stats.items()})
        if it%5==0 or it==1 or a.smoke:
            atomic(out/'status.json',row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        if it%50==0:save('last.pt',it)
        if not a.smoke and time.time()-last_eval>=a.eval_minutes*60:
            save(f'ckpt_{it:06d}.pt',it);ev(it)
    save('last.pt',last_it)
    if not a.smoke:ev(last_it,'final')
    accepted=False
    if not a.smoke and a.final_n:
        finals=[]
        for seed in [17172421,17172437]:
            start=time.time();rows=evaluate(actor,a.final_n,seed,e.z_min,e.z_max,a.eval_steps,full=True);bind(e.arm)
            eval_seconds+=time.time()-start
            passed=all(r['completed']>=.95*r['n'] and (r['velocity_pass'] if r['final_mode']=='velocity' else r['strict_hold_end'])>=.95*r['n'] for r in rows)
            report=dict(iteration=last_it,seed=seed,rows=rows,all_tasks_passed=passed,accepted=False)
            atomic(out/f'final_independent_{seed}.json',report);finals.append(passed)
        accepted=all(finals)
        atomic(out/'final_validation.json',dict(two_seeds_passed=accepted,accepted=accepted,
            validation_scope='Specified finite simulator test matrix; not proof for every continuous cylinder point',
            real_robot_deployed=False))
    atomic(out/'DONE.json',dict(iteration=last_it,actor_updates=actor_updates,stage=stage,stop_reason=stop,
        actor_changed=digest(actor)!=origin['actor'],critic_changed=digest(critic)!=origin['critic'],wm_changed=digest(wm)!=origin['wm'],
        changed_since_launch={k:digest(m)!=loaded_at_start[k] for k,m in [('actor',actor),('critic',critic),('wm',wm)]},
        elapsed_seconds=time.time()-started,evaluation_seconds=eval_seconds,accepted=accepted,smoke=a.smoke))

if __name__=='__main__':
    try:main()
    except BaseException:
        import sys
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
            atomic(out/'ERROR.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
