"""Finite implementation checks before launching; not a trained-policy pass."""
import argparse,json,math,time,ast,importlib.util
from pathlib import Path
import numpy as np
import torch
import mujoco
from bootstrap import HERE
from common import write_json,sha256,tensor_hash
from contract import *
from task import TiltLocomotionTask,TiltLocomotionCfg
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import quat_from_euler_xyz,quat_to_mat,mv
from sampling import sample_targets,positive_huber

def repair_checks():
    pinned=json.loads((HERE/'unchanged_contract.json').read_text())
    assert all(sha256(HERE/name)==h for name,h in pinned.items())
    previous=HERE.parent/'b2_tilt_locomotion'
    def methods(path):
        cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='TiltLocomotionEnv')
        return {x.name:ast.dump(x,include_attributes=False) for x in cls.body if isinstance(x,ast.FunctionDef)}
    old,new=methods(previous/'task.py'),methods(HERE/'task.py')
    physical_names=['__init__','step','reset_envs','_randomize','_finish_resets','_noise','set_target']
    assert all(old[name]==new[name] for name in physical_names)
    torch.manual_seed(79219)
    targets,labels=sample_targets(100000,'cpu',return_labels=True)
    planar=targets[:,:2].norm(dim=-1)
    assert planar.max()<=4.000001 and targets[:,1].abs().max()<=1.
    assert targets[:,3:5].abs().max()<=MAX_ANGLE+1e-7
    assert (targets[:,2].abs()<=yaw_sampling_bound(planar)+1e-6).all()
    counts=[int((labels==j).sum()) for j in range(3)]
    assert 29000<counts[0]<31000 and 29000<counts[1]<31000 and 39000<counts[2]<41000
    high=(planar>=2.9999)&(targets[:,3:5].abs().amin(-1)>=math.radians(14.99))
    corners=(planar>3.999)&(targets[:,3:5].abs().amin(-1)>math.radians(19.99))
    assert high.sum()>30000 and corners.sum()>6000
    assert (corners&(targets[:,0]<0)).sum()>2500 and (corners&(targets[:,0]>0)).sum()>2500
    # Each sign combination is explicitly exercised without importing eval cases.
    quadrants={}
    for r in (-1,1):
      for p in (-1,1):
        n=int((high&(targets[:,3]*r>0)&(targets[:,4]*p>0)).sum())
        assert n>6000;quadrants[f'{r},{p}']=n
    m=mujoco.MjModel.from_xml_path(str(HERE/'b2_tilt_locomotion.xml'))
    meta=json.loads((HERE/'model.json').read_text());kin=FastTreeKinematics(m,'cpu')
    feet=[m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    kwargs=dict(height=meta['standing_height_m'],foot_offsets=m.geom_pos[feet])
    args=(kin,torch.tensor(Q0),torch.tensor(m.jnt_range[1:],dtype=torch.float32),torch.tensor(ACTION_SCALE),TiltLocomotionCfg(),'cpu')
    task=TiltLocomotionTask(*args,**kwargs)
    spec=importlib.util.spec_from_file_location('old_tilt_task',previous/'task.py');module=importlib.util.module_from_spec(spec)
    __import__('sys').modules[spec.name]=module;spec.loader.exec_module(module)
    old_task=module.TiltLocomotionTask(*args,**kwargs)
    v=torch.tensor([[0.,0.,0.],[.2,0.,0.],[1.,0.,0.],[4.,0.,0.]],requires_grad=True)
    s=dict(pos=torch.tensor([0.,0.,meta['standing_height_m']]).expand(4,3),rot=torch.eye(3).expand(4,3,3),q=torch.tensor(Q0).expand(4,12),v=v,w=torch.zeros(4,3),qd=torch.zeros(4,12))
    xi=torch.zeros(4,32);xi[:,0]=1;xi[:,16:19]=1;ctx=torch.zeros(4,CONTEXT_DIM);a=torch.zeros(4,12)
    new_an=task.command_analytics(s,ctx,a,a,xi);old_an=old_task.command_analytics(s,ctx,a,a,xi)
    for key in ['hard','soft','obs','feats','goal']:assert torch.equal(new_an[key],old_an[key]),key
    slip=new_an['terms']['contact_slip'];grad=torch.autograd.grad(slip.sum(),v,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and (grad[1:,0]<0).all() and grad[-1,0].abs()>.1
    assert (slip[1:]<old_an['terms']['contact_slip'][1:]).all()
    for axis,speeds,key in [(0,[3.95,4.05],'planar_overspeed'),(1,[.975,1.025],'lateral_overspeed')]:
        vec=torch.zeros(4,3);vec[:,axis]=torch.tensor([*speeds,*[-x for x in speeds]]);vec.requires_grad_();s['v']=vec
        term=task.command_analytics(s,ctx,a,a,xi)['terms'][key]
        g=torch.autograd.grad(term.sum(),vec)[0]
        assert (g[:,axis]*vec[:,axis]<0).all() and torch.isfinite(g).all()
    # Paired reversal uses a longer hold, while evaluation mode never samples.
    return dict(contract_files_unchanged=True,physics_methods_unchanged=physical_names,
        sampler_counts=counts,high_speed_high_tilt=int(high.sum()),exact_4mps_both20=int(corners.sum()),
        tilt_sign_quadrants=quadrants,non_saturating_slip_gradient=grad[:,0].tolist(),
        unchanged_observation_termination_and_features=True,speed_margin_gradient=True)

def analytic_checks():
    torch.manual_seed(73129)
    values=torch.randn(10000,5)*10;limited=constrain_targets(values)
    assert limited[:,:2].norm(dim=-1).max()<=4.+1e-6
    assert limited[:,0].abs().max()<=4. and limited[:,1].abs().max()<=1.
    assert limited[:,3:5].abs().max()<=MAX_ANGLE+1e-7
    assert torch.equal(values[:,2],limited[:,2])
    assert command(4,1,100,20,-20)[2]==100
    ctx=torch.zeros(4,CONTEXT_DIM);ctx[:,12]=.25
    ctx[:,7:12]=torch.tensor([command(4,1,2,20,-20)])
    sequence=[]
    for i in range(450):ctx=advance_command(ctx);sequence.append(ctx.clone())
    assert torch.allclose(ctx[:,:3],ctx[:,7:10],atol=2e-5)
    assert torch.allclose(ctx[:,3:5],ctx[:,10:12],atol=2e-5)
    assert policy_goal(ctx).shape==(4,16)
    seq=torch.stack(sequence)
    assert seq[:,:,:2].norm(dim=-1).max()<=4.+1e-6
    assert ((seq[1:,:,:2]-seq[:-1,:,:2])/DT).norm(dim=-1).max()<LINEAR_ACCELERATION+1e-4
    m=mujoco.MjModel.from_xml_path(str(HERE/'b2_tilt_locomotion.xml'));meta=json.loads((HERE/'model.json').read_text())
    assert np.allclose(m.opt.gravity,[0,0,-9.81]);assert abs(m.opt.timestep-.005)<1e-8
    d=mujoco.MjData(m);d.qpos[:]=[0,0,1.,1,0,0,0]+Q0;d.ctrl[:]=np.array(Q0)[POLICY_TO_SDK]
    mujoco.mj_step(m,d);assert d.qvel[2]<-.04
    kin=FastTreeKinematics(m,'cpu');feet=[m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    task=TiltLocomotionTask(kin,torch.tensor(Q0),torch.tensor(m.jnt_range[1:],dtype=torch.float32),torch.tensor(ACTION_SCALE),TiltLocomotionCfg(),'cpu',height=meta['standing_height_m'],foot_offsets=m.geom_pos[feet])
    r=torch.tensor([.2,-.2,.25,-.25],requires_grad=True);p=torch.tensor([-.2,.2,.25,-.25],requires_grad=True)
    yaw=torch.tensor([.4,1.,-2.,2.5]);quat=quat_from_euler_xyz(r,p,yaw);R=quat_to_mat(quat)
    assert torch.allclose(angles(R),torch.stack([r,p],-1),atol=1e-6)
    velocity=torch.tensor([[.1,-.1,0.]]).expand(4,3).clone().requires_grad_()
    H=quat_to_mat(quat_from_euler_xyz(yaw*0,yaw*0,yaw));world_v=mv(H,velocity)
    assert torch.allclose(heading_velocity(R,world_v),velocity,atol=1e-6)
    command_context=torch.zeros(4,CONTEXT_DIM);command_context[:,3]=r.detach();command_context[:,4]=p.detach()
    command_context[:,2]=1.4;command_context[:,5:7]=torch.tensor([.15,-.2])
    world_w=mv(R,target_body_rate(command_context))
    assert torch.allclose(euler_rates(R,world_w),torch.tensor([.15,-.2,1.4]).expand(4,3),atol=1e-6)
    s=dict(pos=torch.tensor([0.,0.,meta['standing_height_m']]).expand(4,3),rot=R,q=torch.tensor(Q0).expand(4,12),v=world_v,w=world_w,qd=torch.zeros(4,12))
    xi=torch.zeros(4,32);xi[:,0]=1;xi[:,16:19]=1;actions=torch.zeros(4,12)
    neutral=torch.zeros(4,CONTEXT_DIM);terms=task.command_analytics(s,neutral,actions,actions,xi)['terms']
    gr,gp=torch.autograd.grad((terms['attitude']+terms['attitude_error']).sum(),(r,p),retain_graph=True)
    assert (gr*r<0).all() and (gp*p<0).all()
    gv=torch.autograd.grad((terms['linear_tracking']+terms['linear_error']).sum(),velocity)[0]
    assert (gv[:,:2]*velocity[:,:2]<0).all()
    fk,_=task.foot_state(task.body_kin(s));errors=[]
    for i in range(4):
        d.qpos[:]=np.r_[s['pos'][i].numpy(),quat[i].detach().numpy(),s['q'][i].numpy()];mujoco.mj_forward(m,d)
        errors.append(float(np.abs(fk[i].detach().numpy()-d.geom_xpos[feet]).max()))
    assert max(errors)<2e-6
    return dict(command_limits=True,roll_pitch_simultaneous=True,unclipped_yaw_rate_interface=True,
        yaw_euler_rate_verified=True,heading_velocity_tilt_invariant=True,tracking_gradients=True,
        gravity=[0,0,-9.81],freefall_qvel_z=float(-9.81*.005),fk_max_error=max(errors),command_transition=True)

def main(out,production_batch=False):
    start=time.time();torch.set_num_threads(2)
    report=dict(analytic=analytic_checks(),repair=repair_checks(),physical_acceptance=False)
    from train import CommandTrainer,Config
    cfg=Config(out=str(out),num_envs=128,buffer_steps=128,wm_batch=512 if production_batch else 32,
        pi_batch=1024 if production_batch else 32,horizon=32,wm_batches=10,critic_epochs=5,compile_wm=False,smoke=True,eval_every=0)
    trainer=CommandTrainer(cfg);env=trainer.env
    assert trainer.buf.size==0 and all(len(o.state)==0 for o in (trainer.opt_wm,trainer.opt_pi,trainer.opt_v))
    hashes={k:tensor_hash(getattr(trainer,k)) for k in ('wm','policy','critic')}
    source=torch.load(cfg.init_checkpoint,map_location='cuda',weights_only=False)
    for name in ('wm','policy'):
        assert all(torch.equal(v,source[name][k]) for k,v in getattr(trainer,name).state_dict().items())
    assert any(not torch.equal(v,source['critic'][k]) for k,v in trainer.critic.state_dict().items())
    del source
    for i in range(600):trainer.collect();trainer.it+=1
    for _ in range(2):
        trainer.update_wm();trainer.update_policy();trainer.updates+=1
    assert tensor_hash(trainer.policy)==hashes['policy']
    assert len(trainer.opt_pi.state)==0 and trainer.actor_updates==0
    assert tensor_hash(trainer.critic)!=hashes['critic']
    prefit_frozen_actor=True
    env.reset();env.autoreset=False
    env.set_target(command(3,.8,.5,17.3,-19.2));pre=env.context.clone()
    env.step(torch.zeros(env.num_envs,12,device='cuda'))
    assert torch.allclose(env.context,advance_command(pre),atol=1e-6)
    assert torch.allclose(env.goal_buf,policy_goal(env.context),atol=1e-6)
    assert torch.allclose(env.xi[:,20:],env.prev_action)
    assert torch.allclose(env.sim.model.geom_friction[:,0,0],env.friction)
    assert torch.allclose(env.sim.model.geom_friction[:,env.foot_geoms,0],env.friction[:,None].expand(-1,4))
    env.eval_mode=False;env.event_left[:]=1;env.event_count[:]=1;env.individual[:]=True
    prior=env.context[:,7:12].clone();env.prepare_commands()
    assert torch.equal(env.context[:,7:12],-prior)
    assert int(env.event_left.min())>=400 and int(env.event_left.max())<=500
    env.event_left[:]=1;env.event_count[:]=2;env.prepare_commands();assert torch.count_nonzero(env.context[:,7:12])==0
    env.autoreset=True;env.reset()
    targets=env.sample_targets(10000)
    assert targets[:,:2].norm(dim=-1).max()<=4.000001
    assert (targets[:,3:5].abs()>math.radians(19.9)).all(-1).sum()>20
    assert (targets[:,0]>3.).sum()>20 and (targets[:,0]<-3.).sum()>20
    assert (targets[:,1]>.8).sum()>20 and (targets[:,1]<-.8).sum()>20
    a=torch.zeros(4,12,device='cuda',requires_grad=True);state={k:v[:4].clone() for k,v in env.state.items()}
    xi=env.xi[:4].clone();xi[:,19]=1
    grad=torch.autograd.grad(trainer.wm.step(state,a,xi)['q'].square().sum(),a)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    stats=[]
    trainer.it=cfg.actor_warmup_iters
    weight_before={name:[p.detach().clone() for p in getattr(trainer,name).parameters()] for name in ('policy','wm','critic')}
    for _ in range(3):
        trainer.collect();st=trainer.update_wm();st.update(trainer.update_policy());trainer.updates+=1
        assert all(math.isfinite(float(v)) for v in st.values());stats.append(st)
    after={k:tensor_hash(getattr(trainer,k)) for k in hashes}
    assert all(hashes[k]!=after[k] for k in hashes)
    assert all(any(not torch.equal(a,b) for a,b in zip(weight_before[name],getattr(trainer,name).parameters())) for name in weight_before)
    assert trainer.actor_updates==3
    assert all(torch.isfinite(v).all() for k in hashes for v in getattr(trainer,k).state_dict().values())
    trainer.save(str(Path(out)/'validated.pt'))
    from export import export
    error=export(Path(out)/'validated.pt',Path(out)/'validated_policy.pt')
    original=json.loads((HERE/'engine_provenance.json').read_text())['original_hashes']
    unchanged=all(sha256(HERE/'engine/supermimic'/name)==h for name,h in original.items() if name not in ('task.py','replay.py'))
    assert unchanged
    report.update(real_updates=3,horizon=32,k=8,production_batch=production_batch,stats=stats,
        initial_hashes=hashes,final_hashes=after,engine_learning_files_unchanged=unchanged,
        actor_wm_critic_updated=True,friction_floor_and_feet_verified=True,
        exact_actor_wm_continuation=True,source=trainer.source_meta,fresh_critic_replay_optimizers=True,
        prefit_frozen_actor=prefit_frozen_actor,reversal_schedule_verified=True,
        export_max_error=error,seconds=time.time()-start)
    write_json(Path(out)/'VALIDATED.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--production-batch',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);main(args.out,args.production_batch)
