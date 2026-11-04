"""Finite implementation validation, not a trained skill acceptance test."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
from bootstrap import HERE,ROOT
from contract import *
from common import write_json,tensor_hash,sha256
from task import AttitudeTask,AttitudeCfg
from supermimic.kinematics import FastTreeKinematics
from supermimic.rotations import quat_from_euler_xyz,quat_to_mat
from supermimic.task import flat_to_state,state_to_flat

def analytic_checks():
    model=mujoco.MjModel.from_xml_path(str(HERE/'b2_attitude.xml'))
    meta=json.loads((HERE/'model.json').read_text())
    kin=FastTreeKinematics(model,'cpu')
    feet=[model.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    task=AttitudeTask(kin,torch.tensor(Q0),torch.tensor(model.jnt_range[1:],dtype=torch.float32),torch.tensor(ACTION_SCALE),AttitudeCfg(),'cpu',height=meta['standing_height_m'],foot_offsets=model.geom_pos[feet])
    rp=torch.tensor([[.1,0.],[-.1,0.],[0.,.1],[0.,-.1]],requires_grad=True)
    R=quat_to_mat(quat_from_euler_xyz(rp[:,0],rp[:,1],torch.zeros(4)))
    assert torch.allclose(angles(R),rp,atol=1e-6)
    s=dict(pos=torch.tensor([0.,0.,meta['standing_height_m']]).expand(4,3),rot=R,q=torch.tensor(Q0).expand(4,12),
           v=torch.zeros(4,3),w=torch.zeros(4,3),qd=torch.zeros(4,12))
    xi=torch.zeros(4,32);xi[:,0]=1;xi[:,16:19]=1
    context=torch.zeros(4,6);actions=torch.zeros(4,12)
    r=task.command_analytics(s,context,actions,actions,xi)
    grad=torch.autograd.grad(r['terms']['attitude'].sum()+r['terms']['attitude_error'].sum(),rp,retain_graph=True)[0]
    assert (grad*rp<=0).all() and grad.abs().max()>0
    context[:,:2]=rp.detach()*2
    r=task.command_analytics(s,context,actions,actions,xi)
    grad2=torch.autograd.grad(r['terms']['attitude'].sum()+r['terms']['attitude_error'].sum(),rp)[0]
    assert (grad2*rp>=0).all()
    flat=state_to_flat(s);roundtrip=flat_to_state(flat)
    assert flat.shape[-1]==37
    # Upstream quaternion conversion floors zero components at 5e-7.
    assert max(float((roundtrip[k]-s[k]).detach().abs().max()) for k in s)<2e-6
    # Compare differentiable foot FK with independent MuJoCo FK at all four limits.
    fk=[]
    for axis in range(2):
      for degrees in [-30,30]:
        row=next(r for r in json.loads((HERE/'feasibility.json').read_text())['conditions'] if r['axis']==['roll','pitch'][axis] and r['degrees']==degrees)
        a=torch.zeros(1,3);a[0,axis]=math.radians(degrees);quat=quat_from_euler_xyz(a[:,0],a[:,1],a[:,2])
        st={k:v[:1].detach().clone() for k,v in s.items()};st['q']=torch.tensor(row['q'])[None];st['rot']=quat_to_mat(quat)
        foot,_=task.foot_state(task.body_kin(st));d=mujoco.MjData(model)
        d.qpos[:]=np.r_[st['pos'][0].numpy(),quat[0].numpy(),st['q'][0].numpy()];mujoco.mj_forward(model,d)
        error=float(np.max(np.abs(foot[0].numpy()-d.geom_xpos[feet])));assert error<2e-6;fk.append(error)
    # A 60-degree reversal must settle within the evaluator's 3-second window.
    ctx=torch.tensor([[-MAX_ANGLE,0,0,0,MAX_ANGLE,0.]])
    trajectory=[]
    for _ in range(150):ctx=advance_command(ctx);trajectory.append(ctx.clone())
    assert abs(float(ctx[0,0])-MAX_ANGLE)<math.radians(.5)
    sequence=torch.cat(trajectory);assert sequence[:,2:4].abs().max()<=MAX_RATE+1e-6
    assert ((sequence[1:,2:4]-sequence[:-1,2:4])/DT).abs().max()<MAX_ACCEL+1e-4
    # Uniform continuous targets and all directed changes, rather than four discrete labels.
    return dict(angle_signs=True,tracking_gradients=True,state_roundtrip=True,max_fk_error=max(fk),command_reversal_error_deg=abs(float(ctx[0,0])-MAX_ANGLE)*180/math.pi,command_rate_acceleration=True)

def main(out,production_batch=False):
    torch.set_num_threads(2);started=time.time();report=dict(analytic=analytic_checks(),physical_acceptance=False)
    from train import CommandTrainer,Config
    cfg=Config(out=str(out),num_envs=128,buffer_steps=128,wm_batch=512 if production_batch else 32,
        pi_batch=1024 if production_batch else 32,horizon=32,wm_batches=10,critic_epochs=5,
        compile_wm=False,smoke=True,eval_every=0)
    trainer=CommandTrainer(cfg)
    assert trainer.buf.size==0 and all(len(o.state)==0 for o in (trainer.opt_wm,trainer.opt_pi,trainer.opt_v))
    before={k:tensor_hash(getattr(trainer,k)) for k in ('wm','policy','critic')}
    for i in range(100):trainer.collect();trainer.it+=1
    env=trainer.env
    # Test exact command context progression and replay of the applied noisy action.
    env.context[:,4:6]=torch.tensor([math.radians(17.3),0.],device='cuda')
    pre=env.context.clone();prior=env.prev_action.clone()
    action=torch.zeros(env.num_envs,12,device='cuda');state,info=env.step(action)
    assert torch.allclose(env.context,advance_command(pre),atol=1e-6)
    assert torch.allclose(env.xi[:,20:],env.prev_action)
    assert torch.allclose(env.sim.model.geom_friction[:,0,0],env.friction)
    assert torch.allclose(env.sim.model.geom_friction[:,env.foot_geoms,0],env.friction[:,None].expand(-1,4))
    # Endpoint gradients survive the delay context; changing actions changes WM inputs.
    a=torch.zeros(4,12,device='cuda',requires_grad=True)
    state4={k:v[:4].clone() for k,v in env.state.items()};xi=env.xi[:4].clone();xi[:,19]=1
    loss=trainer.wm.step(state4,a,xi)['q'].square().sum()
    grad=torch.autograd.grad(loss,a)[0];assert grad.abs().sum()>0 and torch.isfinite(grad).all()
    stats=[]
    for j in range(3):
        trainer.collect();st=trainer.update_wm();st.update(trainer.update_policy());trainer.updates+=1
        assert all(math.isfinite(float(v)) for v in st.values());stats.append(st)
    after={k:tensor_hash(getattr(trainer,k)) for k in before}
    assert all(before[k]!=after[k] for k in before)
    assert all(torch.isfinite(v).all() for name in before for v in getattr(trainer,name).state_dict().values())
    trainer.save(str(Path(out)/'validated.pt'))
    report.update(device='cuda',horizon=32,wm_window=8,real_updates=3,production_batch=production_batch,
        initial_hashes=before,final_hashes=after,stats=stats,seconds=time.time()-started,random_initialization=True,
        actor_wm_critic_updated=True,friction_mixing_verified=True,command_transition_parity=True,
        engine_learning_files_unchanged=all(sha256(HERE/'engine/supermimic'/f)==h for f,h in json.loads((HERE/'engine_provenance.json').read_text())['original_hashes'].items() if f not in ('task.py','replay.py')))
    assert report['engine_learning_files_unchanged']
    write_json(Path(out)/'VALIDATED.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--production-batch',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);main(a.out,a.production_batch)
