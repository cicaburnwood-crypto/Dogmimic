"""Finite CPU/formula checks and real H32 optimizer updates, never skill claims."""
import argparse
import ast
import dataclasses
import os
from pathlib import Path
import time
import torch
from common import HERE,ROOT,write_json,tensor_hash
from sm_original.task import TaskCfg,Task,NJ
from sm_original.train import TrainCfg
from train import ScratchTrainer

def checks(trainer):
    source=ast.parse((HERE/'sources/pre_hardware_task.py.txt').read_text())
    target=ast.parse((HERE/'sm_original/task.py').read_text())
    def reward_ast(tree):
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Task')
        return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='reward'))
    assert reward_ast(source)==reward_ast(target),'Published tracking reward changed'
    c=TaskCfg()
    assert [c.w_body_pos,c.w_body_ori,c.w_body_lin,c.w_body_ang,c.w_root_pos,c.w_root_ori,c.w_action_rate,c.w_joint_limit]==[1.,1.,1.,1.,.5,.5,-.1,-2.]
    assert [c.std_body_pos,c.std_body_ori,c.std_body_lin,c.std_body_ang,c.std_root_pos,c.std_root_ori]==[.3,.4,1.,3.14,.3,.4]
    env=trainer.env
    ref=trainer.lib.frame(torch.zeros(4,dtype=torch.long,device=trainer.device))
    # This differentiable check runs on CPU, independent of CUDA.
    import mujoco
    from sm_original.kinematics import FastTreeKinematics
    root=1 if trainer.robot_kind=='B2' else env._find_body('pelvis')
    cpu_kin=FastTreeKinematics(env.mj_model,'cpu',root_body=root)
    cpu_task=Task(cpu_kin,env.q0.cpu(),torch.stack([env.task.qmin,env.task.qmax],-1).cpu(),env.action_scale.cpu(),c,'cpu')
    ref={k:v.cpu() for k,v in ref.items()}
    refkin=cpu_task.body_kin(ref)
    a=torch.zeros(4,NJ);zero=a.clone()
    r0,terms=cpu_task.reward(ref,refkin,ref,refkin,a,zero)
    assert torch.isfinite(r0).all() and (r0<=c.max_step_reward+1e-6).all()
    # Move only absolute root X: re-anchored terms stay equal, root tracking worsens.
    x=torch.full((4,),.07,requires_grad=True)
    state=dict(ref);state['pos']=ref['pos']+torch.stack([x,x*0,x*0],-1)
    r,_=cpu_task.reward(state,cpu_task.body_kin(state),ref,refkin,a,zero)
    grad=torch.autograd.grad(r.sum(),x)[0]
    eps=1e-4
    def val(delta):
        s=dict(ref);s['pos']=ref['pos']+torch.tensor([.07+delta,0.,0.])
        return cpu_task.reward(s,cpu_task.body_kin(s),ref,refkin,a,zero)[0]
    fd=(val(eps)-val(-eps))/(2*eps)
    assert (grad<0).all() and torch.allclose(grad,fd,atol=1e-4,rtol=.02)
    # Critic weights may be frozen, but actor bootstrap must retain state gradients.
    xV=torch.randn(4,trainer.critic_in,device=trainer.device,requires_grad=True)
    for p in trainer.critic.parameters():p.requires_grad_(False)
    gv=torch.autograd.grad(trainer.critic(xV).sum(),xV)[0]
    assert torch.isfinite(gv).all() and float(gv.norm())>0
    for p in trainer.critic.parameters():p.requires_grad_(True)
    return dict(reward_identical_to_paper_implementation=True,reference_reward=r0.tolist(),
                cpu_reward_autograd=grad.tolist(),cpu_reward_finite_difference=fd.tolist(),
                critic_state_gradient_norm=float(gv.norm()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--production',action='store_true');a=p.parse_args()
    robot=os.environ['SM_RESET_ROBOT'];torch.set_num_threads(2);os.chdir(ROOT)
    motion='experiments/b2_cycle/b2_full_cycle.npz' if robot=='B2' else 'data/processed/g1_standing_jump_060m.npz'
    cfg=TrainCfg(motions=motion,out=a.out,seed=2026110301,num_envs=1024 if a.production else 64,
                 buffer_steps=512 if a.production else 128,wm_batch=1024 if a.production else 64,
                 pi_batch=2048 if a.production else 64,wm_batches=10 if a.production else 2,
                 horizon=32,k=8,critic_epochs=5 if a.production else 1,
                 critic_minibatches=4 if a.production else 2,compile_wm=a.production,
                 warmup_steps=64,log_every=100,use_critic=True,amp=True)
    started=time.time();t=ScratchTrainer(cfg)
    if not a.production:t.task.compile_sizes.clear()
    formula=checks(t)
    before={k:tensor_hash(v.state_dict()) for k,v in [('actor',t.policy),('wm',t.wm),('critic',t.critic)]}
    for _ in range(64):t.collect()
    observed={k:float(n.count) for k,n in [('obs',t.policy.obs_norm),('goal',t.policy.goal_norm)]}
    assert min(observed.values())>0
    window=t.buf.sample_windows(64,8)
    assert int(window[-1].sum())>0
    if robot=='B2':
        assert t.env.xi.shape[1]==44
        assert set(t.env.loads.tolist())=={0.,5.}
        force=t.env.push_vectors.norm(dim=-1)
        assert float(force.min())>=7.999 and float(force.max())<=16.001
        xi=t.env.xi.clone();next_xi=t.task.advance_context(xi)
        assert torch.equal(xi[:,:-1],next_xi[:,:-1])
        assert torch.allclose(next_xi[:,-1]-xi[:,-1],torch.full_like(xi[:,-1],.02),atol=1e-6)
        from sm_original.rotations import yaw_mat,mv,mm
        state=t.env.state;rot=yaw_mat(torch.full((t.env.num_envs,),.4,device=t.device))
        shifted=dict(state);shifted['rot']=mm(rot,state['rot'])
        other=xi.clone();schedule=other[:,18:43].reshape(-1,5,5)
        schedule[:,:,2:]=mv(rot[:,None],schedule[:,:,2:].clone())
        assert torch.allclose(t.task.model_context(state,xi),t.task.model_context(shifted,other),atol=2e-6)
    logs=[]
    for i in range(3):
        t.collect();wm=t.update_wm()
        w_before=tensor_hash(t.wm.net.state_dict())
        pi=t.update_policy()
        assert tensor_hash(t.wm.net.state_dict())==w_before,'Actor changed WM weights'
        t.updates+=1
        logs.append(dict(wm=wm,policy=pi))
        print('VALIDATION_UPDATE',robot,t.rank,i,wm['wm_loss'],pi['pi_gnorm'],pi['v_loss'],flush=True)
    after={k:tensor_hash(v.state_dict()) for k,v in [('actor',t.policy),('wm',t.wm),('critic',t.critic)]}
    assert all(before[k]!=after[k] for k in before)
    assert all(torch.isfinite(p).all() for m in (t.wm,t.policy,t.critic) for p in m.parameters())
    if t.world>1:
        import torch.distributed as dist
        gathered=[None]*t.world;dist.all_gather_object(gathered,after)
        assert all(x==after for x in gathered),'DDP weights diverged'
    result=dict(robot=robot,passed=True,kind='production_shape_compiled_three_updates' if a.production else 'CPU_checks_and_CUDA_H32_three_updates',
                world=t.world,formula=formula,initialization=t._initialization,before=before,after=after,
                normalizers_learned=observed,valid_windows=int(window[-1].sum()),updates=logs,
                max_memory_gb=torch.cuda.max_memory_allocated()/1e9,seconds=time.time()-started)
    write_json(Path(a.out)/f'validation_rank{t.rank}.json',result)
    t._barrier()
    if t.world>1:
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__=='__main__':main()
