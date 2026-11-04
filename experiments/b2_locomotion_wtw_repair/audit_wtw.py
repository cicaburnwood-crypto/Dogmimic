"""Parity against unmodified upstream methods, not a second copy of our port."""
import ast,json,hashlib,math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from supermimic.rotations import quat_to_mat,mtv,quat_mul as qmul_wxyz
from wtw_rewards import reward_terms,combine,CONFIG,SCALES
from gait_clock import foot_phases,desired_contacts,advance_phase,policy_goal,FREQUENCY,SWING_HEIGHT,DURATION
from contract import DT,Q0,ACTION_SCALE,constrain_command

HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor/walk_these_ways'
def quat_conjugate(q):return torch.cat([-q[...,:3],q[...,3:]],-1)
def quat_mul(a,b):return qmul_wxyz(a[..., [3,0,1,2]],b[..., [3,0,1,2]])[..., [1,2,3,0]]
def quat_from_angle_axis(angle,axis):
    return torch.cat([torch.sin(angle[...,None]/2)*axis,torch.cos(angle[...,None]/2)],-1)
def quat_rotate_inverse(q,v):return mtv(quat_to_mat(q[..., [3,0,1,2]]),v)
def quat_apply_yaw(q,v):
    z=torch.zeros_like(q[...,:2]);yaw=torch.cat([z,q[...,2:]],-1)
    yaw=yaw/yaw.norm(dim=-1,keepdim=True)
    return (quat_to_mat(yaw[..., [3,0,1,2]])*v[...,None,:]).sum(-1)

def load_original():
    ns=dict(torch=torch,np=np,quat_apply_yaw=quat_apply_yaw,quat_conjugate=quat_conjugate,
            quat_mul=quat_mul,quat_rotate_inverse=quat_rotate_inverse,quat_from_angle_axis=quat_from_angle_axis)
    tree=ast.parse((VENDOR/'go1_gym/envs/rewards/corl_rewards.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CoRLRewards')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'upstream/corl_rewards.py','exec'),ns)
    tree=ast.parse((VENDOR/'go1_gym/envs/base/legged_robot.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LeggedRobot')
    for name in ('compute_reward','_step_contact_targets'):
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream/legged_robot.py','exec'),ns)
    return ns

def run():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1907)
    src=json.loads((VENDOR/'SOURCE.json').read_text())
    for path,meta in src['files'].items():assert hashlib.sha256((VENDOR/path).read_bytes()).hexdigest()==meta['sha256'],path
    n=256;original=load_original()
    qraw=torch.randn(n,4,requires_grad=True);quat=qraw/qraw.norm(dim=-1,keepdim=True);R=quat_to_mat(quat)
    action=torch.randn(n,12).tanh().requires_grad_();prev=torch.randn(n,12).tanh();lastlast=torch.randn(n,12).tanh()
    prev[:16]=0;lastlast[:32]=0
    fp=torch.randn(n,4,3).mul(.2).requires_grad_();fv=torch.randn(n,4,3).mul(.2).requires_grad_()
    aux=torch.randn(n,32);aux[:,:24]*=20;aux[:,24:]=(torch.rand(n,8)>.9).double();aux.requires_grad_()
    pos=torch.randn(n,3)*.1;pos[:,2]+=.6
    s=dict(pos=pos,rot=R,q=torch.tensor(Q0)+torch.randn(n,12)*.2,qd=torch.randn(n,12),v=torch.randn(n,3)*.3,w=torch.randn(n,3)*.2)
    prior=dict(qd=s['qd']+torch.randn(n,12)*.02)
    cmd=torch.randn(n,3)*.4;cmd[:16]=0
    phase=torch.rand(n);bias=torch.randn(n,12)*.002;contacts=torch.rand(n,4)>.5
    q0=torch.tensor(Q0);scale=torch.tensor(ACTION_SCALE);qmin=q0-1.;qmax=q0+1.
    height=.6;width=.4;length=.66
    ours,contact=reward_terms(s,prior,cmd,action,prev,lastlast,bias,phase,aux,contacts,fp,fv,q0,scale,qmin,qmax,height,width,length)
    commands=torch.zeros(n,15);commands[:,:3]=cmd;commands[:,4]=FREQUENCY;commands[:,5]=.5;commands[:,8]=DURATION
    commands[:,9]=SWING_HEIGHT;commands[:,12]=width;commands[:,13]=length
    target=lambda a:(q0+scale*a-bias).clamp(qmin+.02,qmax-.02)
    force=torch.cat([aux[:,12:24].reshape(n,4,3),torch.stack([torch.zeros_like(aux[:,24:])]*2+[(aux[:,24:]>.5).double()*.2],-1)],1)
    e=NS(num_envs=n,device='cpu',commands=commands,dt=DT,
         cfg=NS(rewards=NS(**{**CONFIG['rewards'],'base_height_target':height}),commands=NS(num_commands=15,pacing_offset=False),env=NS(observe_gait_commands=True)),
         base_lin_vel=mtv(R,s['v']),base_ang_vel=mtv(R,s['w']),torques=aux[:,:12]*aux.new_tensor(CONFIG['b2_scale_adapter']['torque_to_go1_equivalent']),
         last_dof_vel=prior['qd'],dof_vel=s['qd'],actions=action,last_actions=prev,last_last_actions=lastlast,
         dof_pos=s['q'],dof_pos_limits=torch.stack([q0-.9,q0+.9],-1),
         base_pos=pos,base_quat=quat[:,[1,2,3,0]],contact_forces=force,
         feet_indices=torch.arange(4),penalised_contact_indices=torch.arange(4,12),
         foot_indices=foot_phases(phase,cmd),desired_contact_states=desired_contacts(foot_phases(phase,cmd)),
         # Radius offset only affects clearance; Raibert uses only XY after yaw.
         foot_positions=fp-torch.tensor([0.,0.,.012]),foot_velocities=fv,
         projected_gravity=-R[:,2,:],gravity_vec=torch.tensor([0.,0.,-1.]).expand(n,-1),
         joint_pos_target=target(action),last_joint_pos_target=target(prev),last_last_joint_pos_target=target(lastlast),
         num_dof=12,num_actuated_dof=12,last_contacts=contacts.clone())
    rewards=original['CoRLRewards'](e)
    expected={name:getattr(rewards,'_reward_'+name)() for name in SCALES}
    errors={k:float((ours[k]-expected[k]).detach().abs().max()) for k in SCALES}
    for k in SCALES:torch.testing.assert_close(ours[k],expected[k],rtol=1e-6,atol=1e-8,msg=k,check_dtype=False)
    gradients={}
    inputs=[action,fp,fv,aux,qraw]
    for k in SCALES:
        if not ours[k].requires_grad:continue
        a=torch.autograd.grad(ours[k].sum(),inputs,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(expected[k].sum(),inputs,retain_graph=True,allow_unused=True)
        errs=[]
        for x,y,inp in zip(a,b,inputs):
            x=torch.zeros_like(inp) if x is None else x;y=torch.zeros_like(inp) if y is None else y
            torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7,msg=k+' gradient');errs.append(float((x-y).abs().max()))
        gradients[k]=max(errs)
    e.last_contacts=contacts.clone();e.reward_names=list(SCALES);e.reward_functions=[getattr(rewards,'_reward_'+k) for k in SCALES]
    e.reward_scales={k:v*DT for k,v in SCALES.items()}
    e.rew_buf=torch.zeros(n);e.rew_buf_pos=torch.zeros(n);e.rew_buf_neg=torch.zeros(n)
    e.episode_sums={k:torch.zeros(n) for k in [*SCALES,'total']};e.command_sums={k:torch.zeros(n) for k in [*SCALES,'lin_vel_raw','ang_vel_raw','lin_vel_residual','ang_vel_residual','ep_timesteps']}
    original['compute_reward'](e);r,_=combine(ours)
    torch.testing.assert_close(r,e.rew_buf,rtol=1e-6,atol=1e-10)
    # Published moving clock is executed verbatim and compared, including wrap.
    e.gait_indices=phase.clone();e.clock_inputs=torch.zeros(n,4);e.doubletime_clock_inputs=torch.zeros(n,4)
    e.halftime_clock_inputs=torch.zeros(n,4);e.desired_contact_states=torch.zeros(n,4)
    original['_step_contact_targets'](e)
    move_cmd=cmd.clone();move_cmd[:,0]=1.
    pp=foot_phases(advance_phase(phase,move_cmd),move_cmd)
    torch.testing.assert_close(pp,e.foot_indices,rtol=0,atol=1e-12)
    torch.testing.assert_close(desired_contacts(pp),e.desired_contact_states,rtol=1e-6,atol=1e-8)
    torch.testing.assert_close(policy_goal(move_cmd,advance_phase(phase,move_cmd))[:,3:],e.clock_inputs,rtol=1e-6,atol=1e-8)
    # Stop interface: all four feet have stance target; yaw remains unbounded.
    stop=torch.zeros(n,3);assert (desired_contacts(foot_phases(phase,stop))>.999).all()
    for x in np.linspace(-5,5,21):
      for y in np.linspace(-2,2,9):
        out=constrain_command(x,y,100.)
        assert math.hypot(*out[:2])<=2.+1e-12 and abs(out[1])<=.6 and out[2]==100.
    # Acceptance invariance, independent of all the new rewards.
    old=HERE.parent/'b2_locomotion_support'
    assert (HERE/'gait.py').read_bytes()==(old/'gait.py').read_bytes()
    def defs(p):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
    x,y=defs(HERE/'evaluate.py'),defs(old/'evaluate.py')
    for key in ('cases','evaluate'):assert x[key]==y[key],key
    result=dict(passed=True,source_commit=CONFIG['source_commit'],terms=len(SCALES),samples=n,
                max_term_abs_error=errors,gradient_max_abs_error=gradients,
                combination_max_abs_error=float((r-e.rew_buf).abs().max()),
                upstream_moving_clock_parity=True,stop_four_foot_stance=True,command_constraints_pass=True,
                physical_acceptance_ast_identical=True,gait_sha256=hashlib.sha256((HERE/'gait.py').read_bytes()).hexdigest())
    (HERE/'reward_parity_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':run()
