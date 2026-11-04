"""Physical-state, observation and analytic-reward parity before learning smoke."""
import torch
from supermimic.world_model.heads import analytic_observation
from supermimic.world_model.wm_trainer import reference_at
from supermimic.robot_layout import NJ,SD,XD,OD,GF


@torch.no_grad()
def validate_env(tr):
    from mjlab.tasks.tracking import mdp
    from mjlab.envs import mdp as base_mdp
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    env=tr.env;obs,_=env.reset();rec=tr.recorder;dev=tr.device
    assert obs['policy'].shape==(env.num_envs,10*OD)
    assert obs['goal'].shape==(env.num_envs,10*GF)
    robot=env.scene['robot'];act=env.action_manager.get_term('joint_pos')
    assert tuple(act._target_names)==tuple(robot.joint_names), (act._target_names,robot.joint_names)
    assert torch.allclose(tr.lib.default_joint_pos,robot.data.default_joint_pos[0])
    cfg=SceneEntityCfg('robot',joint_names=('.*',));cfg.resolve(env.scene)
    worst={};obsworst=0.
    for step in range(16):
        obs,*_=env.step(.02*torch.randn(env.num_envs,NJ,device=dev))
        s=rec.state_io.read();xi=rec.ctx.read()
        assert s.shape==(env.num_envs,SD) and xi.shape==(env.num_envs,XD)
        ref=reference_at(tr.lib,tr.command.frame)
        ours=tr.wm.heads.reward(s,env.action_manager.action,env.action_manager.prev_action,ref)
        theirs={
            'motion_global_root_pos':.5*mdp.motion_global_anchor_position_error_exp(env,'motion',.3),
            'motion_global_root_ori':.5*mdp.motion_global_anchor_orientation_error_exp(env,'motion',.4),
            'motion_body_pos':mdp.motion_relative_body_position_error_exp(env,'motion',.3),
            'motion_body_ori':mdp.motion_relative_body_orientation_error_exp(env,'motion',.4),
            'motion_body_lin_vel':mdp.motion_global_body_linear_velocity_error_exp(env,'motion',1.),
            'motion_body_ang_vel':mdp.motion_global_body_angular_velocity_error_exp(env,'motion',3.14),
            'action_rate_l2':-.1*base_mdp.action_rate_l2(env),
            'joint_limit':-2.*base_mdp.joint_pos_limits(env,cfg),
        }
        for k,v in theirs.items():
            e=float((ours[k]/.02-v).abs().max());worst[k]=max(worst.get(k,0.),e)
        clean=analytic_observation(s,env.action_manager.action,xi,tr.lib.default_joint_pos)
        start=0
        for width,noise in ((NJ,.01),(NJ,.5),(3,.2),(3,.05),(NJ,0.)):
            newest=obs['policy'][:,start*10:(start+width)*10].reshape(env.num_envs,10,width)[:,-1]
            err=(newest-clean[:,start:start+width]).abs().max().item()
            obsworst=max(obsworst,max(0,err-noise));start+=width
        assert torch.allclose(obs['goal'],tr.lib.goal_window(tr.command.frame,10),atol=1e-6)
    assert max(worst.values())<3e-4,worst
    assert obsworst<3e-5,obsworst
    w=tr.buffer.sample_windows(64,8)
    assert w.states.shape==(9,64,SD) and w.a_exec.shape==(8,64,NJ)
    assert .03<float((w.a_exec-w.a_clean).std())<.2
    result=dict(reward_max_errors=worst,observation_noise_excess=obsworst,goal_alignment=True,
                joint_action_order=list(robot.joint_names),window_shape=list(w.states.shape),context_dim=XD)
    if NJ==12:
        assert set(env.b2_load.tolist())=={0.,5.}
        assert float(env.b2_mass_scale.min())>=.9 and float(env.b2_mass_scale.max())<=1.1
        result.update(payload_counts={str(kg):int((env.b2_load==kg).sum()) for kg in (0.,5.)},
                      mass_scale_range=[float(env.b2_mass_scale.min()),float(env.b2_mass_scale.max())])
    print('ADAPTER_CHECKS',result,flush=True)
    return result
