"""Add three feedback inputs without changing the inherited policy's output."""
import hashlib
from pathlib import Path
import torch

def expand_policy(old,model):
    dst=model.state_dict()
    for name,value in old.items():
        if name=='encoder.0.weight':
            dst[name].zero_();dst[name][:,:3]=value
        elif name in ('goal_norm.mean','goal_norm.var'):
            dst[name][:3]=value
        else:dst[name].copy_(value)
    model.load_state_dict(dst)

def expand_critic(old,model,feat_dim):
    dst=model.state_dict()
    old_dim=old['norm.mean'].numel()
    mapping=torch.cat([torch.arange(feat_dim+3),torch.arange(feat_dim+6,old_dim+3)]).to(dst['norm.mean'].device)
    for name,value in old.items():
        if name=='net.0.weight':
            dst[name].zero_();dst[name][:,mapping]=value
        elif name in ('norm.mean','norm.var'):
            dst[name][mapping]=value
        else:dst[name].copy_(value)
    model.load_state_dict(dst)

def load_own_v1(trainer,path):
    ck=torch.load(path,map_location=trainer.device,weights_only=False)
    if ck.get('task_kind')!='reference_free_command_v1' or not ck.get('no_motion_data'):
        raise ValueError('Only our audited, reference-free v1 checkpoint can initialize v2')
    trainer.wm.load_state_dict(ck['wm'])
    trainer.opt_wm.load_state_dict(ck['opt_wm'])
    expand_policy(ck['policy'],trainer.policy)
    expand_critic(ck['critic'],trainer.critic,trainer.feat_dim)
    # Fresh actor/critic optimizer moments because the objective and inputs changed.
    return dict(checkpoint=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                source_iteration=ck['it'],source_train_seconds=ck['train_time'],
                source_launch_wall_seconds=ck['launch_wall_seconds'],external_pretraining=False,
                inherited=['policy','world_model','critic','world_model_optimizer'],
                reset=['policy_optimizer','critic_optimizer','replay','phase_iteration_counter'])
