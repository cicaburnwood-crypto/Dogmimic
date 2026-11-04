"""Portable inference artifact for the unchanged strict evaluator's separate venv."""
import copy
import json
from pathlib import Path
import torch
from common import sha256


def export(trainer, checkpoint, dest, robot):
    p=copy.deepcopy(trainer.policy).cpu().eval()
    nj=len(trainer.lib.default_joint_pos)
    obs=torch.randn(3,10*(3*nj+6));goal=torch.randn(3,10*(2*nj+10))
    class Inference(torch.nn.Module):
        def __init__(self,p):super().__init__();self.p=p
        def forward(self,obs,goal):return self.p(obs,goal)
    wrapper=Inference(p)
    with torch.no_grad():
        scripted=torch.jit.trace(wrapper,(obs,goal),check_trace=True)
        # Batch size changes in independent evaluation.
        o=torch.randn(64,obs.shape[1]);g=torch.randn(64,goal.shape[1])
        err=(scripted(o,g)-p(o,g)).abs().max().item()
        assert err<1e-6,err
    action=trainer.env.action_manager.get_term('joint_pos')
    scale=torch.as_tensor(action.scale).cpu()
    if scale.ndim==2:scale=scale[0]
    if scale.ndim==0:scale=scale.repeat(nj)
    assert tuple(action._target_names)==tuple(trainer.env.scene['robot'].joint_names), 'Action/joint order differs'
    meta=dict(backend='NJU',robot=robot,iteration=trainer.iteration,checkpoint_sha256=sha256(checkpoint),
        q0=trainer.lib.default_joint_pos.cpu().tolist(),scale=scale.tolist(),
        goal_feat=trainer.lib.goal_feat.cpu().tolist(),export_max_abs_error=err,
        observation_layout='term-major history10; joints,vel,body gyro,gravity,previous clean action',goal_offset=0)
    dest=Path(dest);tmp=dest.with_suffix('.tmp')
    torch.jit.save(scripted,str(tmp),_extra_files={'metadata.json':json.dumps(meta)})
    tmp.replace(dest)
    return meta
