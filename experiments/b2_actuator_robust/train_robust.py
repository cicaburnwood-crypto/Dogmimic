import os,time,json,hashlib,dataclasses
from pathlib import Path
import torch,numpy as np
from supermimic import train as base
from supermimic.networks import WorldModel
from train import SkillTrainer
from train_recovery import RecoveryTrainer,RecoveryWorldModel
from train_payload import TimedContextReplay
from stage_policy import StagePolicy
from robust_env import make_env,XI_DIM,SPEED,POWER
from evaluate_cycle import evaluate,CRITERIA
START=time.time();HERE=Path(__file__).resolve().parent
class RobustWM(WorldModel):
    def __init__(self,task,feat_dim,xi_dim):super().__init__(task,feat_dim,22)
    def inputs(self,s,a,xi,feats=None):
        canonical=RecoveryWorldModel.canonical_context(s,xi)
        return super().inputs(s,a,torch.cat([canonical[...,:21],xi[...,30:31]],-1),feats)
base.XI_DIM=XI_DIM;base.WorldModel=RobustWM;base.Policy=lambda o,g:StagePolicy(o,g)
class Trainer(RecoveryTrainer):
    def __init__(self,cfg,device='cuda'):
        self.source_path=Path(os.environ['B2_INIT_CHECKPOINT']);self.source_hash=hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        assert self.source_hash=='14c5f8ba3eac277e8aeb865c5b5deee7e93d669162de49c5e719892e84a78208'
        SkillTrainer.__init__(self,cfg,device)
        ck=torch.load(self.source_path,map_location=device,weights_only=False);self.policy.load_state_dict(ck['policy']);self.source_iteration=int(ck['it'])
        self.policy.teacher.requires_grad_(True)
        self.opt_pi=torch.optim.Adam(self.policy.parameters(),lr=cfg.lr*.1,betas=(.9,.95),fused=True)
        self.actor_lr_scale=1.;self.teacher_weight=.03;self.best_quality=-1.;self.passed=0;self.guard_rollbacks=0
    def _make_env(self):
        e=make_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma);self.tcfg=e.tcfg;return e
    def _make_buffer(self):return TimedContextReplay(self.cfg.num_envs,self.cfg.buffer_steps,XI_DIM,self.device)
    def _extra_state(self):
        return dict(task_kind='b2_actuator_robust_v1',robot='B2',source_checkpoint=str(self.source_path),source_sha256=self.source_hash,source_iteration=getattr(self,'source_iteration',14500),initialization='own actor continued; fresh WM/replay/optimizers, no critic',actor_rise_trainable=True,actuator_budget_experimental=True,speed_soft_rad_s=SPEED,power_abs_mechanical_w=POWER,torque_nm=[160,160,240]*4,criteria=CRITERIA,policy_architecture=dict(type='stage_guarded_residual',encoder=[512,256,128],decoder=[512,256,128],hind_limit_rad=.008,max_joint_delta_rad=.04))
    def _should_stop(self):return Path(self.cfg.out,'STOP').exists() or time.time()-START>self.cfg.wall_hours*3600
    def update_policy(self):
        if self.it<int(os.environ.get('B2_WM_PREFIT','1000')):return dict(actor_prefit=1.)
        return super().update_policy()
    @torch.no_grad()
    def run_eval(self,full=False):
        out=Path(self.cfg.out);cp=out/f'evaluated_{self.it:07d}.pt';self.save(str(cp))
        if not hasattr(self,'eval_env'):self.eval_env=make_env(self.lib,self.device,64,evaluate=True,kg=0.)
        e=self.eval_env;cases=[(kg,1.,1.,False) for kg in [0,5,10,15]]
        if full:cases += [(0,fr,m,asym) for fr in [.4,.7,1.,1.3] for m in [.9,1.,1.1] for asym in [False,True]]
        results=[]
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for kg,fr,m,asym in cases:
                e.fixed_loads.fill_(kg);e.fixed_mass=m;e.fixed_friction=fr;e.asymmetric_eval=asym;e.max_speed.zero_();e.max_power.zero_()
                result,arr=evaluate(self.policy,e,seed=20261103)
                result.update(kg=kg,friction=fr,mass_scale=m,asymmetric=asym,peak_speed=float(e.max_speed.max()),peak_mechanical_power=float(e.max_power.max()))
                result['actuator_pass']=result['peak_speed']<=SPEED and result['peak_mechanical_power']<=POWER+1
                results.append(result)
        good=all(r['success_rate']>=.95 and r['nominal_success'] and r['actuator_pass'] for r in results)
        summary=dict(it=self.it,wall_seconds=time.time()-START,full=full,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),all_passed=good,conditions=results)
        (out/'latest_eval.json').write_text(json.dumps(summary,indent=2));(out/'eval').mkdir(exist_ok=True);(out/'eval'/f'it_{self.it:07d}_{int(full)}.json').write_text(json.dumps(summary))
        with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        score=sum(r['success_rate'] for r in results[:4]);print('PHYSICAL_EVAL',self.it,full,[(r['kg'],round(64*r['success_rate']),r['peak_speed']) for r in results],flush=True)
        if score>self.best_quality:
            self.best_quality=score;self.save(str(out/'best.pt'));summary['best_sha256']=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest();(out/'best_eval.json').write_text(json.dumps(summary,indent=2))
        if good and not full:return self.run_eval(full=True)
        self.passed=self.passed+1 if full and good else 0
        if self.passed>=2:
            self.save(str(out/'accepted.pt'));(out/'TARGET_MET.json').write_text(json.dumps(summary));self.cfg.iterations=self.it
if __name__=='__main__':
    torch.set_num_threads(2);os.sched_setaffinity(0,set(range(24,28)))
    cfg=base.parse_args();assert not cfg.use_critic
    t=Trainer(cfg);Path(cfg.out,'run_meta.json').write_text(json.dumps(t._extra_state(),indent=2));
    smoke=os.environ.get('B2_SMOKE')=='1'
    before=next(t.policy.teacher.decoder.parameters()).detach().clone()
    if not smoke:t.run_eval()
    t.run()
    if not smoke:t.run_eval(full=True)
    if smoke:
        delta=float((next(t.policy.teacher.decoder.parameters()).detach()-before).abs().max())
        assert delta>0 and t.updates>0
        assert torch.allclose(t.env.sim.model.geom_friction[:,:,0],t.env.friction[:,None].expand_as(t.env.sim.model.geom_friction[:,:,0]))
        Path(cfg.out,'smoke_audit.json').write_text(json.dumps(dict(passed=True,updates=t.updates,actor_rise_weight_delta=delta,friction_min=float(t.env.friction.min()),friction_max=float(t.env.friction.max()),mass_min=float(t.env.mass_scale.min()),mass_max=float(t.env.mass_scale.max()),xi_dim=t.env.xi.shape[-1])))
    Path(cfg.out,'EVAL_DONE.json').write_text(json.dumps(dict(end_unix=time.time(),wall_seconds=time.time()-START,it=t.it)))
