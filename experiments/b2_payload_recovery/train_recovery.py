"""Conservative SuperMimic continuation with physical evaluation rollback."""
import dataclasses,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from supermimic import train as base
from supermimic.networks import WorldModel,run_mlp,apply_delta_c
from supermimic.rotations import heading_mat,mtv
from supermimic.task import NJ,flat_to_state
from train_payload import PayloadTrainer,TimedContextReplay
from evaluate_payloads import run_condition,compact
from evaluate_pushes import schedule
from guarded_policy import GuardedPolicy
from recovery_env import XI_DIM,make_recovery_env

START=time.time();HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


class RecoveryWorldModel(WorldModel):
    def __init__(self,task,feat_dim,xi_dim):super().__init__(task,feat_dim,21)
    @staticmethod
    def canonical_context(s,xi):
        H=heading_mat(s['rot'])
        anchors=xi[...,21:25].reshape(*xi.shape[:-1],2,2)
        rel=anchors-s['pos'][...,:2].unsqueeze(-2)
        rel3=torch.cat([rel,torch.zeros_like(rel[...,:1])],-1)
        h=mtv(H.unsqueeze(-3),rel3)[...,:2].flatten(-2)
        c=mtv(H,xi[...,25:28]-s['pos'])
        return torch.cat([xi[...,:17],mtv(H,xi[...,17:20]),xi[...,20:21],h,c,xi[...,28:30]],-1)
    def inputs(self,s,a,xi,feats=None):
        # Anchors are reward memory; they do not affect physical dynamics.
        return super().inputs(s,a,self.canonical_context(s,xi)[...,:21],feats)


base.XI_DIM=XI_DIM
base.WorldModel=RecoveryWorldModel
base.Policy=lambda obs,goal:GuardedPolicy(obs,goal,max_joint_delta_rad=float(os.environ.get('B2_RESIDUAL_MAX_RAD','.04')))


class RecoveryTrainer(PayloadTrainer):
    def __init__(self,cfg,device='cuda'):
        super().__init__(cfg,device)
        self.wm_prefit_source=ROOT/'runs/b2_payload_stable_cycle_v3_20261030_232447/wm_prefit.pt'
        ck=torch.load(self.wm_prefit_source,map_location=device,weights_only=False)
        self.wm.load_state_dict(ck['wm'])  # Physics-only prefit, before any actor updates.
        self.actor_lr_scale=1.;self.guard_rollbacks=0;self.teacher_weight=.03
        self.safe_path=Path(cfg.out)/'safe.pt'
        self.eval_n=int(os.environ.get('B2_EVAL_TRIALS','64'))

    def _make_env(self):
        env=make_recovery_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma)
        self.tcfg=env.tcfg;return env

    def _make_buffer(self):return TimedContextReplay(self.cfg.num_envs,self.cfg.buffer_steps,XI_DIM,self.device)

    def _should_stop(self):return Path(self.cfg.out,'STOP').exists() or time.time()-START>=self.cfg.wall_hours*3600

    def _extra_state(self):
        d=super()._extra_state()
        d.update(task_kind='b2_payload_recovery_v4',
                 policy_architecture=dict(type='guarded_residual',encoder=[512,256,128],decoder=[512,256,128],
                                          max_joint_delta_rad=float(self.policy.max_joint_delta_rad)),
                 policy_normalizers_frozen=True,teacher_checkpoint=str(self.source_path),
                 anchor_definition='actual hind feet at t>=4s; actual loaded COM at t>=13s; reset-local',
                 xi_layout=['physical context[21]','hind_foot_anchor_xy[4]','return_COM_anchor_xyz[3]','anchor_valid[2]'],
                 actor_lr_scale=getattr(self,'actor_lr_scale',1.),guard_rollbacks=getattr(self,'guard_rollbacks',0),
                 teacher_regularization=getattr(self,'teacher_weight',.03),
                 critic_bootstrap=False,wall_seconds=time.time()-START)
        if hasattr(self,'wm_prefit_source'):d['world_model_prefit_source']=str(self.wm_prefit_source)
        return d

    def update_policy(self):
        cfg,task,lib,wm=self.cfg,self.task,self.lib,self.wm
        B,H,K=cfg.pi_batch,cfg.horizon,self.tcfg.future;dev=self.device
        s0,xi,frame,hist,prev=self.buf.sample_starts(B,self.tcfg.history,task)
        with torch.no_grad():
            raw=frame[:,None]+torch.arange(H+K+1,device=dev)
            idx=lib.clamp_to_clip(raw,frame);valid=(raw==idx)[:,:H+1].float()
            ref=lib.frame(idx)
            rkin=self.env.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
        def at(t):return {k:v[:,t] for k,v in ref.items()}
        def future(t):return {k:v[:,t+1:t+1+K] for k,v in ref.items()}
        def kat(t):return tuple(v[:,t] for v in rkin)
        for p in wm.net.parameters():p.requires_grad_(False)
        s=flat_to_state(s0);hist=hist+task.sample_obs_noise((B,self.tcfg.history))
        an=task.analytics_fast(s,at(0),kat(0),future(0),prev,prev,xi[:,1:1+NJ])
        alive=torch.ones(B,device=dev);J=torch.zeros(B,device=dev);context=xi
        trust=torch.zeros(B,device=dev);residual_sq=0.;terms={}
        for t in range(H):
            physical=self.context_at(xi,t,self.tcfg.dt)
            xt=torch.cat([physical,context[:,21:]],-1)
            a,prior=self.policy.with_prior(hist.flatten(1),an['goal'])
            delta=run_mlp(wm.net,wm.in_norm(wm.inputs(s,a,xt,an['feats'])))*wm.out_norm.std+wm.out_norm.mean
            s1=apply_delta_c(s,delta)
            an=task.transition_fast(s,s1,at(t+1),kat(t+1),future(t+1),a,prev,xt)
            mask=valid[:,t+1]
            J=J+cfg.gamma**t*alive*an['r']*mask
            correction=(a-prior)*.7
            trust=trust+correction.square().mean(-1)/.04**2*mask
            residual_sq=residual_sq+correction.detach().square().mean()/H
            for key,v in an['terms'].items():terms[key]=terms.get(key,0.)+v.detach().mean()/H
            alive=alive*(1-an['soft'])
            context=task.advance_context(s1,idx[:,t+1],xt)
            obs=an['obs']+task.sample_obs_noise((B,))
            hist=torch.cat([hist[:,1:],obs[:,None]],1);s,prev=s1,a
        loss=-J.mean()/H+self.teacher_weight*trust.mean()/H
        self.opt_pi.zero_grad(set_to_none=True);loss.backward()
        params=[p for g in self.opt_pi.param_groups for p in g['params']]
        self.sync_grads(params);norm=torch.nn.utils.clip_grad_norm_(params,.5)
        for group in self.opt_pi.param_groups:group['lr']=base.lr_at(cfg,self.it)*self.actor_lr_scale*.1*group.get('lr_multiplier',1.)
        self.opt_pi.step()
        for p in wm.net.parameters():p.requires_grad_(True)
        stats=dict(pi_J=float(J.mean()),pi_loss=float(loss),pi_gnorm=float(norm),img_survival=float(alive.mean()),
                   teacher_action_rms_rad=float(residual_sq.sqrt()),teacher_loss=float(trust.mean()/H),
                   actor_lr=base.lr_at(cfg,self.it)*self.actor_lr_scale*.1,
                   load_min_kg=float(self.env.loads.min()),load_max_kg=float(self.env.loads.max()),
                   load_mean_kg=float(self.env.loads.mean()),pushed_transitions=self.env.pushed_transitions)
        stats.update({f'img_r_{k}':float(v) for k,v in terms.items()})
        return stats

    @torch.no_grad()
    def run_eval(self):
        out=Path(self.cfg.out);n=self.eval_n
        # Immutable evaluation provenance, even when this actor is rejected.
        checkpoint=out/f'evaluated_{self.it:07d}.pt';self.save(str(checkpoint))
        planned,_=schedule(n,self.lib.num_frames-1,.02,20261110)
        results=[];rollouts={}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for kg,path in self.models.items():
                result,arr,_=run_condition(self.policy,self.lib,path,n,20261109,planned,True)
                result.update(payload_kg=kg,it=self.it,source_iteration=self.source_iteration,model_path=str(path),
                              force_range_n=[8,16],pushes_per_trial=5,policy_frozen=True,checkpoint=str(checkpoint))
                results.append(result);rollouts[kg]=arr
        worst=min(r['success_rate'] for r in results if r['payload_kg']>0)
        avg=sum(r['success_rate'] for r in results)/4
        cycles=sum(r['cycle_success_rate'] for r in results)/4
        drift=sum((r['return_metrics_completed_trials']['com_position_std_norm'] or 1.) for r in results)/4
        quality=1000*worst+100*avg+10*cycles-drift
        # Baseline completes every trial. Reject loss of the full movement, even
        # when the new shaped reward or value prediction increases.
        reasons=[]
        for r in results:
            if r['completion_rate']<.95:reasons.append(f"{r['payload_kg']}kg_completion_below_95pct")
        if results[0]['success_rate']<.95:reasons.append('unloaded_stability_regressed')
        if results[1]['success_rate']<.75:reasons.append('5kg_stability_regressed')
        if results[2]['cycle_success_rate']<.90:reasons.append('10kg_full_cycle_regressed')
        if results[3]['cycle_success_rate']<.80:reasons.append('15kg_full_cycle_regressed')
        accepted=not reasons
        summary=dict(it=self.it,source_iteration=self.source_iteration,wall_seconds=time.time()-START,train_seconds=self.train_time,
                     worst_loaded_success=worst,quality=quality,accepted=accepted,guard_reasons=reasons,
                     conditions=[compact(r) for r in results],guard_rollbacks=self.guard_rollbacks,
                     checkpoint=str(checkpoint),pass_target='>=95% plus nominal pass at every load, twice')
        with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        (out/'latest_eval.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('RECOVERY_EVAL '+json.dumps(dict(it=self.it,accepted=accepted,guard_reasons=reasons,
                                               loads=[dict(kg=r['payload_kg'],strict=round(n*r['success_rate']),complete=round(n*r['completion_rate']),trials=n) for r in results])),flush=True)
        if self.it==0:
            assert accepted,'Source actor did not pass baseline movement checks'
        if accepted:
            self.save(str(self.safe_path))
            (out/'safe_eval.json').write_text(json.dumps(summary,indent=2)+'\n')
            if quality>self.best_quality:
                self.best_quality=quality;self.save(str(out/'best.pt'))
                (out/'best_eval.json').write_text(json.dumps(summary,indent=2)+'\n')
                for result in results:
                    kg=result['payload_kg'];d=out/'best_trials'/f'{kg}kg_push';d.mkdir(parents=True,exist_ok=True)
                    (d/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
                    np.savez_compressed(d/'rollout.npz',**rollouts[kg],dt=.02)
        else:
            safe=torch.load(self.safe_path,map_location=self.device,weights_only=False)
            self.policy.load_state_dict(safe['policy']);self.opt_pi.load_state_dict(safe['opt_pi'])
            self.wm.load_state_dict(safe['wm']);self.opt_wm.load_state_dict(safe['opt_wm'])
            self.actor_lr_scale*=.5;self.guard_rollbacks+=1;self.passed=0
            print('REGRESSION_ROLLBACK '+json.dumps(dict(rejected_iteration=self.it,restored_iteration=safe['it'],actor_lr_scale=self.actor_lr_scale)),flush=True)
            self.save(str(out/'after_rollback.pt'))
            if self.guard_rollbacks>=4:
                (out/'GUARD_STOP').write_text('Four rejected updates; safe actor preserved. Requires diagnosis.\n')
                self.cfg.iterations=self.it
        good=accepted and all(r['success_rate']>=.95 and r['nominal_success'] for r in results)
        self.passed=self.passed+1 if good else 0
        if self.passed>=2:
            print('PAYLOAD_STABILITY_TARGET_REACHED',flush=True);self.cfg.iterations=self.it


if __name__=='__main__':
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);first=(0 if gpu<4 else 16)+(gpu%4)*4
    os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
    cfg=base.parse_args();assert not cfg.use_critic
    trainer=RecoveryTrainer(cfg)
    Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,gpu=gpu,cfg=dataclasses.asdict(cfg),**trainer._extra_state()),indent=2)+'\n')
    trainer.run_eval()
    trainer.run()
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,
                                                          train_seconds=trainer.train_time,iterations=trainer.it),indent=2)+'\n')
