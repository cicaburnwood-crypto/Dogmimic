"""Continue our B2 actor/WM under variable physical payloads and transient pushes."""
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from supermimic import train as base
from supermimic.networks import WorldModel,run_mlp,apply_delta_c,state_delta_c,state_error
from supermimic.rotations import heading_mat,mtv
from supermimic.task import NJ,flat_to_state
from supermimic.replay import ReplayBuffer
from train import SkillTrainer
from train_cycle import CycleTrainer
from cycle_env import CycleLibrary
from evaluate_payloads import build_model,run_condition,compact
from evaluate_pushes import schedule
from payload_env import XI_DIM,make_payload_env

HERE=Path(__file__).resolve().parent
START=time.time()
base.MotionLibrary=CycleLibrary
base.XI_DIM=XI_DIM  # This process only; actor inputs retain their original shape.


class PayloadWorldModel(WorldModel):
    @staticmethod
    def canonical_context(s,xi):
        # The original dynamics features/deltas remove global heading. Express
        # external forces in the same heading frame so yaw is not hidden context.
        return torch.cat([xi[...,:17],mtv(heading_mat(s['rot']),xi[...,17:20]),xi[...,20:]],-1)

    def inputs(self,s,a,xi,feats=None):
        return super().inputs(s,a,self.canonical_context(s,xi),feats)


base.WorldModel=PayloadWorldModel


class TimedContextReplay(ReplayBuffer):
    def sample_windows(self,batch,k):
        assert self.size>=k
        env=torch.randint(0,self.n,(batch,),device=self.device)
        age0=torch.randint(k-1,self.size,(batch,),device=self.device)
        idx=self._age_index(age0[:,None]-torch.arange(k,device=self.device))
        e=env[:,None].expand(-1,k)
        ep=self.ep[idx,e]
        mask=(ep==ep[:,:1]).all(-1)&(ep[:,0]>=0)
        # Forces can start/end within a WM training window. Never repeat the
        # first transition's force context across all k targets.
        return self.s[idx[:,0],env],self.a[idx,e],self.s1[idx,e],self.xi[idx,e],mask


def expand_world_model(wm,old):
    state=wm.state_dict()
    old_dim=old['in_norm.mean'].numel()
    for key,value in old.items():
        if key=='net.0.weight':
            state[key].zero_();state[key][:,:old_dim].copy_(value)
        elif key in ('in_norm.mean','in_norm.var'):
            state[key][:old_dim].copy_(value)
        else:
            state[key].copy_(value)
    # Explicit normalized units keep the new privileged inputs well conditioned.
    # Count is reset to a moderate prior so changing loaded dynamics can adapt.
    state['in_norm.mean'][old_dim:]=torch.tensor([2/3,0,0,0,0],device=wm.in_norm.mean.device)
    state['in_norm.var'][old_dim:]=torch.tensor([1/27,.05,.05,.05,.01],device=wm.in_norm.mean.device)
    state['in_norm.count'].fill_(100000.)
    state['out_norm.count'].fill_(100000.)
    wm.load_state_dict(state)
    assert torch.equal(wm.net[0].weight[:,:old_dim],old['net.0.weight'])
    assert torch.count_nonzero(wm.net[0].weight[:,old_dim:])==0


class PayloadTrainer(CycleTrainer):
    def __init__(self,cfg,device='cuda'):
        assert not cfg.resume
        self.source_path=Path(os.environ['B2_INIT_CHECKPOINT'])
        self.source_hash=hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        SkillTrainer.__init__(self,cfg,device)
        ck=torch.load(self.source_path,map_location=self.device,weights_only=False)
        assert ck['task_kind']=='b2_stable_full_cycle_v2'
        self.policy.load_state_dict(ck['policy'])
        expand_world_model(self.wm,ck['wm'])
        self.source_iteration=int(ck['it'])+int(ck.get('source_iteration',0))
        self.best_quality=-float('inf');self.passed=0
        self.models={}
        (Path(cfg.out)/'models').mkdir(exist_ok=True)
        for kg in (0,5,10,15):
            path=Path(cfg.out)/'models'/f'b2_payload_{kg}kg.xml'
            build_model(kg,path);self.models[kg]=path
        print('INITIALIZED_FROM '+str(self.source_path)+' (actor + expanded WM, fresh critic/replay/optimizers)',flush=True)

    def _make_env(self):
        env=make_payload_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma)
        self.tcfg=env.tcfg
        return env

    def _make_buffer(self):
        return TimedContextReplay(self.cfg.num_envs,self.cfg.buffer_steps,XI_DIM,self.device)

    def _extra_state(self):
        return dict(task_kind='b2_payload_stable_cycle_v3',robot='B2',source_checkpoint=str(self.source_path),
                    source_sha256=self.source_hash,source_iteration=getattr(self,'source_iteration',8000),
                    initialization='own stable full-cycle v2 actor and world model',external_pretraining=False,
                    fresh_critic_replay_optimizers=True,payload_range_kg=[5,15],payload_distribution='continuous uniform, resampled each episode',
                    pushes=dict(force_range_n=[8,16],duration_s=[.12,.24],events_per_full_cycle=5,axes='world XYZ',application='trunk COM'),
                    xi_layout=['friction','encoder_bias[12]','COM_offset[3]','payload_kg/15','force_xyz/16','remaining_pulse_seconds'],
                    task_cfg=dataclasses.asdict(self.tcfg),policy_architecture=dict(encoder=[512,256,128],decoder=[512,256,128]),
                    invalid_physics_transitions=int(self.env.invalid_physics_count),payload_resamples=self.env.randomizations,
                    pushed_transitions=self.env.pushed_transitions,wall_seconds=time.time()-START)

    def _should_stop(self):
        return Path(self.cfg.out,'STOP').exists() or time.time()-START>=self.cfg.wall_hours*3600

    def update_wm(self):
        cfg,wm=self.cfg,self.wm
        stats=dict(wm_loss=0.,wm_err1=0.,wm_errk=0.)
        for bi in range(cfg.wm_batches):
            s0,A,S1,xi,mask=self.buf.sample_windows(cfg.wm_batch,cfg.k)
            if bi==0 and self.updates%cfg.norm_update_every==0:
                with torch.no_grad():
                    prev=flat_to_state(s0);xs=[];ds=[]
                    for j in range(cfg.k):
                        nxt=flat_to_state(S1[:,j])
                        xs.append(wm.inputs(prev,A[:,j],xi[:,j])[mask])
                        ds.append(state_delta_c(prev,nxt)[mask]);prev=nxt
                    wm.in_norm.update(torch.cat(xs));wm.out_norm.update(torch.cat(ds))
            loss,e1,ek=self._wm_loss(s0,A,S1,xi,mask.float())
            self.opt_wm.zero_grad(set_to_none=True);loss.backward()
            self.sync_grads(list(wm.net.parameters()))
            torch.nn.utils.clip_grad_norm_(wm.net.parameters(),1.);self.opt_wm.step()
            for key,v in zip(stats,(loss.detach(),e1,ek)):stats[key]+=v/cfg.wm_batches
        return {k:float(v) for k,v in stats.items()}

    def wm_window_loss(self,s0,A,S1,xi,m):
        s=flat_to_state(s0);loss=0.;errors=[]
        for j in range(self.cfg.k):
            s=self.wm.step(s,A[:,j],xi[:,j])
            e=(state_error(s,flat_to_state(S1[:,j]))/self.wm.out_norm.std).square().mean(-1)
            loss=loss+e;errors.append(e.detach())
        denom=m.sum().clamp_min(1)
        return (loss*m).sum()/denom,(errors[0]*m).sum()/denom,(errors[-1]*m).sum()/denom

    @staticmethod
    def context_at(xi,t,dt):
        remaining=(xi[...,20:21]-t*dt).clamp_min(0.)
        force=xi[...,17:20]*(remaining>1e-6)
        return torch.cat([xi[...,:17],force,remaining],-1)

    def update_policy(self):
        cfg,task,lib,wm=self.cfg,self.task,self.lib,self.wm
        B,H,K=cfg.pi_batch,cfg.horizon,self.tcfg.future;dev=self.device
        s0,xi,frame,hist,prev=self.buf.sample_starts(B,self.tcfg.history,task)
        with torch.no_grad():
            raw=frame[:,None]+torch.arange(H+K+1,device=dev)
            idx=lib.clamp_to_clip(raw,frame);valid=(raw==idx)[:,:H+1].float()
            ref=lib.frame(idx)
            rkin=self.env.kin_c(ref['pos'],ref['rot'],ref['q'],ref['v'],ref['w'],ref['qd'])
            rem=lib.remaining(idx[:,:H+1]).float().clamp(max=100)/100.
        def at(t):return {k:v[:,t] for k,v in ref.items()}
        def future(t):return {k:v[:,t+1:t+1+K] for k,v in ref.items()}
        def kat(t):return tuple(v[:,t] for v in rkin)
        for p in wm.net.parameters():p.requires_grad_(False)
        s=flat_to_state(s0);hist=hist+task.sample_obs_noise((B,self.tcfg.history))
        an=task.analytics_fast(s,at(0),kat(0),future(0),prev,prev,xi[:,1:1+NJ])
        c=torch.ones(B,device=dev);J=torch.zeros(B,device=dev)
        rec={k:[] for k in ('x','r','d','m','c')};terms={}
        std,mean=wm.out_norm.std,wm.out_norm.mean
        for t in range(H):
            xt=self.context_at(xi,t,self.tcfg.dt)
            goal,feats=an['goal'],an['feats']
            rec['x'].append(self.critic_input(feats,goal,wm.canonical_context(s,xt),rem[:,t]).detach())
            a=self.policy(hist.flatten(1),goal)
            delta=run_mlp(wm.net,wm.in_norm(wm.inputs(s,a,xt,feats)))*std+mean
            s1=apply_delta_c(s,delta)
            an=task.transition_fast(s,s1,at(t+1),kat(t+1),future(t+1),a,prev,xt)
            r,d=an['r'],an['soft'];mask=valid[:,t+1]
            J=J+cfg.gamma**t*c*r*mask
            rec['r'].append((r*mask).detach());rec['d'].append(d.detach());rec['m'].append(mask);rec['c'].append(c.detach())
            for k,v in an['terms'].items():terms[k]=terms.get(k,0.)+v.detach().mean()/H
            c=c*(1-d)
            obs=an['obs']+task.sample_obs_noise((B,))
            hist=torch.cat([hist[:,1:],obs[:,None]],1);s,prev=s1,a
        xH=self.critic_input(an['feats'],an['goal'],wm.canonical_context(s,self.context_at(xi,H,self.tcfg.dt)),rem[:,H])
        if cfg.use_critic:J=J+cfg.gamma**H*c*self.critic(xH)*valid[:,H]
        loss=-J.mean()/H
        self.opt_pi.zero_grad(set_to_none=True);loss.backward()
        params=[p for g in self.opt_pi.param_groups for p in g['params']]
        self.sync_grads(params);gnorm=torch.nn.utils.clip_grad_norm_(params,1.)
        torch.nn.utils.clip_grad_value_(params,.1);self.opt_pi.step()
        for p in wm.net.parameters():p.requires_grad_(True)
        self.critic.zero_grad(set_to_none=True)
        stats=dict(pi_J=J.mean().item(),pi_loss=loss.item(),pi_gnorm=gnorm.item(),img_survival=c.mean().item(),
                   load_min_kg=float(self.env.loads.min()),load_max_kg=float(self.env.loads.max()),
                   load_mean_kg=float(self.env.loads.mean()),pushed_transitions=self.env.pushed_transitions)
        stats.update({f'img_r_{k}':v.item() for k,v in terms.items()})
        if cfg.use_critic:stats.update(self.update_critic(rec,xH.detach(),valid))
        return stats

    @torch.no_grad()
    def run_eval(self):
        out=Path(self.cfg.out);n=64
        planned,_=schedule(n,self.lib.num_frames-1,.02,20261110)
        results=[];rollouts={}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for kg,path in self.models.items():
                result,arr,_=run_condition(self.policy,self.lib,path,n,20261109,planned,True)
                result.update(payload_kg=kg,it=self.it,source_iteration=self.source_iteration,model_path=str(path),
                              force_range_n=[8,16],pushes_per_trial=5,policy_frozen=True,checkpoint=str(out/'best.pt'))
                results.append(result);rollouts[kg]=arr
        worst=min(r['success_rate'] for r in results if r['payload_kg']>0)
        avg=sum(r['success_rate'] for r in results)/len(results)
        cycles=sum(r['cycle_success_rate'] for r in results)/len(results)
        drift=sum((r['return_metrics_completed_trials']['com_position_std_norm'] or 1.) for r in results)/len(results)
        quality=1000*worst+100*avg+10*cycles-drift
        summary=dict(it=self.it,source_iteration=self.source_iteration,wall_seconds=time.time()-START,
                     train_seconds=self.train_time,worst_loaded_success=worst,quality=quality,
                     conditions=[compact(r) for r in results],pass_target='>=95% and nominal pass at 0/5/10/15 kg with pushes, twice')
        with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        print('PAYLOAD_EVAL '+json.dumps(summary),flush=True)
        (out/'latest_eval.json').write_text(json.dumps(summary,indent=2)+'\n')
        if quality>self.best_quality:
            self.best_quality=quality;self.save(str(out/'best.pt'))
            (out/'best_eval.json').write_text(json.dumps(summary,indent=2)+'\n')
            for result in results:
                kg=result['payload_kg'];d=out/'best_trials'/f'{kg}kg_push';d.mkdir(parents=True,exist_ok=True)
                (d/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
                np.savez_compressed(d/'rollout.npz',**rollouts[kg],dt=.02)
        good=all(r['success_rate']>=.95 and r['nominal_success'] for r in results)
        self.passed=self.passed+1 if good else 0
        if self.passed>=2:
            print('RANDOM_PAYLOAD_AND_PUSH_STABILITY_TARGET_REACHED',flush=True)
            self.cfg.iterations=self.it


if __name__=='__main__':
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);first=(0 if gpu<4 else 16)+(gpu%4)*4
    os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
    cfg=base.parse_args();trainer=PayloadTrainer(cfg)
    Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,gpu=gpu,cfg=dataclasses.asdict(cfg),**trainer._extra_state()),indent=2)+'\n')
    if not os.environ.get('B2_SKIP_INITIAL_EVAL'):trainer.run_eval()
    prefit=int(os.environ.get('B2_WM_PREFIT_UPDATES','100'))
    if prefit:
        for _ in range(cfg.warmup_steps):trainer.collect()
        cfg.warmup_steps=0
        for i in range(prefit):
            trainer.updates=i
            stats=trainer.update_wm()
            assert all(np.isfinite(v) for v in stats.values()),stats
            if (i+1)%25==0:print('WM_PREFIT '+json.dumps(dict(update=i+1,**stats)),flush=True)
        trainer.updates=0
        trainer.save(str(Path(cfg.out,'wm_prefit.pt')))
        print('WM_PREFIT_COMPLETE; beginning joint actor/WM training',flush=True)
    trainer.run()
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,
                                                          train_seconds=trainer.train_time,iterations=trainer.it),indent=2)+'\n')
