"""Continue B2 world-model / BPTT learning with transition acceleration costs."""
import dataclasses,json,os,time,hashlib
from pathlib import Path
import numpy as np
import torch
from supermimic import train as base
from supermimic.networks import run_mlp,apply_delta_c
from supermimic.task import NJ,flat_to_state
from train import SkillTrainer
from cycle_env import CycleLibrary,make_cycle_env
from evaluate_cycle import evaluate

START=time.time()
base.MotionLibrary=CycleLibrary

class CycleTrainer(SkillTrainer):
    def __init__(self,cfg,device='cuda'):
        assert not cfg.resume
        self.source_path=Path(os.environ['B2_INIT_CHECKPOINT'])
        self.source_hash=hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        super().__init__(cfg,device)
        ck=torch.load(self.source_path,map_location=self.device,weights_only=False)
        assert ck['robot']=='B2' and ck['it']==4000
        self.policy.load_state_dict(ck['policy']);self.wm.load_state_dict(ck['wm'])
        self.source_iteration=ck['it']
        # Replay, adaptive sampling, critic, and all optimizer state start fresh.
        # The value function from a different reward must not seed this critic.
        self.best_quality=-float('inf');self.passed=0
        print('INITIALIZED_FROM '+str(self.source_path)+' (policy + world model only)',flush=True)
    def _make_env(self):
        env=make_cycle_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma)
        self.tcfg=env.tcfg;return env
    def _extra_state(self):
        return dict(task_kind='b2_stable_full_cycle_v2',robot='B2',reference_source='our original procedural full-cycle target',
                    initialization='continued_from_own_b2_iteration_4000',source_checkpoint=str(self.source_path),source_sha256=self.source_hash,
                    source_iteration=4000,external_pretraining=False,fresh_critic_and_replay=True,
                    task_cfg=dataclasses.asdict(self.tcfg),policy_architecture=dict(encoder=[512,256,128],decoder=[512,256,128]),
                    invalid_physics_transitions=int(self.env.invalid_physics_count),wall_seconds=time.time()-START)
    def _should_stop(self):
        return Path(self.cfg.out,'STOP').exists() or time.time()-START>=self.cfg.wall_hours*3600
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
        # Initial features need no fictitious previous transition or acceleration.
        an=task.analytics_fast(s,at(0),kat(0),future(0),prev,prev,xi[:,1:1+NJ])
        c=torch.ones(B,device=dev);J=torch.zeros(B,device=dev)
        rec={k:[] for k in ('x','r','d','m','c')};terms={}
        std,mean=wm.out_norm.std,wm.out_norm.mean
        for t in range(H):
            goal,feats=an['goal'],an['feats']
            rec['x'].append(self.critic_input(feats,goal,xi,rem[:,t]).detach())
            a=self.policy(hist.flatten(1),goal)
            delta=run_mlp(wm.net,wm.in_norm(torch.cat([feats,a,xi],-1)))*std+mean
            s1=apply_delta_c(s,delta)
            an=task.transition_fast(s,s1,at(t+1),kat(t+1),future(t+1),a,prev,xi)
            r,d=an['r'],an['soft'];mask=valid[:,t+1]
            J=J+cfg.gamma**t*c*r*mask
            rec['r'].append((r*mask).detach());rec['d'].append(d.detach());rec['m'].append(mask);rec['c'].append(c.detach())
            for k,v in an['terms'].items():terms[k]=terms.get(k,0.)+v.detach().mean()/H
            c=c*(1-d)
            obs=an['obs']+task.sample_obs_noise((B,))
            hist=torch.cat([hist[:,1:],obs[:,None]],1);s,prev=s1,a
        xH=self.critic_input(an['feats'],an['goal'],xi,rem[:,H])
        if cfg.use_critic:J=J+cfg.gamma**H*c*self.critic(xH)*valid[:,H]
        loss=-J.mean()/H
        self.opt_pi.zero_grad(set_to_none=True);loss.backward()
        params=[p for g in self.opt_pi.param_groups for p in g['params']]
        self.sync_grads(params);gnorm=torch.nn.utils.clip_grad_norm_(params,1.)
        torch.nn.utils.clip_grad_value_(params,.1);self.opt_pi.step()
        for p in wm.net.parameters():p.requires_grad_(True)
        self.critic.zero_grad(set_to_none=True)
        stats=dict(pi_J=J.mean().item(),pi_loss=loss.item(),pi_gnorm=gnorm.item(),img_survival=c.mean().item())
        stats.update({f'img_r_{k}':v.item() for k,v in terms.items()})
        if cfg.use_critic:stats.update(self.update_critic(rec,xH.detach(),valid))
        return stats
    @torch.no_grad()
    def run_eval(self):
        if not hasattr(self,'cycle_eval_env'):self.cycle_eval_env=make_cycle_env(self.lib,self.device,32,evaluate=True)
        result,trajectory=evaluate(self.policy,self.cycle_eval_env,seed=20261101+self.it)
        result.update(it=self.it,source_iteration=4000,wall_seconds=time.time()-START,train_seconds=self.train_time)
        with open(Path(self.cfg.out,'eval.jsonl'),'a') as f:f.write(json.dumps(result)+'\n')
        print('CYCLE_EVAL '+json.dumps({k:v for k,v in result.items() if k!='trials_detail'}),flush=True)
        acc=(result['stand_metrics_completed_trials']['com_acc_rms'] or 100.)
        quality=1000*result['success_rate']+100*result['cycle_success_rate']+10*result['stable_hindstand_rate']+result['mean_norm_return']-.1*acc
        if quality>self.best_quality:
            self.best_quality=quality;self.save(str(Path(self.cfg.out,'best.pt')))
            Path(self.cfg.out,'best_eval.json').write_text(json.dumps(result,indent=2)+'\n')
            np.savez_compressed(Path(self.cfg.out,'best_rollout.npz'),**trajectory,dt=.02)
        self.passed=self.passed+1 if result['success_rate']>=.9 and result['nominal_success'] else 0
        if self.passed>=2:
            print('FULL_CYCLE_AND_STABILITY_TARGET_REACHED',flush=True);self.cfg.iterations=self.it

if __name__=='__main__':
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);first=(0 if gpu<4 else 16)+(gpu%4)*4
    os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
    cfg=base.parse_args();trainer=CycleTrainer(cfg)
    Path(cfg.out,'run_meta.json').write_text(json.dumps(dict(start_unix=START,gpu=gpu,cfg=dataclasses.asdict(cfg),**trainer._extra_state()),indent=2)+'\n')
    trainer.run_eval()  # Baseline of old weights on the new reward and full cycle.
    trainer.run()
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,train_seconds=trainer.train_time,iterations=trainer.it),indent=2)+'\n')
