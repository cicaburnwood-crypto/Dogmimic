import sys,copy,os,json,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
# Resolve this candidate environment even when imported by an external audit.
sys.path.insert(0,str(HERE))
sys.path.insert(1,str(ROOT/'experiments/b2_actuator_robust'))
from train_robust import Trainer,base,START,make_env,evaluate,SPEED,POWER
from train_recovery import torch,flat_to_state,run_mlp,apply_delta_c,NJ,RecoveryWorldModel
from supermimic.networks import WorldModel
from actuator_filter import advance_context as advance_filter,COMMAND_RATE,COMMAND_POWER
class FilteredWM(WorldModel):
    def __init__(self,task,feat_dim,xi_dim):
        assert xi_dim==43
        super().__init__(task,feat_dim,34)
    def inputs(self,s,a,xi,feats=None):
        canonical=RecoveryWorldModel.canonical_context(s,xi)
        target=advance_filter(xi,a,self.task.q0,self.task.action_scale,self.task.qmin,self.task.qmax)[...,31:43]
        effective_action=(target-self.task.q0+xi[...,1:13])/self.task.action_scale
        return super().inputs(s,effective_action,torch.cat([canonical[...,:21],xi[...,30:43]],-1),feats)
base.WorldModel=FilteredWM
base.XI_DIM=43
class ProgressTrainer(Trainer):
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
                progress=dict(it=self.it,full=full,cases_done=len(results),cases_total=len(cases),current_case=dict(kg=kg,friction=fr,mass_scale=m,asymmetric=asym),updated=time.time())
                tmp=out/'eval_progress.tmp';tmp.write_text(json.dumps(progress));tmp.replace(out/'eval_progress.json')
        good=all(r['success_rate']>=.95 and r['nominal_success'] and r['actuator_pass'] for r in results)
        summary=dict(it=self.it,wall_seconds=time.time()-START,full=full,checkpoint=str(cp),sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),all_passed=good,conditions=results)
        (out/'latest_eval.json').write_text(json.dumps(summary,indent=2));(out/'eval').mkdir(exist_ok=True);(out/'eval'/f'it_{self.it:07d}_{int(full)}.json').write_text(json.dumps(summary))
        with (out/'eval.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        score=(int(good),min(r['completion_rate'] for r in results[:4]),min(r['success_rate'] for r in results[:4]),sum(r['success_rate'] for r in results[:4]),-max(r['peak_speed'] for r in results[:4]));print('PHYSICAL_EVAL',self.it,full,[(r['kg'],round(64*r['success_rate']),r['peak_speed']) for r in results],flush=True)
        if score>self.best_quality:
            self.best_quality=score;self.save(str(out/'best.pt'));summary['best_sha256']=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest();(out/'best_eval.json').write_text(json.dumps(summary,indent=2))
        if good and not full:return self.run_eval(full=True)
        self.passed=self.passed+1 if full and good else 0
        if self.passed>=2:
            self.save(str(out/'accepted.pt'));(out/'TARGET_MET.json').write_text(json.dumps(summary));self.cfg.iterations=self.it

class GuardedTrainer(ProgressTrainer):
    def __init__(self,cfg,device='cuda'):
        if os.environ.get('B2_WM_SOURCE'):
            raise ValueError('Filtered candidate requires a fresh world model')
        super().__init__(cfg,device)
        policy_source=os.environ.get('B2_POLICY_SOURCE')
        if policy_source:
            cp=Path(policy_source);ck=torch.load(cp,map_location=device,weights_only=False)
            self.policy.load_state_dict(ck['policy']);self.source_path=cp;self.source_hash=hashlib.sha256(cp.read_bytes()).hexdigest();self.source_iteration=int(ck['it'])
        self.frozen_anchor=copy.deepcopy(self.policy).requires_grad_(False).eval()
        self.actor_lr_scale=.2;self.teacher_weight=.05
        source=os.environ.get('B2_WM_SOURCE')
        if source:raise ValueError('Filtered candidate requires fresh 34-dimensional WM; B2_WM_SOURCE must be unset')
        self.best_quality=(-1.,-1.,-1.,-1.,float("-inf"))
        self.baseline=None;self.declines=0
    def _extra_state(self):
        d=super()._extra_state();d.update(task_kind='b2_balance_repair',world_model_action='clamped_target_normalized_no_slew',fixed_full_policy_anchor=True,command_rate_rad_s=None,artificial_target_slew_removed=True,command_power_w=COMMAND_POWER,context_dim=43,world_model_initialization='fresh',runtime_requires_same_filter=False,actuator_loss='worst_joint_endpoint_speed_and_enveloped_power',world_model_source=os.environ.get('B2_WM_SOURCE'),actor_lr_scale=getattr(self,'actor_lr_scale',.2));return d
    def update_policy(self):
        if self.it<int(os.environ.get('B2_WM_PREFIT','2000')):return dict(actor_prefit=1.)
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
            a=self.policy(hist.flatten(1),an['goal'])
            # Freeze reference parameters, not its derivatives with respect to imagined state.
            prior=self.frozen_anchor(hist.flatten(1),an['goal'])
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
            next_filter=advance_filter(xt,a,task.q0,task.action_scale,task.qmin,task.qmax)[:,31:43]
            context=torch.cat([context[:,:31],next_filter],-1)
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


    def run_eval(self,full=False):
        super().run_eval(full)
        p=Path(self.cfg.out);d=json.loads((p/'latest_eval.json').read_text())
        rates=[r['completion_rate'] for r in d['conditions'][:4]]
        if self.baseline is None:self.baseline=rates
        regressed=self.it>0 and any(a<b-.1 for a,b in zip(rates,self.baseline))
        self.declines=self.declines+1 if regressed else 0
        if self.declines>=2:
            (p/'GUARD_STOP.json').write_text(json.dumps(dict(it=self.it,reason='two consecutive physical completion regressions >10 percentage points',baseline=self.baseline,current=rates)))
            (p/'STOP').write_text('Physical regression; preserve best and diagnose.\n')
if __name__=='__main__':
    torch.set_num_threads(2);os.sched_setaffinity(0,set(range(24,28)))
    cfg=base.parse_args();t=GuardedTrainer(cfg)
    smoke=os.environ.get('B2_SMOKE')=='1'
    before={k:v.detach().clone() for k,v in t.frozen_anchor.state_dict().items()}
    if not smoke:t.run_eval()
    t.run()
    if not smoke:t.run_eval(full=True)
    if smoke:
        assert all(torch.equal(v,t.frozen_anchor.state_dict()[k]) for k,v in before.items())
        assert any(not torch.equal(v,t.policy.state_dict()[k]) for k,v in before.items())
        assert all(torch.isfinite(v).all() for v in t.policy.state_dict().values())
        assert all(p.grad is None and not p.requires_grad for p in t.frozen_anchor.parameters())
        Path(cfg.out,'smoke_audit.json').write_text(json.dumps(dict(passed=True,fixed_anchor_unchanged=True,actor_updated=True,updates=t.updates)))
    Path(cfg.out,'EVAL_DONE.json').write_text(json.dumps(dict(it=t.it,wall_seconds=time.time()-START)))
