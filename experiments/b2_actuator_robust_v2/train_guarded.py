import sys,copy,os,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(1,str(ROOT/'experiments/b2_actuator_robust'))
from train_robust import Trainer,base,START
from train_recovery import torch,flat_to_state,run_mlp,apply_delta_c,NJ
class GuardedTrainer(Trainer):
    def __init__(self,cfg,device='cuda'):
        super().__init__(cfg,device)
        self.frozen_anchor=copy.deepcopy(self.policy).requires_grad_(False).eval()
        self.actor_lr_scale=.2;self.teacher_weight=.05
        source=os.environ.get('B2_WM_SOURCE')
        if source:self.wm.load_state_dict(torch.load(source,map_location=device,weights_only=False)['wm'])
        self.baseline=None;self.declines=0
    def _extra_state(self):
        d=super()._extra_state();d.update(task_kind='b2_actuator_robust_v2',fixed_full_policy_anchor=True,world_model_source=os.environ.get('B2_WM_SOURCE'),actor_lr_scale=getattr(self,'actor_lr_scale',.2));return d
    def update_policy(self):
        if self.it<int(os.environ.get('B2_WM_PREFIT','1000')):return dict(actor_prefit=1.)
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
            with torch.no_grad():prior=self.frozen_anchor(hist.flatten(1),an['goal'])
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
    before=next(t.frozen_anchor.teacher.decoder.parameters()).detach().clone()
    if not smoke:t.run_eval()
    t.run()
    if not smoke:t.run_eval(full=True)
    if smoke:
        assert torch.equal(before,next(t.frozen_anchor.teacher.decoder.parameters()).detach())
        assert not torch.equal(before,next(t.policy.teacher.decoder.parameters()).detach())
        Path(cfg.out,'smoke_audit.json').write_text(json.dumps(dict(passed=True,fixed_anchor_unchanged=True,actor_updated=True,updates=t.updates)))
    Path(cfg.out,'EVAL_DONE.json').write_text(json.dumps(dict(it=t.it,wall_seconds=time.time()-START)))
