"""Continue qualified first-seed C1 endpoint without resetting replay/optimizer clock."""
import argparse,dataclasses,gc,json,os,shutil,subprocess,time
from pathlib import Path
from common import ROOT,DESIGN,setup,read_json,write_json,sha256,isolated_rng
from selection import score,RegressionGuard,validate_accepted
p=argparse.ArgumentParser();p.add_argument('--robot',required=True);p.add_argument('--pair',required=True);p.add_argument('--out',required=True)
a=p.parse_args();robot=a.robot;pair=Path(a.pair);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
assert read_json(pair.parent/'pilot_comparison.json')['qualified']
report=read_json(pair/'result.json');initial_loop=report['branches']['C1']['endpoint']
assert report['seed']==1202601 and initial_loop>0
Trainer,base,_=setup(robot)
import torch
from state import TrainerSnapshot,verify_replay
torch.set_num_threads(2)
c=read_json(pair/'manifest.json')['cfg'];c.update(out=str(out),resume='',use_critic=True,eval_every=0,lr=.0001)
cfg=base.TrainCfg(**c);t=Trainer(cfg,'cuda')
restored=TrainerSnapshot.load_into(t,pair/'C1'/f'endpoint_rank{t.rank}.pt')
assert t.actor_updates==report['branches']['C1']['actor_updates']
replay=verify_replay(t.env,t.policy);assert replay['passed'],replay
del restored;gc.collect();t.core_phase='C1_qualified_long'
start=time.time();train_seconds=0.;eval_seconds=0.;last_loop=initial_loop;bad=0;history=[]
best=read_json(pair/'C1/best_eval.json');reason='iteration_budget';stale=0
guard=RegressionGuard(report['branches']['C1']['initial'])
if t.rank==0:write_json(out/'resume.json',dict(first_seed=1202601,endpoint=initial_loop,actor_updates=t.actor_updates,
    clock=t.it,world=t.world,complete_state_restored=True,physical_replay=replay))

def evaluate(cp,folder,pack='DEV-A',full=False):
    t._barrier();r=None;begin=time.time()
    if t.rank==0:
        if robot=='B2':
            from evaluation import evaluate as do_eval
            with isolated_rng(t.device):r=do_eval(robot,cp,folder,pack,full)
            gc.collect();torch.cuda.empty_cache()
        else:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='6')
            for k in ['RANK','LOCAL_RANK','WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','LOCAL_WORLD_SIZE']:env.pop(k,None)
            cmd=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'experiments/supermimic_core/evaluation.py'),
                 '--robot',robot,'--checkpoint',str(cp),'--out',str(folder),'--pack',pack]
            with open(Path(folder).with_suffix('.log'),'w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
            r=read_json(Path(folder)/'result.json')
        assert sha256(cp)==r['checkpoint_sha256']
    if t.world>1:
        import torch.distributed as dist
        objs=[r];dist.broadcast_object_list(objs,0);r=objs[0]
    t._barrier();return r,time.time()-begin

for loop in range(initial_loop+1,20001):
    if any((x/'STOP').exists() for x in (out,out.parent,out.parent.parent)):reason='user_stop';break
    if train_seconds>=9000:reason='remaining_long_training_budget';break
    if time.time()-start>=21600:reason='wall_time_guard';break
    begin=time.time();t.it=500+loop
    for opt in (t.opt_wm,t.opt_pi,t.opt_v):
        for g in opt.param_groups:g['lr']=base.lr_at(cfg,t.it)
    t.collect();t.update_credit+=.4;stats={}
    while t.update_credit>=1:
        t.update_credit-=1;stats.update(t.update_wm());stats.update(t.update_policy());t.updates+=1
    torch.cuda.synchronize();spent=time.time()-begin;train_seconds+=spent;t.train_time+=spent;last_loop=loop
    if loop%100==0 and t.rank==0:
        write_json(out/'progress.json',dict(actor_enabled_loops=loop,actor_updates=t.actor_updates,train_seconds=train_seconds,
            inherited_train_seconds=t.train_time-train_seconds,eval_seconds=eval_seconds,wm_gradient_steps=t.wm_gradient_steps,
            critic_fits=t.critic_fits,updated=time.time(),**stats))
    if loop%1000==0:
        cp=out/f'candidate_{loop:06d}.pt'
        if t.rank==0:t.save(str(cp))
        t._barrier();result,secs=evaluate(cp,out/f'eval_{loop:06d}');eval_seconds+=secs;history.append(result)
        stop_for_regression=guard.update(result,best)
        if score(result)>score(best):
            best=result;stale=0
            if t.rank==0:shutil.copy2(cp,out/'best.pt');write_json(out/'best_eval.json',best)
        else:stale+=1
        if t.rank==0:write_json(out/'latest_eval.json',result)
        if stop_for_regression:reason='persistent_physical_regression';break
        if robot=='B2':
            key_completely_failed=all(not c['actuator_pass'] for c in result['conditions'])
        else:
            key_completely_failed=any(c['passed']==0 for c in result['conditions'])
        if stale>=3 and key_completely_failed:reason='three_evaluations_without_improvement_and_complete_key_failure';break
        if result['all_passed']:reason='candidate_ready_for_confirmation';break
if t.rank==0:t.save(str(out/'last.pt'))
t._barrier()
final,secs=evaluate(out/'last.pt',out/'final_DEV_A',full=True);eval_seconds+=secs
if t.rank==0:write_json(out/'final_eval.json',final)
# Use the same immutable candidate bytes for both held-out confirmations.
accepted=False
if reason!='user_stop' and final['all_passed']:
    confirmations=[]
    for pack in ['HOLD-A','HOLD-B']:
        r,secs=evaluate(out/'last.pt',out/('confirm_'+pack),pack,True);eval_seconds+=secs;confirmations.append(r)
    accepted=validate_accepted(robot,confirmations,sha256(out/'last.pt'))
    if accepted and t.rank==0:
        shutil.copy2(out/'last.pt',out/'accepted.pt')
        write_json(out/'accepted_eval.json',dict(checkpoint_sha256=sha256(out/'accepted.pt'),confirmations=confirmations))
# Retain the physical best inherited from the pilot if no long candidate won.
if t.rank==0:
    if not (out/'best.pt').exists():shutil.copy2(pair/'C1/best.pt',out/'best.pt');write_json(out/'best_eval.json',best)
    write_json(out/'DONE',dict(reason=reason,actor_enabled_loops=last_loop,actor_updates=t.actor_updates,accepted=accepted,
        train_seconds=train_seconds,eval_seconds=eval_seconds,elapsed=time.time()-start,best_sha256=sha256(out/'best.pt')))
t._barrier()
