"""Three independent fresh-policy SuperMimic reproduction experiments."""
import os,time,json,dataclasses
from pathlib import Path
import numpy as np
import torch
from supermimic import train as base
from supermimic.networks import Policy
from supermimic.task import state_to_flat
from skill_env import make_env,IS_B2

START=time.time()
# Smaller actor is the previously tested throughput setting. World model remains
# the reproduction's 3 x 2048 deterministic dynamics model and BPTT algorithm.
base.Policy=lambda obs,goal:Policy(obs,goal,enc_hidden=(512,256,128),dec_hidden=(512,256,128))

class SkillTrainer(base.Trainer):
    def _make_env(self):
        env=make_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma)
        self.tcfg=env.tcfg
        return env
    def _extra_state(self):
        return dict(task_kind=os.environ['SM_TASK'],robot='B2' if IS_B2 else 'G1',reference_source='original procedural target',pretrained_policy=False,
                    task_cfg=dataclasses.asdict(self.tcfg),policy_architecture=dict(encoder=[512,256,128],decoder=[512,256,128]),
                    invalid_physics_transitions=int(self.env.invalid_physics_count.item()),wall_seconds=time.time()-START)
    def _should_stop(self):
        return Path(self.cfg.out,'STOP').exists() or time.time()-START>=self.cfg.wall_hours*3600
    @torch.no_grad()
    def collect(self,random=False):
        env=self.env;obs,goal=env.policy_inputs()
        self.policy.obs_norm.update(obs);self.policy.goal_norm.update(goal)
        a=self.policy(obs,goal);a=a+self.cfg.action_noise*torch.randn_like(a)
        s=state_to_flat(env.state);xi,frame,ep=env.xi.clone(),env.frame.clone(),env.ep_id.clone()
        nxt,info=env.step(a);ep[env.last_step_invalid]=-1
        self.buf.add(s,a,state_to_flat(nxt),xi,frame,ep);self.env_steps+=env.num_envs
        if 'done_ep_return' in info:
            self.ep_returns.append(info['done_ep_return'].mean().item());self.ep_norm_returns.append(info['done_norm_return'].mean().item())
            self.ep_terms.append(info['done_terminated'].float().mean().item());self.ep_lens.append(info['done_ep_len'].float().mean().item())
            self.ep_step_rewards.append((info['done_ep_return']/(info['done_ep_len']*self.tcfg.max_step_reward)).mean().item())
    @torch.no_grad()
    def run_eval(self):
        if not hasattr(self,'skill_eval_env'):
            self.skill_eval_env=make_env(self.lib,self.device,32,evaluate=True)
            self.best_quality=-float('inf');self.passed=0
        env=self.skill_eval_env;torch.manual_seed(self.cfg.seed+100000+self.it)
        ids=torch.arange(32,device=self.device);env.reset()
        env.reset_envs(ids,torch.zeros(32,device=self.device,dtype=torch.long));env._finish_resets(ids,force=True)
        # Nominal trial 0; independent small initial perturbations in the others.
        env.sim.data.qpos[1:,env.q_adr]+=.006*torch.randn(31,len(env.q_adr),device=self.device)
        env._finish_resets(ids,force=True)
        live=torch.ones(32,dtype=torch.bool,device=self.device);returns=torch.zeros(32,device=self.device)
        error=torch.zeros(32,device=self.device);frames=torch.zeros(32,device=self.device)
        traj=[];foots=[];alive=[];self.policy.eval()
        T=int(self.lib.lengths[0])-1
        for t in range(T):
            a=self.policy(*env.policy_inputs());s,info=env.step(a)
            ref=self.lib.frame(torch.full((32,),t+1,device=self.device,dtype=torch.long))
            error+=live*(s['q']-ref['q']).square().mean(-1);frames+=live
            returns+=live*info['reward']
            traj.append(state_to_flat(s).cpu().numpy());foots.append(env.task.foot_positions(env.task.body_kin(s)).cpu().numpy())
            live=live&~info['terminated'];alive.append(live.cpu().numpy())
        q=np.asarray(traj);feet=np.asarray(foots);alive=np.asarray(alive);rmse=(error/frames.clamp_min(1)).sqrt().cpu().numpy()
        complete=alive.all(0);kind=os.environ['SM_TASK']
        extras={}
        if kind=='g1_jump_full_turn':
            quat=q[:,:,3:7];w,x,y,z=np.moveaxis(quat,-1,0)
            yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),axis=0)
            turn=(yaw[-1]-yaw[0])*180/np.pi
            air=(feet[:,:, :,2].min(-1)>.09)&alive
            airtime=air.sum(0)*.02
            stable=(np.linalg.norm(q[-30:,:,36:39],axis=-1).mean(0)<.3)
            success=complete&(abs(turn-360)<30)&(airtime>.20)&stable
            extras=dict(turn_degrees=turn.tolist(),airtime_s=airtime.tolist(),stable_landing=stable.tolist())
        elif IS_B2:
            quat=q[:,:,3:7];w,x,y,z=np.moveaxis(quat,-1,0)
            up=2*(x*z-w*y)
            stance=(up>np.sin(np.deg2rad(70)))&(feet[:,:,:2,2].min(-1)>.30)&(feet[:,:,2:,2].max(-1)<.09)&alive
            hold=stance[-250:].mean(0)
            success=complete&(hold>.95)
            extras=dict(last_five_seconds_upright_fraction=hold.tolist())
        else:
            success=complete&(rmse<.22)
        result=dict(it=self.it,task=kind,success_rate=float(success.mean()),nominal_success=bool(success[0]),completion_rate=float(complete.mean()),joint_rmse_rad=float(rmse.mean()),mean_norm_return=float((returns/(T*self.tcfg.max_step_reward)).mean()),wall_seconds=time.time()-START,train_seconds=self.train_time,**extras)
        with open(Path(self.cfg.out,'eval.jsonl'),'a') as f:f.write(json.dumps(result)+'\n')
        print('SKILL_EVAL '+json.dumps(result),flush=True)
        quality=100*result['success_rate']+10*result['completion_rate']+result['mean_norm_return']-result['joint_rmse_rad']
        if quality>self.best_quality:
            self.best_quality=quality;self.save(str(Path(self.cfg.out,'best.pt')))
            Path(self.cfg.out,'best_eval.json').write_text(json.dumps(result,indent=2)+'\n')
            np.savez_compressed(Path(self.cfg.out,'best_rollout.npz'),state=q,feet=feet,alive=alive,dt=.02)
        self.passed=self.passed+1 if result['success_rate']>=.9 and result['nominal_success'] else 0
        if self.passed>=2:self.cfg.iterations=self.it
        self.policy.train()

if __name__=='__main__':
    physical=int(os.environ['CUDA_VISIBLE_DEVICES']);first=(0 if physical<4 else 16)+(physical%4)*4
    os.sched_setaffinity(0,set(range(first,first+4)));torch.set_num_threads(2)
    cfg=base.parse_args()
    assert not cfg.resume,'These runs must start from random policy and world-model weights'
    trainer=SkillTrainer(cfg)
    meta=dict(start_unix=START,physical_gpu=physical,cfg=dataclasses.asdict(cfg),**trainer._extra_state())
    Path(cfg.out,'run_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    trainer.run()
    Path(cfg.out,'timing.json').write_text(json.dumps(dict(start_unix=START,end_unix=time.time(),wall_seconds=time.time()-START,train_seconds=trainer.train_time,iterations=trainer.it),indent=2)+'\n')
