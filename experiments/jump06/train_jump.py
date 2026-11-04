import json
import os
from pathlib import Path
import torch
from supermimic.train import Trainer,parse_args
from jump_task import make_env
from evaluate_jump import run_trials

class JumpTrainer(Trainer):
    def _make_env(self):
        env=make_env(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma)
        self.tcfg=env.tcfg
        return env
    def run_eval(self):
        if not hasattr(self,'jump_eval_env'):
            self.jump_eval_env=make_env(self.lib,self.device,32,evaluate=True)
            self.best_quality=-1.
            self.passed_evals=0
        self.policy.eval()
        result,trajectory=run_trials(self.policy,self.lib,self.jump_eval_env,seed=20261027+self.it)
        self.policy.train()
        result.update(it=self.it,updates=self.updates,train_hours=self.train_time/3600)
        with open(Path(self.cfg.out)/'jump_eval.jsonl','a') as f:f.write(json.dumps(result)+'\n')
        print('JUMP_EVAL '+json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
        quality=100*result['success_rate']+10*result['completion_rate']+result['mean_norm_return']
        if quality>self.best_quality:
            self.best_quality=quality
            self.save(str(Path(self.cfg.out)/'best.pt'))
            import numpy as np
            np.savez_compressed(Path(self.cfg.out)/'best_rollout.npz',**trajectory)
            (Path(self.cfg.out)/'best_eval.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['nominal_success'] and result['success_rate']>=.9:
            self.passed_evals+=1
        else:self.passed_evals=0
        if self.passed_evals>=2:
            print('TARGET_REACHED: two evaluations with nominal success and >=90% success.',flush=True)
            self.cfg.iterations=self.it

if __name__=='__main__':
    cfg=parse_args()
    print(cfg,flush=True)
    JumpTrainer(cfg).run()
