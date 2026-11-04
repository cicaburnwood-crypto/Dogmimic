import sys,os,json,time,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'engine'))
import torch
from supermimic.train import Trainer,parse_args
from robust_env import RobustEnv
class RobustTrainer(Trainer):
 def _make_env(self):
  e=RobustEnv(self.lib,self.device,self.cfg.num_envs,self.cfg.gamma);self.tcfg=e.tcfg;return e
 def __init__(self,cfg):
  super().__init__(cfg)
  src=Path(os.environ['INITIAL_POLICY']);ck=torch.load(src,map_location=self.device,weights_only=False);self.policy.load_state_dict(ck['policy'])
  if self.rank==0:
   (Path(cfg.out)/'initialization.json').write_text(json.dumps(dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_iteration=ck['it'],policy='own previous policy',world_model='random',critic='random',optimizer='new',replay='empty'),indent=2));self.save(str(Path(cfg.out)/'initial.pt'));self.run_eval()
 def update_policy(self):
  if self.it<int(os.environ.get('WM_ONLY_STEPS','1000')):return {'policy_warmup':1.}
  return super().update_policy()
 def run_eval(self):
  p=Path(self.cfg.out);tmp=p/'candidate.tmp.pt';self.save(str(tmp));tmp.replace(p/'candidate.pt')
  t=p/'eval_request.tmp';t.write_text(json.dumps(dict(it=self.it,created=time.time())));t.replace(p/'eval_request.json')
 def _should_stop(self):
  return super()._should_stop() or (Path(self.cfg.out)/'STOP').exists()
if __name__=='__main__':
 cfg=parse_args();t=RobustTrainer(cfg);t.run()
