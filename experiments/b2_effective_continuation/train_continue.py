"""Continue own effective-action Actor/WM with fresh replay and optimizers."""
import sys,os,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/b2_effective_action_candidate'))
import train_guarded as parent
from train_guarded import torch,base
SOURCE_SHA='f5aae854c90fe26f15ccc484505b7a303c43f630466164bfb9d25fb42d19d4ad'
class Trainer(parent.GuardedTrainer):
    def __init__(self,cfg,device='cuda'):
        super().__init__(cfg,device)
        self.anchor_source_hash=self.source_hash
        source=Path(os.environ['B2_CONTINUE_CHECKPOINT'])
        self.continue_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        assert self.continue_hash==SOURCE_SHA
        ck=torch.load(source,map_location=device,weights_only=False)
        assert ck['world_model_action']=='effective_filtered_target_normalized'
        self.policy.load_state_dict(ck['policy']);self.wm.load_state_dict(ck['wm'])
        self.source_path=source;self.source_hash=self.continue_hash;self.source_iteration=int(ck['it'])
        self.policy.obs_norm.until=0;self.policy.goal_norm.until=0
    def _extra_state(self):
        d=super()._extra_state()
        d.update(task_kind='b2_effective_continuation',world_model_initialization='continued own effective-action WM',world_model_source=str(self.source_path),source_sha256=self.source_hash,source_iteration=self.source_iteration,anchor_source_sha256=getattr(self,'anchor_source_hash',None),replay_initialization='empty',optimizer_initialization='new')
        return d
if __name__=='__main__':
    torch.set_num_threads(2);os.sched_setaffinity(0,set(range(24,28)))
    cfg=base.parse_args();t=Trainer(cfg);out=Path(cfg.out)
    (out/'run_meta.json').write_text(json.dumps(t._extra_state(),indent=2))
    t.run_eval();t.run();t.run_eval(full=True)
    (out/'EVAL_DONE.json').write_text(json.dumps(dict(it=t.it,wall_seconds=time.time()-parent.START)))
