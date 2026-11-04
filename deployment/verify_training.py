"""Check that the upstream smoke run completed real finite gradient updates."""
import json
import math
import sys
from pathlib import Path
import torch

run = Path(sys.argv[1])
assert (run / 'DONE').is_file(), 'Training did not finish normally'
rows = [json.loads(x) for x in (run / 'log.jsonl').read_text().splitlines()]
assert rows and rows[-1]['updates'] > 0
for row in rows:
    assert all(not isinstance(v, (int, float)) or math.isfinite(v) for v in row.values()), row
ck = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
assert ck['updates'] > 0
for name in ('wm', 'policy', 'critic'):
    for key, value in ck[name].items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), (name, key)
assert ck['opt_wm']['state'] and ck['opt_pi']['state'] and ck['opt_v']['state']
report = {'run': str(run), 'iterations': ck['it'], 'updates': ck['updates'],
          'env_steps': ck['env_steps'], 'training_seconds': ck['train_time'],
          'peak_torch_memory_gb': rows[-1]['mem_gb'], 'finite_logs_and_weights': True,
          'world_model_policy_critic_optimizers_updated': True}
print(json.dumps(report, indent=2))
(run / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')
