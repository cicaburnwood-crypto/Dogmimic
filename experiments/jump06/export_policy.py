"""Package a validated checkpoint for simulation inference and record provenance."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from jump_task import JumpTaskCfg


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location='cpu')
    result = json.loads(Path(args.evaluation).read_text())
    assert result['it'] == ck['it']
    assert result['nominal_success'] and result['success_rate'] >= .9
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = {k: ck[k] for k in ('policy', 'cfg', 'it', 'updates', 'env_steps', 'train_time')}
    policy['task_cfg'] = dataclasses.asdict(JumpTaskCfg())
    torch.save(policy, out / 'policy.pt')
    source = list(Path('experiments/jump06').glob('*.py'))
    source += [Path('experiments/jump06/train.sh'), Path('data/processed/g1_standing_jump_060m.npz')]
    manifest = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=sha256(args.checkpoint),
        policy_sha256=sha256(out / 'policy.pt'),
        iterations=ck['it'], updates=ck['updates'], env_steps=ck['env_steps'],
        training_minutes=ck['train_time'] / 60,
        task_cfg=policy['task_cfg'], train_cfg=ck['cfg'],
        packages={name: importlib.metadata.version(name) for name in ('torch', 'mujoco', 'mujoco-warp', 'mjlab', 'warp-lang', 'numpy')},
        source_sha256={str(path): sha256(path) for path in source},
        evaluation={k: v for k, v in result.items() if k != 'trials'},
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(policy=str(out / 'policy.pt'), iterations=ck['it'], training_minutes=manifest['training_minutes']), indent=2))


if __name__ == '__main__':
    main()
