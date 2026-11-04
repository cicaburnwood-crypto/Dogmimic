import importlib.metadata as md
import json
from pathlib import Path
import platform

import mujoco
import mujoco_warp
import numpy as np
import torch
import warp as wp

assert torch.cuda.is_available()
x = torch.randn(256, 256, device='cuda', requires_grad=True)
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = (x @ x.T).square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
torch.cuda.synchronize()
wp.init()
assert wp.get_device('cuda:0').is_cuda
m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><light pos="0 0 3"/><geom type="plane" size="2 2 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
d = mujoco.MjData(m)
for _ in range(10):
    mujoco.mj_step(m, d)
with mujoco.Renderer(m, height=64, width=64) as renderer:
    renderer.update_scene(d)
    frame = renderer.render()
assert frame.shape == (64, 64, 3) and frame.max() > 0
report = {
    'python': platform.python_version(),
    'packages': {p: md.version(p) for p in ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'numpy', 'scipy', 'rsl-rl-lib', 'tensordict')},
    'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'capability': torch.cuda.get_device_capability(0),
    'torch_arch_list': torch.cuda.get_arch_list(),
    'checks': {'cuda_bf16_forward_backward': True, 'warp_cuda': True, 'mujoco_egl_render': True},
}
print(json.dumps(report, indent=2))
Path('logs/runtime-report.json').write_text(json.dumps(report, indent=2)+'\n')
