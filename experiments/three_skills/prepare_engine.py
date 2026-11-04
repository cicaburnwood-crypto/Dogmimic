"""Create an isolated, dimension-generalized copy of the installed reproduction."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = ROOT / 'code/supermimic'
if not source.exists():
    source = ROOT / 'upstream/code/supermimic'
dest = HERE / 'engine/supermimic'
dest.mkdir(parents=True, exist_ok=True)
for p in source.glob('*.py'):
    shutil.copy2(p, dest / p.name)

def change(name, before, after):
    p = dest / name
    s = p.read_text()
    assert before in s, (name, before)
    p.write_text(s.replace(before, after))

change('task.py', 'NJ = 29', 'import os\nNJ = 12 if os.environ.get("SM_ROBOT") == "b2" else 29\nif NJ == 12:\n  TRACKED_BODIES = ("base_link", "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf", "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf")')
for a,b in [('7:36','7:7+NJ'),('36:39','7+NJ:10+NJ'),('39:42','10+NJ:13+NJ'),('42:71','13+NJ:13+2*NJ')]:
    change('task.py', a, b)
change('replay.py', 'SDIM = 71', 'SDIM = 13 + 2 * NJ')
# A quadruped's forward axis points vertically during a rear stand. Its lateral
# axis provides a nonsingular heading at that pose, consistently in WM and task.
change('rotations.py', 'import torch', 'import torch\nimport os\nB2_HEADING = os.environ.get("SM_ROBOT") == "b2"')
change('rotations.py', 'return yaw_mat(yaw_from_mat(R))', 'if B2_HEADING:\n    return yaw_mat(torch.atan2(-R[..., 0, 1], R[..., 1, 1]))\n  return yaw_mat(yaw_from_mat(R))')
# An empty valid-window sample must not poison running statistics with NaNs.
change('networks.py', 'x = x.reshape(-1, x.shape[-1]).float()', 'x = x.reshape(-1, x.shape[-1]).float()\n    if x.shape[0] == 0:\n      return')
print(dest)
