from pathlib import Path
import shutil
b=Path(__file__).resolve().parent
shutil.copytree(b.parents[1]/'code/supermimic',b/'engine/supermimic',dirs_exist_ok=True)
p=b/'engine/supermimic/task.py';s=p.read_text();import re
s,n=re.subn(r'XI_DIM\s*=\s*[^\n]+','XI_DIM = 42  # 33 original + 9 robust dynamics parameters',s,count=1);assert n;p.write_text(s)
p=b/'engine/supermimic/sim_env.py';s=p.read_text();old='''    self.sim.data.ctrl[:] = target[:, self.act_to_joint]
    for _ in range(self.decimation):
      self.sim.step()''';assert old in s;s=s.replace(old,'''    self.physics_step(target)''');pos=s.index('  def step(self, action: torch.Tensor):');s=s[:pos]+'''  def physics_step(self, target):
    self.sim.data.ctrl[:] = target[:, self.act_to_joint]
    for _ in range(self.decimation): self.sim.step()

'''+s[pos:];p.write_text(s)
