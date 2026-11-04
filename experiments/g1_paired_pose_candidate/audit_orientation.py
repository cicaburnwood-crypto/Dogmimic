import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(1,str(root/'experiments/g1_boundary_sampling_candidate/engine'))
import torch
from supermimic.rotations import quat_from_euler_xyz,quat_mul
from pose_orientation import paired_orientation
q=torch.tensor(json.loads((root/'experiments/g1_boundary_sampling_candidate/measured_start.json').read_text())['quaternion'])
torch.manual_seed(137);n=1000
r=quat_from_euler_xyz(torch.randn(n)*.1,torch.randn(n)*.2,torch.randn(n))
noise=quat_from_euler_xyz(torch.randn(n)*.008,torch.randn(n)*.008,torch.randn(n)*.008);current=quat_mul(noise,r)
a=paired_orientation(current,r,q,torch.zeros(n));assert torch.allclose(a,current,atol=1e-6)
b=paired_orientation(current,r,q,torch.rand(n));assert torch.allclose(b.norm(dim=-1),torch.ones(n),atol=1e-5)
full=paired_orientation(r,r,q,torch.ones(n))
# Heading invariant tilt: z component of rotated local z is 1-2(x*x+y*y).
assert torch.allclose(1-2*(full[:,1]**2+full[:,2]**2),torch.full((n,),float(1-2*(q[1]**2+q[2]**2))),atol=1e-5)
blend=torch.rand(n);assert torch.allclose(paired_orientation(current,r,q,blend),paired_orientation(current,r,-q,blend),atol=1e-6)
print(json.dumps(dict(passed=True,samples=n,zero_blend_preserves_noise=True,unit_quaternions=True,measured_tilt_endpoint=True,quaternion_sign_invariant=True,scope='helper algebra only, not dynamics')))
