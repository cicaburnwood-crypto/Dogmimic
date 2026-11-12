"""Offline hashes, MuJoCo assets, V18 export parity, and P4 load. No robot IO."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'apps/b2-v18-live'),str(ROOT/'dog_bundle')]
from policy import Manager,PhysicalGeometry,commands
from height_runtime import HeightLocomotionController
manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
 p=ROOT/name
 with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=expected:raise ValueError(f'Hash mismatch: {name}. Run git lfs pull.')
g=PhysicalGeometry();manager=Manager();f=np.load(ROOT/'apps/b2-v18-live/models/parity.npz')
a=np.stack([manager(x) for x in f['obs']]);error=float(abs(a-f['action']).max())
assert error<2e-5,error
c=np.stack([commands(x) for x in a]);assert np.max(abs(c-f['commands']))<2e-5
lower=HeightLocomotionController(ROOT/'dog_bundle/models/locomotion')
print(json.dumps({'files_verified':len(manifest),'manager_max_abs_error':error,'mujoco_nq':g.m.nq,'mujoco_nv':g.m.nv,'P4_loaded':True,'hardware_commands_sent':0},indent=2))
