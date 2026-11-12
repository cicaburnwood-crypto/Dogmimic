"""Consume exported 99-D observation(s), emit V18 12-D actions; no hardware IO."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'apps/b2-v18-live'))
from policy import Manager,commands
p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,help='JSON array of 99 floats; omitted uses deterministic fixture')
a=p.parse_args();obs=np.asarray(json.loads(a.observation.read_text())) if a.observation else np.load(ROOT/'apps/b2-v18-live/models/parity.npz')['obs'][0]
u=Manager()(obs)
print(json.dumps({'action':u.tolist(),'body_request':commands(u).tolist(),'note':'Raw arm action needs stateful velocity/acceleration/geometry projection; not motor commands.'},indent=2))
