"""Hardware backend factory for server.py --backend backend:create_backend."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'deployments/g1_actions'))
from runner import Backend,create_backend

class DisconnectedTransport:
    def snapshot(self):
        raise RuntimeError('G1 尚未连接')

def create_offline_backend():
    # Load exactly the same actors as the SDK backend, without initializing DDS.
    return Backend(DisconnectedTransport())
