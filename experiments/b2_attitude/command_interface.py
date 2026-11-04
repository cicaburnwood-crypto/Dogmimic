"""Simulation/inference command interface; does not connect to robot hardware."""
from collections import deque
import math
import torch
from contract import DT,advance_command,target_from_degrees

class AttitudeCommands:
    def __init__(self,device='cpu'):
        self.context=torch.zeros(1,6,device=device)
        self.pending=deque();self.phase='idle';self.elapsed=0.

    def _begin(self,target,phase='execute'):
        self.context[:,4:6]=torch.tensor(target,device=self.context.device)
        self.phase=phase;self.elapsed=0.

    def trigger(self,axis,degrees,standing_stable):
        """Single action: smooth tilt, hold, recover to stable neutral."""
        self.sequence([(axis,degrees)],standing_stable)

    def sequence(self,actions,standing_stable):
        targets=[target_from_degrees(axis,float(degrees)) for axis,degrees in actions]
        if not targets:return
        if not standing_stable:raise ValueError('Start the sequence from stable standing')
        self.pending=deque(targets[1:]);self._begin(targets[0])

    def stream(self,axis,degrees,standing_stable=False):
        """Start from standing, then replace targets continuously without a reset."""
        target=target_from_degrees(axis,float(degrees))
        if self.phase=='idle' and not standing_stable:raise ValueError('Start from stable standing')
        self.pending.clear();self._begin(target,'stream')

    def recover(self):
        self.pending.clear();self._begin([0.,0.],'recover')

    def policy_goal(self):return self.context.clone()

    def advance(self,standing_stable):
        """Call once AFTER applying the policy for one 20ms simulator step."""
        self.context=advance_command(self.context);self.elapsed+=DT
        if self.phase=='execute' and self.elapsed>=4.-1e-8:
            self._begin([0.,0.],'recover')
        elif self.phase=='recover':
            neutral=self.context[0,:2].abs().max()<math.radians(.5)
            slow=self.context[0,2:4].abs().max()<math.radians(1.)
            if standing_stable and neutral and slow:
                if self.pending:self._begin(self.pending.popleft())
                else:self.phase='idle'
        return self.policy_goal()

