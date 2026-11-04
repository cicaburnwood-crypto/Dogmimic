from recovery_env import RecoveryTask
from stage_policy import append_phase


class StageTask(RecoveryTask):
    def goal(self, state, future):
        return append_phase(super().goal(state, future), future)
