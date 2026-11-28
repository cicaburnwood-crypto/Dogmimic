"""ONNX adapters for the archived M1 baselines and vendor M20S policy."""

import numpy as np
import onnxruntime as ort

from sim import DT, LEGS, M1_HOME, M1_SCALE, RUNTIME, WHEELS
from native_terrain import TERRAIN_NAME, observe_terrain

BASELINES = {
    "classic": "model_21000_08_07_18_05_11.onnx",
    "cts": "policy_cts.onnx",
}
CTS_WIDTH = 61
HISTORY_STEPS = 5


def session(file):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(file), opts, providers=["CPUExecutionProvider"])


class M1Controller:
    def __init__(self, mode="cts", checkpoint=None, *, period=None, continuous_phase=False):
        self.mode = mode
        self.period = period if period is not None else (0.7 if mode == "classic" else 0.8)
        self.continuous_phase = continuous_phase
        policy = checkpoint or RUNTIME / "m1/models" / BASELINES[mode]
        self.model = session(policy)
        self.use_terrain = any(value.name == TERRAIN_NAME for value in self.model.get_overridable_initializers())
        self.reset()

    def reset(self):
        self.active = False
        self.still = 0.0
        self.nominal_phase = np.array([0, np.pi, np.pi, 0])
        self.phase = self.nominal_phase.copy()
        self.phase_obs = np.zeros(8, dtype=np.float32)
        self.previous = np.zeros(16, dtype=np.float32)
        self.history = np.zeros((HISTORY_STEPS, 49 if self.mode == "classic" else CTS_WIDTH),
                                dtype=np.float32)
        self.terrain = None

    def observe(self, robot, command, walking=False):
        if self.use_terrain:
            self.terrain = observe_terrain(robot)
        yaw_or_side = abs(command[1]) >= (0.15 if walking and self.mode == "classic" else 0.05)
        yaw_or_side |= abs(command[2]) >= 0.05
        if yaw_or_side or (walking and abs(command[0]) >= 0.05):
            self.active = True
            self.still = 0.0
        else:
            self.still += DT
            if self.still >= (1.5 if walking else 0.5):
                self.active = False
        if self.active or self.continuous_phase:
            self.phase = (self.phase + DT * 2 * np.pi / self.period) % (2 * np.pi)
        else:
            self.phase[:] = self.nominal_phase
        if self.active:
            self.phase_obs[:] = 0.1 * np.stack((np.cos(self.phase), np.sin(self.phase)), axis=-1).ravel()
        else:
            self.phase_obs.fill(0)

        gravity, _, _ = robot.tip_and_wheels()
        angles = robot.data.qpos[robot.qadr[LEGS]] - M1_HOME[LEGS]
        joint_vel = robot.data.qvel[robot.vadr]
        velocity_index = WHEELS if self.mode == "classic" else np.r_[LEGS, WHEELS]
        obs = np.concatenate((robot.data.qvel[3:6] * 0.25, gravity,
                              command * [2, 2, 0.25], angles, joint_vel[velocity_index] * 0.05,
                              self.previous, self.phase_obs)).astype(np.float32)
        return obs if self.mode == "classic" else obs.clip(-100, 100)

    def action(self, observation):
        obs = observation[None, :]
        history = self.history.reshape(1, -1)
        inputs = {"obs": obs, "obs_history": history}
        if self.terrain is not None:
            inputs[TERRAIN_NAME] = self.terrain[None, :]
        action = self.model.run(["actions"], inputs)[0].flatten()
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Invalid M1 policy output.")
        self.history[:-1] = self.history[1:].copy()
        self.history[-1] = observation
        self.previous = action if self.mode == "classic" else action.clip(-100, 100)
        return self.previous


# Simulator and SDK use separate calibration layers. These constants match the
# M20S SDK's MuJoCo bridge and M20SInterface, in hardware order FL, FR, HL, HR.
SIM_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1])
SIM_OFFSET = np.deg2rad([-25, 229, 160, 0, 25, -131, -200, 0, -25, -229, -160, 0, 25, 131, 200, 0])
SDK_DIR = np.array([1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1])
SDK_OFFSET = np.deg2rad([-25, -131, 160, 0, 25, -131, 160, 0, -25, 131, -160, 0, 25, 131, -160, 0])
M20S_BIRTH = np.array([-0.438, -1.16, 2.76, 0, 0.438, -1.16, 2.76, 0,
                       -0.438, 1.16, -2.76, 0, 0.438, 1.16, -2.76, 0])
M20S_HOME = np.array([0, -0.3, 0.6, 0, 0, -0.3, 0.6, 0, 0, 0.3, -0.6, 0, 0, 0.3, -0.6, 0])
M20S_POLICY_ORDER = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15])
# Interchange between hardware order and the experiment's common FL, HL, FR, HR order.
M20S_SWAP = np.array([0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15])


class M20SController:
    def __init__(self):
        self.model = session(RUNTIME / "m20s-min/policy/policy.onnx")
        self.sign = SIM_DIR * SDK_DIR
        nominal = (M20S_BIRTH - SIM_OFFSET) * self.sign + SDK_OFFSET
        offset = SDK_OFFSET.copy()
        for indices, limit in ((np.arange(1, 16, 4), 140), (np.arange(2, 16, 4), 164)):
            offset[indices] -= np.where(abs(nominal[indices]) > np.deg2rad(limit),
                                        np.sign(nominal[indices]) * 2 * np.pi, 0)
        self.bias = offset - SIM_OFFSET * self.sign
        self.kp = np.tile([80, 80, 80, 0], 4)
        self.kd = np.tile([2, 2, 2, 0.6], 4)
        self.default = ((M20S_HOME - self.bias) * self.sign)[M20S_SWAP]
        self.reset()

    def reset(self):
        self.previous = np.zeros(16, dtype=np.float32)

    def desired(self, robot, command):
        position = robot.data.qpos[robot.qadr][M20S_SWAP] * self.sign + self.bias
        position[WHEELS] = 0
        speed = robot.data.qvel[robot.vadr][M20S_SWAP] * self.sign
        gravity, _, _ = robot.tip_and_wheels()
        obs = np.concatenate((robot.data.qvel[3:6] * 0.25, gravity, command,
                              (position - M20S_HOME)[M20S_POLICY_ORDER],
                              speed[M20S_POLICY_ORDER] * 0.05, self.previous))
        self.previous = self.model.run(["actions"], {"obs": obs.astype(np.float32)[None, :]})[0].flatten()
        if not np.isfinite(self.previous).all():
            raise ValueError("M20S emitted nonfinite action.")
        action = np.empty(16)
        action[M20S_POLICY_ORDER] = self.previous
        action *= np.tile([0.125, 0.25, 0.25, 5], 4)
        action[LEGS] += (M20S_HOME - self.bias)[LEGS]
        return (action * self.sign)[M20S_SWAP]
