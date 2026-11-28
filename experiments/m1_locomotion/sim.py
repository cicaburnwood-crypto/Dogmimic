"""Headless MuJoCo dynamics and named joint contracts for M1 and M20S."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
DT = 0.02
M1_NAMES = tuple(f"{leg}_{axis}" for leg in ("LF", "LH", "RF", "RH")
                 for axis in ("HAA", "HFE", "KNEE", "WHEEL"))
M20S_NAMES = tuple(f"{leg}_{axis}_joint" for leg in ("fl", "hl", "fr", "hr")
                   for axis in ("hipx", "hipy", "knee", "wheel"))
WHEELS = np.arange(3, 16, 4)
LEGS = np.setdiff1d(np.arange(16), WHEELS)
M1_HOME = np.array([0, 0.9, -1.6, 0, 0, -0.9, 1.6, 0] * 2)
M1_STAND = np.array([0, 0.8, -1.28, 0, 0, -0.8, 1.28, 0] * 2)
M1_SETTLE_KP = np.tile([150, 150, 150, 0], 4)
M1_SETTLE_KD = np.tile([5, 5, 5, 3], 4)
M1_SCALE = np.tile([0.5, 0.5, 0.5, 10.0], 4)
M1_KP = np.tile([40, 40, 40, 0], 4)
M1_KD = np.tile([2, 2, 2, 3], 4)
M1_HARDWARE_STAND = np.tile([0., 1., -1.435, 0., 0., -1., 1.435, 0.], 2)
M1_HARDWARE_STAND_KP = np.tile([150., 150., 150., 0.], 4)
M1_HARDWARE_STAND_KD = np.tile([2., 2., 2., 1.], 4)
M1_REMOTE_SMOOTH = .99**10  # Ten 500 Hz filter updates per 50 Hz policy tick.

M1_FLAT = RUNTIME / "m1/robot/robot_model/M1-robotdog-asm09/flat.xml"
M1_HEADLESS = RUNTIME / "m1/headless.xml"
M20S_FLAT = RUNTIME / "m20s-min/M20S_description/m20s_mjcf/mjcf/M20S.xml"


def body_matrix(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.ascontiguousarray(quat, dtype=np.float64))
    return matrix.reshape(3, 3)


def se2_twist(state):
    """Heading-frame horizontal velocity and heading derivative, in SI units."""
    rotation = body_matrix(state[3:7])
    heading = rotation[:2, 0]
    heading_sq = max(float(heading @ heading), 1e-6)
    forward = heading / np.sqrt(heading_sq)
    left = np.array([-forward[1], forward[0]])
    omega = state[26:29]
    forward_rate = np.cross(omega, rotation[:, 0])[:2]
    yaw_rate = (heading[0] * forward_rate[1] - heading[1] * forward_rate[0]) / heading_sq
    return np.array([state[23:25] @ forward, state[23:25] @ left, yaw_rate])


class Robot:
    def __init__(self, xml, names, wheel_radius):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.keys = tuple(names)
        self.joints = np.array([self.model.joint(name).id for name in names])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.actuator = np.array([np.where(self.model.actuator_trnid[:, 0] == j)[0].item()
                                  for j in self.joints])
        if (self.model.nq, self.model.nv, self.model.nu) != (23, 22, 16):
            raise ValueError("Expected a floating base with 16 actuators.")
        self.wheel_radius = wheel_radius
        self.wheel_bodies = self.model.jnt_bodyid[self.joints[WHEELS]]
        self.floor = self.model.geom("ground" if names == M1_NAMES else "floor").id
        self.step_count = round(DT / self.model.opt.timestep)
        self.ctrl_limit = self.model.actuator_ctrlrange[self.actuator]
        self.last_effort = np.zeros(16)

    def reset(self, pose, height, state=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = height
        self.data.qpos[3] = 1
        self.data.qpos[self.qadr] = pose
        if state is not None:
            self.data.qpos[:7] = state[:7]
            self.data.qpos[self.qadr] = state[7:23]
            self.data.qvel[:3] = state[23:26]
            self.data.qvel[3:6] = body_matrix(state[3:7]).T @ state[26:29]
            self.data.qvel[self.vadr] = state[29:45]
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        qpos, qvel = self.data.qpos, self.data.qvel
        world_omega = body_matrix(qpos[3:7]) @ qvel[3:6]
        return np.concatenate((qpos[:7], qpos[self.qadr], qvel[:3], world_omega,
                               qvel[self.vadr]))

    def velocity(self):
        return se2_twist(self.state())

    def step(self, desired, kp, kd):
        q_target = np.array(desired)
        q_target[WHEELS] = 0.0
        dq_target = np.zeros(16)
        dq_target[WHEELS] = desired[WHEELS]
        for _ in range(self.step_count):
            effort = kp * (q_target - self.data.qpos[self.qadr])
            effort += kd * (dq_target - self.data.qvel[self.vadr])
            self.last_effort = effort.clip(self.ctrl_limit[:, 0], self.ctrl_limit[:, 1])
            self.data.ctrl[self.actuator] = self.last_effort
            mujoco.mj_step(self.model, self.data)
        # Update site, geometry and sensor readings from the state at the control boundary.
        mujoco.mj_forward(self.model, self.data)

    def tip_and_wheels(self):
        rotation = body_matrix(self.data.qpos[3:7])
        wheel_axis = self.data.xaxis[self.joints[WHEELS]]
        wheel_tangent = wheel_axis @ rotation[:, 1]
        wheel_tangent *= self.data.qvel[self.vadr[WHEELS]] * self.wheel_radius
        support = np.zeros(4, dtype=bool)
        for contact in self.data.contact:
            if contact.geom1 != self.floor and contact.geom2 != self.floor:
                continue
            other = contact.geom2 if contact.geom1 == self.floor else contact.geom1
            support |= self.wheel_bodies == self.model.geom_bodyid[other]
        return -rotation[2], wheel_tangent, support

    def valid(self):
        quat = self.data.qpos[3:7]
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
                    and self.data.qpos[2] > 0.20 and body_matrix(quat)[2, 2] > 0.65)


def step_m1_handoff(robot, action, tick):
    """Apply the M1's 0.8-second transition from StandUp to Uni Exp."""
    target = M1_HOME + M1_SCALE * action
    entry_ticks = round(.8 / DT)
    if tick >= entry_ticks:
        robot.step(target, M1_KP, M1_KD)
        return
    u = tick / entry_ticks
    blend = u * u * (3. - 2. * u)
    kp = (1. - blend) * M1_HARDWARE_STAND_KP + blend * M1_KP
    kd = (1. - blend) * M1_HARDWARE_STAND_KD + blend * M1_KD
    target[LEGS] = ((1. - blend) * M1_HARDWARE_STAND_KP[LEGS] * M1_HARDWARE_STAND[LEGS]
                    + blend * M1_KP[LEGS] * target[LEGS]) / kp[LEGS]
    target[WHEELS] *= blend * M1_KD[WHEELS] / kd[WHEELS]
    robot.step(target, kp, kd)
