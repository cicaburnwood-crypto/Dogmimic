"""M1's local terrain observations and simulation descent guidance."""
import mujoco
import numpy as np

from native_world import MAX_STAIRS, Course
from sim import LEGS, WHEELS, body_matrix

TERRAIN_NAME = "terrain"
# Validity, sampled height span and grade, trunk height and vertical velocity,
# forward/lateral velocity, yaw rate, downhill heading error and lateral offset.
TERRAIN_SIZE = 10
PROBES = (-.65, -.32, 0., .32, .64, .88)
TRUNK_COM = (.0012616, .000052895, .025714)
WHEEL_SIGNS = (-1., -1., 1., 1.)
FLAT_CAP = (.28, .30, .30, .18) * 4
DESCENT_LEG_CAP = (.10, .14, .16) * 4
DESCENT_GAIN_SPAN = (.50, .23, .18)
PROFILE_HIDDEN = 96
GUIDE_PARAMETERS = {"heading_kp": 2.2, "yaw_kd": .65, "lane_gain": 1.10, "wheel_speed_kp": .28}


def surface(points, context):
    rise, tread, count, edge, half_width, _ = context
    x, y = points.T
    steps = np.arange(MAX_STAIRS)
    risers = np.clip((x[:, None]-edge-tread*steps)/.065 + .5, 0., 1.)
    lane = np.clip((half_width-np.abs(y))/.08+.5, 0., 1.)
    end = np.clip((edge+tread*count+2.7-x)/.08+.5, 0., 1.)
    return rise*np.sum(risers*(steps < count), axis=-1)*lane*end


def terrain_context(robot):
    model = robot.model
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"step_{i+1}") for i in range(MAX_STAIRS)]
    steps = [geom for geom in ids if geom >= 0]
    if not steps:
        return Course().context()
    pos = model.geom_pos[steps[0]]
    size = model.geom_size[steps[0]]
    return Course(rise=2*size[2], tread=2*size[0], steps=len(steps), edge=pos[0]-size[0],
                  half_width=size[1], friction=model.geom_friction[steps[0], 0]).context()


def observe_terrain(robot):
    if not hasattr(robot, "locomotion_terrain"):
        robot.locomotion_terrain = terrain_context(robot)
    pos = robot.data.qpos[:3]
    rotation = body_matrix(robot.data.qpos[3:7])
    horizontal = rotation[:2, 0]
    direction = horizontal / max(np.linalg.norm(horizontal), 1e-6)
    heights = surface(pos[:2] + np.asarray(PROBES)[:, None]*direction, robot.locomotion_terrain)
    offset = rotation @ TRUNK_COM
    omega = rotation @ robot.data.qvel[3:6]
    vz = robot.data.qvel[2] + np.cross(omega, offset)[2]
    heading = np.arctan2(direction[1], direction[0])
    aim = np.pi + np.arctan(GUIDE_PARAMETERS["lane_gain"]*pos[1])
    error = np.arctan2(np.sin(aim-heading), np.cos(aim-heading))
    return np.asarray([1., heights.max()-heights.min(), heights[-1]-heights[0],
                       pos[2]+offset[2]-heights[2], vz, *robot.velocity(), error, pos[1]], np.float32)


def level_guidance(command, terrain, obs, action):
    weight = np.clip(abs(command[0])/.12, 0., 1.)
    previous = obs[25:41]
    delta = np.zeros(16)
    delta[LEGS] = .46*(previous[LEGS]-action[LEGS])
    extension = np.clip(1.65*(.465-terrain[3])-.085*terrain[4], -.15, .16)
    signs = np.array([1., -1., 1., -1.])
    delta[2::4] += signs*extension
    delta[1::4] -= .52*signs*extension
    return weight*delta


def descent_guidance(command, terrain):
    """Guide the previous actor using heading and forward-speed feedback in MuJoCo."""
    gains = GUIDE_PARAMETERS
    turn = np.clip(gains["heading_kp"]*terrain[8]-gains["yaw_kd"]*terrain[7], -.8, .8)
    drive = gains["wheel_speed_kp"]*(command[0]-terrain[5])
    weight = np.clip(command[0]/.06, 0., 1.)
    delta = np.zeros(16, dtype=np.float64)
    delta[WHEELS] = weight*(drive + .275*turn*np.asarray(WHEEL_SIGNS))/.95
    return delta
