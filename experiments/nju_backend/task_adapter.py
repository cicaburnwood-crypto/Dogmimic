"""Robot/task integration only. NJU learning losses and optimizers are untouched."""
from pathlib import Path
import os
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields, RecomputeLevel
from supermimic.robot_layout import NJ

HERE = Path(__file__).resolve().parent
LEGACY = Path(os.environ.get('NJU_LEGACY_ROOT', '/home/liudiwen/projects/supermimic_vibe'))


def b2_xml():
    return HERE / 'assets/b2.xml'


def prepare_b2_asset():
    p = b2_xml(); p.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(LEGACY/'experiments/supermimic_scratch/models/b2_train.xml')
    root = tree.getroot()
    for tag in ('keyframe', 'sensor'):
        elem = root.find(tag)
        if elem is not None: root.remove(elem)
    world = root.find('worldbody')
    for elem in list(world):
        if elem.tag != 'body': world.remove(elem)
    tree.write(p, encoding='unicode')
    return p


def load_task_model():
    if NJ == 29:
        from supermimic.data.motion_convert import load_g1_model
        return load_g1_model()
    return mujoco.MjModel.from_xml_path(str(b2_xml()))


def b2_entity():
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    m = load_task_model()
    joints = [m.joint(i).name for i in range(1, m.njnt)]
    # Preserve the pre-hardware XML PD, torque and target limits exactly.
    actuators = (XmlActuatorCfg(target_names_expr=tuple(joints)),)
    return EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_file(str(b2_xml())),
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.6), joint_pos={'.*_hip_joint':0., '.*_thigh_joint':.7, '.*_calf_joint':-1.4}, joint_vel={'.*':0.}),
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=.9))


@requires_model_fields('geom_friction')
def all_contact_friction(env, env_ids, lo=.3, hi=1.2):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    mu = lo + (hi-lo)*torch.rand(len(ids), device=env.device)
    env.sim.model.geom_friction[ids, :, 0] = mu[:, None]


@requires_model_fields('body_mass', 'body_ipos', 'body_inertia', 'body_iquat', recompute=RecomputeLevel.set_const)
def b2_payload(env, env_ids):
    from mjlab.utils.lab_api.math import quat_from_matrix
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    n, dev = len(ids), env.device
    m = env.sim.mj_model
    robot = env.scene['robot']
    root = int(robot.indexing.body_ids[robot.body_names.index('base_link')])
    if not hasattr(env, 'b2_load'):
        env.b2_load = torch.zeros(env.num_envs, device=dev)
        env.b2_mass_scale = torch.ones(env.num_envs, device=dev)
    mass = torch.where(torch.rand(n, device=dev)<.5, 0., 5.).double()
    scale = .9 + .2*torch.rand(n, device=dev)
    env.b2_load[ids] = mass.float(); env.b2_mass_scale[ids] = scale
    base_mass = float(m.body_mass[root])
    base_com = torch.tensor(m.body_ipos[root], dtype=torch.float64, device=dev)
    R_np = np.empty(9); mujoco.mju_quat2Mat(R_np, m.body_iquat[root])
    R = torch.tensor(R_np.reshape(3,3), device=dev)
    inertia0 = R @ torch.diag(torch.tensor(m.body_inertia[root],device=dev)) @ R.T
    center_box = torch.tensor([0.,0.,.18],device=dev,dtype=torch.float64)
    half = torch.tensor([.18,.12,.10],device=dev,dtype=torch.float64)
    center = (base_mass*base_com + mass[:,None]*center_box)/(base_mass+mass[:,None])
    eye = torch.eye(3,device=dev,dtype=torch.float64)
    def shift(mass, d):
        return mass[:,None,None]*(d.square().sum(-1)[:,None,None]*eye-d[:,:,None]*d[:,None,:])
    inertia = inertia0 + mass[:,None,None]*torch.diag((half.square().sum()-half.square())/3.)
    inertia += shift(torch.full_like(mass,base_mass), base_com-center)+shift(mass,center_box-center)
    moments, axes = torch.linalg.eigh(inertia)
    axes[:,:,0] *= torch.where(torch.linalg.det(axes)<0,-1.,1.)[:,None]
    for key in ('body_mass','body_inertia','body_ipos','body_iquat'):
        getattr(env.sim.model,key)[ids] = torch.tensor(getattr(m,key),dtype=torch.float32,device=dev)
    env.sim.model.body_mass[ids,root] = (base_mass+mass).float()
    env.sim.model.body_ipos[ids,root] = center.float() + (torch.rand(n,3,device=dev)*2-1)*.005
    env.sim.model.body_inertia[ids,root] = moments.float()
    env.sim.model.body_iquat[ids,root] = quat_from_matrix(axes).float()
    env.sim.model.body_mass[ids] *= scale[:,None]
    env.sim.model.body_inertia[ids] *= scale[:,None,None]


class B2Pushes:
    """Five 8–16 N force pulses per clip, retained from the light-payload task.

    As in NJU's optional push events, forces are exogenous disturbances; the
    actor receives no privileged force, mass or payload input.
    """
    def __init__(self, cfg, env):
        self.env = env; self.robot = env.scene['robot']
        self.body_ids = [self.robot.body_names.index('base_link')]
        n, dev = env.num_envs, env.device
        self.start = torch.zeros(n,5,device=dev); self.end = self.start.clone()
        self.force = torch.zeros(n,5,3,device=dev)

    def reset(self, env_ids=None):
        ids = torch.arange(self.env.num_envs,device=self.env.device) if env_ids is None else env_ids
        n, dev = len(ids), self.env.device
        lo = torch.tensor([1.5,4.5,7.2,10.2,13.1],device=dev)
        hi = torch.tensor([2.7,5.5,8.2,11.5,13.8],device=dev)
        self.start[ids] = ((lo+(hi-lo)*torch.rand(n,5,device=dev))/.02).round()*.02
        self.end[ids] = self.start[ids]+torch.randint(6,13,(n,5),device=dev)*.02
        d = torch.randn(n,5,3,device=dev); d /= d.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        self.force[ids] = d*(8.+8.*torch.rand(n,5,1,device=dev))
        z = torch.zeros(n,1,3,device=dev)
        self.robot.write_external_wrench_to_sim(z,z,env_ids=ids,body_ids=self.body_ids)

    def __call__(self, env, env_ids, **params):
        c = env.command_manager.get_term('motion'); lib = c.library
        t = (c.frame-lib.clip_start[lib.frame_clip[c.frame]])[:,None]*.02
        f = (((t>=self.start)&(t<self.end))[:,:,None]*self.force).sum(1)[:,None]
        self.robot.write_external_wrench_to_sim(f,torch.zeros_like(f),body_ids=self.body_ids)


def adapt_env_cfg(cfg, a):
    # Match the requested friction under MuJoCo mixing, including the floor.
    cfg.events['foot_friction'] = EventTermCfg(mode='reset', func=all_contact_friction,
        params={'lo':.4 if NJ==12 else .3, 'hi':1.3 if NJ==12 else 1.2})
    if NJ == 29:
        return cfg
    cfg.scene.entities['robot'] = b2_entity()
    cfg.actions['joint_pos'].scale = .7
    cfg.commands['motion'].pose_range = {'x':(-.01,.01),'y':(-.01,.01),'z':(-.003,.003),
        'roll':(-.02,.02),'pitch':(-.02,.02),'yaw':(-.02,.02)} if a.reset_noise else {}
    cfg.commands['motion'].velocity_range = {k:(-.05,.05) for k in ('x','y','z','roll','pitch','yaw')} if a.reset_noise else {}
    cfg.commands['motion'].joint_position_range = (-.02,.02) if a.reset_noise else (0.,0.)
    cfg.events.pop('base_com',None)
    if 'encoder_bias' in cfg.events:
        cfg.events['encoder_bias'].params['bias_range'] = (-.003,.003)
    cfg.events['payload'] = EventTermCfg(mode='reset', func=b2_payload)
    cfg.events['pushes'] = EventTermCfg(mode='step', func=B2Pushes)
    cfg.recorders['replay'].params['com_body_name'] = 'base_link'
    cfg.viewer.body_name = 'base_link'
    cfg.sim.nconmax = 48; cfg.sim.njmax = 320
    return cfg
