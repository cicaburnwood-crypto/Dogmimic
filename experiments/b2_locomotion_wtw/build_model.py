"""B2 geometry/limits only. No Unitree policy or reference trajectory is loaded."""
import json, hashlib, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from contract import Q0,KP,KD,POLICY_JOINTS,POLICY_TO_SDK

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'three_skills/vendor/unitree_mujoco/unitree_robots/b2/b2.xml'

def build():
    root=ET.parse(SOURCE).getroot()
    root.find('compiler').set('meshdir',str(SOURCE.parent/'assets'))
    ET.SubElement(root,'option',timestep='.005',integrator='implicitfast',iterations='10',ls_iterations='20')
    wb=root.find('worldbody')
    ET.SubElement(wb,'geom',name='floor',type='plane',size='0 0 .1',friction='1 .005 .0001',rgba='.3 .35 .4 1')
    ET.SubElement(wb,'light',pos='0 0 5',dir='0 0 -1',directional='true')
    joints={j.get('name'):j for j in root.findall('.//joint') if j.get('name')}
    for a in root.find('actuator'):
        force=a.get('ctrlrange');a.tag='position'
        a.set('kp',str(KP));a.set('kv',str(KD));a.set('forcerange',force)
        a.set('ctrlrange',joints[a.get('joint')].get('range'))
    for leg in ('FL','FR','RL','RR'):
        for g in root.find(f'.//body[@name="{leg}_calf"]').findall('geom'):
            if g.get('type')=='sphere':
                g.set('name',leg+'_foot_collision');g.set('friction','1 .005 .0001')
    for tag in ('keyframe','sensor'):
        e=root.find(tag)
        if e is not None:root.remove(e)
    path=HERE/'b2_locomotion.xml';ET.ElementTree(root).write(path,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    assert [m.joint(i).name for i in range(1,13)]==POLICY_JOINTS
    assert list(m.actuator_trnid[:,0]-1)==POLICY_TO_SDK
    d.qpos[:]=[0,0,0,1,0,0,0]+Q0;mujoco.mj_forward(m,d)
    feet=[m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    z=float(max(m.geom_size[f,0]-d.geom_xpos[f,2] for f in feet))+.002
    meta=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              source_use='geometry, mass, joint limits only; no policy or motion files',
              model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mass_kg=float(m.body_mass.sum()),
              standing_height_m=z,q0=Q0,joint_names=POLICY_JOINTS,policy_to_sdk=POLICY_TO_SDK,
              joint_limits=m.jnt_range[1:].tolist(),torque_limits=m.actuator_forcerange[:,1].tolist(),kp=KP,kd=KD)
    (HERE/'model.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))
    return meta

if __name__=='__main__':build()
