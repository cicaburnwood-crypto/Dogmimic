"""B2 geometry/PD only; no old policy, motion reference or NJU imports."""
import json
import xml.etree.ElementTree as ET
import mujoco
from bootstrap import HERE, ROOT
from common import sha256,write_json
from contract import Q0,KP,KD,POLICY_JOINTS,POLICY_TO_SDK

def build():
    source=ROOT/'experiments/three_skills/vendor/unitree_mujoco/unitree_robots/b2/b2.xml'
    root=ET.parse(source).getroot()
    root.find('compiler').set('meshdir',str(source.parent/'assets'))
    ET.SubElement(root,'option',timestep='.005',integrator='implicitfast',iterations='10',ls_iterations='20')
    world=root.find('worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='0 0 .1',friction='1 .005 .0001',rgba='.3 .35 .4 1')
    ET.SubElement(world,'light',pos='0 0 5',dir='0 0 -1',directional='true')
    joints={j.get('name'):j for j in root.findall('.//joint') if j.get('name')}
    for a in root.find('actuator'):
        force=a.get('ctrlrange');a.tag='position';a.set('kp',str(KP));a.set('kv',str(KD))
        a.set('forcerange',force);a.set('ctrlrange',joints[a.get('joint')].get('range'))
    for leg in ('FL','FR','RL','RR'):
        for geom in root.find(f'.//body[@name="{leg}_calf"]').findall('geom'):
            if geom.get('type')=='sphere':geom.set('name',leg+'_foot_collision')
    for name in ('keyframe','sensor'):
        element=root.find(name)
        if element is not None:root.remove(element)
    path=HERE/'b2_attitude.xml';ET.ElementTree(root).write(path,encoding='unicode')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    assert [m.joint(i).name for i in range(1,13)]==POLICY_JOINTS
    assert list(m.actuator_trnid[:,0]-1)==POLICY_TO_SDK
    d.qpos[:]=[0,0,0,1,0,0,0]+Q0;mujoco.mj_forward(m,d)
    feet=[m.geom(l+'_foot_collision').id for l in ('FL','FR','RL','RR')]
    height=float(max(m.geom_size[f,0]-d.geom_xpos[f,2] for f in feet))+.002
    d.qpos[2]=height;mujoco.mj_forward(m,d)
    result=dict(source=str(source),source_sha256=sha256(source),model_sha256=sha256(path),
                standing_height_m=height,q0=Q0,foot_positions=d.geom_xpos[feet].tolist(),
                mass_kg=float(m.body_mass.sum()),joint_names=POLICY_JOINTS,policy_to_sdk=POLICY_TO_SDK,
                joint_limits=m.jnt_range[1:].tolist(),torque_limits=m.actuator_forcerange[:,1].tolist(),kp=KP,kd=KD)
    write_json(HERE/'model.json',result)
    print(json.dumps(result))

if __name__=='__main__':build()

