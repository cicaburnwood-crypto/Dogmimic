import json
import mujoco
import numpy as np
from pathlib import Path
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from PIL import Image, ImageDraw

spec=get_g1_robot_cfg().spec_fn()
spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[5,5,.1],rgba=[.75,.77,.8,1])
spec.worldbody.add_light(pos=[0,-3,5],dir=[0,0,-1])
m=spec.compile();d=mujoco.MjData(m)
canvas=Image.new('RGB',(6*320,3*260),'white'); draw=ImageDraw.Draw(canvas)
for row,(clip,a,b) in enumerate([('jumps1_subject1',353,368),('jumps1_subject1',611,629),('jumps1_subject2',5286,5299)]):
    z=np.load(f'experiments/jump06/{clip}_scan.npz'); q=z['qpos'];v=z['qvel'];h=z['foot_height'];xy=z['foot_xy'].mean(axis=1)
    print(clip)
    for off in [-75,-50,-30,-20,-10,0,b-a,30,40,50,65,80,100]:
        t=a+off
        d.qpos[:]=q[t];mujoco.mj_kinematics(m,d)
        R=d.xmat[1].reshape(3,3)
        print(off, 'root',q[t,:3].round(3).tolist(),'foot_d', (xy[t]-xy[a-50]).round(3).tolist(),'foot_h',h[t].round(3).tolist(),'vel',v[t,:3].round(3).tolist(),'heading',R[:2,0].round(2).tolist())
    camera=mujoco.MjvCamera();camera.lookat[:]=[*(xy[a-50]+xy[b])/2,.65];camera.distance=2.7;camera.azimuth=120;camera.elevation=-15
    with mujoco.Renderer(m,height=230,width=320) as renderer:
        for col,off in enumerate([-50,-10,5,b-a,30,65]):
            d.qpos[:]=q[a+off];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera)
            canvas.paste(Image.fromarray(renderer.render()),(col*320,row*260+30))
            draw.text((col*320+5,row*260+5),f'{clip} frame {a+off}',fill='black')
canvas.save('experiments/jump06/reference_candidates.png')
