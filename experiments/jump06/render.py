import argparse
from pathlib import Path
import mujoco
import numpy as np
import mediapy as media
from PIL import Image,ImageDraw,ImageFont
from mjlab.asset_zoo.robots import get_g1_robot_cfg

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--trial',type=int,default=0);p.add_argument('--label',default='G1 standing broad jump | target 0.60 m');args=p.parse_args()
z=np.load(args.input);q=z['qpos'];q=q[:,args.trial] if q.ndim==3 else q
spec=get_g1_robot_cfg().spec_fn()
initial_model=spec.compile();initial_data=mujoco.MjData(initial_model)
initial_data.qpos[:]=q[0];mujoco.mj_forward(initial_model,initial_data)
foot_geoms=[i for i in range(initial_model.ngeom) if '_foot' in (mujoco.mj_id2name(initial_model,mujoco.mjtObj.mjOBJ_GEOM,i) or '') and 'collision' in (mujoco.mj_id2name(initial_model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
origin=initial_data.geom_xpos[foot_geoms,:2].mean(axis=0)
spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[5,5,.1],rgba=[.75,.78,.82,1])
for x,c in [(0,[.2,.5,1,1]),(.6,[.1,.8,.3,1])]:
    spec.worldbody.add_geom(name=f'marker_{x}',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.008,.45,.001],pos=[origin[0]+x,origin[1],.001],rgba=c,contype=0,conaffinity=0)
spec.worldbody.add_light(pos=[0,-2,4],dir=[0,0,-1],diffuse=[.8,.8,.8],ambient=[.4,.4,.4])
m=spec.compile();m.vis.global_.offwidth=720;m.vis.global_.offheight=480;d=mujoco.MjData(m)
cam=mujoco.MjvCamera();cam.lookat[:]=[origin[0]+.3,origin[1],.8];cam.distance=2.8;cam.azimuth=90;cam.elevation=-14
frames=[]
with mujoco.Renderer(m,height=480,width=720) as renderer:
    for t in range(0,len(q),2):
        d.qpos[:]=q[t];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam)
        im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle([0,0,720,55],fill=(20,25,35))
        draw.text((15,10),args.label,fill='white')
        draw.text((15,30),f't={t/50:.2f} s | BLUE=start | GREEN=0.60 m | simulation',fill='white')
        frames.append(np.array(im))
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
media.write_video(args.out,frames,fps=25)
if 'foot_height' in z:
    peak=int(np.argmax(z['foot_height'][:,args.trial].min(-1)))
else:
    peak=67
selected=np.array([0,max(0,peak-12)//2,peak//2,min(len(q)-1,peak+12)//2,max(0,len(q)-50)//2,len(frames)-1])
contact=Image.new('RGB',(3*720,2*480))
for j,i in enumerate(selected):contact.paste(Image.fromarray(frames[i]),((j%3)*720,(j//3)*480))
contact.save(str(Path(args.out).with_suffix('.png')))
print(args.out)
