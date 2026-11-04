import argparse
from pathlib import Path
import numpy as np
import mujoco
import mediapy as media
from PIL import Image,ImageDraw,ImageFont
from mjlab.asset_zoo.robots import get_g1_robot_cfg

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--label',default='G1 trained sprint policy');args=p.parse_args()
z=np.load(args.input);q=z['qpos'][:,args.trial];v=z['root_velocity'][:,args.trial];fps=float(z['fps']);target=float(z['targets'][args.trial])
spec=get_g1_robot_cfg().spec_fn()
spec.worldbody.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[200,10,.1],rgba=[.72,.76,.8,1])
for x in range(int(q[:,0].min())-3,int(q[:,0].max())+5):
    major=x%5==0
    spec.worldbody.add_geom(name=f'marker_{x}',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.02 if major else .008,1.2,.001],pos=[x,0,.001],rgba=[.1,.5,.9,1] if major else [.48,.52,.57,1],contype=0,conaffinity=0)
spec.worldbody.add_light(pos=[0,-2,4],dir=[0,0,-1],type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,diffuse=[.8,.8,.8],ambient=[.4,.4,.4])
m=spec.compile();m.vis.global_.offwidth=960;m.vis.global_.offheight=640;d=mujoco.MjData(m)
cam=mujoco.MjvCamera();cam.distance=3.;cam.azimuth=90;cam.elevation=-12
frames=[]
font=ImageFont.truetype('DejaVuSans.ttf',17)
with mujoco.Renderer(m,height=640,width=960) as renderer:
    for t in range(0,len(q),2):
        d.qpos[:]=q[t];mujoco.mj_forward(m,d);cam.lookat[:]=[q[t,0]+.25,0,.8]
        renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
        draw.rectangle([0,0,960,62],fill=(20,25,35))
        draw.text((16,8),args.label,fill='white',font=font)
        a=max(0,t-50);speed=(q[t,0]-q[a,0])/max((t-a)/fps,.02)
        draw.text((16,34),f't={t/fps:.2f}s | target {target:.1f}m/s | last 1s {speed:.2f}m/s | x={q[t,0]-q[0,0]:.1f}m | marks 1m / BLUE 5m',fill='white',font=font)
        frames.append(np.asarray(im))
Path(args.out).parent.mkdir(parents=True,exist_ok=True);media.write_video(args.out,frames,fps=fps/2)
sheet=Image.new('RGB',(960*3,640*2))
for j,t in enumerate([0,1,2,4,8,11.9]):
    sheet.paste(Image.fromarray(frames[min(len(frames)-1,int(t*fps/2))]),((j%3)*960,(j//3)*640))
sheet.save(str(Path(args.out).with_suffix('.png')))
print(args.out)
