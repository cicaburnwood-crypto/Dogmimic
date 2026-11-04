"""Render recorded policy evaluation states, without changing their motion."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import mjlab  # Configures mediapy's bundled ffmpeg executable.
import mediapy as media
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
p.add_argument('--out',required=True)
p.add_argument('--trial',type=int,default=0)
p.add_argument('--rollout-name',default='best_rollout.npz')
p.add_argument('--evaluation-name',default='best_eval.json')
p.add_argument('--baseline',default='')
p.add_argument('--label',default='')
p.add_argument('--note',default='')
a=p.parse_args()
run=Path(a.run);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
data=np.load(run/a.rollout_name)
state=data['state'][:,a.trial];q=state[:,:19]
alive=data['alive'][:,a.trial];dt=float(data['dt'])
feet=data['feet'][:,a.trial]
evaluation=json.loads((run/a.evaluation_name).read_text())
is_cycle='com_acc' in data
has_push='force_world' in data
baseline=np.load(a.baseline) if a.baseline else None
first_failure=int(np.flatnonzero(~alive)[0]) if not alive.all() else len(q)-1
baseline_failure=(int(np.flatnonzero(~baseline['alive'][:,a.trial])[0]) if not baseline['alive'][:,a.trial].all() else len(q)-1) if baseline is not None else None
assert q.shape[1]==19 and np.isfinite(q).all()
spec=mujoco.MjSpec.from_file(str(evaluation.get('model_path',Path(__file__).parent/'b2_train.xml')))
spec.add_texture(name='sky',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=[.45,.55,.65],rgb2=[.82,.87,.91],width=256,height=1536)
spec.geom('floor').rgba=[.72,.76,.80,1]
for axis in range(2):
    for i in range(-8,9):
        pos=[0.,0.,.0006];pos[axis]=i*.25
        size=[2.,2.,.0005];size[axis]=.003
        spec.worldbody.add_geom(name=f'grid_{axis}_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,size=size,pos=pos,
                               rgba=[.58,.63,.68,1],contype=0,conaffinity=0)
spec.worldbody.add_light(pos=[1,-3,4],dir=[-.2,.4,-1],type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                         diffuse=[.7,.7,.7],ambient=[.35,.35,.35])
m=spec.compile();m.vis.global_.offwidth=640;m.vis.global_.offheight=640
m.light_diffuse[:]*=.45
m.light_ambient[:]=.10
m.vis.headlight.diffuse[:]=.25
m.vis.headlight.ambient[:]=.18
d=mujoco.MjData(m)
opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0]
cameras=[]
for az in ((145,145) if baseline is not None else (90,145)):
    cam=mujoco.MjvCamera();cam.distance=3.15;cam.azimuth=az;cam.elevation=-12
    cam.lookat[:]=[-.20,0.,.8];cameras.append(cam)
font=ImageFont.truetype('DejaVuSans.ttf',20)
small=ImageFont.truetype('DejaVuSans.ttf',17)
frames=[]
with mujoco.Renderer(m,height=640,width=640) as renderer:
    for i in range(0,len(q),2):
        frame=Image.new('RGB',(1280,720),(20,25,35))
        for j,cam in enumerate(cameras):
            if j==0 and baseline is not None:
                qi=min(i,baseline_failure);d.qpos[:]=baseline['state'][qi,a.trial,:19]
            else:
                qi=min(i,first_failure);d.qpos[:]=q[qi]
            mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=cam,scene_option=opt)
            if has_push and (baseline is None or j==1) and i<=first_failure:
                force=data['force_world'][i,a.trial]
                if np.linalg.norm(force)>0:
                    scene=renderer.scene;geom=scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array([1.,.3,.03,1.],dtype=np.float32))
                    end=d.xipos[1].copy();begin=end-force*.035
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.025,begin,end)
                    scene.ngeom+=1
            frame.paste(Image.fromarray(renderer.render()),(640*j,80))
        draw=ImageDraw.Draw(frame)
        title=(f'B2 v2 | full cycle | +{evaluation["it"]:,} training iterations | actual policy | 1x speed' if is_cycle
               else f'Unitree B2 | trained policy rollout | iteration {evaluation["it"]:,} | 1x speed')
        if has_push:title=f'B2 v2 | paired random-push test | trial {a.trial} (preselected) | frozen policy | 1x speed'
        if a.label:title=a.label
        draw.text((20,10),title,font=font,fill='white')
        w,x,y,z=q[i,3:7]
        inclination=np.degrees(np.arcsin(np.clip(2*(x*z-w*y),-1,1)))
        front=float(feet[i,:2,2].min())
        if is_cycle:
            now=(i+1)*dt
            phase=('FOUR-LEG STAND' if now<1 else 'RISE' if now<3.5 else 'HIND-LEG HOLD' if now<9.5 else 'RETURN TO FOUR LEGS' if now<12.5 else 'FOUR-LEG SETTLE')
            acc=data['com_acc'][i,a.trial]
            subtitle=f't={now:.2f}s   |   {phase}   |   COM acceleration XYZ [{acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f}] m/s^2'
            if has_push:
                force=data['force_world'][i,a.trial];magnitude=float(np.linalg.norm(force))
                subtitle=f't={now:.2f}s | {phase} | PUSH {magnitude:.1f} N | force XYZ [{force[0]:+.1f}, {force[1]:+.1f}, {force[2]:+.1f}] N'
        else:
            subtitle=f't={(i+1)*dt:.2f}s   |   body inclination {inclination:.1f} deg   |   front feet height {front:.2f} m   |   nominal trial {a.trial}'
        draw.text((20,43),subtitle,font=small,fill=(190,213,235))
        draw.text((18,92),'NO PUSH (matched control)' if baseline is not None else 'SIDE VIEW',font=small,fill=(240,246,252),stroke_width=1,stroke_fill=(45,55,65))
        draw.text((658,92),'RANDOM PUSH' if has_push else 'OBLIQUE VIEW',font=small,fill=(255,190,90) if has_push else (240,246,252),stroke_width=1,stroke_fill=(45,55,65))
        if has_push:
            acc=data['com_acc'][min(i,first_failure),a.trial]
            draw.rectangle([640,688,1280,720],fill=(20,25,35))
            draw.text((653,694),f'COM a XYZ [{acc[0]:+.3f}, {acc[1]:+.3f}, {acc[2]:+.3f}] m/s^2',font=small,fill=(190,213,235))
        if a.note:
            draw.rectangle([0,688,640,720],fill=(100,65,12))
            draw.text((13,694),a.note,font=small,fill='white')
        for j,failed in enumerate((baseline_failure if baseline is not None else first_failure,first_failure)):
            is_failed=(baseline is not None and j==0 and not baseline['alive'][i,a.trial]) or ((baseline is None or j==1) and not alive[i])
            if is_failed:
                draw.rectangle([j*640,119,j*640+640,155],fill=(150,25,25))
                draw.text((j*640+15,127),f'FAILED at {(failed+1)*dt:.2f}s - terminal state frozen',font=small,fill='white')
        frames.append(np.asarray(frame))
media.write_video(str(out),frames,fps=1/(2*dt),codec='h264')
times=(0.,6.,11.,15.4) if is_cycle else (0.,2.,8.8)
if has_push:
    push_times=np.flatnonzero(np.linalg.norm(data['force_world'][:,a.trial],axis=-1)>0)
    if len(push_times):times=(0.,(push_times[0]+1)*dt,(push_times[len(push_times)//2]+1)*dt,15.4)
sheet=Image.new('RGB',(1280,720*len(times)))
for j,t in enumerate(times):
    sheet.paste(Image.fromarray(frames[min(len(frames)-1,int(t/(2*dt)))]),(0,j*720))
sheet.save(out.with_suffix('.png'))
metadata=dict(source=str(run/a.rollout_name),checkpoint=evaluation.get('checkpoint',str(run/'best.pt')),iteration=evaluation['it'],trial=a.trial,
              all_frames_active=bool(alive.all()),frames=len(frames),fps=1/(2*dt),duration_s=len(frames)*2*dt,views=2,
              actual_policy_trajectory=True,reference_motion_visualization=False)
if a.label:metadata['label']=a.label
if a.note:metadata['note']=a.note
if 'trials_detail' in evaluation:
    chosen=evaluation['trials_detail'][a.trial]
    metadata.update(strict_success=chosen['success'],failure_reasons=chosen.get('failure_reasons',[]),
                    hind_foot_xy_drift_m=chosen.get('hind_foot_xy_drift_m'),model_path=evaluation.get('model_path'))
if is_cycle:
    metadata.update(source_iteration=evaluation['source_iteration'],full_cycle=True,
                    total_policy_training_iterations=evaluation['source_iteration']+evaluation['it'],
                    evaluation_success_rate=evaluation['success_rate'],nominal_success=evaluation['nominal_success'])
if has_push:metadata.update(paired_control=a.baseline,force_range_n=evaluation['force_range_n'],pushes_per_trial=evaluation['pushes_per_trial'],failed_episodes_frozen_not_reset=True)
out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata),flush=True)
print(str(out),flush=True)
