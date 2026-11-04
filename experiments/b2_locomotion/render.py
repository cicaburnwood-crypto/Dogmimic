"""Render saved actual-physics rollouts; preserve all poses and 1x timing."""
import argparse
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mjlab  # Registers the available ffmpeg binary with mediapy.
import mediapy as media
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
OVERVIEW=[
    ('x+2','Forward / reverse | 2.00 m/s'),
    ('y+0.6','Sideways | 0.60 m/s'),
    ('diagonal+1_+1_2','Diagonal | total speed 2.00 m/s'),
    ('spin+3','Turn in place | 3.00 rad/s'),
    ('combined_1_+1','Move + turn | xy (1.00, 0.60), yaw 1.00'),
    ('combined_1.9_+1','Move + turn | xy (1.90, 0.60), yaw 1.00'),
]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def model_for_render(width,height):
    tree=ET.parse(HERE/'b2_locomotion.xml').getroot()
    asset=tree.find('asset')
    ET.SubElement(asset,'texture',name='preview_checker',type='2d',builtin='checker',
                  rgb1='.69 .74 .79',rgb2='.77 .81 .85',width='512',height='512')
    ET.SubElement(asset,'material',name='preview_floor',texture='preview_checker',texrepeat='2 2',texuniform='true',reflectance='0')
    tree.find('.//geom[@name="floor"]').set('material','preview_floor')
    tree.find('.//geom[@name="floor"]').set('rgba','1 1 1 1')
    spec=mujoco.MjSpec.from_string(ET.tostring(tree,encoding='unicode'))
    spec.add_texture(name='preview_sky',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.50,.61,.72],rgb2=[.86,.90,.94],width=256,height=1536)
    spec.worldbody.add_light(pos=[2,-3,5],dir=[-.3,.4,-1],type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[.55,.55,.55],ambient=[.25,.25,.25])
    m=spec.compile();m.vis.global_.offwidth=width;m.vis.global_.offheight=height
    m.vis.quality.shadowsize=4096
    m.light_diffuse[:]*=.55;m.light_ambient[:]=.12
    m.vis.headlight.diffuse[:]=.35;m.vis.headlight.ambient[:]=.22
    expected=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
    assert m.nq==19 and m.nv==18 and [m.joint(i).name for i in range(1,13)]==expected
    return m

def phase(t):
    if t<1.5:return 'STAND',(.45,.85,.7)
    if t<5.5:return 'MOVE',(.3,.7,1.)
    if t<8.5:return 'STOP + SETTLE',(.45,.85,.7)
    if t<12.5:return 'REVERSE COMMAND',(.95,.65,.3)
    return 'STOP + SETTLE',(.45,.85,.7)

def body_metrics(s,rotation):
    # Same horizontal heading definition used by the B2 training task.
    heading=np.arctan2(-rotation[0,1],rotation[1,1]);c=np.cos(heading);sn=np.sin(heading)
    v=s[19:22]
    return np.array([c*v[0]+sn*v[1],-sn*v[0]+c*v[1],(rotation.T@s[22:25])[2]])

def arrow(scene,start,end,color,width=.014):
    if np.linalg.norm(end-start)<.015:return
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array(color,dtype=np.float32))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,width,start,end);scene.ngeom+=1

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=('overview','detail'),default='overview');p.add_argument('--trial',type=int,default=0)
    p.add_argument('--case',default='x+2');p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    start=time.time();run=Path(a.run);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    ev=json.loads((run/'best_eval.json').read_text());record=np.load(run/'best_rollout.npz')
    checkpoint_hash=sha(run/'best.pt');assert checkpoint_hash==ev['checkpoint_sha256']
    source_hash=sha(run/'best_rollout.npz');dt=float(record['dt']);T=len(record['state']);fps=round(1/dt)
    assert T==850 and abs(dt-.02)<1e-9
    chosen=OVERVIEW if a.mode=='overview' else [(a.case,'SIDE VIEW'),(a.case,'OBLIQUE VIEW')]
    names=[g['name'] for g in ev['cases']];entries=[]
    for name,label in chosen:
        ci=names.index(name);matches=np.flatnonzero((record['case_index']==ci)&(record['trial_index']==a.trial))
        assert len(matches)==1,(name,a.trial)
        col=int(matches[0]);state=record['state'][:,col]
        assert np.isfinite(state).all() and record['alive'][:,col].all()
        assert np.max(np.abs(np.linalg.norm(state[:,3:7],axis=-1)-1))<1e-5
        row=ev['cases'][ci]['trials'][a.trial]
        entries.append(dict(name=name,label=label,column=col,case_index=ci,state=state,
                            commands=record['commands'][:,col],success=bool(record['success'][col]),
                            evaluation=row,passed=ev['cases'][ci]['passed']))
    if a.mode=='overview':width,height=1920,1080;vw,vh=640,422;header=88;rowh=484;columns=3
    else:width,height=1280,720;vw,vh=640,560;header=88;rowh=620;columns=2
    m=model_for_render(vw,vh);d=mujoco.MjData(m)
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,1,0,0,0]
    cams=[]
    for j in range(len(entries)):
        cam=mujoco.MjvCamera();cam.distance=3.15 if a.mode=='overview' else 3.05
        cam.azimuth=135. if a.mode=='overview' else (90. if j==0 else 140.)
        cam.elevation=-22.;cams.append(cam)
    title_font=ImageFont.truetype('DejaVuSans.ttf',25 if a.mode=='overview' else 22)
    font=ImageFont.truetype('DejaVuSans.ttf',19);small=ImageFont.truetype('DejaVuSans.ttf',17)
    dark=(18,25,35);muted=(185,205,223);gold=(255,196,86);blue=(92,209,240)
    sample_indices=[0,174,274,424,549,849];snapshots={};progress=time.time()
    def frame_at(i,renderer):
        frame=Image.new('RGB',(width,height),dark);draw=ImageDraw.Draw(frame)
        title=f'B2 locomotion | best policy {ev["it"]:,} | actual physics recording | 1x speed'
        draw.text((18,9),title,font=title_font,fill='white')
        ph,col=phase(i*dt);color=tuple(int(x*255) for x in col)
        draw.text((18,47),f't={(i+1)*dt:05.2f}s / {T*dt:.2f}s | {ph} | fixed trial {a.trial} | ground pattern fixed in world',font=font,fill=color)
        for j,e in enumerate(entries):
            x=(j%columns)*vw;y=header+(j//columns)*rowh;s=e['state'][i];cmd=e['commands'][i]
            d.qpos[:]=s[:19];mujoco.mj_forward(m,d)
            R=d.xmat[1].reshape(3,3);actual=body_metrics(s,R)
            cam=cams[j];cam.lookat[:]=[s[0],s[1],.43]
            renderer.update_scene(d,camera=cam,scene_option=opt)
            heading=np.arctan2(-R[0,1],R[1,1]);c=np.cos(heading);sn=np.sin(heading)
            vector=np.array([c*cmd[0]-sn*cmd[1],sn*cmd[0]+c*cmd[1],0.])
            # Display-only ground arrows; recorded robot poses are never changed.
            begin=np.array([s[0],s[1],.04]);arrow(renderer.scene,begin,begin+.5*vector,[1.,.65,.08,.85])
            draw.rectangle((x,y,x+vw-1,y+60),fill=(27,37,49))
            label=e['label'] if a.mode=='overview' else e['label']+' | forward / reverse 2.00 m/s'
            draw.text((x+12,y+5),label,font=font,fill='white')
            draw.text((x+12,y+31),f'cmd  x {cmd[0]:+.2f}  y {cmd[1]:+.2f} m/s  yaw {cmd[2]:+.2f} rad/s',font=small,fill=gold)
            frame.paste(Image.fromarray(renderer.render()),(x,y+60))
            draw=ImageDraw.Draw(frame)
            bottom=y+60+vh-52
            draw.rectangle((x,bottom,x+vw-1,bottom+51),fill=dark)
            draw.text((x+12,bottom+4),f'actual x {actual[0]:+.2f}  y {actual[1]:+.2f} m/s  yaw {actual[2]:+.2f} rad/s',font=small,fill=blue)
            draw.text((x+12,bottom+27),f'Case: {e["passed"]}/16 PASS | trial {a.trial}: {"PASS" if e["success"] else "FAIL"}',font=small,fill=(160,225,190))
        draw.rectangle((0,height-5,int(width*(i+1)/T),height-1),fill=color)
        return frame
    with mujoco.Renderer(m,height=vh,width=vw) as renderer:
        if a.preview_only:
            for i in sample_indices:snapshots[i]=frame_at(i,renderer)
        else:
            with media.VideoWriter(str(out),shape=(height,width),fps=fps,codec='h264') as writer:
                for i in range(T):
                    frame=frame_at(i,renderer);writer.add_image(np.asarray(frame))
                    if i in sample_indices:snapshots[i]=frame.copy()
                    if time.time()-progress>20:
                        print(json.dumps(dict(mode=a.mode,frame=i+1,total=T,elapsed_seconds=round(time.time()-start,1))),flush=True);progress=time.time()
    sheet=Image.new('RGB',(1280,3*int(640*height/width)),dark)
    for k,i in enumerate(sample_indices):
        thumb=snapshots[i].resize((640,int(640*height/width)))
        sheet.paste(thumb,((k%2)*640,(k//2)*thumb.height))
    sheet.save(out.with_suffix('.png'));snapshots[174].save(out.with_name(out.stem+'_frame.png'))
    meta=dict(source=str(run/'best_rollout.npz'),source_sha256=source_hash,checkpoint=str(run/'best.pt'),checkpoint_sha256=checkpoint_hash,
              iteration=ev['it'],mode=a.mode,trial=a.trial,trial_selection='fixed nominal trial zero; not chosen for score' if a.trial==0 else 'explicit trial index',
              cases=[dict(name=e['name'],case_index=e['case_index'],success=e['success'],case_passed=e['passed'],trials=16,metrics=e['evaluation']) for e in entries],
              model=str(HERE/'b2_locomotion.xml'),model_sha256=sha(HERE/'b2_locomotion.xml'),frames=T,fps=fps,duration_s=T*dt,
              width=width,height=height,playback_speed=1.,all_frames_active=True,actual_policy_trajectory=True,
              motion_reference_used=False,pose_edits=False,time_interpolation=False,first_state_time_s=dt,
              camera='fixed world orientation, translates with recorded root XY',render_wall_seconds=time.time()-start,
              physics_evaluation_core_passed=ev['core_passed'],physics_evaluation_core_total=ev['core_total'])
    if not a.preview_only:out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),mode=a.mode,preview_only=a.preview_only,frames=T,fps=fps,duration_s=T*dt,iteration=ev['it'],render_seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
