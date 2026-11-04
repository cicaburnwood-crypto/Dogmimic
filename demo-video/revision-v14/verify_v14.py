from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from edit_v14 import ROOT, BASE, OLD, NEW, HUMAN_2X, FPS, SR, SPF, read_audio, activity, save, map_local_frame

cv2.setNumThreads(2)
jobs = json.loads((ROOT/'output/timeline.json').read_text())
manifest = json.loads((ROOT/'output/edit-manifest.json').read_text())
entries = json.loads((ROOT/'audio/narration-manifest.json').read_text())


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
                                              '-show_chapters','-of','json',str(path)]))


def frame(cap,n):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(n))
    ok,im=cap.read()
    assert ok,n
    return im


def source_mapping(j):
    result=[];boundaries=[0]
    for a,b,m in j['retime']:
        if j['id']=='clip-4' and a==977:
            result.extend((str(HUMAN_2X),f) for f in range(1348,1684))
        else:
            ns=np.arange(b-a,dtype=np.int64)
            selected=ns[(ns*m)%(b-a)<m]+a
            assert len(selected)==m
            result.extend((j['previous_output'],int(f)) for f in selected)
        boundaries.append(len(result))
    return result,boundaries


checks=[]
for j in jobs:
    p=probe(j['output']);v=next(s for s in p['streams'] if s['codec_type']=='video')
    assert (int(v['nb_frames']),v['width'],v['height'],v['avg_frame_rate'])==(j['frames'],1920,1080,'60/1')
    if not j['changed']:
        assert j['output']==j['previous_output']
        continue
    mapping,boundaries=source_mapping(j)
    assert len(mapping)==j['frames']
    samples={0,len(mapping)//2,len(mapping)-1}
    for boundary in boundaries[1:-1]:
        samples.update([boundary-1,boundary])
    for closeup in manifest['closeups']:
        if closeup['clip']!=j['id']:continue
        a,b=map(int,closeup['output_local_frames'])
        samples.update(np.linspace(a,b-1,5,dtype=int).tolist())
    if j['id']=='clip-4':
        samples.update([940,1040,1140,1275,1276,1350])
    srcs={path:cv2.VideoCapture(path) for path in set(p for p,_ in mapping)}
    dst=cv2.VideoCapture(j['output'])
    for n in sorted(samples):
        path,old_n=mapping[n]
        before,after=frame(srcs[path],old_n),frame(dst,n)
        mask=np.ones(before.shape[:2],bool)
        if j['id']=='clip-3' and 2812<=old_n<3516:
            mask[68:198,8:184]=False
        if j['id']=='clip-4' and path==str(HUMAN_2X):
            mask[111:208,355:493]=False
        delta=after.astype(float)-before.astype(float)
        mae=float(np.abs(delta[mask]).mean())
        mse=float(np.square(delta[mask]).mean())
        psnr=float(10*np.log10(255**2/mse)) if mse else 100
        assert mae<2 and psnr>39,(j['id'],n,old_n,mae,psnr)
        checks.append(dict(id=j['id'],output_frame=int(n),source=path,source_frame=old_n,mae=mae,psnr_db=psnr))
    for cap in srcs.values():cap.release()
    dst.release()
    print(f"Verified {j['id']}: {j['frames']} frames",flush=True)

p=probe(NEW)
v=next(s for s in p['streams'] if s['codec_type']=='video')
a=next(s for s in p['streams'] if s['codec_type']=='audio')
assert (int(v['nb_frames']),v['width'],v['height'],v['avg_frame_rate'])==(10667,1920,1080,'60/1')
assert abs(float(v['duration'])-10667/60)<1e-5
assert abs(float(a['duration'])-10667/60)<.025
assert len(p['chapters'])==11
for exp,got in zip(manifest['chapters'],p['chapters']):
    assert abs(exp['start']-float(got['start_time']))<=.000501
    assert abs(exp['end']-float(got['end_time']))<=.000501
for c in manifest['closeups']:
    a,b=c['previous_local_frames'];lo,hi=c['output_local_frames']
    assert abs((hi-lo)-(b-a)/1.2)<=.5
assert manifest['closeup_count']==5

old_manifest=json.loads((BASE/'revision-v13/output/edit-manifest.json').read_text())
black=[]
for item in old_manifest['black_transitions']:
    j=next(j for j in jobs if j['id']==item['id'])
    a,b=item['output_local_frames']
    lo,hi=map_local_frame(j,a),map_local_frame(j,b)
    assert hi-lo==b-a
    black.append(dict(id=j['id'],output_frames=[lo,hi],duration=(hi-lo)/60))

mix=read_audio(ROOT/'audio/full-mix-v14.wav')
old_mix=read_audio(BASE/'revision-v13/audio/full-mix-v13.wav')
assert len(mix)==10667*SPF
residual=mix.copy()
for j in jobs:
    if j['id'] not in ['clip-1','clip-2','before-3','before-4','before-5']:continue
    a=j['start_frame']*SPF;b=a+j['frames']*SPF;c=round(j['previous_start']*SR)
    assert np.array_equal(mix[a:b],old_mix[c:c+b-a]),j['id']
    residual[a:b]=0
last_end=0
voice_checks=[]
for e in entries:
    a=read_audio(e['audio']);lo=round(e['start']*SR);hi=lo+len(a)
    before=read_audio(e['previous_audio'])
    assert lo>=last_end
    assert np.max(np.abs(mix[lo:hi]-a))<2e-7
    if e['kind']=='callout':
        assert e['tempo_multiplier']==1.2
        assert abs(len(a)-len(before)/1.2)<=.5
        assert activity(a)[1]<len(a)
    else:
        assert e['tempo_multiplier']==1
        assert np.array_equal(a,before)
    residual[lo:hi]-=a
    last_end=hi
    voice_checks.append(dict(id=e['id'],start=e['start'],end=e['end'],tempo=e['tempo_multiplier']))
assert np.abs(residual).max()<2e-7
assert all(e['id']!='wheeled-lighting' for e in entries)
w3=next(c for c in manifest['closeups'] if c['id']=='wheeled-lighting')
assert np.abs(mix[round(w3['output_global_seconds'][0]*SR):round(jobs[3]['end']*SR)]).max()==0
quad=jobs[-1]
mapping,_=source_mapping(quad)
assert mapping[-1][1]==2243
assert quad['previous_frames']-(mapping[-1][1]+1)==72
assert np.abs(mix[-SR//2:]).max()==0


def packets(path):
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_packets','-show_entries','packet=data_hash','-show_data_hash','sha256','-of','json',str(path)]))
    return [x['data_hash'] for x in p['packets']]


with ThreadPoolExecutor(max_workers=2) as pool:
    old_packets,new_packets=list(pool.map(packets,[OLD,NEW]))
for j in jobs:
    if j['changed']:continue
    a=j['start_frame'];b=a+j['frames'];c=round(j['previous_start']*FPS)
    assert new_packets[a:b]==old_packets[c:c+j['frames']],j['id']
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),
                '-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)

preview=[]
for c in manifest['closeups']:
    lo,hi=c['output_global_seconds']
    preview.append((c['id'],(lo+hi)/2))
wheel=next(j for j in jobs if j['id']=='clip-3')
human=next(j for j in jobs if j['id']=='clip-4')
preview += [('Wheeled Case 3 - 2.5X',wheel['start']+50),
            ('Elevator approach - 2X',human['start']+17),
            ('Elevator door crossing - 2X',human['start']+21.2),
            ('After elevator - 3X',human['start']+22.5),
            ('Last retained frame',jobs[-1]['end']-1/60)]
canvas=Image.new('RGB',(1920,5*565),(20,20,20));draw=ImageDraw.Draw(canvas)
cap=cv2.VideoCapture(str(NEW))
for i,(label,t) in enumerate(preview):
    im=Image.fromarray(cv2.cvtColor(frame(cap,round(t*60)),cv2.COLOR_BGR2RGB)).resize((960,540))
    x,y=i%2*960,i//2*565
    canvas.paste(im,(x,y+25));draw.text((x+8,y+6),f'{label} | {t:.3f}s',fill='white')
cap.release();canvas.save(ROOT/'inspect/final-contact.jpg',quality=94)
report=dict(output=str(NEW),frames=10667,duration_seconds=10667/60,resolution=[1920,1080],fps=60,
            closeups_accelerated=5,existing_callout_voices_accelerated=4,closeup_and_voice_tempo=1.2,
            wheeled_case3_moving_speed=2.5,humanoid_case2_before_exit_speed=2,
            humanoid_case2_after_exit_speed=3,tail_removed_frames=72,
            source_frame_checks=checks,narration_checks=voice_checks,black_transitions=black,
            unchanged_slides_and_black_transition_video_and_audio=True,
            narration_overlap=False,field_audio_silent=True,full_decode='passed',
            bytes=NEW.stat().st_size,sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
save(ROOT/'output/verification.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['source_frame_checks','narration_checks','black_transitions']},indent=2),flush=True)
