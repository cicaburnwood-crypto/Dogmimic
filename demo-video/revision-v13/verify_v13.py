from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import cv2
import numpy as np

from edit_v13 import ROOT, BASE, OLD, NEW, FPS, SR, SPF, read_audio, activity, save

cv2.setNumThreads(2)
jobs=json.loads((ROOT/'output/timeline.json').read_text())
manifest=json.loads((ROOT/'output/edit-manifest.json').read_text())
entries=json.loads((ROOT/'audio/narration-manifest.json').read_text())


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
                                              '-show_chapters','-of','json',str(path)]))


def frame(cap,n):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(n));ok,im=cap.read();assert ok,n
    return im


checks=[]
for j in jobs:
    p=probe(j['output']);v=next(s for s in p['streams'] if s['codec_type']=='video')
    assert int(v['nb_frames'])==j['frames'],j['id']
    assert (v['width'],v['height'],v['avg_frame_rate'])==(1920,1080,'60/1')
    if not j['changed']:
        assert j['output']==j['previous_output']
        continue
    mapping=[];boundaries=[0]
    for a,b,m in j['retime']:
        ns=np.arange(b-a,dtype=np.int64)
        selected=ns[(ns*m)%(b-a)<m]+a
        assert len(selected)==m
        mapping.extend(selected.tolist());boundaries.append(len(mapping))
    samples={0,len(mapping)//2,len(mapping)-1}
    for boundary in boundaries[1:-1]:samples.update([boundary-1,boundary])
    if j['id']=='clip-3':samples.update([1800,2000,2200,2400,2600])
    src=cv2.VideoCapture(j['previous_output']);dst=cv2.VideoCapture(j['output'])
    for n in sorted(samples):
        old_frame=mapping[n];before=frame(src,old_frame);after=frame(dst,n)
        mask=np.ones(before.shape[:2],bool)
        if j['id']=='clip-3' and 1789<=old_frame<3371:mask[68:198,8:184]=False
        delta=after.astype(float)-before.astype(float)
        mae=float(np.abs(delta[mask]).mean());mse=float(np.square(delta[mask]).mean())
        psnr=float(10*np.log10(255**2/mse)) if mse else 100
        assert mae<2 and psnr>39,(j['id'],n,old_frame,mae,psnr)
        checks.append(dict(id=j['id'],output_frame=n,source_frame=old_frame,mae=mae,psnr_db=psnr))
    src.release();dst.release()
    print(f"Verified {j['id']}: {j['frames']} frames",flush=True)

p=probe(NEW);v=next(s for s in p['streams'] if s['codec_type']=='video')
assert (int(v['nb_frames']),v['width'],v['height'],v['avg_frame_rate'])==(10932,1920,1080,'60/1')
assert abs(float(v['duration'])-182.2)<1e-5
assert len(p['chapters'])==11
for exp,got in zip(manifest['chapters'],p['chapters']):
    assert abs(exp['start']-float(got['start_time']))<=.000501
    assert abs(exp['end']-float(got['end_time']))<=.000501
assert manifest['black_transition_count']==len(manifest['black_transitions'])==8
for item in manifest['black_transitions']:
    a,b=item['previous_local_frames'];lo,hi=item['output_local_frames']
    assert abs((hi-lo)-(b-a)/1.5)<=.5

mix=read_audio(ROOT/'audio/full-mix-v13.wav')
old_mix=read_audio(BASE/'revision-v12/audio/full-mix-v12.wav')
assert len(mix)==10932*SPF
for ident in ['clip-1','clip-2','clip-5']:
    j=next(j for j in jobs if j['id']==ident)
    a=j['start_frame']*SPF;b=a+j['frames']*SPF;c=round(j['previous_start']*SR)
    assert np.array_equal(mix[a:b],old_mix[c:c+b-a]),ident
last_end=0
voice_checks=[]
for e in entries:
    a=read_audio(e['audio']);lo=round(e['start']*SR);hi=lo+len(a)
    assert lo>=last_end
    assert np.max(np.abs(mix[lo:hi]-a))<2e-7
    if e['kind']=='case-title':
        before=read_audio(e['previous_audio'])
        assert e['tempo_multiplier']==1.5
        assert abs(len(a)-len(before)/1.5)<=.5
    else:
        assert e['tempo_multiplier']==1
    last_end=hi
    voice_checks.append(dict(id=e['id'],start=e['start'],end=e['end'],tempo=e['tempo_multiplier']))
card_checks=[]
for j in jobs:
    if j['kind']!='card':continue
    a=mix[j['start_frame']*SPF:(j['start_frame']+j['frames'])*SPF]
    start,end=activity(a)
    assert start/SR<.06 and 0<j['duration']-end/SR<.15
    card_checks.append(dict(id=j['id'],duration=j['duration'],voice_start=start/SR,voice_end=end/SR))


def packet_hashes(path):
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
          '-show_packets','-show_entries','packet=data_hash','-show_data_hash','sha256','-of','json',str(path)]))
    return [x['data_hash'] for x in p['packets']]


with ThreadPoolExecutor(max_workers=2) as pool:
    old_packets,new_packets=list(pool.map(packet_hashes,[OLD,NEW]))
assert new_packets[:3420]==old_packets[:3420]
assert new_packets[-2316:]==old_packets[-2316:]
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),
                '-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
report=dict(output=str(NEW),frames=10932,duration_seconds=182.2,resolution=[1920,1080],fps=60,
            black_title_cards_accelerated=8,black_title_card_and_voice_speed=1.5,
            wheeled_case2_moving_speed=6,source_frame_checks=checks,narration_checks=voice_checks,
            transition_audio_checks=card_checks,unchanged_slides_video_and_audio=True,
            unchanged_quadruped_video_and_audio=True,narration_overlap=False,full_decode='passed',
            bytes=NEW.stat().st_size,sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
save(ROOT/'output/verification.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['source_frame_checks','narration_checks']},indent=2),flush=True)
