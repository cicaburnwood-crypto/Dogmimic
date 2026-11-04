from pathlib import Path
import hashlib
import json
import subprocess

import cv2
import numpy as np
import soundfile as sf

from edit_v12 import ROOT, NEW, FPS, SR, SPF, activity, read_audio, save_json

cv2.setNumThreads(2)
jobs = json.loads((ROOT/'output/timeline.json').read_text())
manifest = json.loads((ROOT/'output/edit-manifest.json').read_text())
entries = json.loads((ROOT/'audio/narration-manifest.json').read_text())
assert [s['duration'] for s in manifest['slides']] == [15,15,10,12]
assert manifest['title']['duration'] == 5
assert 4.9 < manifest['title']['speech_seconds'][1] < 5


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe','-v','error','-show_streams','-show_chapters','-show_format','-of','json',str(path)]))


def frame(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(f))
    ok, im = cap.read()
    assert ok, f
    return im


checks=[]
for job in jobs:
    data=probe(job['output'])
    v=next(s for s in data['streams'] if s['codec_type']=='video')
    assert int(v['nb_frames'])==job['frames'], (job['id'],v['nb_frames'],job['frames'])
    assert v['avg_frame_rate']=='60/1' and (v['width'],v['height'])==(1920,1080)
    assert abs(float(v['duration'])-job['duration'])<1e-5
    mapping=[];boundaries=[0]
    for a,b,m in job['retime']:
        ns=np.arange(b-a,dtype=np.int64)
        keep=ns[(ns*m)%(b-a)<m]+a
        assert len(keep)==m
        mapping.extend(keep.tolist());boundaries.append(len(mapping))
    assert len(mapping)==job['frames']
    samples={0,len(mapping)//2,len(mapping)-1}
    for b in boundaries[1:-1]:samples.update([b-1,b])
    if job['id']=='clip-3':samples.update([180,300,470,471,1200])
    if job['id']=='clip-4':samples.update([293,294,500,956,957,1016,1017,1200])
    before=cv2.VideoCapture(job['previous_output']);after=cv2.VideoCapture(job['output'])
    for f in sorted(samples):
        old_frame=mapping[f]
        src=frame(before,old_frame);dst=frame(after,f)
        allowed=np.ones(src.shape[:2],bool)
        if job['id']=='clip-3' and 180<=old_frame<1900:
            allowed[68:198,8:184]=False
        if job['id']=='clip-4' and 294<=old_frame<1288:
            allowed[4:109,289:441]=False
        if job['id']=='clip-4' and 1348<=old_frame<2098:
            allowed[111:208,355:493]=False
        delta=dst.astype(float)-src.astype(float)
        error=float(np.abs(delta[allowed]).mean())
        mse=float(np.square(delta[allowed]).mean())
        psnr=float(10*np.log10(255**2/mse)) if mse else 100
        assert error<2 and psnr>39,(job['id'],f,old_frame,error,psnr)
        checks.append(dict(id=job['id'],output_frame=f,source_frame=old_frame,
                           mean_error_outside_badge=error,psnr_db=psnr))
    before.release();after.release()
    print(f"Verified {job['id']}: {job['frames']} frames",flush=True)

full=probe(NEW)
v=next(s for s in full['streams'] if s['codec_type']=='video')
assert (v['width'],v['height'],v['avg_frame_rate'],int(v['nb_frames']))==(1920,1080,'60/1',11812)
assert abs(float(v['duration'])-11812/FPS)<1e-5
assert len(full['chapters'])==11
for expected,actual in zip(manifest['chapters'],full['chapters']):
    # MP4 chapter timestamps use milliseconds, even with 60 kHz video timebase.
    num,den=map(int,actual['time_base'].split('/'))
    quantum=num/den
    assert abs(expected['start']-float(actual['start_time']))<=quantum/2+1e-6
    assert abs(expected['end']-float(actual['end_time']))<=quantum/2+1e-6

mix=read_audio(ROOT/'audio/full-mix-v12.wav')
assert len(mix)==11812*SPF
voice_checks=[]
last_end=0
for e in entries:
    a=read_audio(e['audio']);start=round(e['start']*SR);end=start+len(a)
    assert start>=last_end
    assert np.abs(mix[start:end]-a).max()<2e-7
    voice_checks.append(dict(id=e['id'],start=e['start'],end=e['end'],full_utterance_preserved=True))
    last_end=end
card_checks=[]
for card in manifest['cards']:
    j=next(j for j in jobs if j['id']==card['id'])
    a=mix[j['start_frame']*SPF:(j['start_frame']+j['frames'])*SPF]
    first,end=activity(a)
    assert first/SR<.06 and 0<j['duration']-end/SR<.11
    card_checks.append(dict(id=j['id'],duration=j['duration'],leading_silence=first/SR,
                            trailing_silence=j['duration']-end/SR))
last=jobs[-1]
dog_audio=mix[last['start_frame']*SPF:].copy()
e=next(e for e in entries if e['id']=='quadruped-memory')
voice=read_audio(e['audio']);start=round(e['start']*SR)-last['start_frame']*SPF
dog_audio[start:start+len(voice)]-=voice
assert np.abs(dog_audio).max()<2e-7
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),
                '-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
report=dict(output=str(NEW),frames=11812,duration_seconds=11812/FPS,resolution=[1920,1080],
            fps=60,chapters=11,slide_duration_seconds=[5,15,15,10,12],
            source_frame_checks=checks,narration_checks=voice_checks,transition_checks=card_checks,
            narration_overlap=False,quadruped_original_audio_muted=True,full_decode='passed',
            bytes=NEW.stat().st_size,sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
save_json(ROOT/'output/verification.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['source_frame_checks','narration_checks']},indent=2),flush=True)
