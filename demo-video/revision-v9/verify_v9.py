from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import cv2
import numpy as np
import soundfile as sf
from PIL import Image,ImageDraw
from edit_v9 import ROOT,BASE,OLD,NEW,SOURCE,SEGMENT,START,END,REMOVED,SHIFT,NEW_FRAMES,DURATION,SR


def packets(path):
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets',
             '-show_entries','packet=data_hash','-show_data_hash','sha256','-of','json',str(path)]))
    return [s['data_hash'] for s in p['packets']]


def frame(cap,f):
    cap.set(1,f);ok,im=cap.read();assert ok,f
    return im


with ThreadPoolExecutor(max_workers=2) as pool:
    a,b=list(pool.map(packets,[OLD,NEW]))
assert len(a)==14552 and len(b)==NEW_FRAMES
assert a[:4842]==b[:4842]
assert a[9538:]==b[9388:]
p=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters',
                                    '-show_format','-of','json',str(NEW)]))
v=next(s for s in p['streams'] if s['codec_type']=='video')
assert (v['width'],v['height'],v['avg_frame_rate'],int(v['nb_frames']))==(1920,1080,'60/1',NEW_FRAMES)
assert abs(float(v['duration'])-DURATION)<1e-5
assert abs(float(p['format']['duration'])-DURATION)<.002
assert len(p['chapters'])==8

source=cv2.VideoCapture(str(SOURCE));new=cv2.VideoCapture(str(SEGMENT))
samples=[120,1200,2490,3662,3800,3840,3842,3843,3852,3872,3902,3932,3962,3990,3992,3994,4200,4545]
sheet=Image.new('RGB',(1920,1800),(25,25,25));draw=ImageDraw.Draw(sheet);checks={}
for i,f in enumerate(samples):
    oldf=f if f<START else START+2*(f-START) if f<START+REMOVED else f+REMOVED
    im=frame(new,f);ref=frame(source,oldf)
    delta=float(np.abs(im.astype(float)-ref).mean())
    assert delta<1.5,(f,oldf,delta)
    checks[str(f)]={'source_frame':oldf,'mean_absolute_difference':delta}
    x=i%3*640;y=i//3*300
    sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((533,300)),(x,y))
    draw.text((x+536,y+8),f'{f/60:.3f}s',fill='white')
    draw.text((x+536,y+28),f'src {oldf}',fill='white')
sheet.save(ROOT/'inspect/verification-storyboard.jpg',quality=93)
source.release();new.release()

mix,sr=sf.read(ROOT/'audio/full-mix-v9.wav',dtype='float32',always_2d=True)
assert sr==SR and len(mix)==NEW_FRAMES*(SR//60)
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(NEW),'-map','0:a:0','-ar',str(SR),'-ac','2','-f','f32le','-'])
decoded=np.frombuffer(raw,'<f4').reshape(-1,2)
assert len(decoded)>=len(mix) and np.max(np.abs(decoded))<1
silent_peak=float(np.max(np.abs(decoded[round(144.68*SR):round(156.45*SR)])))
assert silent_peak<1e-5
entries=json.loads((ROOT/'audio/narration-manifest.json').read_text())
assert not any(e['id']=='wheeled-lighting' for e in entries)
assert any(e['id']=='wheeled-case-3' for e in entries)
audio_checks=[]
for e in entries:
    lo=round(e['start']*SR);hi=round(e['end']*SR)
    corr=float(np.corrcoef(mix[lo:hi,0],decoded[lo:hi,0])[0,1])
    assert corr>.98,(e['id'],corr)
    audio_checks.append(dict(id=e['id'],start=e['start'],end=e['end'],correlation=corr))
quad=next(e for e in entries if e['id']=='quadruped-memory')
assert np.max(np.abs(decoded[round(197.5*SR):round((quad['start']-.08)*SR)]))<1e-5
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
report=dict(duration_seconds=DURATION,frames=NEW_FRAMES,resolution=[1920,1080],fps=60,chapters=8,
            closeup_original_duration_seconds=5,closeup_new_duration_seconds=2.5,
            closeup_speed_multiplier=2,removed_callout_narration=True,case3_title_narration_preserved=True,
            closeup_and_following_case3_silent_peak=silent_peak,
            other_segments_video_packets_identical_to_v8=True,frame_mapping_checks=checks,audio_checks=audio_checks,
            last_clip_field_audio_muted=True,full_decode='passed',bytes=NEW.stat().st_size,
            sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
(ROOT/'output/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
