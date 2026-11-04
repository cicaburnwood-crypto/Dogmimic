from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import cv2
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from edit_v8 import ROOT, BASE, SOURCE, OLD, NEW, SEGMENT, FPS, SR, DURATION, NEW_FRAMES, REMOVED_SECONDS


def packets(path):
    j = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=data_hash', '-show_data_hash', 'sha256',
        '-of', 'json', str(path)]))
    return [p['data_hash'] for p in j['packets']]


def get(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, im = cap.read()
    assert ok, frame
    return im


with ThreadPoolExecutor(max_workers=2) as pool:
    a, b = list(pool.map(packets, [OLD, NEW]))
assert len(a) == 14922 and len(b) == NEW_FRAMES
assert a[:9718] == b[:9718]
assert a[12186:] == b[11816:]

p = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error', '-show_streams', '-show_chapters', '-show_format', '-of', 'json', str(NEW)]))
v = next(s for s in p['streams'] if s['codec_type'] == 'video')
assert (v['width'], v['height'], v['avg_frame_rate'], int(v['nb_frames'])) == (1920,1080,'60/1',NEW_FRAMES)
assert abs(float(v['duration']) - DURATION) < 1e-5
assert abs(float(p['format']['duration']) - DURATION) < .002
assert len(p['chapters']) == 8

# Inspect the last title frame, first banner frame, and both sides of the splice.
cap = cv2.VideoCapture(str(SEGMENT))
frames = [1280,1288,1347,1348,1440,1526,1528,1530,1532,1534,1560,1620,1740,1860,1980,2097]
sheet = Image.new('RGB', (1920,1200), (25,25,25))
draw = ImageDraw.Draw(sheet)
for i,f in enumerate(frames):
    im = get(cap,f)
    x,y = i%4*480,i//4*300
    sheet.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((480,270)),(x,y))
    draw.text((x+5,y+275),f'new local {f/60:.3f}s / global {(9718+f)/60:.3f}s',fill='white')
    if f in [1348,1440,1528,1530,1532,1740,2097]:
        cv2.imwrite(str(ROOT/'inspect'/f'final-{f}.png'),im)
sheet.save(ROOT/'inspect/final-storyboard.jpg',quality=95)

# Every retained moving frame is a continuous sequence, except for the single
# splice over the inserted freeze. Check the transformation around that splice.
oldcap = cv2.VideoCapture(str(SOURCE))
transform_errors = {}
for f in [1348,1440,1526,1528,1530,1532,1560,1740,2097]:
    source_frame = f if f < 1530 else f+370
    ref = get(oldcap, source_frame)
    expected = cv2.resize(ref,(1728,972),interpolation=cv2.INTER_LANCZOS4)
    actual = get(cap,f)[108:1080,96:1824]
    delta = float(np.abs(expected.astype(float)-actual).mean())
    transform_errors[str(f)] = dict(source_frame=source_frame, mean_absolute_difference=delta)
    assert delta < 2, (f,delta)
for f in [0,600,1000,1288,1347]:
    expected=get(oldcap,f);actual=get(cap,f)
    assert float(np.abs(expected.astype(float)-actual).mean()) < 2
oldcap.release()

# Search the complete Case 2 moving portion for repeated full-frame freezes.
cap.set(1,1348)
prev=None;repeat=0;longest=0
for f in range(1348,2098):
    ok,im=cap.read();assert ok
    small=cv2.resize(im[108:,350:1570],(610,486))
    delta=float(np.abs(small.astype(float)-prev).mean()) if prev is not None else 999
    repeat=repeat+1 if delta < .07 else 0
    longest=max(longest,repeat)
    prev=small
cap.release()
assert longest/FPS < .3, ('Unexpected static stretch',longest/FPS)

raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(NEW),'-map','0:a:0','-ar',str(SR),'-ac','2','-f','f32le','-'])
decoded=np.frombuffer(raw,'<f4').reshape(-1,2)
mix,sr=sf.read(ROOT/'audio/full-mix-v8.wav',dtype='float32',always_2d=True)
assert sr==SR and len(mix)==NEW_FRAMES*(SR//FPS) and len(decoded)>=len(mix)
assert np.max(np.abs(decoded)) < 1
entries=json.loads((ROOT/'audio/narration-manifest.json').read_text())
assert not any(e['id']=='humanoid-anchor' for e in entries)
checks=[]
for e in entries:
    lo=round(e['start']*SR);hi=round(e['end']*SR)
    corr=float(np.corrcoef(mix[lo:hi,0],decoded[lo:hi,0])[0,1])
    assert corr > .98,(e['id'],corr)
    checks.append(dict(id=e['id'],start=e['start'],end=e['end'],correlation=corr))
assert float(np.max(np.abs(decoded[round(187.55*SR):round(196.9*SR)]))) < 1e-5
quad=next(e for e in entries if e['id']=='quadruped-memory')
assert float(np.max(np.abs(decoded[round((206.15-REMOVED_SECONDS)*SR):round((quad['start']-.08)*SR)]))) < 1e-5

subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
report=dict(duration_seconds=DURATION,frames=NEW_FRAMES,resolution=[1920,1080],fps=60,
            chapters=8,removed_seconds=REMOVED_SECONDS,
            other_segments_video_packets_identical_to_v7=True,
            frame_transform_checks=transform_errors,case2_longest_near_identical_run_seconds=longest/FPS,
            audio_checks=checks,deleted_callout_audio_removed=True,last_clip_field_audio_muted=True,
            full_decode='passed',bytes=NEW.stat().st_size,sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
(ROOT/'output/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
