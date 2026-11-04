"""Change only the last quadruped clip's layout border to black."""
from pathlib import Path
import json
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
SOURCE=BASE/'redaction/manual-v2/segments/08-clip-5.mp4'
OLD=BASE/'revision-v9/output/EchoNav_full_demo_mosaic_manual_v9.mp4'
SEGMENT=ROOT/'segments/08-clip-5.mp4'
NEW=ROOT/'output/EchoNav_full_demo_mosaic_manual_v10.mp4'
# Manually inspected 1920 x 1080 layout: outside margins and inter-panel gaps.
# Include the original pale antialiasing fringe (one pixel into each panel).
BORDERS=[(0,0,1920,180),(0,898,1920,1080),(0,180,22,898),(1898,180,1920,898),
         (1301,180,1313,898),(1635,180,1647,422),(1313,421,1898,434)]
cv2.setNumThreads(2)


def zoom_rectangle(im,f):
    # Locate the existing red outline of the BEV enlargement only, so the
    # black layout dividers cannot cut through an animated overlay. No OCR.
    if not 27.15 <= f/60 <= 30.6:
        return None
    q=im[175:905,1000:1905].astype(np.int16)
    b,g,r=q[:,:,0],q[:,:,1],q[:,:,2]
    red=(r>100)&(g<160)&(b<160)&(r-g>35)&(r-b>35)
    xs=np.where(red.sum(0)>90)[0]
    ys=np.where(red.sum(1)>180)[0]
    if len(xs) and len(ys):
        return (1000+int(xs.min()),175+int(ys.min()),
                min(1899,1000+int(xs.max())+1),min(899,175+int(ys.max())+1))
    return None


def edit(im,f):
    out=im.copy()
    for x0,y0,x1,y1 in BORDERS:
        out[y0:y1,x0:x1]=0
    rect=zoom_rectangle(im,f)
    if rect:
        x0,y0,x1,y1=rect
        out[y0:y1,x0:x1]=im[y0:y1,x0:x1]
    return out,rect


def preview():
    cap=cv2.VideoCapture(str(SOURCE))
    boxes={}
    for t in [0,16,27.3,27.5,28.5,30.2,30.4,36,42.5]:
        f=round(t*60);cap.set(1,f);ok,im=cap.read();assert ok
        out,rect=edit(im,f)
        cv2.imwrite(str(ROOT/'inspect'/f'preview-{f}.png'),out)
        boxes[str(t)]=rect
    cap.release()
    print(json.dumps(boxes,indent=2))


def render():
    cap=cv2.VideoCapture(str(SOURCE));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','bgr24',
         '-video_size','1920x1080','-framerate','60','-i','pipe:0','-an','-c:v','libx264',
         '-preset','fast','-crf','15','-profile:v','high','-level:v','4.2','-threads','4',
         '-pix_fmt','yuv420p','-video_track_timescale','60000','-map_metadata','-1',
         '-map_chapters','-1','-movflags','+faststart',str(SEGMENT)]
    boxes={};st=time.monotonic()
    with (ROOT/'inspect/render.log').open('w') as log:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for f in range(count):
            ok,im=cap.read();assert ok,f
            out,rect=edit(im,f)
            if rect:boxes[str(f)]=rect
            p.stdin.write(out.tobytes())
            if f%600==0:print(f'{f}/{count} frames, {time.monotonic()-st:.1f}s',flush=True)
        p.stdin.close();assert p.wait()==0
    cap.release()
    assert count==2556
    (ROOT/'inspect/zoom-rectangles.json').write_text(json.dumps(boxes,indent=2))
    print(f'Rendered {count} frames in {time.monotonic()-st:.1f}s',flush=True)


def finish():
    jobs=json.loads((BASE/'revision-v9/output/timeline.json').read_text())
    jobs[-1]['output']=str(SEGMENT)
    (ROOT/'output/timeline.json').write_text(json.dumps(jobs,indent=2))
    concat=ROOT/'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0',
        '-i',str(concat),'-i',str(OLD),'-map','0:v:0','-map','1:a:0','-map_metadata','1',
        '-map_chapters','1','-c','copy','-movflags','+faststart',str(NEW)],check=True)
    (ROOT/'output/edit-manifest.json').write_text(json.dumps(dict(
        previous_version=str(OLD),output=str(NEW),edited_clip='Quadruped Test Case',
        interval_seconds=[197.43333333333334,240.03333333333333],
        border_color='black',border_rectangles=BORDERS,
        animated_BEV_overlay='Preserved above the border layer',
        audio='Bit-for-bit copy of v9',duration_seconds=240.03333333333333,
        frames=14402,redactions='Existing manually redacted frames reused.'),indent=2))
    print(NEW,flush=True)


if __name__=='__main__':
    {'preview':preview,'render':render,'finish':finish}[sys.argv[1]]()
