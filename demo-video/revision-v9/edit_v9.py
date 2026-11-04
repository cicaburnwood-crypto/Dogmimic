"""Double the speed of the Wheeled Case 3 close-up and remove its narration."""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OLD=BASE/'revision-v8/output/EchoNav_full_demo_mosaic_manual_v8.mp4'
SOURCE=BASE/'revision-v6/segments/04-clip-3.mp4'
SEGMENT=ROOT/'segments/04-clip-3.mp4'
NEW=ROOT/'output/EchoNav_full_demo_mosaic_manual_v9.mp4'
FPS=60
SR=48000
WHEELED_START=4842
START=3842
END=4142
REMOVED=(END-START)//2
SHIFT=REMOVED/FPS
NEW_FRAMES=14552-REMOVED
DURATION=NEW_FRAMES/FPS


def render():
    # The five-second enlargement, hold and return animation occupies exactly
    # 300 frames. Keep alternate frames only inside that interval, then reset
    # continuous 60-fps timestamps. Everything after it resumes the old speed.
    vf=(f"select='lt(n,{START})+gte(n,{END})+"
        f"between(n,{START},{END-1})*not(mod(n-{START},2))',"
        "setpts=N/(60*TB),setsar=1")
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-threads','4',
         '-i',str(SOURCE),'-an','-vf',vf,'-c:v','libx264','-preset','fast',
         '-crf','15','-profile:v','high','-level:v','4.2','-threads','4',
         '-pix_fmt','yuv420p','-r','60','-fps_mode','cfr','-video_track_timescale','60000',
         '-map_metadata','-1','-map_chapters','-1','-movflags','+faststart',
         '-progress',str(ROOT/'inspect/render-progress.txt'),str(SEGMENT)]
    (ROOT/'inspect/render-command.json').write_text(json.dumps(cmd,indent=2))
    st=time.monotonic()
    with (ROOT/'inspect/render.log').open('w') as log:
        subprocess.run(cmd,check=True,stderr=log)
    print(f'Rendered in {time.monotonic()-st:.1f}s',flush=True)


def audio():
    mix,sr=sf.read(BASE/'revision-v8/audio/full-mix-v8.wav',dtype='float32',always_2d=True)
    assert sr==SR
    entries=json.loads((BASE/'revision-v8/audio/narration-manifest.json').read_text())
    removed=next(e for e in entries if e['id']=='wheeled-lighting')
    voice,_=sf.read(removed['audio'],dtype='float32',always_2d=True)
    start=round(removed['start']*SR)
    assert np.max(np.abs(mix[start:start+len(voice)]-voice))<2e-7
    mix[start:start+len(voice)]=0
    a=(WHEELED_START+START)*(SR//FPS)
    b=(WHEELED_START+END)*(SR//FPS)
    assert np.max(np.abs(mix[a:b]))==0
    # No remaining audio in the sped-up interval: halve its silent duration.
    mix=np.concatenate([mix[:a],np.zeros(((END-START-REMOVED)*(SR//FPS),2),np.float32),mix[b:]])
    assert len(mix)==NEW_FRAMES*(SR//FPS)
    sf.write(ROOT/'audio/full-mix-v9.wav',mix,SR,subtype='PCM_24')
    adjusted=[]
    for e in entries:
        if e['id']=='wheeled-lighting':continue
        e=e.copy()
        if e['start']>=b/SR:
            for key in ['start','end','requested_start','end_by']:
                if key in e:e[key]-=SHIFT
        adjusted.append(e)
    (ROOT/'audio/narration-manifest.json').write_text(json.dumps(adjusted,indent=2))
    print('Close-up narration removed; later audio shifted earlier by 2.5 seconds.',flush=True)


def finish():
    jobs=json.loads((BASE/'revision-v8/output/timeline.json').read_text())
    offset=0
    for j in jobs:
        if j['id']=='clip-3':
            j['frames']-=REMOVED
            j['output']=str(SEGMENT)
        j['duration']=j['frames']/FPS
        j['start']=offset/FPS
        offset+=j['frames']
        j['end']=offset/FPS
    assert offset==NEW_FRAMES
    (ROOT/'output/timeline.json').write_text(json.dumps(jobs,indent=2))
    concat=ROOT/'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    metadata=';FFMETADATA1\ntitle=EchoNav - Full Research Demo\n'
    for j in jobs:
        metadata+=(f"[CHAPTER]\nTIMEBASE=1/60000\nSTART={round(j['start']*60000)}\n"
                   f"END={round(j['end']*60000)}\ntitle={j['title']}\n")
    meta=ROOT/'inspect/chapters.txt';meta.write_text(metadata)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0',
                    '-i',str(concat),'-i',str(ROOT/'audio/full-mix-v9.wav'),'-f','ffmetadata','-i',str(meta),
                    '-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2',
                    '-c:v','copy','-c:a','aac','-b:a','192k','-ar',str(SR),'-ac','2',
                    '-movflags','+faststart',str(NEW)],check=True)
    manifest=dict(previous_version=str(OLD),output=str(NEW),
        original_closeup_local_frames=[START,END],
        original_closeup_global_seconds=[(WHEELED_START+START)/FPS,(WHEELED_START+END)/FPS],
        new_closeup_global_seconds=[(WHEELED_START+START)/FPS,(WHEELED_START+START+REMOVED)/FPS],
        closeup_speed_multiplier=2,original_closeup_duration_seconds=5,new_closeup_duration_seconds=2.5,
        removed_narration='Our model and OmniVLA both passed the test under these lighting conditions.',
        case3_title_narration='Preserved',later_video_and_audio_shift_seconds=-SHIFT,
        preserved_edits='All prior source replacements and mosaics; Humanoid Case 2 Dynamic Avoidance banner and removed freeze.',
        duration_seconds=DURATION,frames=NEW_FRAMES)
    (ROOT/'output/edit-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(NEW,flush=True)


if __name__=='__main__':
    {'render':render,'audio':audio,'finish':finish}[sys.argv[1]]()
