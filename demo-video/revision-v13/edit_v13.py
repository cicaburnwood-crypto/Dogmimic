"""Wheeled Case 2 at 6X; all eight black title cards and voices at 1.5X."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import json
import math
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'revision-v12'))
from edit_v12 import badge, select_expression, read_audio, map_local_frame, activity

FPS=60
SR=48000
SPF=800
OLD=BASE/'revision-v12/output/EchoNav_full_demo_mosaic_manual_v12.mp4'
NEW=ROOT/'output/EchoNav_full_demo_mosaic_manual_v13.mp4'
SPECS={
    'before-3':[(0,114,76)],
    'before-4':[(0,99,66)],
    'before-5':[(0,91,61)],
    'clip-3':[(0,60,40),(60,1609,1549),(1609,1669,40),(1669,1789,120),
              (1789,2079,97),(2079,2379,300),(2379,2845,155),
              (2845,3371,351),(3371,3431,40),(3431,4255,824)],
    'clip-4':[(0,60,40),(60,957,897),(957,1017,40),(1017,1517,500)],
}


def save(path,value):
    path.write_text(json.dumps(value,indent=2))


def accelerate_voice(path,name,target_samples=None):
    original=read_audio(path)
    samples=target_samples or round(len(original)/1.5)
    out=ROOT/'audio'/f'{name}.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(path),'-af',
                    f'atempo=1.5,apad,atrim=end_sample={samples}',
                    '-ar',str(SR),'-ac','2','-c:a','pcm_s24le',str(out)],check=True)
    a=read_audio(out)
    assert len(a)==samples
    first,end=activity(a)
    assert end<len(a), (name,'Speech reaches the very last sample')
    return out


def prepare():
    previous=json.loads((BASE/'revision-v12/output/timeline.json').read_text())
    jobs=[];offset=0
    for old in previous:
        j=copy.deepcopy(old)
        j['previous_output']=old['output']
        j['previous_start']=old['start']
        j['previous_end']=old['end']
        j['previous_frames']=old['frames']
        j['retime']=SPECS.get(j['id'],[(0,j['frames'],j['frames'])])
        j['changed']=j['id'] in SPECS
        j['frames']=sum(s[2] for s in j['retime'])
        j['duration']=j['frames']/FPS
        j['start_frame']=offset;j['start']=offset/FPS
        offset+=j['frames'];j['end']=offset/FPS
        if j['changed']:
            j['output']=str(ROOT/'segments'/Path(old['output']).name)
        jobs.append(j)
    assert offset==10932
    save(ROOT/'output/timeline.json',jobs)
    old_mix=read_audio(BASE/'revision-v12/audio/full-mix-v12.wav')
    mix=np.zeros((offset*SPF,2),np.float32)
    audio_tracks=[]
    for j in jobs:
        if j['id'] not in ['clip-1','clip-2','before-3','before-4','before-5']:
            continue
        old_start=round(j['previous_start']*SR)
        a=old_mix[old_start:old_start+j['previous_frames']*SPF]
        if j['kind']=='card':
            path=accelerate_voice(BASE/'revision-v12/audio'/f"{j['id']}.wav",j['id'],j['frames']*SPF)
            a=read_audio(path)
        else:
            path=ROOT/'audio'/f"{j['id']}-unchanged.wav"
            sf.write(path,a,SR,subtype='PCM_24')
        assert len(a)==j['frames']*SPF
        start=j['start_frame']*SPF
        mix[start:start+len(a)]=a
        audio_tracks.append(dict(id=j['id'],audio=str(path),start=j['start'],end=j['end'],
                                 tempo=1.5 if j['kind']=='card' else 1))
    originals=json.loads((BASE/'revision-v12/audio/narration-manifest.json').read_text())
    entries=[];last_end=0
    for old in originals:
        e=copy.deepcopy(old)
        ident='clip-3' if e['id'].startswith('wheeled-') else ('clip-5' if e['id'].startswith('quadruped-') else 'clip-4')
        j=next(j for j in jobs if j['id']==ident)
        local_frame=(old['start']-j['previous_start'])*FPS
        requested=j['start']+map_local_frame(j,local_frame)/FPS
        start=max(requested,last_end+.18)
        if e['kind']=='case-title':
            path=accelerate_voice(e['audio'],e['id'])
            e['previous_audio']=e['audio'];e['audio']=str(path)
            e['tempo_multiplier']=1.5
        else:
            e['tempo_multiplier']=1.0
        a=read_audio(e['audio']);lo=round(start*SR);hi=lo+len(a)
        assert hi<=round(j['end']*SR)
        assert np.abs(mix[lo:hi]).max()==0,e['id']
        mix[lo:hi]+=a
        e.update(previous_start=old['start'],previous_end=old['end'],
                 start=lo/SR,end=hi/SR,duration=len(a)/SR,requested_start=requested,
                 narration_speed_preserved=e['tempo_multiplier']==1)
        entries.append(e);last_end=hi/SR
    assert np.abs(mix).max()<1
    sf.write(ROOT/'audio/full-mix-v13.wav',mix,SR,subtype='PCM_24')
    save(ROOT/'audio/base-tracks.json',audio_tracks)
    save(ROOT/'audio/narration-manifest.json',entries)
    black=json.loads((ROOT/'inspect/black-transitions.json').read_text())
    transitions=[]
    for item in black:
        j=next(j for j in jobs if j['id']==item['id'])
        a,b=item['local_frames']
        lo=map_local_frame(j,a);hi=map_local_frame(j,b)
        assert abs((hi-lo)-(b-a)/1.5)<=.5
        transitions.append(dict(id=item['id'],previous_local_frames=[a,b],
                                previous_global_seconds=item['global_seconds'],
                                output_local_frames=[lo,hi],
                                output_global_seconds=[j['start']+lo/FPS,j['start']+hi/FPS],
                                speed_multiplier=1.5))
    save(ROOT/'output/edit-manifest.json',dict(previous_version=str(OLD),output=str(NEW),
        duration_seconds=offset/FPS,frames=offset,black_transitions=transitions,
        black_transition_count=8,black_transition_audio_speed_multiplier=1.5,
        wheeled_case2_moving_intervals=[
            dict(previous_local_frames=[1789,2079],previous_speed=2,new_speed=6,new_frames=97),
            dict(previous_local_frames=[2379,2845],previous_speed=2,new_speed=6,new_frames=155),
            dict(previous_local_frames=[2845,3371],previous_speed=4,new_speed=6,new_frames=351)],
        wheeled_case2_explanation_local_frames=[2079,2379],explanation_duration_seconds=5,
        preserved='Prior source replacements, mosaics, slide timing, black quadruped borders, 3X humanoid motion and muted field audio.',
        audio='Pitch-preserving 1.5X title/card voices; other narration remapped to the moving video without additional pauses.'))
    print(json.dumps(dict(frames=offset,duration=offset/FPS,black_transitions=transitions),indent=2),flush=True)


def render_one(j):
    prefix=badge('6X',(12,72,168,122),96,1789,3371) if j['id']=='clip-3' else ''
    vf=prefix+f"select='{select_expression(j['retime'])}',setpts=N/(60*TB),setsar=1"
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-threads','3',
         '-i',j['previous_output'],'-an','-vf',vf,'-frames:v',str(j['frames']),
         '-c:v','libx264','-preset','fast','-crf','15','-profile:v','high','-level:v','4.2',
         '-threads','4','-pix_fmt','yuv420p','-r','60','-fps_mode','cfr',
         '-video_track_timescale','60000','-map_metadata','-1','-map_chapters','-1',
         '-movflags','+faststart',j['output']]
    save(ROOT/'inspect'/f"{j['id']}-command.json",cmd)
    start=time.monotonic();print(f"Rendering {j['id']}",flush=True)
    with (ROOT/'inspect'/f"{j['id']}-render.log").open('w') as log:
        subprocess.run(cmd,stderr=log,check=True)
    print(f"Rendered {j['id']} in {time.monotonic()-start:.1f}s",flush=True)


def render():
    jobs=json.loads((ROOT/'output/timeline.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render_one,[j for j in jobs if j['changed']]))


def finish():
    jobs=json.loads((ROOT/'output/timeline.json').read_text())
    concat=ROOT/'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    old_chapters=json.loads((BASE/'revision-v12/output/edit-manifest.json').read_text())['chapters']
    chapters=[]
    for j in jobs:
        if j['id']=='clip-2':
            chapters.extend(copy.deepcopy(old_chapters[1:5]))
        else:
            chapters.append(dict(title=j['title'],start=j['start'],end=j['end']))
    text=';FFMETADATA1\ntitle=EchoNav - Full Research Demo\n'
    for c in chapters:
        text+=(f"[CHAPTER]\nTIMEBASE=1/60000\nSTART={round(c['start']*60000)}\n"
               f"END={round(c['end']*60000)}\ntitle={c['title']}\n")
    meta=ROOT/'inspect/chapters.txt';meta.write_text(text)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning',
                    '-f','concat','-safe','0','-i',str(concat),'-i',str(ROOT/'audio/full-mix-v13.wav'),
                    '-f','ffmetadata','-i',str(meta),'-map','0:v:0','-map','1:a:0',
                    '-map_metadata','2','-map_chapters','2','-c:v','copy','-c:a','aac',
                    '-b:a','192k','-ar',str(SR),'-ac','2','-movflags','+faststart',str(NEW)],check=True)
    manifest=json.loads((ROOT/'output/edit-manifest.json').read_text())
    manifest['chapters']=chapters;save(ROOT/'output/edit-manifest.json',manifest)
    print(NEW,flush=True)


if __name__=='__main__':
    {'prepare':prepare,'render':render,'finish':finish}[sys.argv[1]]()
