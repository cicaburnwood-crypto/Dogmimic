"""Render original Google slide layers and unretimed original GIFs to H.264."""
from pathlib import Path
import json, math, subprocess, textwrap
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
scripts=json.loads((ROOT/'audio/narration-manifest.json').read_text())
visuals=json.loads((ROOT/'render/visuals.json').read_text())
fps=60;rate=24000
parts=[];captions=[];timeline=[];offset=0
for script,visual in zip(scripts,visuals):
    i=script['index'];speech,sr=sf.read(ROOT/f'audio/slide-{i}.wav',dtype='float32')
    assert sr==rate
    lead=.30
    duration=math.ceil(max(lead+len(speech)/sr+.75,visual.get('cycle_seconds',0)+.15)*fps)/fps
    track=np.zeros(round(duration*sr),np.float32)
    start=round(lead*sr);track[start:start+len(speech)]=speech
    parts.append(track)
    timeline.append({'slide':i,'start':offset,'end':offset+duration,'seconds':duration,'audio_start':offset+lead,'audio_end':offset+lead+len(speech)/sr,'original_animation_seconds':visual.get('cycle_seconds')})
    for seg in script['segments']:
        captions.append((offset+lead+seg['start'],offset+lead+seg['end'],seg['text']))
    offset+=duration
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-filter_complex_threads','2']
    if 'gif' in visual:
        x,y,w,h=visual['bounds']
        cmd+=['-loop','1','-framerate',str(fps),'-i',visual['under'],'-ignore_loop','1','-i',visual['gif'],'-loop','1','-framerate',str(fps),'-i',visual['over']]
        # The original GIF runs at native speed exactly once. The last frame remains
        # visible for any brief narration tail; no GIF frames are regenerated.
        filt=f'[1:v]scale={w}:{h}:flags=lanczos,setsar=1[g];[0:v][g]overlay={x}:{y}:eof_action=repeat:format=auto[b];[b][2:v]overlay=0:0:format=auto,fps={fps},format=yuv420p[v]'
    else:
        cmd+=['-loop','1','-framerate',str(fps),'-i',visual['static']]
        filt=f'[0:v]fps={fps},format=yuv420p[v]'
    cmd+=['-filter_complex',filt,'-map','[v]','-frames:v',str(round(duration*fps)),'-an','-c:v','libx264','-preset','medium','-crf','17','-threads','4','-pix_fmt','yuv420p','-r',str(fps),'-video_track_timescale','60000',str(ROOT/f'render/slide-{i}.mp4')]
    print(f'Rendering slide {i}: {duration:.2f} seconds',flush=True)
    subprocess.run(cmd,check=True)

sf.write(ROOT/'audio/narration-combined.wav',np.concatenate(parts),rate,subtype='PCM_24')
(ROOT/'render/timeline.json').write_text(json.dumps(timeline,indent=2))
def timestamp(s):
    ms=round(s*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);sec,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{sec:02},{ms:03}'
srt='\n\n'.join(f'{n}\n{timestamp(a)} --> {timestamp(b)}\n'+textwrap.fill(t,width=66) for n,(a,b,t) in enumerate(captions,1))+'\n'
(OUT/'EchoNav_demo.en.srt').write_text(srt)
concat=ROOT/'render/concat.txt'
concat.write_text(''.join(f"file 'slide-{i}.mp4'\n" for i in range(1,5)))
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(ROOT/'render/silent.mp4')],check=True)
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','info','-i',str(ROOT/'audio/narration-combined.wav'),'-af','loudnorm=I=-16:TP=-1.5:LRA=7','-ar','48000','-ac','2','-c:a','pcm_s24le',str(OUT/'EchoNav_narration.wav')],check=True,stderr=(ROOT/'render/audio-normalization.log').open('w'))
metadata=';FFMETADATA1\ntitle=EchoNav - Narrated Research Demo\ncomment=Original slide visuals and animation assets; offline synthetic narration.\n'
titles=['Visual memory and hidden space prediction','Selective visual memory','Predict space behind obstacles','Simulation and real-robot validation']
for t,title in zip(timeline,titles):
    metadata+=f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(t['start']*1000)}\nEND={round(t['end']*1000)}\ntitle={title}\n"
(ROOT/'render/chapters.txt').write_text(metadata)
subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-i',str(ROOT/'render/silent.mp4'),'-i',str(OUT/'EchoNav_narration.wav'),'-f','ffmetadata','-i',str(ROOT/'render/chapters.txt'),'-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2','-c:v','copy','-c:a','aac','-b:a','192k','-metadata:s:a:0','language=eng','-movflags','+faststart',str(OUT/'EchoNav_demo_1080p.mp4')],check=True)
print(f'Completed: {offset:.2f} seconds. {OUT}',flush=True)
