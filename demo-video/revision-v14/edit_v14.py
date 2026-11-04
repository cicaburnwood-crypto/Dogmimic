"""Retiming of five close-ups, Wheeled Case 3, elevator approach, and tail."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import json
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'revision-v12'))
from edit_v12 import badge, select_expression, read_audio, map_local_frame, activity

FPS, SR, SPF = 60, 48000, 800
OLD = BASE / 'revision-v13/output/EchoNav_full_demo_mosaic_manual_v13.mp4'
NEW = ROOT / 'output/EchoNav_full_demo_mosaic_manual_v14.mp4'
HUMAN_2X = BASE / 'revision-v8/segments/06-clip-4.mp4'
# Source ranges refer to v13 segment frames and have exclusive ends.
# 2X elevator footage comes directly from the already-redacted v8 master,
# recovering original intermediate frames instead of duplicating v13 frames.
SPECS = {
    'clip-3': [(0,869,869), (869,1169,250), (1169,1846,677),
               (1846,2146,250), (2146,2812,666), (2812,2962,125),
               (2962,3516,443)],
    'clip-4': [(0,54,54), (54,274,183), (274,977,703),
               (977,1201,336), (1201,1477,276)],
    'clip-5': [(0,1392,1392), (1392,1584,160), (1584,2244,660)],
}
CLOSEUPS = [
    ('wheeled-memory', 'clip-3', 869,1169),
    ('wheeled-prediction', 'clip-3',1846,2146),
    ('wheeled-lighting', 'clip-3',2812,2962),
    ('humanoid-glass', 'clip-4',54,274),
    ('quadruped-memory', 'clip-5',1392,1584),
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2))


def accelerate_voice(path, name):
    original = read_audio(path)
    samples = round(len(original)/1.2)
    out = ROOT / 'audio' / f'{name}.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(path),'-af',
                    f'atempo=1.2,apad,atrim=end_sample={samples}',
                    '-ar',str(SR),'-ac','2','-c:a','pcm_s24le',str(out)], check=True)
    result = read_audio(out)
    assert len(result) == samples
    assert activity(result)[1] < len(result)
    return out


def prepare():
    old_jobs = json.loads((BASE/'revision-v13/output/timeline.json').read_text())
    jobs, offset = [], 0
    for old in old_jobs:
        j = copy.deepcopy(old)
        j.update(previous_output=old['output'], previous_start=old['start'],
                 previous_end=old['end'], previous_frames=old['frames'])
        j['retime'] = SPECS.get(j['id'], [(0,j['frames'],j['frames'])])
        j['changed'] = j['id'] in SPECS
        j['frames'] = sum(z[2] for z in j['retime'])
        j['duration'] = j['frames']/FPS
        j['start_frame'], j['start'] = offset, offset/FPS
        offset += j['frames']
        j['end'] = offset/FPS
        if j['changed']:
            j['output'] = str(ROOT/'segments'/Path(old['output']).name)
        jobs.append(j)
    assert offset == 10667
    save(ROOT/'output/timeline.json', jobs)
    old_mix = read_audio(BASE/'revision-v13/audio/full-mix-v13.wav')
    mix = np.zeros((offset*SPF,2),np.float32)
    base_tracks = []
    for j in jobs:
        if j['id'] not in ['clip-1','clip-2','before-3','before-4','before-5']:
            continue
        lo = round(j['previous_start']*SR)
        a = old_mix[lo:lo+j['previous_frames']*SPF]
        assert len(a) == j['frames']*SPF
        path = ROOT/'audio'/f"{j['id']}-unchanged.wav"
        sf.write(path,a,SR,subtype='PCM_24')
        start = j['start_frame']*SPF
        mix[start:start+len(a)] = a
        base_tracks.append(dict(id=j['id'],audio=str(path),start=j['start'],end=j['end'],tempo=1))
    entries, last_end = [], 0
    originals = json.loads((BASE/'revision-v13/audio/narration-manifest.json').read_text())
    for old in originals:
        e = copy.deepcopy(old)
        ident = 'clip-3' if e['id'].startswith('wheeled-') else ('clip-5' if e['id'].startswith('quadruped-') else 'clip-4')
        j = next(j for j in jobs if j['id'] == ident)
        requested = j['start'] + map_local_frame(j,(old['start']-j['previous_start'])*FPS)/FPS
        start = max(requested,last_end+.18)
        e['previous_audio'] = old['audio']
        e['tempo_multiplier'] = 1.2 if e['kind'] == 'callout' else 1
        if e['tempo_multiplier'] == 1.2:
            e['audio'] = str(accelerate_voice(old['audio'],e['id']))
        a = read_audio(e['audio'])
        lo = round(start*SR); hi = lo+len(a)
        assert hi <= round(j['end']*SR), e['id']
        assert np.abs(mix[lo:hi]).max() == 0, e['id']
        mix[lo:hi] = a
        e.update(previous_start=old['start'],previous_end=old['end'],start=lo/SR,end=hi/SR,
                 duration=len(a)/SR,requested_start=requested,narration_speed_preserved=e['tempo_multiplier']==1)
        entries.append(e);last_end=hi/SR
    assert np.abs(mix).max() < 1
    sf.write(ROOT/'audio/full-mix-v14.wav',mix,SR,subtype='PCM_24')
    save(ROOT/'audio/base-tracks.json',base_tracks)
    save(ROOT/'audio/narration-manifest.json',entries)
    closeups = []
    for name,ident,a,b in CLOSEUPS:
        j = next(j for j in jobs if j['id'] == ident)
        lo,hi = map_local_frame(j,a),map_local_frame(j,b)
        closeups.append(dict(id=name,clip=ident,previous_local_frames=[a,b],
                             previous_global_seconds=[j['previous_start']+a/FPS,j['previous_start']+b/FPS],
                             output_local_frames=[lo,hi],output_global_seconds=[j['start']+lo/FPS,j['start']+hi/FPS],
                             tempo_multiplier=1.2,narration=name!='wheeled-lighting'))
    j = next(j for j in jobs if j['id']=='clip-4')
    save(ROOT/'output/edit-manifest.json',dict(
        previous_version=str(OLD),output=str(NEW),frames=offset,duration_seconds=offset/FPS,
        closeups=closeups,closeup_count=5,callout_narration_tempo_multiplier=1.2,
        wheeled_case3_moving_speed=2.5,wheeled_case3_previous_moving_speed=2,
        wheeled_case3_moving_previous_local_frames=[2962,3516],
        humanoid_case2_before_exit=dict(previous_local_frames=[977,1201],new_speed=2,
            previous_speed=3,source=str(HUMAN_2X),source_frames=[1348,1684],
            output_global_seconds=[j['start']+map_local_frame(j,977)/FPS,j['start']+map_local_frame(j,1201)/FPS]),
        humanoid_case2_after_exit_speed=3,tail_removed_seconds=1.2,tail_removed_frames=72,
        original_v13_tail_cut_global_seconds=[181,182.2],
        preserved='Existing mosaics, source replacements, slide durations, black title cards and their voices, quadruped black borders, Dynamic Avoidance banner and muted field audio.',
        audio='Four existing close-up narrations accelerated 1.2X with unchanged pitch; other voices unchanged; no new pauses. Wheeled Case 3 close-up remains silent.'))
    print(json.dumps(dict(frames=offset,duration=offset/FPS,closeups=closeups),indent=2),flush=True)


def render_one(j):
    cmd = ['ffmpeg','-y','-hide_banner','-loglevel','warning','-threads','3','-i',j['previous_output']]
    if j['id']=='clip-4':
        # v13 frame 1201 maps exactly to v8 frame 1684. This splice restores
        # 2X through the door crossing and continues at the existing 3X.
        pre = select_expression(j['retime'][:3])
        first = int(map_local_frame(j,977)); end = first+336
        graph = (
            f"[0:v]split=2[p][q];[p]select='{pre}',setpts=N/(60*TB)[pre];"
            "[1:v]trim=start_frame=1348:end_frame=1684,setpts=N/(60*TB)[elevator];"
            "[q]trim=start_frame=1201:end_frame=1477,setpts=N/(60*TB)[post];"
            "[pre][elevator][post]concat=n=3:v=1:a=0,setpts=N/(60*TB),"
            +badge('2X',(359,115,130,89),74,first,end)+"setsar=1[v]"
        )
        cmd += ['-threads','3','-i',str(HUMAN_2X),'-filter_complex_threads','2',
                '-filter_complex',graph,'-map','[v]']
    else:
        prefix = badge('2.5X',(12,72,168,122),70,2812,3516) if j['id']=='clip-3' else ''
        vf = prefix+f"select='{select_expression(j['retime'])}',setpts=N/(60*TB),setsar=1"
        cmd += ['-vf',vf]
    cmd += ['-an','-frames:v',str(j['frames']),'-c:v','libx264','-preset','fast',
            '-crf','15','-profile:v','high','-level:v','4.2','-threads','4','-pix_fmt','yuv420p',
            '-r','60','-fps_mode','cfr','-video_track_timescale','60000','-map_metadata','-1',
            '-map_chapters','-1','-movflags','+faststart',
            '-progress',str(ROOT/'inspect'/f"{j['id']}-progress.txt"),j['output']]
    save(ROOT/'inspect'/f"{j['id']}-command.json",cmd)
    print(f"Rendering {j['id']}: {j['frames']} frames",flush=True)
    start = time.monotonic()
    with (ROOT/'inspect'/f"{j['id']}-render.log").open('w') as log:
        subprocess.run(cmd,stderr=log,check=True)
    print(f"Rendered {j['id']} in {time.monotonic()-start:.1f}s",flush=True)


def render():
    jobs = json.loads((ROOT/'output/timeline.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render_one,[j for j in jobs if j['changed']]))


def finish():
    jobs = json.loads((ROOT/'output/timeline.json').read_text())
    concat = ROOT/'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    old_chapters = json.loads((BASE/'revision-v13/output/edit-manifest.json').read_text())['chapters']
    chapters=[]
    for j in jobs:
        if j['id']=='clip-2':
            chapters.extend(copy.deepcopy(old_chapters[1:5]))
        else:
            chapters.append(dict(title=j['title'],start=j['start'],end=j['end']))
    meta=';FFMETADATA1\ntitle=EchoNav - Full Research Demo\n'
    for c in chapters:
        meta += (f"[CHAPTER]\nTIMEBASE=1/60000\nSTART={round(c['start']*60000)}\n"
                 f"END={round(c['end']*60000)}\ntitle={c['title']}\n")
    path=ROOT/'inspect/chapters.txt';path.write_text(meta)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning',
                    '-f','concat','-safe','0','-i',str(concat),'-i',str(ROOT/'audio/full-mix-v14.wav'),
                    '-f','ffmetadata','-i',str(path),'-map','0:v:0','-map','1:a:0',
                    '-map_metadata','2','-map_chapters','2','-c:v','copy','-c:a','aac',
                    '-b:a','192k','-ar',str(SR),'-ac','2','-movflags','+faststart',str(NEW)],check=True)
    manifest=json.loads((ROOT/'output/edit-manifest.json').read_text())
    manifest['chapters']=chapters;save(ROOT/'output/edit-manifest.json',manifest)
    print(NEW,flush=True)


if __name__=='__main__':
    {'prepare':prepare,'render':render,'finish':finish}[sys.argv[1]]()
