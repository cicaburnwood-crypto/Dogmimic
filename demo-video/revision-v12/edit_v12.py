"""Retiming requested after v11. Keep all previously composited redactions."""
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

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
FPS = 60
SR = 48000
SPF = SR // FPS
OLD = BASE / 'revision-v11/output/EchoNav_full_demo_mosaic_manual_v11.mp4'
NEW = ROOT / 'output/EchoNav_full_demo_mosaic_manual_v12.mp4'
FONT = subprocess.check_output(['fc-match', '-f', '%{file}', 'Arial']).decode()

# (source start frame, exclusive source end, output frame count).
# The first humanoid explanation finishes at 4.9 s, before motion accelerates.
SPECS = {
    'clip-1': [(0, 300, 300)],
    'clip-2': [(0, 1059, 900), (1059, 2264, 900), (2264, 3194, 600), (3194, 4207, 720)],
    'clip-3': [(0, 180, 180), (180, 762, 291), (762, 4546, 3784)],
    'clip-4': [(0, 294, 294), (294, 1288, 663), (1288, 1348, 60), (1348, 2098, 500)],
    # v11 global 200.0--208.0 s corresponds to local frames 154--634.
    'clip-5': [(0, 154, 154), (154, 634, 240), (634, 2556, 1922)],
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def read_audio(path):
    a, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == SR and a.shape[1] == 2, (path, sr, a.shape)
    return a


def activity(a):
    idx = np.flatnonzero(np.max(np.abs(a), axis=1) > .004)
    assert len(idx)
    return int(idx[0]), int(idx[-1]) + 1


def trim_voice(a):
    first, end = activity(a)
    lo, hi = max(0, first-round(.04*SR)), min(len(a), end+round(.08*SR))
    return a[lo:hi], [lo/SR, hi/SR]


def stretch(a, samples, name):
    raw = ROOT / 'audio' / f'{name}-input.wav'
    out = ROOT / 'audio' / f'{name}.wav'
    sf.write(raw, a, SR, subtype='PCM_24')
    tempo = len(a) / samples
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(raw), '-af',
                    f'atempo={tempo:.12f},apad,atrim=end_sample={samples}',
                    '-ar', str(SR), '-ac', '2', '-c:a', 'pcm_s24le', str(out)], check=True)
    result = read_audio(out)
    assert len(result) == samples
    return out, tempo


def prepare():
    old_jobs = json.loads((BASE / 'revision-v11/output/timeline.json').read_text())
    old_mix = read_audio(BASE / 'revision-v9/audio/full-mix-v9.wav')
    assert len(old_mix) == 14402*SPF
    title, title_trim = trim_voice(read_audio(BASE / 'assembly/audio/title.wav'))
    title_file, title_tempo = stretch(title, 300*SPF, 'title')
    title_activity = [v/SR for v in activity(read_audio(title_file))]
    assert 4.8 < title_activity[1] <= 5.0, title_activity
    audio = {'clip-1': [dict(file=str(title_file), start=0, samples=300*SPF)]}
    details = dict(title=dict(duration=5, trimmed_source_seconds=title_trim,
                             tempo=title_tempo, speech_seconds=title_activity), slides=[], cards=[])
    audio['clip-2'] = []
    offset = 0
    for i, (a, b, n) in enumerate(SPECS['clip-2'], 2):
        speech = old_mix[(455+a)*SPF:(455+b)*SPF]
        path, tempo = stretch(speech, n*SPF, f'slide-{i}')
        audio['clip-2'].append(dict(file=str(path), start=offset, samples=n*SPF))
        details['slides'].append(dict(page=i, source_local_frames=[a,b], output_frames=n,
                                      duration=n/FPS, tempo=tempo, speech_seconds=[v/SR for v in activity(read_audio(path))]))
        offset += n*SPF
    for ident in ['before-3', 'before-4', 'before-5']:
        a, bounds = trim_voice(read_audio(BASE / f'assembly/audio/{ident}.wav'))
        frames = math.ceil(len(a)/SPF)
        result = np.zeros((frames*SPF,2), np.float32)
        result[:len(a)] = a
        path = ROOT / 'audio' / f'{ident}.wav'
        sf.write(path, result, SR, subtype='PCM_24')
        audio[ident] = [dict(file=str(path), start=0, samples=len(result))]
        SPECS[ident] = [(0, frames, frames)]
        details['cards'].append(dict(id=ident, frames=frames, duration=frames/FPS,
                                     trimmed_source_seconds=bounds, speech_seconds=[v/SR for v in activity(result)]))
    jobs = []
    total = 0
    for old in old_jobs:
        j = copy.deepcopy(old)
        ident = j['id']
        specs = SPECS[ident]
        j['previous_output'] = j['output']
        j['previous_start'] = j['start']
        j['previous_end'] = j['end']
        j['previous_frames'] = j['frames']
        j['retime'] = specs
        j['frames'] = sum(z[2] for z in specs)
        j['duration'] = j['frames']/FPS
        j['start_frame'] = total
        j['start'] = total/FPS
        total += j['frames']
        j['end'] = total/FPS
        j['output'] = str(ROOT/'segments'/Path(j['previous_output']).name)
        jobs.append(j)
    save_json(ROOT/'output/timeline.json', jobs)
    save_json(ROOT/'audio/base-tracks.json', audio)
    details.update(frames=total, duration=total/FPS,
                   wheeled_3x_interval_local_frames=[180,762], wheeled_output_speed='6X',
                   humanoid_3x_intervals_local_frames=[[294,1288],[1348,2098]],
                   quadruped_v11_global_interval=[200,208], quadruped_speed_multiplier=2)
    save_json(ROOT/'output/edit-manifest.json', details)
    print(json.dumps(details, indent=2), flush=True)


def select_expression(specs):
    parts = []
    for a,b,m in specs:
        term = f'between(n,{a},{b-1})'
        if m != b-a:
            term += f'*lt(mod((n-{a})*{m},{b-a}),{m})'
        parts.append(term)
    return '+'.join(parts)


def badge(text, rect, size, first, end):
    x,y,w,h = rect
    enable = f'between(n,{first},{end-1})'
    return (f"drawbox=x={x}:y={y}:w={w}:h={h}:color=0x242424:t=fill:enable='{enable}',"
            f"drawtext=fontfile='{FONT}':text='{text}':fontcolor=white:fontsize={size}:"
            f"x={x}+({w}-tw)/2:y={y}+({h}-th)/2:enable='{enable}',")


def video_filter(job):
    prefix = ''
    if job['id'] == 'clip-3':
        prefix = badge('6X', (12,72,168,122), 96, 180, 1900)
    elif job['id'] == 'clip-4':
        prefix = badge('3X', (293,8,144,97), 82, 294, 1288)
        prefix += badge('3X', (359,115,130,89), 74, 1348, 2098)
    return prefix+f"select='{select_expression(job['retime'])}',setpts=N/(60*TB),setsar=1"


def render_one(job):
    ident = job['id']
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-threads', '3',
           '-i', job['previous_output'], '-an', '-vf', video_filter(job),
           '-frames:v', str(job['frames']), '-c:v', 'libx264', '-preset', 'fast',
           '-crf', '15', '-profile:v', 'high', '-level:v', '4.2', '-threads', '4',
           '-pix_fmt', 'yuv420p', '-r', '60', '-fps_mode', 'cfr',
           '-video_track_timescale', '60000', '-map_metadata', '-1', '-map_chapters', '-1',
           '-movflags', '+faststart', '-progress', str(ROOT/'inspect'/f'{ident}-progress.txt'), job['output']]
    save_json(ROOT/'inspect'/f'{ident}-command.json', cmd)
    start = time.monotonic()
    print(f'Rendering {ident}: {job["frames"]} frames', flush=True)
    with (ROOT/'inspect'/f'{ident}-render.log').open('w') as log:
        subprocess.run(cmd, stderr=log, check=True)
    print(f'Rendered {ident} in {time.monotonic()-start:.1f}s', flush=True)


def render():
    jobs = json.loads((ROOT/'output/timeline.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(render_one, jobs))


def map_local_frame(job, f):
    offset = 0
    for a,b,m in job['retime']:
        if f < b:
            return offset + (f-a)*m/(b-a)
        offset += m
    assert abs(f-job['previous_frames']) < 1e-4, (job['id'],f)
    return offset


def mix_audio():
    jobs = json.loads((ROOT/'output/timeline.json').read_text())
    index = {j['id']:j for j in jobs}
    mix = np.zeros((sum(j['frames'] for j in jobs)*SPF,2), np.float32)
    base_tracks = json.loads((ROOT/'audio/base-tracks.json').read_text())
    for ident, tracks in base_tracks.items():
        job = index[ident]
        for entry in tracks:
            a = read_audio(entry['file'])
            assert len(a) == entry['samples']
            start = job['start_frame']*SPF+entry['start']
            mix[start:start+len(a)] += a
    old_entries = json.loads((BASE/'revision-v9/audio/narration-manifest.json').read_text())
    old_mix = read_audio(BASE/'revision-v9/audio/full-mix-v9.wav')
    # Verify that the experiment sections contain only the known narration.
    residual = old_mix.copy()
    for e in old_entries:
        a = read_audio(e['audio']);start = round(e['start']*SR)
        residual[start:start+len(a)] -= a
    for ident in ['clip-3','clip-4','clip-5']:
        j = index[ident]
        lo, hi = round(j['previous_start']*SR), round(j['previous_end']*SR)
        assert np.abs(residual[lo:hi]).max() < 1e-5, ident
    entries = []
    last_end = 0
    for original in old_entries:
        e = copy.deepcopy(original)
        ident = 'clip-3' if e['id'].startswith('wheeled-') else ('clip-5' if e['id'].startswith('quadruped-') else 'clip-4')
        j = index[ident]
        source_frame = (e['start']-j['previous_start'])*FPS
        requested = j['start']+map_local_frame(j,source_frame)/FPS
        start_seconds = max(requested,last_end+.18)
        a = read_audio(e['audio'])
        start = round(start_seconds*SR)
        end = start+len(a)
        assert end <= round(j['end']*SR), e['id']
        assert np.max(np.abs(mix[start:end])) == 0, e['id']
        mix[start:end] += a
        for key in ['requested_start','end_by']:
            e.pop(key,None)
        e.update(previous_start=original['start'], previous_end=original['end'],
                 requested_start=requested, start=start/SR, end=end/SR,
                 duration=len(a)/SR, narration_speed_preserved=True)
        entries.append(e)
        last_end = end/SR
    assert np.max(np.abs(mix)) < 1
    sf.write(ROOT/'audio/full-mix-v12.wav', mix, SR, subtype='PCM_24')
    save_json(ROOT/'audio/narration-manifest.json',entries)
    print('Audio aligned with new video timeline.', flush=True)


def finish():
    jobs = json.loads((ROOT/'output/timeline.json').read_text())
    concat = ROOT/'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    chapters = []
    for j in jobs:
        if j['id']=='clip-2':
            start = j['start_frame']
            titles = ['Visual memory and hidden space prediction', 'Selective visual memory',
                      'Predict space behind obstacles', 'Simulation and real-robot validation']
            for title, (a,b,n) in zip(titles,j['retime']):
                chapters.append(dict(title=title,start=start/FPS,end=(start+n)/FPS))
                start += n
        else:
            chapters.append(dict(title=j['title'],start=j['start'],end=j['end']))
    meta = ';FFMETADATA1\ntitle=EchoNav - Full Research Demo\n'
    for ch in chapters:
        meta += (f"[CHAPTER]\nTIMEBASE=1/60000\nSTART={round(ch['start']*60000)}\n"
                 f"END={round(ch['end']*60000)}\ntitle={ch['title']}\n")
    metadata = ROOT/'inspect/chapters.txt'
    metadata.write_text(meta)
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning',
                    '-f','concat','-safe','0','-i',str(concat),'-i',str(ROOT/'audio/full-mix-v12.wav'),
                    '-f','ffmetadata','-i',str(metadata),'-map','0:v:0','-map','1:a:0',
                    '-map_metadata','2','-map_chapters','2','-c:v','copy','-c:a','aac',
                    '-b:a','192k','-ar',str(SR),'-ac','2','-movflags','+faststart',str(NEW)],check=True)
    manifest = json.loads((ROOT/'output/edit-manifest.json').read_text())
    manifest.update(previous_version=str(OLD),output=str(NEW),chapters=chapters,
                    redactions='Reuse v11 composited video frames, including restored face pixelation and black quadruped borders.',
                    audio='Pitch-preserving slide speedup; original voice retained; no new video pauses.',
                    humanoid_intro='Keep first explanation through 4.9 seconds; accelerate subsequent motion from 2X to 3X.',
                    last_clip_field_audio='Remains muted')
    save_json(ROOT/'output/edit-manifest.json',manifest)
    print(NEW,flush=True)


if __name__=='__main__':
    {'prepare':prepare,'render':render,'audio':mix_audio,'finish':finish}[sys.argv[1]]()
