"""Remove the Humanoid Case 2 explanatory freeze and add a top banner."""
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / 'revision-v7/output/EchoNav_full_demo_mosaic_manual_v7.mp4'
SOURCE = BASE / 'revision-v4/segments/06-clip-4.mp4'
NEW = ROOT / 'output/EchoNav_full_demo_mosaic_manual_v8.mp4'
SEGMENT = ROOT / 'segments/06-clip-4.mp4'
FONT = '/home/wolfie/.local/share/micromamba/fonts/DejaVuSans.ttf'
FPS = 60
SR = 48000
HUMANOID_START_FRAME = 9718
CASE2_FOOTAGE_START = 1348
CUT_FIRST = 1530
CUT_END = 1900
REMOVED_FRAMES = CUT_END - CUT_FIRST
REMOVED_SECONDS = REMOVED_FRAMES / FPS
NEW_FRAMES = 14922 - REMOVED_FRAMES
DURATION = NEW_FRAMES / FPS


def render():
    # At the splice, the last clean pre-overlay frame is 1528/1529, and
    # frame 1900 is the first clean moving frame after the fade-out.
    vf = (
        f"select='lt(n,{CUT_FIRST})+gte(n,{CUT_END})',setpts=N/(60*TB),"
        f"scale=w='if(lt(n,{CASE2_FOOTAGE_START}),1920,1728)':"
        f"h='if(lt(n,{CASE2_FOOTAGE_START}),1080,972)':eval=frame:flags=lanczos,"
        "pad=w=1920:h=1080:x=(ow-iw)/2:y=oh-ih:color=black:eval=frame,"
        f"drawtext=fontfile={FONT}:text='Dynamic Avoidance':fontcolor=white:"
        "fontsize=52:x=(w-tw)/2:y=(108-th)/2:"
        f"enable='gte(n,{CASE2_FOOTAGE_START})',setsar=1"
    )
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-threads', '4',
        '-i', str(SOURCE), '-an', '-vf', vf,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-profile:v', 'high',
        '-level:v', '4.2', '-threads', '4', '-pix_fmt', 'yuv420p', '-r', '60',
        '-fps_mode', 'cfr', '-video_track_timescale', '60000',
        '-map_metadata', '-1', '-map_chapters', '-1', '-movflags', '+faststart',
        '-progress', str(ROOT / 'inspect/render-progress.txt'), str(SEGMENT)]
    (ROOT / 'inspect/render-command.json').write_text(json.dumps(cmd, indent=2))
    start = time.monotonic()
    with (ROOT / 'inspect/render.log').open('w') as log:
        subprocess.run(cmd, check=True, stderr=log)
    print(f'Rendered updated humanoid segment in {time.monotonic()-start:.1f}s', flush=True)


def audio():
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    ort.disable_telemetry_events()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(BASE / 'models/kokoro-v1.0.onnx'),
                                  sess_options=opts, providers=['CPUExecutionProvider'])
    tts = Kokoro.from_session(session, str(BASE / 'models/voices-v1.0.bin'))
    a, sr = tts.create('Dynamic avoidance.', voice='bm_george', speed=1.10, lang='en-gb')
    active = np.flatnonzero(np.abs(a) > .004)
    a = a[max(0, active[0]-round(.04*sr)):min(len(a), active[-1]+round(.12*sr))]
    raw = ROOT / 'audio/dynamic-avoidance-raw.wav'
    wav = ROOT / 'audio/dynamic-avoidance.wav'
    sf.write(raw, a, sr, subtype='PCM_24')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(raw), '-af',
                    'loudnorm=I=-16:TP=-1.5:LRA=7', '-ar', str(SR), '-ac', '2',
                    '-c:a', 'pcm_s24le', str(wav)], check=True)
    narration, sr = sf.read(wav, dtype='float32', always_2d=True)
    mix, sr = sf.read(BASE / 'revision-v7/audio/full-mix-v7.wav', dtype='float32', always_2d=True)
    assert sr == SR
    entries = json.loads((BASE / 'revision-v7/audio/narration-manifest.json').read_text())
    removed = next(e for e in entries if e['id'] == 'humanoid-anchor')
    # This window contains only the old explanation; its underlying source is silent.
    lo = round(removed['start'] * SR)
    old_audio, _ = sf.read(removed['audio'], dtype='float32', always_2d=True)
    assert np.max(np.abs(mix[lo:lo+len(old_audio)]-old_audio)) < 2e-7
    mix[lo:lo+len(old_audio)] = 0
    start = next(e['end'] for e in entries if e['id'] == 'humanoid-case-2') + .18
    lo = round(start * SR)
    assert np.max(np.abs(mix[lo:lo+len(narration)])) == 0
    assert start + len(narration)/SR < (HUMANOID_START_FRAME+CUT_FIRST)/FPS
    mix[lo:lo+len(narration)] = narration
    a = (HUMANOID_START_FRAME+CUT_FIRST) * (SR//FPS)
    b = (HUMANOID_START_FRAME+CUT_END) * (SR//FPS)
    mix = np.concatenate([mix[:a], mix[b:]])
    assert len(mix) == NEW_FRAMES * (SR//FPS)
    sf.write(ROOT / 'audio/full-mix-v8.wav', mix, SR, subtype='PCM_24')
    result = []
    for e in entries:
        if e['id'] == 'humanoid-anchor':
            continue
        e = e.copy()
        if e['start'] >= b/SR:
            for key in ['start', 'end', 'requested_start', 'end_by']:
                if key in e:
                    e[key] -= REMOVED_SECONDS
        result.append(e)
    result.append(dict(id='dynamic-avoidance', kind='case-subtitle', text='Dynamic Avoidance',
                       spoken='Dynamic avoidance.', start=start, duration=len(narration)/SR,
                       end=start+len(narration)/SR, audio=str(wav), voice='bm_george', speed=1.1))
    result.sort(key=lambda e:e['start'])
    (ROOT / 'audio/narration-manifest.json').write_text(json.dumps(result, indent=2))
    print(f'Audio aligned. New banner narration {start:.3f}–{start+len(narration)/SR:.3f}s', flush=True)


def finish():
    original_jobs = json.loads((BASE / 'assembly/output/timeline.json').read_text())
    sources = [Path(line[6:-1]) for line in (BASE / 'revision-v6/segments/concat.txt').read_text().splitlines()
               if line.startswith("file '")]
    jobs = []
    offset = 0
    for job, source in zip(original_jobs, sources):
        job = job.copy()
        if job['id'] == 'clip-4':
            source = SEGMENT
            job['frames'] -= REMOVED_FRAMES
        job['duration'] = job['frames']/FPS
        job['output'] = str(source)
        job['start'] = offset/FPS
        offset += job['frames']
        job['end'] = offset/FPS
        jobs.append(job)
    assert offset == NEW_FRAMES
    (ROOT / 'output/timeline.json').write_text(json.dumps(jobs, indent=2))
    concat = ROOT / 'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    metadata = ';FFMETADATA1\ntitle=EchoNav - Full Research Demo\n'
    for j in jobs:
        metadata += (f"[CHAPTER]\nTIMEBASE=1/60000\nSTART={round(j['start']*60000)}\n"
                     f"END={round(j['end']*60000)}\ntitle={j['title']}\n")
    meta = ROOT / 'inspect/chapters.txt'
    meta.write_text(metadata)
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'concat', '-safe', '0', '-i', str(concat),
        '-i', str(ROOT / 'audio/full-mix-v8.wav'), '-f', 'ffmetadata', '-i', str(meta),
        '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '2', '-map_chapters', '2',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', str(SR), '-ac', '2',
        '-movflags', '+faststart', str(NEW)], check=True)
    manifest = dict(previous_version=str(OLD), output=str(NEW),
        removed_local_frames=[CUT_FIRST,CUT_END], removed_seconds=REMOVED_SECONDS,
        removed_original_global_seconds=[(HUMANOID_START_FRAME+CUT_FIRST)/FPS,
                                         (HUMANOID_START_FRAME+CUT_END)/FPS],
        removed_effects=['Both occupancy charts', 'Historical and latest prediction insets',
                        'Full explanatory callout', 'Fade-in', 'Freeze', 'Fade-out'],
        removed_narration='The latest-frame anchor enables timely responses to the elevator door opening and closing.',
        banner='Dynamic Avoidance', banner_top_height=108, picture_scale=.9,
        banner_new_global_seconds=[(HUMANOID_START_FRAME+CASE2_FOOTAGE_START)/FPS,
                                    (HUMANOID_START_FRAME+2468-REMOVED_FRAMES)/FPS],
        duration_seconds=DURATION, frames=NEW_FRAMES,
        redactions='Existing manually redacted frames reused; labels and faces remain covered.',
        playback='All moving footage kept at its previous speed; only the explanatory freeze is removed.',
        audio='Removed old callout narration; added Dynamic Avoidance in the existing voice; later audio shifted with the edit.')
    (ROOT / 'output/edit-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(NEW, flush=True)


if __name__ == '__main__':
    {'render':render, 'audio':audio, 'finish':finish}[sys.argv[1]]()
