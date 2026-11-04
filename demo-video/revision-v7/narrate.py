"""Read manually verified case titles and full-sentence callouts over v6.

This revision changes audio only. The video stream, timestamps, chapters,
existing visual effects, redactions and source replacements are preserved.
"""
from pathlib import Path
import json
import math
import subprocess
import time

import numpy as np
import soundfile as sf
import onnxruntime as ort
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'revision-v6/output/EchoNav_full_demo_mosaic_manual_v6.mp4'
OUTPUT = ROOT / 'output/EchoNav_full_demo_mosaic_manual_v7.mp4'
SR = 48000
DURATION = 248.7
VOICE = 'bm_george'

# Read from the rendered v6 frames, without OCR. Times are global seconds.
ENTRIES = [
    dict(id='wheeled-case-1', kind='case-title', start=80.75,
         text='Case 1: Multiple Glass doors and misleading turning point',
         spoken='Case one. Multiple glass doors and misleading turning point.'),
    dict(id='wheeled-memory', kind='callout', start=100.65,
         text='Memory enables our model to guide the robot back to the corridor after exploring the room and finding no way out.'),
    dict(id='wheeled-case-2', kind='case-title', start=112.416666667,
         text='Case 2: Goal outside the room',
         spoken='Case two. Goal outside the room.'),
    dict(id='wheeled-prediction', kind='callout', start=120.50,
         text='Predicted free space in unobserved areas enables our model to plan a clear path through the doorway.'),
    dict(id='wheeled-case-3', kind='case-title', start=141.783333333,
         text='Case 3: Narrow space in darkness',
         spoken='Case three. Narrow space in darkness.'),
    dict(id='wheeled-lighting', kind='callout', start=145.15,
         text='Our model and OmniVLA both passed the test under these lighting conditions.',
         spoken='Our model and Omni V L A both passed the test under these lighting conditions.'),
    dict(id='humanoid-case-1', kind='case-title', start=162.016666667,
         text='Humanoid Test Case 1', spoken='Humanoid test case one.'),
    dict(id='humanoid-glass', kind='callout', start=163.65,
         text='Our model detects the glass door and predicts free space behind it.'),
    dict(id='humanoid-case-2', kind='case-title', start=183.516666667,
         text='Humanoid Test Case 2', spoken='Humanoid test case two.'),
    dict(id='humanoid-anchor', kind='callout', start=187.80,
         text='The latest-frame anchor enables timely responses to the elevator door opening and closing.'),
    dict(id='quadruped-memory', kind='callout', start=233.55, end_by=248.45,
         text='The robot detects obstacles ahead and explores to its left in search of a better route. As this region enters its field of view, additional obstacles are detected. Our memory mechanism uses historical observations to prevent the robot from becoming trapped in local minima.'),
]


def trim_boundary_silence(a, sr):
    active = np.flatnonzero(np.abs(a) > .004)
    if len(active):
        a = a[max(0, active[0] - round(.04 * sr)):
              min(len(a), active[-1] + round(.12 * sr))]
    return a


def synthesize():
    ort.disable_telemetry_events()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(ROOT.parent / 'models/kokoro-v1.0.onnx'),
        sess_options=opts, providers=['CPUExecutionProvider'])
    tts = Kokoro.from_session(session, str(ROOT.parent / 'models/voices-v1.0.bin'))
    entries = []
    cachefile = ROOT / 'audio/narration-manifest.json'
    cached = {e['id']: e for e in json.loads(cachefile.read_text())} if cachefile.exists() else {}
    for entry in ENTRIES:
        e = entry.copy()
        e.setdefault('spoken', e['text'])
        e['requested_start'] = e['start']
        # Let the preceding title finish without simultaneous speakers.
        if entries and e['start'] < entries[-1]['end'] + .18:
            e['start'] = entries[-1]['end'] + .18
        prior = cached.get(e['id'])
        if (prior and prior['spoken'] == e['spoken'] and prior['start'] == e['start']
                and Path(prior['audio']).exists()):
            entries.append(prior)
            print(f"{e['id']}: using verified generated clip", flush=True)
            continue
        speed = 1.10
        st = time.monotonic()
        a, sr = tts.create(e['spoken'], voice=VOICE, speed=speed, lang='en-gb')
        a = trim_boundary_silence(a, sr)
        e['standard_speed_duration'] = len(a) / sr
        # The final three-sentence caption must finish before the existing end.
        # Regenerate at a modestly faster rate; never truncate speech or video.
        limit = e.get('end_by', DURATION) - e['start']
        while len(a) / sr > limit:
            speed = math.ceil(speed * (len(a) / sr) / (limit - .12) * 100) / 100
            assert speed <= 1.5, ('Unexpectedly high rate required', speed)
            print(f"{e['id']}: fitting {len(a) / sr:.3f}s into {limit:.3f}s, rate {speed:.2f}", flush=True)
            a, sr = tts.create(e['spoken'], voice=VOICE, speed=speed, lang='en-gb')
            a = trim_boundary_silence(a, sr)
        assert len(a) / sr <= limit, (e['id'], len(a) / sr, limit)
        raw = ROOT / 'audio' / f"{e['id']}-raw.wav"
        normalized = ROOT / 'audio' / f"{e['id']}.wav"
        sf.write(raw, a, sr, subtype='PCM_24')
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(raw),
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=7', '-ar', str(SR), '-ac', '2',
            '-c:a', 'pcm_s24le', str(normalized)], check=True)
        info = sf.info(normalized)
        e.update(voice=VOICE, speed=speed, audio=str(normalized),
                 duration=info.duration, end=e['start'] + info.duration)
        assert e['end'] <= e.get('end_by', DURATION)
        entries.append(e)
        print(f"{e['id']}: {e['start']:.3f}–{e['end']:.3f}, "
              f"speed {speed:.2f}, generated in {time.monotonic() - st:.1f}s", flush=True)
        (ROOT / 'audio/narration-manifest.json').write_text(json.dumps(entries, indent=2))
    return entries


def mix_and_mux(entries):
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(SOURCE), '-map', '0:a:0',
        '-ar', str(SR), '-ac', '2', '-f', 'f32le', '-'])
    original = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
    mix = np.zeros((round(DURATION * SR), 2), dtype=np.float32)
    n = min(len(mix), len(original))
    mix[:n] = original[:n]
    # v6 already has silence in all new speech windows, including the last clip.
    # Assert that, so prior narration cannot accidentally be doubled or ducked.
    for e in entries:
        a, sr = sf.read(e['audio'], dtype='float32', always_2d=True)
        assert sr == SR and a.shape[1] == 2
        start = round(e['start'] * SR)
        end = start + len(a)
        assert end <= len(mix)
        peak = float(np.max(np.abs(mix[start:end])))
        assert peak < 1e-5, (e['id'], 'Existing speech in placement window', peak)
        mix[start:end] += a
    assert np.max(np.abs(mix)) < 1.0
    mixfile = ROOT / 'audio/full-mix-v7.wav'
    sf.write(mixfile, mix, SR, subtype='PCM_24')
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', str(SOURCE), '-i', str(mixfile),
        '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '0', '-map_chapters', '0',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', str(SR), '-ac', '2',
        '-t', str(DURATION), '-movflags', '+faststart', str(OUTPUT)], check=True)
    manifest = dict(
        previous_version=str(SOURCE), output=str(OUTPUT), duration=DURATION,
        video='Original v6 compressed video stream copied without re-encoding or retiming.',
        new_title_count=5, new_callout_count=6,
        already_narrated_sections=['Wheeled robot test cases', 'Humanoid test cases', 'Quadruped test case'],
        voice=VOICE, locale='en-gb', video_pauses_added=0, video_frames_added=0,
        last_clip_original_audio='Remains muted; only the newly requested callout narration is added.',
        script_source='Manually read and visually checked against rendered v6 frames; no OCR.',
        entries=entries)
    (ROOT / 'output/edit-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(OUTPUT, flush=True)


if __name__ == '__main__':
    mix_and_mux(synthesize())
