from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'revision-v6/output/EchoNav_full_demo_mosaic_manual_v6.mp4'
NEW = ROOT / 'output/EchoNav_full_demo_mosaic_manual_v7.mp4'
SR = 48000


def video_evidence(path):
    payload = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
        '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-']).decode().strip()
    packets = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts,dts,duration,size,flags', '-of', 'csv=p=0', str(path)])
    return dict(payload=payload, timestamps_and_packet_sizes_sha256=hashlib.sha256(packets).hexdigest())


with ThreadPoolExecutor(max_workers=2) as pool:
    before, after = list(pool.map(video_evidence, [OLD, NEW]))
assert before == after, (before, after)
probe = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error', '-show_streams', '-show_chapters', '-show_format',
    '-of', 'json', str(NEW)]))
v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
a = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
assert (v['width'], v['height'], v['nb_frames'], v['avg_frame_rate']) == (1920, 1080, '14922', '60/1')
assert float(probe['format']['duration']) == 248.7
assert len(probe['chapters']) == 8
assert a['sample_rate'] == '48000' and a['channels'] == 2

manifest = json.loads((ROOT / 'audio/narration-manifest.json').read_text())
assert len(manifest) == 11
assert sum(e['kind'] == 'case-title' for e in manifest) == 5
assert sum(e['kind'] == 'callout' for e in manifest) == 6
raw = subprocess.check_output([
    'ffmpeg', '-v', 'error', '-i', str(NEW), '-map', '0:a:0',
    '-ar', str(SR), '-ac', '2', '-f', 'f32le', '-'])
decoded = np.frombuffer(raw, '<f4').reshape(-1, 2)
mixed, sr = sf.read(ROOT / 'audio/full-mix-v7.wav', dtype='float32', always_2d=True)
assert sr == SR and len(mixed) == round(248.7 * SR)
assert len(decoded) >= len(mixed)
assert np.max(np.abs(decoded)) < 1.0
details = []
for i, e in enumerate(manifest):
    if i:
        assert e['start'] - manifest[i - 1]['end'] >= .179
    lo = round(e['start'] * SR)
    hi = round(e['end'] * SR)
    corr = float(np.corrcoef(mixed[lo:hi, 0], decoded[lo:hi, 0])[0, 1])
    assert corr > .98, (e['id'], corr)
    details.append(dict(id=e['id'], start=e['start'], end=e['end'],
                        voice=e['voice'], speed=e['speed'], pcm_aac_correlation=corr))

# No field recording should reappear anywhere in the final quadruped clip.
# Its only new sound is the requested three-sentence TTS explanation.
last = manifest[-1]
silent_regions = [(206.15, last['start'] - .08), (last['end'] + .08, 248.65)]
silence = []
for lo, hi in silent_regions:
    peak = float(np.max(np.abs(decoded[round(lo * SR):round(hi * SR)])))
    assert peak < 1e-5, (lo, hi, peak)
    silence.append(dict(start=lo, end=hi, peak=peak))

subprocess.run([
    'ffmpeg', '-v', 'error', '-threads', '4', '-i', str(NEW),
    '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
report = dict(
    duration_seconds=248.7, resolution=[1920, 1080], fps=60, frames=14922,
    chapters=8, video_payload_and_timestamps_identical_to_v6=True,
    video=after, no_new_pauses_or_retiming=True, new_narrations=details,
    last_clip_original_audio_still_muted=silence,
    decoded_audio_peak=float(np.max(np.abs(decoded))), full_decode='passed',
    bytes=NEW.stat().st_size, sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
(ROOT / 'output/verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
