from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess

import cv2
import numpy as np

from edit_v11 import ROOT, OLD, NEW, SOURCE, SEGMENT, PATCH, BOX, FIRST, FRAMES, frame


def payloads(path):
    audio = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
        '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-']).decode().strip()
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=data_hash', '-show_data_hash', 'sha256',
        '-of', 'json', str(path)]))
    return audio, [p['data_hash'] for p in data['packets']]


with ThreadPoolExecutor(max_workers=2) as pool:
    old, new = list(pool.map(payloads, [OLD, NEW]))
assert old[0] == new[0]
assert len(old[1]) == len(new[1]) == 14402
assert old[1][:455] == new[1][:455]
assert old[1][4662:] == new[1][4662:]
probe = json.loads(subprocess.check_output([
    'ffprobe', '-v', 'error', '-show_streams', '-show_chapters',
    '-show_format', '-of', 'json', str(NEW)]))
video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
assert (video['width'], video['height'], video['avg_frame_rate'], video['nb_frames']) == (1920, 1080, '60/1', '14402')
assert abs(float(video['duration']) - 240.03333333333333) < 1e-5
assert len(probe['chapters']) == 8

x0, y0, x1, y1 = BOX
patch = cv2.imread(str(PATCH))
checks = []
outside = np.ones((1080, 1920), bool)
# Exclude a small ring around the edited region to allow chroma subsampling.
outside[y0-4:y1+4, x0-4:x1+4] = False
for f in [0, 1500, 3193, 3194, 3195, 3685, 4206]:
    before = frame(SOURCE, f)
    after = frame(SEGMENT, f)
    expected = before.copy()
    if f >= FIRST:
        expected[y0:y1, x0:x1] = patch
    delta = after.astype(float) - expected.astype(float)
    mae = float(np.abs(delta).mean())
    psnr = float(10 * np.log10(255**2 / np.mean(delta**2)))
    assert mae < 2 and psnr > 39, (f, mae, psnr)
    outside_error = float(np.abs(after.astype(float)-before.astype(float))[outside].mean())
    assert outside_error < 2, (f, outside_error)
    face_error = float(np.abs(delta[y0:y1, x0:x1]).mean())
    if f >= FIRST:
        assert face_error < 5, (f, face_error)
        assert float(after[y0:y1, x0:x1].mean()) > 70
    else:
        assert face_error < 3, (f, face_error)
    checks.append(dict(segment_frame=f, global_seconds=(455+f)/60,
                       pixel_error=mae, psnr_db=psnr, face_error=face_error,
                       outside_edit_error=outside_error, mosaic_active=f>=FIRST))

final = frame(NEW, 4140)
cv2.imwrite(str(ROOT / 'inspect/final-69.png'), final)
cv2.imwrite(str(ROOT / 'inspect/final-thumbnail.png'), final[648:960, 1455:1689])
subprocess.run(['ffmpeg', '-v', 'error', '-threads', '4', '-i', str(NEW),
                '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
report = dict(output=str(NEW), frames=14402, duration_seconds=240.03333333333333,
              resolution=[1920,1080], fps=60, chapters=8,
              audio_identical_to_v10=True, audio_payload_sha256=new[0],
              other_video_segments_identical_to_v10=True,
              source_thumbnail_frame=750, source_thumbnail_seconds=25.0,
              changed_global_interval_seconds=[(455+FIRST)/60, (455+FRAMES)/60],
              checks=checks, full_decode='passed', bytes=NEW.stat().st_size,
              sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
(ROOT / 'output/verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
