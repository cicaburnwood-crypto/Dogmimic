"""Replace the baked-in black face box on the overview slide with pixelation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / 'revision-v10/output/EchoNav_full_demo_mosaic_manual_v10.mp4'
SOURCE = BASE / 'replacement-v3/segments/02-clip-2.mp4'
ORIGINAL = Path('/home/wolfie/Downloads/newvideos/4.MP4')
SEGMENT = ROOT / 'segments/02-clip-2.mp4'
NEW = ROOT / 'output/EchoNav_full_demo_mosaic_manual_v11.mp4'
PATCH = ROOT / 'inspect/face-mosaic-patch.png'
THUMB = (1455, 648, 1689, 960)
BOX = (1616, 670, 1668, 730)
FIRST = 3194
FRAMES = 4207
SOURCE_FRAME = 750
cv2.setNumThreads(2)


def frame(path, f):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read()
    cap.release()
    assert ok, (path, f)
    return im


def prepare():
    # The original thumbnail exactly matches frame 750 (25.0 s) of 4.MP4.
    # Restore those source pixels only inside the already-redacted face area.
    original = frame(ORIGINAL, SOURCE_FRAME)
    thumb = cv2.resize(original[:, :1440], (234, 312), interpolation=cv2.INTER_AREA)
    ref = frame(SOURCE, 3685)
    restored = ref.copy()
    x0, y0, x1, y1 = THUMB
    restored[y0:y1, x0:x1] = thumb
    # Match the 12 x 12 pixel grid used by the existing manual redactions.
    mosaic = cv2.resize(cv2.resize(restored, (160, 90), interpolation=cv2.INTER_AREA),
                        (1920, 1080), interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = BOX
    patch = mosaic[y0:y1, x0:x1]
    cv2.imwrite(str(PATCH), patch)
    preview = ref.copy()
    preview[y0:y1, x0:x1] = patch
    cv2.imwrite(str(ROOT / 'inspect/preview-69.png'), preview)
    cv2.imwrite(str(ROOT / 'inspect/preview-thumbnail.png'), preview[648:960, 1455:1689])
    print(json.dumps(dict(source=str(ORIGINAL), source_frame=SOURCE_FRAME,
                         source_seconds=25.0, target=BOX, grid_pixels=12,
                         global_interval_seconds=[(455+FIRST)/60, (455+FRAMES)/60]), indent=2))


def render():
    x0, y0, x1, y1 = BOX
    # Work directly on the decoded video colors, avoiding an RGB round trip
    # for all of the unedited slide content. The page begins at local frame 3194.
    graph = f"[0:v][1:v]overlay={x0}:{y0}:enable='gte(t,53.233)':format=yuv420:shortest=1[v]"
    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-threads', '4',
           '-i', str(SOURCE), '-loop', '1', '-framerate', '60', '-i', str(PATCH),
           '-filter_complex_threads', '1', '-filter_complex', graph, '-map', '[v]',
           '-an', '-frames:v', str(FRAMES), '-c:v', 'libx264', '-preset', 'fast',
           '-crf', '15', '-profile:v', 'high', '-level:v', '4.2', '-threads', '4',
           '-pix_fmt', 'yuv420p', '-video_track_timescale', '60000',
           '-map_metadata', '-1', '-map_chapters', '-1', '-movflags', '+faststart',
           str(SEGMENT)]
    with (ROOT / 'inspect/render.log').open('w') as log:
        subprocess.run(cmd, stderr=log, check=True)
    print(SEGMENT, flush=True)


def finish():
    jobs = json.loads((BASE / 'revision-v10/output/timeline.json').read_text())
    jobs[1]['output'] = str(SEGMENT)
    (ROOT / 'output/timeline.json').write_text(json.dumps(jobs, indent=2))
    concat = ROOT / 'segments/concat.txt'
    concat.write_text(''.join(f"file '{j['output']}'\nduration {j['duration']:.9f}\n" for j in jobs))
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-f', 'concat', '-safe', '0', '-i', str(concat), '-i', str(OLD),
                    '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '1',
                    '-map_chapters', '1', '-c', 'copy', '-movflags', '+faststart',
                    str(NEW)], check=True)
    manifest = dict(previous_version=str(OLD), output=str(NEW),
                    source_frame_file=str(ORIGINAL), source_frame=SOURCE_FRAME,
                    source_frame_seconds=25.0, thumbnail_rect=THUMB,
                    changed_rect=BOX, pixel_grid=12,
                    global_interval_seconds=[(455+FIRST)/60, (455+FRAMES)/60],
                    patch_sha256=hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                    redaction_method='Manually selected face area; exact source frame; no OCR',
                    audio='Bit-for-bit copy of v10', duration_seconds=240.03333333333333,
                    frames=14402)
    (ROOT / 'output/edit-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(NEW, flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'render': render, 'finish': finish}[sys.argv[1]]()
