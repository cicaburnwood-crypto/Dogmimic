"""Render an unedited, preselected G1 policy evaluation from two fixed cameras."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import mediapy as media
from PIL import Image, ImageDraw, ImageFont
from mjlab.asset_zoo.robots import get_g1_robot_cfg


parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--trial', type=int, default=0)
args = parser.parse_args()
run, output = Path(args.run), Path(args.out)
output.parent.mkdir(parents=True, exist_ok=True)
source = run / 'best_rollout.npz'
record = np.load(source)
state = record['state'][:, args.trial]
feet = record['feet'][:, args.trial]
alive = record['alive'][:, args.trial]
dt = float(record['dt'])
evaluation = json.loads((run / 'best_eval.json').read_text())
assert state.shape[-1] == 71 and np.isfinite(state).all()
assert alive.all(), 'This renderer requires a complete trial; do not hide resets or failures.'
jump = evaluation['task'] == 'g1_jump_full_turn'
title = 'G1 | Jump + 360-degree turn' if jump else 'G1 | Funky Chicken-inspired dance'
w, x, y, z = state[:, 3:7].T
yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
turn = np.degrees(yaw - yaw[0])

spec = get_g1_robot_cfg().spec_fn()
spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[10, 10, .1], rgba=[.73, .77, .81, 1])
spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=[.45, .55, .65], rgb2=[.82, .87, .91], width=256, height=1536)
for axis in range(2):
    for k in range(-8, 9):
        position = [0., 0., .0006]
        position[axis] = k * .25
        size = [2., 2., .0005]
        size[axis] = .002
        spec.worldbody.add_geom(name=f'grid_{axis}_{k}', type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=size, pos=position, rgba=[.59, .64, .69, 1],
                                contype=0, conaffinity=0)
spec.worldbody.add_light(pos=[2, -2, 4], dir=[-.4, .4, -1],
                         type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                         diffuse=[.65, .65, .65], ambient=[.25, .25, .25])
model = spec.compile()
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 640
model.light_diffuse[:] *= .65
model.vis.headlight.diffuse[:] = .35
model.vis.headlight.ambient[:] = .20
data = mujoco.MjData(model)
assert model.nq == 36
root = model.body('pelvis').id
root_address = model.jnt_qposadr[model.body_jntadr[root]]
hinges = [model.body_jntadr[root + 1 + k] for k in range(29)]
addresses = model.jnt_qposadr[hinges]
assert len(set(addresses.tolist())) == 29
foot_ids = [model.body(name).id for name in ('left_ankle_roll_link', 'right_ankle_roll_link')]
options = mujoco.MjvOption()
options.geomgroup[:] = [1, 1, 1, 0, 0, 0]
cameras = []
center = state[:, :2].mean(0)
for angle in (0, 55):
    camera = mujoco.MjvCamera()
    camera.distance = 3.45 if jump else 3.05
    camera.azimuth = angle
    camera.elevation = -10
    camera.lookat[:] = [center[0], center[1], .92 if jump else .74]
    cameras.append(camera)
font = ImageFont.truetype('DejaVuSans.ttf', 23)
small = ImageFont.truetype('DejaVuSans.ttf', 18)
frames, foot_errors = [], []
with mujoco.Renderer(model, height=640, width=640) as renderer:
    for i, s in enumerate(state):
        data.qpos[root_address:root_address + 7] = s[:7]
        data.qpos[addresses] = s[7:36]
        mujoco.mj_forward(model, data)
        foot_errors.append(float(np.max(np.abs(data.xpos[foot_ids] - feet[i]))))
        frame = Image.new('RGB', (1280, 720), (20, 25, 35))
        for view, camera in enumerate(cameras):
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame.paste(Image.fromarray(renderer.render()), (view * 640, 80))
        draw = ImageDraw.Draw(frame)
        draw.text((18, 9), title, font=font, fill='white')
        subtitle = f'Actual trained policy | iteration {evaluation["it"]:,} | nominal trial {args.trial} | 1x speed'
        draw.text((18, 44), subtitle, font=small, fill=(190, 213, 235))
        draw.text((18, 93), 'FRONT VIEW', font=small, fill='white', stroke_width=1, stroke_fill=(45, 55, 65))
        draw.text((658, 93), 'OBLIQUE VIEW', font=small, fill='white', stroke_width=1, stroke_fill=(45, 55, 65))
        draw.rectangle([0, 685, 1280, 720], fill=(20, 25, 35))
        if jump:
            phase = 'AIRBORNE' if feet[i, :, 2].min() > .09 else 'GROUND / RECOVERY'
            detail = f't={(i + 1) * dt:.2f}s | rotation {turn[i]:.1f} deg | {phase}'
        else:
            detail = f't={(i + 1) * dt:.2f}s | original choreography | full 9-second evaluation'
        draw.text((18, 693), detail, font=small, fill=(205, 222, 237))
        frames.append(np.asarray(frame))
# state_to_flat stores a quaternion reconstructed with float32 diagonal square
# roots. Near-zero components incur cancellation, so recorded FK and replay FK
# differ by submillimeter amounts. Keep the saved poses unchanged and record the
# discrepancy; this is a rendering check, not a relaxed task success criterion.
assert max(foot_errors) < 5e-4, f'Recorded state/joint order mismatch: {max(foot_errors)}'
media.write_video(str(output), frames, fps=1 / dt, codec='h264')
times = (0., 1.5, 1.9, 2.15, 2.6, 4.58) if jump else (0., 1.8, 3.6, 5.4, 7.2, 8.98)
sheet = Image.new('RGB', (1280, 360 * len(times)))
for row, t in enumerate(times):
    frame = Image.fromarray(frames[min(len(frames) - 1, int(t / dt))])
    sheet.paste(frame.resize((640, 360)), ((row % 2) * 640, (row // 2) * 360))
sheet = sheet.crop((0, 0, 1280, 1080))
sheet.save(output.with_suffix('.png'))
metadata = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                checkpoint=str(run / 'best.pt'), checkpoint_sha256=hashlib.sha256((run / 'best.pt').read_bytes()).hexdigest(),
                task=evaluation['task'], iteration=evaluation['it'], trial=args.trial,
                frames=len(frames), fps=1 / dt, duration_s=len(frames) * dt, views=2,
                width=1280, height=720, playback_speed=1., all_frames_active=True,
                actual_policy_trajectory=True, reference_motion_visualization=False,
                first_state_time_s=dt, max_foot_fk_error_m=max(foot_errors),
                evaluation_success_rate=evaluation['success_rate'])
output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
