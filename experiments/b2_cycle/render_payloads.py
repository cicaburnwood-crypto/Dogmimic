"""Four physical models, rendered from their recorded trial-zero trajectories."""
import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import mujoco
import mjlab  # Sets the bundled ffmpeg path for mediapy.
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def model_for_render(path):
    spec = mujoco.MjSpec.from_file(str(path))
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.45, .55, .65], rgb2=[.82, .87, .91], width=256, height=1536)
    spec.geom('floor').rgba = [.72, .76, .80, 1]
    for axis in range(2):
        for i in range(-8, 9):
            pos = [0., 0., .0006]
            pos[axis] = i * .25
            size = [2., 2., .0005]
            size[axis] = .003
            spec.worldbody.add_geom(name=f'grid_{axis}_{i}', type=mujoco.mjtGeom.mjGEOM_BOX,
                                   size=size, pos=pos, rgba=[.58, .63, .68, 1], contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[1, -3, 4], dir=[-.2, .4, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=[.7, .7, .7], ambient=[.35, .35, .35])
    model = spec.compile()
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 450
    model.light_diffuse[:] *= .45
    model.light_ambient[:] = .10
    model.vis.headlight.diffuse[:] = .25
    model.vis.headlight.ambient[:] = .18
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--condition', choices=('push', 'no_push'), default='push')
    p.add_argument('--trial', type=int, default=0)
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    training_result = (run / 'best_eval.json').exists() and (run / 'best_trials').is_dir()
    if training_result:
        best = json.loads((run / 'best_eval.json').read_text())
        checkpoint = Path(best['checkpoint'])
        summary = dict(completed=(run / 'DONE').exists(), checkpoint=str(checkpoint),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        trial_root = run / 'best_trials'
        assert a.condition == 'push', 'Stored training evaluations include external pushes'
    else:
        summary = json.loads((run / 'summary.json').read_text())
        trial_root = run
    assert summary['completed']
    title_font = ImageFont.truetype('DejaVuSans.ttf', 22)
    font = ImageFont.truetype('DejaVuSans.ttf', 18)
    small = ImageFont.truetype('DejaVuSans.ttf', 16)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.05, 145, -12
    cam.lookat[:] = [-.20, 0., .8]
    entries, frames = [], []
    with ExitStack() as stack:
        for kg in (0, 5, 10, 15):
            directory = trial_root / f'{kg}kg_{a.condition}'
            with np.load(directory / 'rollout.npz') as packed:
                arr = {key: packed[key] for key in packed.files}
            ev = json.loads((directory / 'evaluation.json').read_text())
            if training_result:
                assert ev['it'] == best['it'] and ev['checkpoint'] == summary['checkpoint']
            model = model_for_render(ev['model_path'])
            renderer = stack.enter_context(mujoco.Renderer(model, height=450, width=640))
            active = arr['alive'][:, a.trial]
            failure = int(np.flatnonzero(~active)[0]) if not active.all() else len(active) - 1
            entries.append(dict(kg=kg, arr=arr, evaluation=ev, model=model, data=mujoco.MjData(model),
                                renderer=renderer, active=active, first_failure=failure))
        dt = float(entries[0]['arr']['dt'])
        for i in range(0, len(entries[0]['active']), 2):
            frame = Image.new('RGB', (1280, 1080), (20, 25, 35))
            draw = ImageDraw.Draw(frame)
            now = (i + 1) * dt
            phase = 'FOUR-LEG STAND' if now < 1 else 'RISE' if now < 3.5 else 'HIND-LEG HOLD' if now < 9.5 else 'RETURN TO FOUR LEGS' if now < 12.5 else 'FOUR-LEG SETTLE'
            title = (f'B2 | best checkpoint {best["it"]:,} | trial {a.trial} (preselected) | 1x speed' if training_result
                     else f'B2 | rigid back payloads | same frozen policy | trial {a.trial} (preselected) | 1x speed')
            draw.text((20, 10), title, font=title_font, fill='white')
            disturbance = 'matched 8-16 N random pulses' if a.condition == 'push' else 'no external pushes'
            draw.text((20, 45), f't={now:.2f}s | {phase} | {disturbance} | actual mass + COM + inertia', font=font, fill=(190, 213, 235))
            for j, e in enumerate(entries):
                x, y = (j % 2) * 640, 80 + (j // 2) * 500
                arr, ev, m, d = e['arr'], e['evaluation'], e['model'], e['data']
                qi = min(i, e['first_failure'])
                d.qpos[:] = arr['state'][qi, a.trial, :19]
                mujoco.mj_forward(m, d)
                renderer = e['renderer']
                renderer.update_scene(d, camera=cam, scene_option=opt)
                force = arr['force_world'][i, a.trial]
                if np.linalg.norm(force) > 0 and e['active'][i]:
                    scene = renderer.scene
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                       np.eye(3).flatten(), np.array([1., .15, .03, 1.], dtype=np.float32))
                    end = d.xipos[1].copy()
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .025, end - force * .035, end)
                    scene.ngeom += 1
                frame.paste(Image.fromarray(renderer.render()), (x, y + 50))
                draw = ImageDraw.Draw(frame)
                success_count = round(ev['success_rate'] * ev['trials'])
                draw.rectangle([x, y, x + 639, y + 49], fill=(29, 37, 49))
                draw.text((x + 12, y + 4), f"+{e['kg']} kg | total {float(m.body_mass.sum()):.1f} kg | strict pass {success_count}/{ev['trials']}",
                          font=font, fill=(255, 190, 90) if e['kg'] else (220, 235, 245))
                acceleration = np.linalg.norm(arr['com_acc'][qi, a.trial])
                detail = f"Trial {a.trial}: {'PASS' if ev['trials_detail'][a.trial]['success'] else 'FAIL'} | push {np.linalg.norm(force):.1f} N | COM |a| {acceleration:.2f} m/s^2"
                draw.text((x + 12, y + 27), detail, font=small, fill=(190, 213, 235))
                if not e['active'][i]:
                    draw.rectangle([x, y + 50, x + 639, y + 85], fill=(150, 25, 25))
                    draw.text((x + 12, y + 58), f"TERMINATED at {(e['first_failure'] + 1) * dt:.2f}s - state frozen",
                              font=small, fill='white')
                elif not ev['trials_detail'][a.trial]['success'] and now >= 13:
                    reasons = ev['trials_detail'][a.trial]['failure_reasons']
                    draw.rectangle([x, y + 465, x + 639, y + 499], fill=(100, 65, 12))
                    label = reasons[0] if reasons else 'strict stability criteria not met'
                    draw.text((x + 12, y + 474), f'Completed but failed: {label}', font=small, fill='white')
            frames.append(np.asarray(frame))
    media.write_video(str(out), frames, fps=1 / (2 * dt), codec='h264')
    sheet = Image.new('RGB', (1280, 1080 * 4))
    for j, t in enumerate((0., 6., 11., 15.4)):
        sheet.paste(Image.fromarray(frames[min(len(frames) - 1, int(t / (2 * dt)))]), (0, 1080 * j))
    sheet.save(out.with_suffix('.png'))
    meta = dict(run=str(run), checkpoint=summary['checkpoint'], checkpoint_sha256=summary['checkpoint_sha256'],
                payloads_kg=[0, 5, 10, 15], condition=a.condition, trial=a.trial,
                actual_policy_trajectory=True, failure_states_frozen=True, policy_frozen=True,
                frames=len(frames), fps=1 / (2 * dt), duration_s=len(frames) * 2 * dt,
                dimensions=[1280, 1080], panels=4)
    if training_result:
        meta.update(iteration=best['it'],training_result=True,selection='fixed nominal trial zero' if a.trial==0 else 'explicit trial index',
                    trial_success_by_payload={str(e['kg']):e['evaluation']['trials_detail'][a.trial]['success'] for e in entries},
                    strict_success_counts={str(e['kg']):round(e['evaluation']['success_rate']*e['evaluation']['trials']) for e in entries})
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)
    print(str(out), flush=True)


if __name__ == '__main__':
    main()
