"""Serve local keyboard control and a live MuJoCo camera for a selected M1 policy."""
from argparse import ArgumentParser
from hashlib import sha256
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
from PIL import Image

from controllers import BASELINES, M1Controller
from native_world import NATIVE_RUNTIME, Course
from sim import (DT, M1_FLAT, M1_HARDWARE_STAND, M1_HARDWARE_STAND_KD, M1_HARDWARE_STAND_KP,
                 M1_HOME, M1_KD, M1_KP, M1_NAMES, M1_REMOTE_SMOOTH, M1_SCALE, M1_SETTLE_KD,
                 M1_SETTLE_KP, M1_STAND, ROOT, RUNTIME, Robot, body_matrix, step_m1_handoff)

WEB = ROOT / "web"
LIMITS = np.array([.60, .15, .35])
COMMAND_TTL = .32
AZIMUTH_SPEED = 100.
ELEVATION_SPEED = 70.
ELEVATION_LIMITS = (-82., -6.)


class Dashboard:
    def __init__(self, checkpoint, name, native=False, descent=False, hardware_preview=False):
        self.policy = M1Controller("classic", checkpoint, period=.8 if native else .7, continuous_phase=native)
        if hardware_preview and self.policy.use_terrain:
            raise ValueError("The hardware preview requires a policy without terrain observations")
        self.hardware_preview = hardware_preview
        self.policy_info = {"name": name, "sha256": sha256(checkpoint.read_bytes()).hexdigest()}
        self.scenes = {"flat": ("平地", M1_FLAT)}
        self.descending_starts = {}
        if native:
            stage = NATIVE_RUNTIME / "view"
            stage.mkdir(parents=True, exist_ok=True)
            for cm in (18,20):
                path = stage / f"stairs-{cm}.xml"
                course = Course(rise=cm/100., steps=10)
                course.write(path, source=M1_FLAT)
                self.scenes[f"stairs-{cm}"] = (f"平地 → {cm} cm 楼梯 → 平台", path)
                if descent:
                    key = f"descend-{cm}"
                    self.scenes[key] = (f"上平台 → 下 {cm} cm 楼梯 → 平地", path)
                    self.descending_starts[key] = (course.descending_start, course.top+.54)
        self.scene = "stairs-20" if native and not hardware_preview else "flat"
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.command = np.zeros(3)
        self.command_time = 0.
        self.view_input = np.zeros(2)
        self.view_time = 0.
        self.view = None
        self.sequence = {}
        self.reset_requested = True
        self.sample = {"pose": None, "ready": False, "fault": None}
        self.frame = 0
        self.frame_time = 0.
        self.image = None

    def _accept_sequence(self, channel, client, sequence):
        key = (channel, client)
        if sequence <= self.sequence.get(key, -1):
            return False
        self.sequence[key] = sequence
        if len(self.sequence) > 96:
            del self.sequence[next(iter(self.sequence))]
        return True

    def drive(self, client, sequence, command):
        with self.lock:
            if self._accept_sequence("drive", client, sequence):
                self.command_time = time.monotonic()
                limits = LIMITS if self.scene == "flat" else np.array([.45,.15,.35])
                self.command = np.clip(command, -limits, limits)

    def orbit(self, client, sequence, direction):
        with self.lock:
            if self._accept_sequence("orbit", client, sequence):
                self.view_time = time.monotonic()
                self.view_input = np.clip(direction, -1., 1.)

    def active(self):
        return self.command if time.monotonic() - self.command_time < COMMAND_TTL else np.zeros(3)

    def request_reset(self, scene=None):
        with self.lock:
            if scene is not None:
                if scene not in self.scenes:
                    raise ValueError("Unknown M1 scene")
                self.scene = scene
            self.command_time = 0.
            self.view_time = 0.
            self.reset_requested = True

    def state(self):
        with self.lock:
            return {**self.sample, "command": self.active().tolist(), "policy": self.policy_info,
                    "camera": time.monotonic() - self.frame_time < 2, "frame": self.frame,
                    "view": self.view,
                    "scene": self.scene, "scene_name": self.scenes[self.scene][0]}

    def run(self):
        try:
            self.simulate()
        except Exception as exc:
            with self.lock:
                self.sample = {**self.sample, "ready": False, "fault": str(exc)}
            raise

    def simulate(self):
        camera = mujoco.MjvCamera()
        deadline = time.monotonic()
        tick, policy_tick, settle, failed = 0, 0, 0, False
        active_scene, renderer, robot = None, None, None
        filtered = np.zeros(3)
        stand, stand_kp, stand_kd = (
            (M1_HARDWARE_STAND, M1_HARDWARE_STAND_KP, M1_HARDWARE_STAND_KD)
            if self.hardware_preview else (M1_STAND, M1_SETTLE_KP, M1_SETTLE_KD)
        )
        try:
            while not self.closed.is_set():
                with self.lock:
                    reset = self.reset_requested
                    self.reset_requested = False
                    command = self.active()
                    scene = self.scene
                    orbit = self.view_input if time.monotonic() - self.view_time < COMMAND_TTL else None
                if scene != active_scene:
                    if renderer is not None:
                        renderer.close()
                        renderer = None
                    robot = Robot(self.scenes[scene][1], M1_NAMES, .095)
                    robot.model.vis.global_.offwidth = 768
                    renderer = mujoco.Renderer(robot.model, width=768, height=432)
                    if active_scene is None:
                        camera.azimuth, camera.elevation = (145., -26.) if scene == "flat" else (136., -21.)
                    active_scene = scene
                    camera.distance = 2.1 if scene == "flat" else 3.45
                    reset = True
                if orbit is not None:
                    horizontal, vertical = orbit
                    camera.azimuth = (camera.azimuth + horizontal * AZIMUTH_SPEED * DT) % 360.
                    elevation = camera.elevation - vertical * ELEVATION_SPEED * DT
                    camera.elevation = min(max(elevation, ELEVATION_LIMITS[0]), ELEVATION_LIMITS[1])
                if reset:
                    if scene in self.descending_starts:
                        start, height = self.descending_starts[scene]
                        robot.reset(stand, height)
                        robot.data.qpos[0] = start
                        robot.data.qpos[3:7] = [0., 0., 0., 1.]
                        mujoco.mj_forward(robot.model, robot.data)
                    else:
                        robot.reset(stand, .54)
                    self.policy.reset()
                    filtered.fill(0.)
                    policy_tick, settle, failed = 0, 60, False
                if settle:
                    robot.step(stand, stand_kp, stand_kd)
                    settle -= 1
                elif not failed:
                    if self.hardware_preview or scene != "flat":
                        filtered = M1_REMOTE_SMOOTH * filtered + (1 - M1_REMOTE_SMOOTH) * command
                        command = filtered
                    action = self.policy.action(self.policy.observe(robot, command))
                    if self.hardware_preview:
                        step_m1_handoff(robot, action, policy_tick)
                        policy_tick += 1
                    else:
                        robot.step(M1_HOME + action * M1_SCALE, M1_KP, M1_KD)
                    failed = not robot.valid()

                x, y, z = robot.data.qpos[:3]
                rotation = body_matrix(robot.data.qpos[3:7])
                vx, vy, yaw = robot.velocity()
                pose = {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "time": robot.data.time,
                        "heading": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                        "speed": float(np.hypot(vx, vy)), "yaw_rate": yaw}
                with self.lock:
                    self.sample = {"pose": pose, "ready": not (settle or failed),
                                   "fault": "按 R 复位" if failed else None}
                if tick % 4 == 0:
                    camera.lookat[:] = (x, y, z - (.16 if scene == "flat" else .04))
                    renderer.update_scene(robot.data, camera=camera)
                    jpg = BytesIO()
                    Image.fromarray(renderer.render()).save(jpg, format="JPEG", quality=80)
                    with self.lock:
                        self.image = jpg.getvalue()
                        self.frame += 1
                        self.frame_time = time.monotonic()
                        self.view = {"azimuth": camera.azimuth, "elevation": camera.elevation,
                                     "target": camera.lookat.tolist()}
                tick += 1
                deadline = max(deadline + DT, time.monotonic() - 2 * DT)
                self.closed.wait(max(0., deadline - time.monotonic()))
        finally:
            if renderer is not None:
                renderer.close()


def serve(dashboard, port):
    index = (ROOT / "keyboard.html").read_text().replace("{{POLICY}}", escape(dashboard.policy_info["name"]))
    scenes = ""
    if len(dashboard.scenes) > 1:
        options = []
        for scene, (name, _) in dashboard.scenes.items():
            selected = " selected" if scene == dashboard.scene else ""
            options.append(f'<option value="{escape(scene)}"{selected}>{escape(name)}</option>')
        scenes = '<label class="scene-choice" for="scene">测试场景</label><select id="scene">'+"".join(options)+"</select>"
    index = index.replace("{{SCENES}}", scenes)
    pages = {
        "/": ("text/html; charset=utf-8", index.encode()),
        "/style.css": ("text/css; charset=utf-8", (WEB / "style.css").read_bytes()),
        "/app.js": ("text/javascript; charset=utf-8", (WEB / "app.js").read_bytes()),
    }

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, kind="text/plain", body=b""):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = self.path.partition("?")[0]
            if path in pages:
                self.respond(200, *pages[path])
            elif path == "/state":
                self.respond(200, "application/json", json.dumps(dashboard.state()).encode())
            elif path == "/camera.jpg":
                with dashboard.lock:
                    image = dashboard.image
                self.respond(200 if image else 503, "image/jpeg", image or b"")
            else:
                self.respond(404)

        def do_POST(self):
            if self.path not in ("/command", "/orbit", "/reset") or self.headers.get_content_type() != "application/json":
                self.respond(404)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://localhost:{port}", f"http://127.0.0.1:{port}"):
                self.respond(403)
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size <= 512:
                    raise ValueError("payload length")
                body = json.loads(self.rfile.read(size))
                if self.path == "/reset":
                    if not isinstance(body, dict):
                        raise ValueError("Expected a reset object")
                    dashboard.request_reset(body.get("scene"))
                else:
                    client, sequence, vector = (body[name] for name in ("client", "sequence", "vector"))
                    command = np.asarray(vector, dtype=float)
                    if not isinstance(client, str) or not 0 < len(client) <= 64:
                        raise ValueError("client ID")
                    orbiting = self.path == "/orbit"
                    width = 2 if orbiting else 3
                    if type(sequence) is not int or sequence < 0 or command.shape != (width,) or not np.isfinite(command).all():
                        raise ValueError("sequence or command")
                    handler = dashboard.orbit if orbiting else dashboard.drive
                    handler(client, sequence, command)
            except (TypeError, KeyError, ValueError, OverflowError):
                self.respond(400)
                return
            self.respond(204)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        worker = threading.Thread(target=dashboard.run, name="m1-simulation")
        worker.start()
        print(f"M1 {dashboard.policy_info['name']} on http://localhost:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            dashboard.closed.set()
            worker.join()


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8767)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--checkpoint", type=Path, help="Preview a different Classic-format candidate")
    modes.add_argument("--hardware-preview", type=Path, help="Preview a Uni Exp hardware candidate on flat ground and stairs")
    modes.add_argument("--baseline", action="store_true", help="Run the original archived Classic controller")
    modes.add_argument("--native", action="store_true", help="Run the accepted M1-only universal actor")
    args = p.parse_args()
    native_release = NATIVE_RUNTIME / "release.json"
    native, descent = False, False
    if args.hardware_preview:
        checkpoint, label = args.hardware_preview, "Uni Exp"
        native = descent = True
    elif args.checkpoint:
        checkpoint, label = args.checkpoint, "M1 候选策略"
    elif not args.baseline and (args.native or native_release.is_file()):
        release = json.loads(native_release.read_text())
        native = True
        descent = bool(release.get("descent_check"))
        checkpoint, label = NATIVE_RUNTIME / release["checkpoint"], release["name"]
        if not release["passed"] or sha256(checkpoint.read_bytes()).hexdigest() != release["sha256"]:
            raise ValueError("Native deployment does not match the accepted policy hash")
    else:
        checkpoint, label = RUNTIME / "m1/models" / BASELINES["classic"], "M1 Classic 原始基线"
    serve(Dashboard(checkpoint, label, native=native, descent=descent, hardware_preview=bool(args.hardware_preview)), args.port)


if __name__ == "__main__":
    main()
