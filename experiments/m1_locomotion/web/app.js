const el = id => document.getElementById(id);
const keys = new Set();
const motion = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"]);
const orbitEnabled = document.documentElement.hasAttribute("data-camera-orbit");
const orbitKeys = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);
const orbit = new Set();
const aliases = orbitEnabled ? {} : {ArrowUp: "KeyW", ArrowDown: "KeyS", ArrowLeft: "KeyQ", ArrowRight: "KeyE"};
const buttons = Array.from(document.querySelectorAll("[data-key]"));
const client = crypto.randomUUID();
let sequence = 0;
let previousFrame = -1;
let lastCameraLoaded = 0;
let cameraRequest = false;

function sendInput(path, vector, keepalive) {
  fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({client, sequence: ++sequence, vector}), keepalive,
  }).catch(() => {});
}

const axis = (pressed, positive, negative) => Number(pressed.has(positive)) - Number(pressed.has(negative));

function publish(keepalive = false) {
  const speed = Number(el("speed").value);
  const x = axis(keys, "KeyW", "KeyS");
  const y = axis(keys, "KeyA", "KeyD");
  const scale = speed / Math.max(1, Math.hypot(x, y));
  sendInput("/command", [x * scale, y * Math.min(scale, 0.45), axis(keys, "KeyQ", "KeyE") * 0.65], keepalive);
}

function publishOrbit(keepalive = false) {
  sendInput("/orbit", [axis(orbit, "ArrowRight", "ArrowLeft"), axis(orbit, "ArrowUp", "ArrowDown")], keepalive);
}

function updateKeys() {
  for (const button of buttons) {
    button.classList.toggle("active", keys.has(button.dataset.key));
  }
  publish(true);
}

function stop() {
  keys.clear();
  updateKeys();
}

function stopOrbit() {
  if (!orbit.size) return;
  orbit.clear();
  publishOrbit(true);
}

function releaseControls() {
  stop();
  stopOrbit();
}

function formControl(target) {
  return target instanceof Element && target.closest("input, select, textarea, [contenteditable='true']");
}

window.addEventListener("keydown", event => {
  if (event.ctrlKey || event.altKey || event.metaKey || formControl(event.target)) return;
  if (orbitEnabled && orbitKeys.has(event.code)) {
    event.preventDefault();
    if (!event.repeat) {
      orbit.add(event.code);
      publishOrbit(true);
    }
    return;
  }
  const key = aliases[event.code] ?? event.code;
  if (key === "Space") {
    event.preventDefault();
    stop();
  } else if (motion.has(key)) {
    event.preventDefault();
    if (!event.repeat) {
      keys.add(key);
      updateKeys();
    }
  }
});
window.addEventListener("keyup", event => {
  if (orbit.delete(event.code)) publishOrbit(true);
  if (keys.delete(aliases[event.code] ?? event.code)) updateKeys();
});
window.addEventListener("blur", releaseControls);
window.addEventListener("pagehide", releaseControls);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseControls(); });
if (orbitEnabled) document.addEventListener("focusin", event => { if (formControl(event.target)) stopOrbit(); });
el("stop").addEventListener("click", stop);
el("speed").addEventListener("input", () => {
  el("speed-value").textContent = `${Number(el("speed").value).toFixed(2)} m/s`;
  if (keys.size) publish();
});

for (const button of buttons) {
  button.addEventListener("pointerdown", event => {
    button.setPointerCapture(event.pointerId);
    keys.add(button.dataset.key);
    updateKeys();
  });
  for (const name of ["lostpointercapture", "pointerup", "pointercancel"]) {
    button.addEventListener(name, () => {
      if (keys.delete(button.dataset.key)) updateKeys();
    });
  }
}

async function refreshCamera(state) {
  if (!state.camera || state.frame === previousFrame || cameraRequest) return;
  cameraRequest = true;
  try {
    const response = await fetch(`/camera.jpg?frame=${state.frame}`, {signal: AbortSignal.timeout(2000)});
    if (!response.ok) return;
    const url = URL.createObjectURL(await response.blob());
    const previous = el("camera").src;
    el("camera").src = url;
    if (previous.startsWith("blob:")) URL.revokeObjectURL(previous);
    previousFrame = state.frame;
    lastCameraLoaded = performance.now();
    el("camera").hidden = false;
  } catch {
    // A stale camera is reported by the status poll.
  } finally {
    cameraRequest = false;
  }
}

async function tick() {
  try {
    const response = await fetch("/state", {signal: AbortSignal.timeout(2000)});
    if (!response.ok) throw new Error("state unavailable");
    const state = await response.json();
    el("connection").classList.toggle("online", state.ready);
    el("connection").textContent = state.ready ? "控制器在线" : "等待控制器";
    if (state.pose) {
      el("sim-time").textContent = state.pose.time.toFixed(1);
      el("velocity").textContent = state.pose.speed.toFixed(2);
      el("yaw-rate").textContent = state.pose.yaw_rate.toFixed(2);
      el("position").textContent = `${state.pose.x.toFixed(2)} / ${state.pose.y.toFixed(2)}`;
      el("heading").textContent = (state.pose.heading * 180 / Math.PI).toFixed(1);
    }
    void refreshCamera(state);
    el("camera-status").hidden = state.camera && lastCameraLoaded > 0
      && performance.now() - lastCameraLoaded < 2100;
    el("camera-status").textContent = "等待仿真相机";
  } catch {
    el("connection").classList.remove("online");
    el("connection").textContent = "仿真已断开";
    el("camera-status").hidden = false;
    el("camera-status").textContent = "相机信号中断";
    if (keys.size) stop();
    stopOrbit();
  }
  setTimeout(tick, 65);
}

setInterval(() => {
  if (keys.size) publish();
  if (orbit.size) publishOrbit();
}, 85);
tick();
