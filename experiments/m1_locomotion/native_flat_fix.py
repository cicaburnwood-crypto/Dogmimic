"""Calibrate the archived flat manoeuvres and brake level wheels in the universal actor."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.compose import add_prefix

from native_world import NATIVE_FILES, NATIVE_RUNTIME, NATIVE_SHA256
from sim import M1_HOME, M1_SCALE, M1_STAND, WHEELS


def export(base, classic, output):
    if base.resolve() == output.resolve():
        raise ValueError("Use a new path for the calibrated ONNX")
    if sha256(classic.read_bytes()).hexdigest() != NATIVE_SHA256["classic"]:
        raise ValueError("Expected the archived M1 flat policy")
    source = base.read_bytes()
    model = onnx.load_model_from_string(source)
    ground = "m1_motion_relative_elevation"
    outputs = {out for node in model.graph.node for out in node.output}
    if ground not in outputs or "m1_flat_upstream_actions" in outputs:
        raise ValueError("Expected a universal ONNX with rolling stance control")
    graph = model.graph
    producer, = [node for node in graph.node if "actions" in node.output]
    upstream = "m1_flat_upstream_actions"
    producer.output[list(producer.output).index("actions")] = upstream

    def const(name, value, dtype=np.float32):
        name = "m1_flat_" + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, name, *inputs, **attrs):
        name = "m1_flat_" + name
        graph.node.append(helper.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def gather(name, source, columns):
        return op("Gather", name, source, const(name+"_index", columns, np.int64), axis=1)

    def peak(name, values):
        return op("ReduceMax", name, op("Abs", name+"_abs", values), axes=[1], keepdims=1)

    zero, one = const("zero", 0.), const("one", 1.)

    def ramp(name, x, lo, hi):
        offset = op("Sub", name+"_offset", x, const(name+"_lo", lo))
        f = op("Div", name+"_fraction", offset, const(name+"_span", hi-lo))
        return op("Clip", name, f, zero, one)

    def inverse(name, x):
        return op("Sub", name, one, x)

    commands = gather("commands", "obs", [6, 7, 8])
    command = op("Mul", "commands_si", commands, const("command_units", [.5, .5, 4.]))
    motion = peak("command_peak", command)
    side = op("Abs", "side_speed", gather("side", command, [1]))
    yaw = op("Abs", "yaw_speed", gather("yaw", command, [2]))
    manoeuvring = op("Max", "manoeuvring", ramp("side_gate", side, .02, .08), ramp("yaw_gate", yaw, .04, .18))

    current_pitch = gather("pitch", "obs", [3])
    recent_pitch = gather("pitch_history", "obs_history", [49*i+3 for i in (2, 3, 4)])
    pitch = peak("pitch_peak", op("Concat", "pitch_samples", current_pitch, recent_pitch, axis=1))
    high_foot = op("ReduceMax", "highest_foot", ground, axes=[1], keepdims=1)
    low_foot = op("ReduceMin", "lowest_foot", ground, axes=[1], keepdims=1)
    height_spread = op("Sub", "foot_height_spread", high_foot, low_foot)
    level = op("Mul", "level_manoeuvre", inverse("pitch_level", ramp("sloped", pitch, .085, .17)),
               inverse("feet_level", ramp("raised", height_spread, .085, .15)))
    blend = op("Mul", "manoeuvre_weight", level, manoeuvring)

    actor = add_prefix(onnx.load(classic), "m1_flat_classic_")
    if actor.opset_import != model.opset_import:
        raise ValueError("Flat and stair ONNX operators must use the same opset")
    gain = np.ones(49, np.float32)
    gain[6:9] = [.88, .82, .90]
    side_bias = np.zeros(49, np.float32)
    side_bias[7] = .07

    def calibrated(name, source, count):
        scaled = op("Mul", name+"_scaled", source, const(name+"_gains", np.tile(gain, count)))
        bias = op("Mul", name+"_side_bias", op("Abs", name+"_absolute", source),
                  const(name+"_side_balance", np.tile(side_bias, count)))
        return op("Sub", name, scaled, bias)

    inputs = {
        "m1_flat_classic_obs": calibrated("classic_observation", "obs", 1),
        "m1_flat_classic_obs_history": calibrated("classic_history", "obs_history", 5),
    }
    for node in actor.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = inputs.get(name, name)
    graph.initializer.extend(actor.graph.initializer)
    graph.node.extend(actor.graph.node)
    difference = op("Sub", "classic_delta", actor.graph.output[0].name, upstream)
    actions = op("Add", "calibrated_actions", upstream, op("Mul", "applied_delta", difference, blend))

    # Use the wheel velocity servos to hold still once the feet are level.
    tilt = peak("body_tilt", gather("body_xy", "obs", [3, 4]))
    neutral = inverse("neutral", ramp("moving", motion, .012, .035))
    settled = op("Mul", "level_stance", inverse("stable_body", ramp("body_tip", tilt, .05, .11)),
                 inverse("supported_feet", ramp("support_gap", height_spread, .026, .060)))
    brake = op("Mul", "brake_weight", neutral, settled)
    wheel_mask = np.zeros(16, np.float32)
    wheel_mask[WHEELS] = 1.
    wheel_brake = op("Mul", "wheel_brake", brake, const("wheel_mask", wheel_mask))
    braked = op("Mul", "braked_actions", actions, inverse("remaining_action", wheel_brake))
    wheels = peak("measured_wheel_speed", gather("wheel_speed", "obs", [21, 22, 23, 24]))
    phase = peak("gait_activity", gather("gait_phase", "obs", list(range(41,49))))
    slow = inverse("slow_wheels", ramp("rolling", wheels, .025, .10))
    gait_activity = ramp("active_gait", phase, .006, .08)
    quiet = inverse("quiet_gait", op("Mul", "gait_effect", gait_activity, const("gait_hold", .45)))
    standing = op("Mul", "standing_weight", brake, op("Mul", "still_wheels_and_gait", slow, quiet))
    hold_action = (M1_STAND-M1_HOME) / M1_SCALE
    delta = op("Sub", "stand_delta", const("standing_action", hold_action), braked)
    hold = op("Mul", "standing_correction", delta, standing)
    graph.node.append(helper.make_node("Add", [braked, hold], ["actions"], name="m1_flat_final_action"))
    for key, value in (("m1_flat_base_sha256",sha256(source).hexdigest()),
                       ("m1_flat_teacher_sha256",NATIVE_SHA256["classic"])):
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = key, value
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    export(args.base, NATIVE_RUNTIME / "teachers" / NATIVE_FILES["classic"], args.output)
