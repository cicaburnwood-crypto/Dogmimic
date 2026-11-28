"""Stabilize faster level travel using the M1's existing hardware observations."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sim import DT, LEGS, M1_HOME, M1_KP, M1_SCALE, WHEELS


def export(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate path for the floor-calibrated policy")
    payload = source.read_bytes()
    model = onnx.load_model_from_string(payload)
    graph = model.graph
    if [value.name for value in graph.input] != ["obs", "obs_history"]:
        raise ValueError("Expected the M1 hardware policy without a terrain input")
    producer, = (node for node in graph.node if "actions" in node.output)
    if producer.name != "m1_stand_final_action":
        raise ValueError("Expected Uni Exp with the standing stabilization")
    upstream = "m1_floor_base_actions"
    producer.output[list(producer.output).index("actions")] = upstream

    def const(name, value, dtype=np.float32):
        name = "m1_floor_" + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, name, *inputs, **attrs):
        name = "m1_floor_" + name
        graph.node.append(helper.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def take(name, source, columns):
        return op("Gather", name, source, const(name + "_columns", columns, np.int64), axis=1)

    zero, one = const("zero", 0.), const("one", 1.)

    def ramp(name, value, low, high):
        offset = op("Sub", name + "_offset", value, const(name + "_min", low))
        fraction = op("Div", name + "_fraction", offset, const(name + "_width", high - low))
        return op("Clip", name, fraction, zero, one)

    def inverse(name, value):
        return op("Sub", name, one, value)

    # Observed forward commands are twice the speed in m/s. Keep the slow
    # commands used on stairs exactly on the original hardware policy.
    forward = "m1_motion_x_command"
    speed = op("Abs", "command_speed", forward)
    reversing = op("Less", "reversing", forward, zero)
    faster = op("Where", "cruise_speed", reversing,
                ramp("reverse_speed", speed, 2 * .48, 2 * .59),
                ramp("forward_speed", speed, 2 * .48, 2 * .69))
    even_wheels = inverse("even_support", ramp("wheel_levels", "m1_motion_elevation_range", .020, .066))
    upright = inverse("near_upright", ramp("body_lean", "m1_motion_tilt", .07, .15))
    still_body = inverse("body_steady", ramp("body_motion", "m1_motion_gyro", .065, .165))
    level = op("Mul", "level", even_wheels, op("Mul", "stable_pose", upright, still_body))
    # A small steering input can already need leg motion to overcome lateral
    # wheel friction. Fade out before 0.02 rad/s of requested yaw.
    yaw = op("Abs", "yaw_command", take("yaw", "obs", [8]))
    no_yaw = inverse("no_yaw", ramp("steering", yaw, .25 * .005, .25 * .02))
    straight = op("Mul", "straight", "m1_motion_straight", no_yaw)
    straight_level = op("Mul", "straight_level", straight, level)
    active = op("Mul", "level_cruise", faster, straight_level)

    # Express additional position and damping feedback as a bounded target
    # offset for the same leg servos that the original policy drives.
    forward_pose = np.tile([0., .81, -1.29, 0., 0., -.81, 1.29, 0.], 2)
    reverse_pose = np.tile([0., .86, -1.40, 0., 0., -.86, 1.40, 0.], 2)
    reference = op("Where", "leg_reference", reversing,
                   const("reverse_pose", (reverse_pose - M1_HOME)[LEGS]),
                   const("forward_pose", (forward_pose - M1_HOME)[LEGS]))
    columns = list(range(9, 21))
    q = take("joint_angle", "obs", columns)
    previous = take("last_joint_angle", "obs_history", [49 * 4 + i for i in columns])
    error = op("Sub", "joint_error", reference, q)
    spring = op("Mul", "load", error, const("extra_position_gain", 135. / M1_KP[LEGS] - 1.))
    displacement = op("Sub", "displacement", q, previous)
    bounded = op("Clip", "bounded_displacement", displacement,
                 const("displacement_min", -.065), const("displacement_max", .065))
    damper = op("Mul", "damping", bounded, const("derivative_gain", -2.1 / (DT * M1_KP[LEGS])))
    previous_gravity = take("last_gravity", "obs_history", [49 * 4 + 5])
    previous_valid = op("Less", "previous_valid", previous_gravity, const("upright_z", -.65))
    damper = op("Where", "damping_when_valid", previous_valid, damper, zero)
    leg_target = op("Div", "leg_action_target", op("Sum", "load_bearing_target", reference, spring, damper),
                    const("leg_scale", M1_SCALE[LEGS]))
    leg_error = op("Sub", "leg_action_error", leg_target, take("base_legs", upstream, LEGS))
    leg_error = op("Clip", "bounded_leg_change", leg_error, const("leg_change_min", -1.4), const("leg_change_max", 1.4))
    embed = np.eye(16, dtype=np.float32)
    legs = op("MatMul", "leg_delta", leg_error, const("leg_map", embed[LEGS]))
    legs = op("Mul", "weighted_legs", legs, const("leg_authority", .93))

    # Match the common rolling speed during straight travel.
    base_wheels = take("base_wheels", upstream, WHEELS)
    wheel_goal = op("Mul", "forward_wheel_goal", forward, const("wheel_speed_scale", .995 / (2 * .095 * 10.)))
    wheel_error = op("Sub", "wheel_error", wheel_goal, base_wheels)
    wheels = op("MatMul", "wheel_delta", wheel_error, const("wheel_map", embed[WHEELS]))
    wheel_authority = op("Where", "wheel_authority", reversing, const("reverse_wheel_gain", .96),
                         const("forward_wheel_gain", .72))
    wheels = op("Mul", "weighted_wheels", wheels, wheel_authority)
    # Match the reverse command before the higher leg stance takes full effect.
    wheel_speed_weight = op("Where", "wheel_speed_weight", reversing,
                            ramp("reverse_wheel_speed", speed, 2 * .48, 2 * .51), faster)
    wheel_active = op("Mul", "wheel_active", wheel_speed_weight, straight_level)
    correction = op("Add", "adjustment", op("Mul", "leg_adjustment", active, legs),
                    op("Mul", "wheel_adjustment", wheel_active, wheels))
    graph.node.append(helper.make_node("Add", [upstream, correction], ["actions"], name="m1_floor_final_action"))
    note = model.metadata_props.add()
    note.key, note.value = "m1_floor_hardware_base_sha256", sha256(payload).hexdigest()
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
