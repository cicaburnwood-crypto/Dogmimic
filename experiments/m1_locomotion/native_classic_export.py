"""Merge the retrained flat actor into the board's two-input Uni Exp graph."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.compose import add_prefix

from native_world import (CLASSIC_RETRAINED_KEYS, CONTACT_ACTION_LIMIT, CONTACT_CLEARANCE,
                          CONTACT_DIRECTION, CONTACT_SIDE, CONTACT_TRAVEL, CONTACT_TRIM_KEYS,
                          CONTACT_YAW, HARDWARE_CLASSIC_SHA256, contact_action_maps)


def classic_actor(source, weights):
    payload = source.read_bytes()
    if sha256(payload).hexdigest() != HARDWARE_CLASSIC_SHA256:
        raise ValueError("Expected the original Classic weights read from the M1")
    model = onnx.load_model_from_string(payload)
    graph = model.graph
    # Use the latent mean, as in the differentiable training actor.
    mixer, = (node for node in graph.node if node.name == "/Add")
    for node in graph.node:
        for i, key in enumerate(node.input):
            if key == mixer.output[0]:
                node.input[i] = mixer.input[0]
    needed, kept = {"actions"}, []
    for node in reversed(graph.node):
        if needed.intersection(node.output):
            needed.update(node.input)
            kept.append(node)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    live_params = [param for param in graph.initializer if param.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(live_params)
    for param in graph.initializer:
        if param.name in weights:
            trained = weights[param.name]
            if tuple(param.dims) != trained.shape or not np.isfinite(trained).all():
                raise ValueError(f"Invalid parameter {param.name}")
            param.CopyFrom(numpy_helper.from_array(trained.astype(np.float32), param.name))
    return add_prefix(model, "m1_retrained_classic_")


def skip_inactive_actors(graph, shape_info):
    """Evaluate both actors only when the support gate blends their outputs."""
    gate = "m1_classic_posture_weight"
    old = "m1_classic_posture_original"
    flat = "m1_retrained_classic_actions"
    archived = "m1_flat_classic_actions"
    manoeuvre = "m1_flat_manoeuvre_weight"
    parameters = list(graph.initializer)

    def upstream(output):
        needed, kept = {output}, []
        for index in range(len(graph.node) - 1, -1, -1):
            node = graph.node[index]
            if needed.intersection(node.output):
                needed.update(node.input)
                kept.append(index)
        return kept[::-1]

    def references(nodes):
        return {key for node in nodes for key in node.input}

    indices = sorted(set(upstream(gate)) | set(upstream(manoeuvre)))
    parent = [graph.node[index] for index in indices]
    shared = set(indices)
    # Keep one latent draw per tick for each original actor, even when that
    # actor is inactive. The original actor graphs use fixed float32 shapes.
    for index, node in enumerate(graph.node):
        if node.op_type != "RandomNormalLike":
            continue
        spec = shape_info[node.input[0]].type.tensor_type
        shape = tuple(dim.dim_value for dim in spec.shape.dim)
        if spec.elem_type != TensorProto.FLOAT or not all(shape):
            raise ValueError("Expected a fixed-shape float32 policy latent")
        key = f"m1_conditional_random_shape_{index}"
        parameters.append(numpy_helper.from_array(np.zeros(shape, dtype=np.float32), key))
        sample = onnx.NodeProto()
        sample.CopyFrom(node)
        sample.input[0] = key
        parent.append(sample)
        shared.add(index)
    old_action, flat_action = "m1_conditional_stair_action", "m1_conditional_flat_action"
    old_needed, flat_needed = "m1_conditional_stair_needed", "m1_conditional_flat_needed"
    archived_needed = "m1_conditional_archived_needed"
    zero, one = "m1_classic_posture_zero", "m1_classic_posture_one"
    parent.extend([
        helper.make_node("Squeeze", [gate], ["m1_conditional_weight"], axes=[0, 1]),
        helper.make_node("Squeeze", [manoeuvre], ["m1_conditional_manoeuvre"], axes=[0, 1]),
        helper.make_node("Less", ["m1_conditional_weight", one], [old_needed]),
        helper.make_node("Greater", ["m1_conditional_weight", zero], [flat_needed]),
        helper.make_node("Greater", ["m1_conditional_manoeuvre", zero], ["m1_conditional_turning"]),
        helper.make_node("And", [old_needed, "m1_conditional_turning"], [archived_needed]),
    ])
    parent_parameters = references(parent) | {zero, one}
    for source, output, condition in ((archived, archived, archived_needed), (old, old_action, old_needed),
                                      (flat, flat_action, flat_needed)):
        steps = [index for index in upstream(source) if index not in shared]
        active = [graph.node[index] for index in steps]
        used = references(active) - parent_parameters
        result_name = output + "_result"
        active.append(helper.make_node("Identity", [source], [result_name]))
        result = helper.make_tensor_value_info(result_name, TensorProto.FLOAT, [1, 16])
        enabled = helper.make_graph(active, output + "_enabled", [], [result],
                                    initializer=[tensor for tensor in parameters if tensor.name in used])
        blank = output + "_zeros"
        zeros = numpy_helper.from_array(np.zeros((1, 16), dtype=np.float32))
        empty = helper.make_graph([helper.make_node("Constant", [], [blank], value=zeros)],
                                  output + "_disabled", [], [helper.make_tensor_value_info(blank, TensorProto.FLOAT, [1, 16])])
        parent.append(helper.make_node("If", [condition], [output], then_branch=enabled, else_branch=empty, name=output))
        shared.update(steps)

    parent.extend([
        helper.make_node("Sub", [flat_action, old_action], ["m1_conditional_action_delta"]),
        helper.make_node("Mul", [gate, "m1_conditional_action_delta"], ["m1_conditional_scaled_delta"]),
        helper.make_node("Add", [old_action, "m1_conditional_scaled_delta"], ["actions"]),
    ])
    used = references(parent)
    del graph.node[:], graph.initializer[:], graph.value_info[:]
    graph.node.extend(parent)
    graph.initializer.extend(tensor for tensor in parameters if tensor.name in used)


def export(base, classic, checkpoint, output, clearance=(.032, .069), tilt=(.065, .125),
           rolling=(.22, .67), wheel_spread=(.29, .66), conditional=False, contact_turn=False,
           gyro=(.10, .24), release_slew=None, release_contact_gain=0.):
    if output.resolve() in (base.resolve(), classic.resolve()):
        raise ValueError("Write the candidate to a separate file")
    if any(not 0 <= low < high < np.inf for low, high in (clearance, tilt, gyro)):
        raise ValueError("Expected increasing clearance, tilt and gyro limits")
    if release_slew is not None and (not contact_turn or not 0 < release_slew < np.inf):
        raise ValueError("The rolling-turn release limit must be positive")
    if not 0 <= release_contact_gain < np.inf or (release_contact_gain and release_slew is None):
        raise ValueError("The release contact correction requires a rolling-turn release limit")
    weights, refinement = {}, {}
    if checkpoint:
        with np.load(checkpoint, allow_pickle=False) as stored:
            expected = set(CLASSIC_RETRAINED_KEYS)
            if contact_turn and set(stored.files) == expected | set(CONTACT_TRIM_KEYS):
                for name, shape, limits in (("contact_gain", (4,), (4., 36.)), ("steering_gain", (2,), (-.27, .27))):
                    values = stored[name]
                    if (values.shape != shape or not np.isfinite(values).all()
                            or np.any(values < limits[0]) or np.any(values > limits[1])):
                        raise ValueError(f"Invalid parameter {name}")
                    refinement[name] = values.astype(np.float32)
            elif set(stored.files) != expected:
                raise ValueError("Expected the two retrained actor layers")
            weights = {key: stored[key] for key in CLASSIC_RETRAINED_KEYS}
    source = base.read_bytes()
    model = onnx.load_model_from_string(source)
    graph = model.graph
    if [i.name for i in graph.input] != ["obs", "obs_history"]:
        raise ValueError("Expected the M1 hardware observation interface")
    original, = (node for node in graph.node if "actions" in node.output)
    if original.name != "m1_stand_final_action":
        raise ValueError("Expected the Uni Exp model with the standing controller")
    old = "m1_classic_posture_original"
    original.output[list(original.output).index("actions")] = old

    def constant(name, value, dtype=np.float32):
        key = "m1_classic_posture_" + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), key))
        return key

    def op(kind, name, *inputs, **attrs):
        key = "m1_classic_posture_" + name
        graph.node.append(helper.make_node(kind, list(inputs), [key], name=key, **attrs))
        return key

    def gather(name, source, columns):
        return op("Gather", name, source, constant(name + "_indices", columns, np.int64), axis=1)

    zero, one = constant("zero", 0.), constant("one", 1.)

    def ramp(name, value, limits):
        lo, hi = limits
        shift = op("Sub", name + "_shift", value, constant(name + "_low", lo))
        fraction = op("Div", name + "_fraction", shift, constant(name + "_width", hi - lo))
        return op("Clip", name, fraction, zero, one)

    def inverse(name, value):
        return op("Sub", name, one, value)

    def peak(name, value):
        return op("ReduceMax", name, op("Abs", name + "_absolute", value), axes=[1], keepdims=1)

    # Retreat to the current stair controller on raised support or a wheel that
    # stalls or spins up relative to its peers. A level approach before contact
    # has the same observations as open floor on this hardware interface.
    command = op("Mul", "command_si", gather("commands", "obs", [6, 7, 8]), constant("units", [.5, .5, 4.]))
    moving = ramp("moving", peak("motion", command), (.032, .135))
    travel = gather("travel", command, [0])
    speed = op("Abs", "speed", travel)
    turning = peak("nonstraight", gather("nonstraight_command", command, [1, 2]))
    slow_travel = inverse("slow_travel", ramp("fast_travel", speed, (.44, .58)))
    straight = inverse("straight", ramp("turning", turning, (.025, .13)))
    risk_speed = op("Mul", "stair_speed", slow_travel, straight)

    direction = op("Sign", "direction", travel)
    # Average each wheel over 80 ms so a one-frame slip on level ground
    # does not repeatedly switch between actors at low speed.
    wheel_axes = list(range(21, 25))
    recent_wheels = [gather("current_wheels", "obs", wheel_axes)]
    recent_wheels.extend(gather(f"wheel_frame_{i}", "obs_history", [49 * i + j for j in wheel_axes])
                         for i in (2, 3, 4))
    measured = op("Mean", "wheel_samples", *recent_wheels)
    wheels = op("Mul", "wheel_motion", direction, measured)
    low = op("ReduceMin", "slowest_wheel", wheels, axes=[1], keepdims=1)
    high = op("ReduceMax", "fastest_wheel", wheels, axes=[1], keepdims=1)
    expected = op("Max", "expected_wheel", op("Mul", "wheel_at_command", speed, constant("wheel_units", .05/.095)),
                  constant("wheel_speed_floor", .036))
    pace = ramp("wheel_pace", op("Div", "rolling_ratio", low, expected), rolling)
    spread = op("Div", "wheel_disagreement", op("Sub", "wheel_difference", high, low), expected)
    even = inverse("wheel_agreement", ramp("varying_wheels", spread, wheel_spread))
    rolling_support = op("Mul", "rolling_support", pace, even)
    traction = inverse("traction", op("Mul", "stall_risk", risk_speed, inverse("wheel_risk", rolling_support)))
    coplanar = inverse("coplanar", ramp("wheel_elevation", "m1_motion_elevation_range", clearance))
    upright = inverse("upright", ramp("lean", "m1_motion_tilt", tilt))
    body = inverse("steady_body", ramp("gyro", "m1_motion_gyro", gyro))
    support = op("Mul", "level_support", coplanar, op("Mul", "stable_body", upright, body))
    eligible = op("Mul", "clearance_and_traction", support, traction)
    if contact_turn:
        lateral = op("Abs", "lateral_speed", gather("side_command", command, [1]))
        travel_active = ramp("arc_travel", speed, (.06, .14))
        lateral_quiet = inverse("arc_side", ramp("arc_side_ramp", lateral, (.015, .080)))
        rolling_arc = op("Mul", "arc_travel_and_side", travel_active, lateral_quiet)
        yaw = op("Abs", "yaw_speed", gather("yaw_command", command, [2]))
        historical_yaw = peak("recent_yaw", gather("yaw_samples", "obs_history", [49*i+8 for i in (2,3,4)]))
        historical_yaw = op("Mul", "recent_yaw_si", historical_yaw, constant("yaw_units", 4.))
        yaw_recent = op("Max", "yaw_request", yaw, historical_yaw)
        steering = ramp("arc_steering", yaw_recent, (.035, .110))
        # After release the controller retains its gait phase briefly. Keep the
        # rolling actor through this transition while longitudinal motion continues.
        gait = peak("residual_gait", gather("gait_phase", "obs", list(range(41,49))))
        recent_gait = ramp("recent_gait", gait, (.006, .068))
        turning_recently = op("Max", "turn_intent", steering, recent_gait)
        arc = op("Mul", "arc", rolling_arc, turning_recently)
        fast = op("Mul", "fast_level", lateral_quiet, ramp("fast_room", speed, (.49, .63)))
        # A slow straight command can approach the first riser. During pure
        # ascent/descent commands the established stair actor keeps authority.
        mode = op("Max", "rolling_authority", arc, fast)
        eligible = op("Mul", "level_rolling_authority", eligible, mode)
    blend = op("Mul", "weight", moving, eligible)

    actor = classic_actor(classic, weights)
    recent = gather("previous_four", "obs_history", list(range(49, 245)))
    history = op("Concat", "classic_history", recent, "obs", axis=1)
    observation = "obs"
    if contact_turn:
        # Match ContactActor.infer: keep the phase for side travel and spins,
        # and suppress it while longitudinal rolling uses the trained branch.
        bits = np.zeros(49, np.float32)
        bits[41:] = 1.
        phase_mask = op("Sub", "phase_mask", one, op("Mul", "phase_to_remove", rolling_arc, constant("phase_bits", bits)))
        history_mask = op("Sub", "history_phase_mask", one,
                          op("Mul", "history_phase_to_remove", rolling_arc, constant("history_phase_bits", np.tile(bits,5))))
        observation = op("Mul", "rolling_observation", observation, phase_mask)
        history = op("Mul", "rolling_history", history, history_mask)
    inputs = {"m1_retrained_classic_obs": observation, "m1_retrained_classic_obs_history": history}
    for node in actor.graph.node:
        for i, key in enumerate(node.input):
            node.input[i] = inputs.get(key, key)
    graph.initializer.extend(actor.graph.initializer)
    graph.node.extend(actor.graph.node)
    flat_action = actor.graph.output[0].name
    if refinement:
        legs, wheel_map = contact_action_maps()
        name = "m1_retrained_classic_unrefined_actions"
        action_node, = (node for node in graph.node if flat_action in node.output)
        action_node.output[list(action_node.output).index(flat_action)] = name
        height = "m1_motion_relative_elevation"
        floor = op("ReduceMin", "min_wheel_elevation", height, axes=[1], keepdims=1)
        above_floor = op("Sub", "wheel_clearance", height, floor)
        gap = op("Relu", "support_error", op("Sub", "unsupported_gap", above_floor, constant("trim_clearance", CONTACT_CLEARANCE)))
        extension = op("Mul", "wheel_extension", gap, constant("extension_gain", refinement["contact_gain"]))
        extension = op("Min", "extension_limit", extension, constant("max_extension", CONTACT_ACTION_LIMIT))
        support_action = op("MatMul", "leg_contact_trim", extension, constant("leg_contact_map", legs))
        direction_weight = ramp("steering_direction_weight", travel, CONTACT_DIRECTION)
        reverse, forward = refinement["steering_gain"]
        gain = op("Add", "steering_gain", constant("reverse_gain", reverse),
                  op("Mul", "directional_gain", direction_weight, constant("steering_gain_span", forward-reverse)))
        steer = op("Mul", "steer_adjustment", gain, gather("signed_yaw", command, [2]))
        wheel_action = op("Mul", "wheel_steering_trim", steer, constant("wheel_steering_map", wheel_map))
        trim_motion = op("Mul", "contact_travel_and_yaw", ramp("contact_travel", speed, CONTACT_TRAVEL),
                        ramp("contact_yaw", yaw, CONTACT_YAW))
        strength = op("Mul", "rolling_contact_authority", trim_motion, inverse("contact_lateral_quiet", ramp("contact_lateral", lateral, CONTACT_SIDE)))
        correction = op("Mul", "contact_correction", strength, op("Add", "contact_and_steer", support_action, wheel_action))
        graph.node.append(helper.make_node("Add", [name, correction], [flat_action], name="m1_classic_posture_refined_actions"))
    delta = op("Sub", "actor_difference", flat_action, old)
    trim = op("Mul", "weighted_delta", blend, delta)
    graph.node.append(helper.make_node("Add", [old, trim], ["actions"], name="m1_classic_posture_actions"))
    for key, value in {"base": sha256(source).hexdigest(), "teacher": HARDWARE_CLASSIC_SHA256,
                       "checkpoint": sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else "reference_only"}.items():
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = "m1_classic_posture_" + key, value
    if contact_turn:
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = "m1_classic_posture_rolling_turn", "learned_wheel_support"
    if refinement:
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = "m1_classic_posture_contact_refinement", "learned_bounded_gains"
    if conditional:
        inferred = onnx.shape_inference.infer_shapes(model).graph
        skip_inactive_actors(graph, {item.name: item for item in inferred.value_info})
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = "m1_classic_posture_execution", "skip_inactive_actors"
    if release_slew is not None:
        # The archived actor can issue a large leg target when it takes over
        # during release. Use the command history and remaining gait phase to
        # cap the leg target change as long as the wheels are near level.
        upstream = "m1_classic_posture_unlimited_actions"
        action_node, = (node for node in graph.node if "actions" in node.output)
        action_node.output[list(action_node.output).index("actions")] = upstream
        decay = ramp("steering_decay", op("Sub", "steering_drop", yaw_recent, yaw), (.0015, .022))
        low_yaw = inverse("low_yaw", ramp("held_steering", yaw, (.027, .087)))
        tail = op("Mul", "release_gait", low_yaw, recent_gait)
        release = op("Max", "release_detected", decay, tail)
        rolling_release = op("Mul", "rolling_release", rolling_arc, release)
        active_release = op("Mul", "level_rolling_release", rolling_release, coplanar)
        previous = gather("previous_action", "obs", list(range(25, 41)))
        target = upstream
        if release_contact_gain:
            height = "m1_motion_relative_elevation"
            floor = op("ReduceMin", "release_floor", height, axes=[1], keepdims=1)
            gap = op("Sub", "release_wheel_gap", height, floor)
            error = op("Relu", "release_wheel_error", op("Sub", "release_clearance_error", gap,
                       constant("release_clearance", CONTACT_CLEARANCE)))
            extension = op("Min", "release_extension_limit", constant("release_extension_max", .32),
                           op("Mul", "release_extension", error, constant("release_contact_gain", release_contact_gain)))
            support_action = op("MatMul", "release_support_action", extension, constant("release_contact_map", contact_action_maps()[0]))
            remaining = op("Sub", "release_elevation_remaining", constant("release_contact_cutoff", .105), "m1_motion_elevation_range")
            scale = op("Div", "release_support_fraction", remaining, constant("release_contact_band", .105-.065))
            support = op("Clip", "release_near_floor", scale, zero, one)
            projected = op("Mul", "release_contact_delta", op("Mul", "release_contact_authority", rolling_release, support), support_action)
            target = op("Add", "release_contact_target", upstream, projected)
        diff = op("Sub", "release_difference", previous, target)
        clipped = op("Clip", "allowed_release_difference", diff,
                     constant("release_lower", -release_slew), constant("release_upper", release_slew))
        overflow = op("Sub", "excess_release_change", diff, clipped)
        legs = op("Mul", "release_leg_change", overflow, constant("release_legs_only", np.tile([1.,1.,1.,0.], 4)))
        correction = op("Mul", "release_trim", active_release, legs)
        graph.node.append(helper.make_node("Add", [target, correction], ["actions"], name="m1_classic_posture_release_actions"))
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = "m1_classic_posture_release_leg_slew", str(release_slew)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for name in ("base", "classic", "checkpoint", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--conditional", action="store_true", help="Skip inactive actors on the M1 board")
    p.add_argument("--contact-turn", action="store_true", help="Deploy the rolling-turn checkpoint with its training phase mask")
    p.add_argument("--clearance", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(.032, .069))
    p.add_argument("--tilt", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(.065, .125))
    p.add_argument("--gyro", type=float, nargs=2, metavar=("LOW", "HIGH"), default=(.10, .24))
    p.add_argument("--release-slew", type=float, metavar="AMOUNT")
    p.add_argument("--release-contact-gain", type=float, default=0., metavar="GAIN")
    options = p.parse_args()
    export(options.base, options.classic, options.checkpoint, options.output,
           clearance=options.clearance, tilt=options.tilt, conditional=options.conditional, contact_turn=options.contact_turn,
           gyro=options.gyro, release_slew=options.release_slew, release_contact_gain=options.release_contact_gain)
