"""Maintain a load-bearing pose when the M1 stops on level support."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sim import DT, LEGS, M1_HOME, M1_KP, M1_SCALE, M1_STAND


def export(base, output):
    if base.resolve() == output.resolve():
        raise ValueError("Use a new path for the standing-stabilized model")
    payload = base.read_bytes()
    model = onnx.load_model_from_string(payload)
    graph = model.graph
    available = {name for node in graph.node for name in node.output}
    standing = "m1_flat_standing_weight"
    if standing not in available or "m1_stand_upstream" in available:
        raise ValueError("Expected an M1 terrain release with level parking support")
    producer, = (node for node in graph.node if "actions" in node.output)
    upstream = "m1_stand_upstream"
    producer.output[list(producer.output).index("actions")] = upstream

    def const(name, data, dtype=np.float32):
        name = "m1_stand_" + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(data, dtype=dtype), name))
        return name

    def op(kind, name, *inputs, **attrs):
        name = "m1_stand_" + name
        graph.node.append(helper.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def gather(name, source, columns):
        return op("Gather", name, source, const(name + "_indices", columns, np.int64), axis=1)

    stiffness, damping = 110., 1.6
    hip, knee = .91, 1.48
    pose = np.tile([0., hip, -knee, 0., 0., -hip, knee, 0.], 2)
    reference = const("joint_reference", (pose - M1_HOME)[LEGS])
    joint_columns = list(range(9, 21))
    measured = gather("measured", "obs", joint_columns)
    error = op("Sub", "pose_error", reference, measured)
    # Convert position and velocity feedback to an offset in the existing PD target.
    spring = op("Mul", "load_support", error, const("extra_stiffness", stiffness / M1_KP[LEGS] - 1.))

    previous_obs = 49 * 4
    previous = gather("previous_pose", "obs_history", [previous_obs + i for i in joint_columns])
    movement = op("Sub", "joint_displacement", measured, previous)
    gravity = gather("previous_gravity_z", "obs_history", [previous_obs + 5])
    vertical = op("Neg", "previous_vertical", gravity)
    # The first observation after reset has no previous joint sample.
    valid = op("Greater", "sample_valid", vertical, const("gravity_threshold", .65))
    movement = op("Clip", "limited_displacement", movement, const("displacement_min", -.07),
                  const("displacement_max", .07))
    damper = op("Mul", "joint_damping", movement, const("extra_damping", -damping / (DT * M1_KP[LEGS])))
    damper = op("Where", "measured_damping", valid, damper, const("zero", 0.))

    # The existing standing blend already releases the correction when moving
    # or when the wheels no longer share level support.
    offset = const("reference_change", (pose - M1_STAND)[LEGS])
    delta = op("Sum", "joint_correction", spring, damper, offset)
    delta = op("Clip", "bounded_correction", delta, const("correction_min", -.65), const("correction_max", .92))
    embed = np.eye(16, dtype=np.float32)[LEGS] / M1_SCALE[LEGS, None]
    delta = op("MatMul", "actuator_correction", delta, const("leg_indices", embed))
    correction = op("Mul", "standing_correction", standing, delta)
    graph.node.append(helper.make_node("Add", [upstream, correction], ["actions"], name="m1_stand_final_action"))
    note = model.metadata_props.add()
    note.key, note.value = "m1_stand_base_sha256", sha256(payload).hexdigest()
    onnx.checker.check_model(model)
    output.parent.mkdir(exist_ok=True, parents=True)
    onnx.save(model, output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.base, args.output)
