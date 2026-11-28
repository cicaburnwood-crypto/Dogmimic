"""Carry the archived Classic ONNX through training and fuse its learned wheel correction."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def weights(source, output):
    base = onnx.load(source)
    np.savez(output, **{t.name: numpy_helper.to_array(t) for t in base.graph.initializer})


def export(source, correction, output):
    model = onnx.load(source)
    graph = model.graph
    base_output = graph.output[0].name
    producer, = [node for node in graph.node if base_output in node.output]
    producer.output[list(producer.output).index(base_output)] = "classic_actions"
    values = dict(np.load(correction, allow_pickle=False))
    for name, value in {
        "cmd_x": [6], "cmd_turn": [7, 8], "wheel_velocity": [21, 22, 23, 24],
        "wheel_history": [28, 32, 36, 40], "phase": list(range(41, 49)),
    }.items():
        values[name] = np.asarray(value, dtype=np.int64)
    values["wheel_mask"] = np.tile([0, 0, 0, 1], 4).astype(np.float32)[None]
    values["gate_epsilon"] = np.array(1.e-5, dtype=np.float32)
    for name, value in values.items():
        graph.initializer.append(numpy_helper.from_array(value, "m1_" + name))

    def node(op, inputs, output, **attrs):
        graph.node.append(helper.make_node(op, inputs, [output], name="m1_" + output, **attrs))

    for name in ("cmd_x", "cmd_turn", "wheel_velocity", "wheel_history", "phase"):
        node("Gather", ["obs", "m1_" + name], "obs_" + name, axis=1)
    for name in ("wheel_velocity", "wheel_history"):
        node("ReduceMean", ["obs_" + name], "mean_" + name, axes=[1], keepdims=1)
    for name in ("cmd_turn", "phase"):
        node("Abs", ["obs_" + name], "abs_" + name)
        node("ReduceMax", ["abs_" + name], "max_" + name, axes=[1], keepdims=1)
    node("Max", ["max_cmd_turn", "max_phase"], "max_gait")
    node("Less", ["max_gait", "m1_gate_epsilon"], "allow_correction")
    node("Cast", ["allow_correction"], "wheel_gate", to=onnx.TensorProto.FLOAT)
    node("Concat", ["obs_cmd_x", "mean_wheel_velocity", "mean_wheel_history"], "drive_input", axis=1)
    node("Gemm", ["drive_input", "m1_w1", "m1_b1"], "drive_hidden", transB=1)
    node("Tanh", ["drive_hidden"], "drive_activation")
    node("Gemm", ["drive_activation", "m1_w2", "m1_b2"], "drive_logit", transB=1)
    node("Tanh", ["drive_logit"], "bounded_drive_adjustment")
    node("Mul", ["bounded_drive_adjustment", "m1_gain"], "scaled_drive_adjustment")
    node("Mul", ["scaled_drive_adjustment", "wheel_gate"], "gated_drive_adjustment")
    node("Mul", ["gated_drive_adjustment", "m1_wheel_mask"], "wheel_adjustment")
    node("Add", ["classic_actions", "wheel_adjustment"], base_output)
    onnx.checker.check_model(model)
    onnx.save(model, output)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--correction", type=Path)
    args = p.parse_args()
    if args.correction:
        export(args.source, args.correction, args.output)
    else:
        weights(args.source, args.output)


if __name__ == "__main__":
    main()
