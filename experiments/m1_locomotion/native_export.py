"""Export the learned full-action correction as part of one native M1 ONNX graph."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from native_world import (FRONT_ACTIONS, FRONT_DEFICIT, FRONT_FORWARD, FRONT_HIDDEN, FRONT_LIMIT,
                          FRONT_PITCH, FRONT_STALL, NATIVE_FILES, NATIVE_RUNTIME, NATIVE_SHA256,
                          observation_reflection)


def export(base, correction, output):
    if sha256(base.read_bytes()).hexdigest() != NATIVE_SHA256["stairs"]:
        raise ValueError("Native export expects the archived M1 stairs actor")
    model = onnx.load(base)
    graph = model.graph
    source, = [node for node in graph.node if "actions" in node.output]
    source.output[list(source.output).index("actions")] = "stairs_initial_actions"

    def value(name, array):
        graph.initializer.append(numpy_helper.from_array(np.asarray(array), "m1_native_"+name))
        return "m1_native_"+name

    def op(kind, inputs, output, **attrs):
        out = "m1_native_"+output
        graph.node.append(helper.make_node(kind, inputs, [out], name=out, **attrs))
        return out

    expected = {"w1": (144,294), "b1": (144,), "w2": (16,144), "b2": (16,),
                "mean": (294,), "inverse_std": (294,), "limit": (16,)}
    with np.load(correction, allow_pickle=False) as checkpoint:
        projected = "forward_correction" in checkpoint.files
        front_lift = any(k.startswith("front_") for k in checkpoint.files)
        if projected:
            expected["forward_correction"] = ()
        if front_lift:
            if not projected:
                raise ValueError("The front-lift actor requires the projected base")
            expected.update(front_w1=(FRONT_HIDDEN,294), front_b1=(FRONT_HIDDEN,),
                            front_w2=(3,FRONT_HIDDEN), front_b2=(3,))
        for k, shape in expected.items():
            if checkpoint[k].shape != shape or not np.isfinite(checkpoint[k]).all():
                raise ValueError(f"Invalid {k} in {correction}")
        weights = {k: value(k, checkpoint[k].astype(np.float32)) for k in expected}

    def constant(name, x):
        return value(name,np.array(x, dtype=np.float32))

    combined = op("Concat", ["obs", "obs_history"], "raw_observation", axis=1)
    centered = op("Sub",[combined, weights["mean"]],"centered")
    scaled = op("Mul",[centered,weights["inverse_std"]],"normalized")
    hidden = op("Gemm",[scaled,weights["w1"],weights["b1"]],"hidden",transB=1)
    features = op("Elu",[hidden],"features")
    logits = op("Gemm",[features,weights["w2"],weights["b2"]],"logits",transB=1)
    bounded = op("Tanh",[logits],"bounded")
    limited = op("Mul",[bounded,weights["limit"]],"limited")

    cmd_x = op("Gather",["obs",value("cmd_x_index",np.array([6],np.int64))],"cmd_x",axis=1)
    cmd_turn = op("Gather",["obs",value("cmd_turn_index",np.array([7,8],np.int64))],"cmd_turn",axis=1)
    low, high = constant("zero",0.), constant("one",1.)
    turn_abs = op("Abs",[cmd_turn],"turn_absolute")
    turn_scaled = op("Mul",[turn_abs,constant("turn_scales",[5.,14.])],"turn_scaled")
    turn_peak = op("ReduceMax",[turn_scaled],"turn_peak",axes=[1],keepdims=1)
    turn = op("Clip",[turn_peak,low,high],"turn_weight")
    drive_input = cmd_x if projected else op("Abs",[cmd_x],"drive_absolute")
    excess = op("Sub",[drive_input,constant("speed_cutoff",.90 if projected else .88)],"drive_excess")
    speed = op("Div",[excess,constant("speed_range",.30 if projected else .28)],"drive_scale")
    fast = op("Clip",[speed,low,high],"high_speed_weight")
    if projected:
        turn_delta = op("Mul",[limited,turn],"steering_correction")
        straight = op("Sub",[high,turn],"straight_factor")
        drive_gate = op("Mul",[straight,fast],"drive_factor")
        drive = op("Mul",[drive_gate,weights["forward_correction"]],"shared_wheel_correction")
        wheels = op("Mul",[drive,constant("wheel_mask",[0.,0.,0.,1.]*4)],"wheel_correction")
        correction = op("Add",[turn_delta,wheels],"correction")
    else:
        activity = op("Max",[fast,turn],"activity")
        additional = op("Mul",[activity,constant("active_fraction",.90)],"additional_authority")
        authority = op("Add",[additional,constant("base_fraction",.10)],"authority")
        correction = op("Mul",[limited,authority],"correction")
    if front_lift:
        def gather(source, name, indices):
            return op("Gather", [source, value(name+"_index", np.asarray(indices, dtype=np.int64))], name, axis=1)

        def ramp(source, name, bounds):
            lo, hi = bounds
            offset = op("Sub",[source,constant(name+"_min",lo)],name+"_offset")
            fraction = op("Div",[offset,constant(name+"_span",hi-lo)],name+"_fraction")
            return op("Clip",[fraction,low,high],name)

        mirror, signs = observation_reflection()
        reflected = gather(combined, "front_reflected", np.concatenate([mirror + 49*i for i in range(6)]))
        reflected = op("Mul",[reflected,constant("front_reflection_sign",np.tile(signs,6))],"front_mirrored")
        pair = op("Concat",[combined,reflected],"front_pair",axis=0)
        pair = op("Sub",[pair,weights["mean"]],"front_centered")
        pair = op("Mul",[pair,weights["inverse_std"]],"front_normalized")
        pair = op("Gemm",[pair,weights["front_w1"],weights["front_b1"]],"front_hidden",transB=1)
        pair = op("Elu",[pair],"front_features")
        pair = op("Gemm",[pair,weights["front_w2"],weights["front_b2"]],"front_logits",transB=1)
        pair = op("Tanh",[pair],"front_bounded")
        pair = op("Mul",[pair,constant("front_limit",FRONT_LIMIT)],"front_limited")
        pair = op("Reshape",[pair,value("front_pair_shape",np.array([1,6],np.int64))],"front_paired_actions")

        prior = gather("obs", "front_previous_wheel_action", [28,32,36,40])
        prior = op("Mul",[prior,constant("front_action_to_observed_speed",.5)],"front_expected_speed")
        speed = gather("obs", "front_measured_speed", [21,22,23,24])
        deficit = op("Sub",[prior,speed],"front_speed_deficits")
        front = op("ReduceMax",[gather(deficit, "front_wheel_deficits", [0,2])],"front_deficit",axes=[1],keepdims=1)
        rear = op("ReduceMean",[gather(deficit, "rear_wheel_deficits", [1,3])],"front_rear_deficit",axes=[1],keepdims=1)
        difference = op("Sub",[front,rear],"front_deficit_difference")
        stall = op("Mul",[ramp(difference,"front_stall",FRONT_STALL),
                          ramp(front,"front_absolute_deficit",FRONT_DEFICIT)],"front_stall_cue")
        pitch = op("Neg",[gather("obs","front_gravity_x",[3])],"front_nose_up")
        cue = op("Max",[ramp(pitch,"front_pitch",FRONT_PITCH),stall],"front_riser_cue")
        authority = op("Mul",[cue,ramp(cmd_x,"front_forward",FRONT_FORWARD)],"front_moving_uphill")
        authority = op("Mul",[authority,straight],"front_authority")
        embed = np.zeros((len(FRONT_ACTIONS),16),np.float32)
        embed[np.arange(len(FRONT_ACTIONS)),list(FRONT_ACTIONS)] = 1.
        mapped = op("MatMul",[pair,value("front_actuators",embed)],"front_actuator_delta")
        front_delta = op("Mul",[mapped,authority],"front_correction")
        correction = op("Add",[correction,front_delta],"trained_correction")
    graph.node.append(helper.make_node("Add", ["stairs_initial_actions", correction], ["actions"], name="m1_universal_actions"))
    metadata = model.metadata_props.add()
    metadata.key, metadata.value = "m1_native_teacher_sha256", ",".join(NATIVE_SHA256.values())
    onnx.checker.check_model(model)
    onnx.save(model,output)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("correction",type=Path)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    export(NATIVE_RUNTIME / "teachers" / NATIVE_FILES["stairs"],args.correction,args.output)
