"""Append learned level stabilisation and descent to the current one-file M1 actor."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from native_terrain import (DESCENT_GAIN_SPAN, DESCENT_LEG_CAP, FLAT_CAP, GUIDE_PARAMETERS,
                            PROFILE_HIDDEN, TERRAIN_NAME, TERRAIN_SIZE, WHEEL_SIGNS)
from sim import LEGS, WHEELS


def export(base, checkpoint, output):
    if base.resolve() == output.resolve():
        raise ValueError("Use a new path for the terrain-trained model")
    source = base.read_bytes()
    model = onnx.load_model_from_string(source)
    graph = model.graph
    if not any(n.name == "m1_flat_final_action" for n in graph.node) or any(i.name == TERRAIN_NAME for i in graph.input):
        raise ValueError("Expected the verified flat/ascent baseline graph")
    if not any("m1_native_front_correction" in n.output for n in graph.node):
        raise ValueError("The base actor must contain the trained forelegs")
    producer, = [node for node in graph.node if "actions" in node.output]
    prior = "m1_stability_upstream_actions"
    producer.output[list(producer.output).index("actions")] = prior
    graph.input.append(helper.make_tensor_value_info(TERRAIN_NAME, TensorProto.FLOAT, [1, TERRAIN_SIZE]))
    # Old callers can omit terrain and receive the frozen baseline's actions.
    graph.initializer.append(numpy_helper.from_array(np.zeros((1,TERRAIN_SIZE),np.float32), TERRAIN_NAME))

    def constant(name, value, dtype=np.float32):
        name = "m1_stability_" + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value,dtype=dtype), name))
        return name

    def op(kind, name, *values, **attrs):
        name = "m1_stability_" + name
        graph.node.append(helper.make_node(kind, list(values), [name], name=name, **attrs))
        return name

    def gather(name, source, columns):
        return op("Gather", name, source, constant(name+"_columns", columns, np.int64), axis=1)

    def peak(name, source):
        return op("ReduceMax", name, op("Abs", name+"_abs", source), axes=[1], keepdims=1)

    zero, one = constant("zero",0.), constant("one",1.)

    def ramp(name, x, a, b):
        offset = op("Sub",name+"_offset",x,constant(name+"_min",a))
        fraction = op("Div",name+"_fraction",offset,constant(name+"_width",b-a))
        return op("Clip",name,fraction,zero,one)

    def inverse(name, x):
        return op("Sub",name,one,x)

    width = 294+TERRAIN_SIZE
    shapes = {"mean": (width,), "inverse_std": (width,)}
    for name, size in (("flat",16),("descent",15)):
        shapes.update({f"{name}_w1": (PROFILE_HIDDEN,width), f"{name}_b1": (PROFILE_HIDDEN,),
                       f"{name}_w2": (size,PROFILE_HIDDEN), f"{name}_b2": (size,)})
    with np.load(checkpoint,allow_pickle=False) as arrays:
        weights = {}
        for name, shape in shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"Invalid trained tensor {name}")
            weights[name] = constant(name,arrays[name])
    raw = op("Concat","inputs","obs","obs_history",TERRAIN_NAME,axis=1)
    norm = op("Mul","normalized",op("Sub","centered",raw,weights["mean"]),weights["inverse_std"])
    features = op("Clip","bounded_inputs",norm,constant("input_min",-8.),constant("input_max",8.))

    def network(name):
        hidden = op("Gemm",name+"_hidden",features,weights[name+"_w1"],weights[name+"_b1"],transB=1)
        hidden = op("Elu",name+"_elu",hidden)
        logits = op("Gemm",name+"_output",hidden,weights[name+"_w2"],weights[name+"_b2"],transB=1)
        return op("Tanh",name+"_bounded",logits)

    flat_delta = op("Mul","flat_delta",network("flat"),constant("flat_limit",FLAT_CAP))
    down = network("descent")
    modifiers = gather("gain_modifiers",down,[0,1,2])
    gain_base = [GUIDE_PARAMETERS[k] for k in ("heading_kp","yaw_kd","wheel_speed_kp")]
    gains = op("Add","gains",constant("reference_gains",gain_base),
               op("Mul","gain_delta",modifiers,constant("gain_span",DESCENT_GAIN_SPAN)))
    kp, kd, speed_kp = [gather(key,gains,[i]) for i,key in enumerate(("heading_kp","yaw_kd","speed_kp"))]
    heading_error = gather("heading_error",TERRAIN_NAME,[8])
    yaw_rate = gather("yaw_rate",TERRAIN_NAME,[7])
    yaw_effort = op("Sub","yaw_effort",op("Mul","heading_feedback",kp,heading_error),
                    op("Mul","yaw_damping",kd,yaw_rate))
    turn = op("Clip","turn_effort",yaw_effort,constant("turn_min",-.8),constant("turn_max",.8))
    x = gather("x_command","obs",[6])
    vx = op("Mul","command_velocity",x,constant("observation_speed_units",.5))
    speed_error = op("Sub","speed_error",vx,gather("measured_velocity",TERRAIN_NAME,[5]))
    drive = op("Mul","wheel_drive",speed_kp,speed_error)
    turn_per_wheel = op("Mul","wheel_turn",turn,constant("wheel_sides",.275*np.asarray(WHEEL_SIGNS)))
    wheel = op("Div","wheel_velocity",op("Add","wheel_feedback",drive,turn_per_wheel),constant("wheel_units",.95))
    embed = np.eye(16,dtype=np.float32)
    wheel = op("MatMul","wheel_delta",wheel,constant("wheel_embed",embed[WHEELS]))
    legs = gather("descent_leg_logits",down,list(range(3,15)))
    legs = op("Mul","descent_leg_bounded",legs,constant("descent_leg_limit",DESCENT_LEG_CAP))
    legs = op("MatMul","leg_delta",legs,constant("leg_embed",embed[LEGS]))
    descent_delta = op("Add","descent_delta",legs,wheel)

    turn_input = op("Mul","turn_input",gather("turn_axes","obs",[7,8]),constant("turn_units",[1.,4.]))
    upright = peak("tilt",gather("gravity_xy","obs",[3,4]))
    valid = gather("terrain_present",TERRAIN_NAME,[0])
    straight = op("Mul","straight",valid,inverse("no_turn",ramp("turn_gate",peak("turn_peak",turn_input),.015,.065)))
    level_span = inverse("level_span",ramp("height_change",gather("height_span",TERRAIN_NAME,[1]),.012,.055))
    level_posture = inverse("level_posture",ramp("lean",upright,.05,.13))
    level = op("Mul","level",level_span,level_posture)
    moving = ramp("moving",op("Abs","abs_speed",x),.06,.24)
    flat_weight = op("Mul","flat_weight",straight,op("Mul","moving_on_level",moving,level))
    drop = op("Neg","downhill_drop",gather("slope_change",TERRAIN_NAME,[2]))
    descending = op("Mul","downhill",ramp("grade",drop,.036,.110),ramp("forward",x,.06,.23))
    down_weight = op("Mul","descent_weight",straight,descending)
    flat = op("Mul","active_flat",flat_delta,flat_weight)
    down = op("Mul","active_descent",descent_delta,down_weight)
    action = op("Add","level_stabilized",prior,flat)
    graph.node.append(helper.make_node("Add",[action,down],["actions"],name="m1_stability_final_action"))
    for key,value in (("m1_stability_base_sha256",sha256(source).hexdigest()),
                      ("m1_stability_trained_sha256",sha256(checkpoint.read_bytes()).hexdigest())):
        metadata = model.metadata_props.add()
        metadata.key,metadata.value = key,value
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    export(args.base,args.checkpoint,args.output)
