"""Damp the M1's leg and body feedback during straight travel on sensed level ground."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from native_terrain import TERRAIN_NAME
from sim import LEGS


def export(base, output):
    if base.resolve() == output.resolve():
        raise ValueError("Use a new path for the cruise-stabilized model")
    payload = base.read_bytes()
    model = onnx.load_model_from_string(payload)
    graph = model.graph
    action, = (node for node in graph.node if "actions" in node.output)
    if action.name != "m1_stability_final_action":
        raise ValueError("Expected the previously trained terrain policy")
    upstream = "m1_cruise_upstream"
    action.output[list(action.output).index("actions")] = upstream

    def constant(name, data, dtype=np.float32):
        name = "m1_cruise_" + name
        graph.initializer.append(numpy_helper.from_array(np.asarray(data, dtype=dtype), name))
        return name

    def op(kind, name, *inputs, **attrs):
        name = "m1_cruise_" + name
        graph.node.append(helper.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def gather(name, src, columns):
        return op("Gather", name, src, constant(name+"_columns", columns, np.int64), axis=1)

    zero, one = constant("zero",0.), constant("one",1.)

    def ramp(name, src, lo, hi):
        a = op("Sub",name+"_offset",src,constant(name+"_min",lo))
        b = op("Div",name+"_fraction",a,constant(name+"_range",hi-lo))
        return op("Clip",name,b,zero,one)

    span = gather("ground_span",TERRAIN_NAME,[1])
    level = op("Sub","level",one,ramp("ground_edge",span,.010,.04))
    tilt = op("ReduceMax","tilt",op("Abs","absolute_tilt",gather("gravity_xy","obs",[3,4])),
              axes=[1],keepdims=1)
    upright = op("Sub","upright",one,ramp("tip",tilt,.14,.29))
    straight = op("Mul","straight_moving","m1_stability_straight","m1_stability_moving")
    active = op("Mul","authority",straight,op("Mul","supported",level,upright))

    leg_mask = np.zeros(16,np.float32)
    leg_mask[LEGS] = 1.
    mirrored = gather("opposite_leg_actions",upstream,[8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7])
    mirrored = op("Mul","reflected_leg_actions",mirrored,constant("mirror_directions",[-1.,1.,1.,1.]*4))
    asymmetry = op("Sub","asymmetry",mirrored,upstream)
    sym = op("Mul","symmetry_delta",asymmetry,constant("symmetry_gain",.5*.38*leg_mask))

    roll = op("Mul","roll_position_feedback",gather("gravity_y","obs",[4]),constant("roll_gain",.65))
    gyro = op("Mul","roll_rate_feedback",gather("gyro_x","obs",[0]),constant("roll_damping",4*.11))
    roll = op("Sub","roll_error",roll,gyro)
    vertical = op("Mul","vertical_velocity_feedback",gather("centre_vz",TERRAIN_NAME,[4]),
                  constant("vertical_damping",-.32))
    low, high = constant("correction_min",-.15),constant("correction_max",.15)
    roll = op("Clip","bounded_roll",roll,low,high)
    vertical = op("Clip","bounded_vertical",vertical,low,high)

    # Opposing hip and knee changes limit fore-aft wheel motion during extension.
    extend = np.array([0.,-.52,1.,0.,0.,.52,-1.,0.]*2,np.float32)
    side = extend * np.repeat([1.,1.,-1.,-1.],4)
    roll_delta = op("Mul","roll_leg_delta",roll,constant("roll_joint_signs",side))
    vertical_delta = op("Mul","vertical_leg_delta",vertical,constant("vertical_joint_signs",extend))
    leg_target = op("Sum","level_leg_target",upstream,sym,roll_delta,vertical_delta)
    previous = gather("previous_actions","obs",list(range(25,41)))
    to_previous = op("Sub","change_to_previous",previous,leg_target)
    slew = op("Mul","slew_delta",to_previous,constant("leg_smoothing",.14*leg_mask))
    delta = op("Sum","leg_delta",sym,roll_delta,vertical_delta,slew)

    weighted = op("Mul","active_delta",delta,active)
    graph.node.append(helper.make_node("Add",[upstream,weighted],["actions"],name="m1_cruise_final_action"))
    metadata = model.metadata_props.add()
    metadata.key, metadata.value = "m1_cruise_base_sha256", sha256(payload).hexdigest()
    onnx.checker.check_model(model)
    output.parent.mkdir(exist_ok=True,parents=True)
    onnx.save(model,output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    export(args.base,args.output)
