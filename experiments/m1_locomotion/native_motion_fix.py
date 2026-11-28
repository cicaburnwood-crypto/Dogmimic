"""Add reverse-turn damping, rolling stance control and a stair assist to the exported actor."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sim import M1_HOME, M1_SCALE, WHEELS


def export(base, output, strength=1.1):
    if not .5 <= strength <= 1.8:
        raise ValueError("Rolling stance gain outside the tested calibration range")
    if base.resolve() == output.resolve():
        raise ValueError("Use a new path for the corrected ONNX")
    payload = base.read_bytes()
    model = onnx.load_model_from_string(payload)
    graph = model.graph
    tensors = {name for node in graph.node for name in node.output}
    steering = "m1_native_steering_correction"
    if steering not in tensors or "m1_motion_roll_weight" in tensors:
        raise ValueError("Expected the original projected M1 universal graph")
    action_node, = [node for node in graph.node if "actions" in node.output]
    action_node.output[list(action_node.output).index("actions")] = "m1_motion_previous_actions"

    def literal(name, value, dtype=np.float32):
        name = "m1_motion_"+name
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, name, *inputs, **attrs):
        name = "m1_motion_"+name
        graph.node.append(helper.make_node(kind,list(inputs),[name],name=name,**attrs))
        return name

    def gather(name, source, indices):
        return op("Gather",name,source,literal(name+"_index",indices,np.int64),axis=1)

    zero, one = literal("zero",0.), literal("one",1.)

    def ramp(name, value, lo, hi):
        a = op("Sub",name+"_offset",value,literal(name+"_lo",lo))
        b = op("Div",name+"_fraction",a,literal(name+"_range",hi-lo))
        return op("Clip",name,b,zero,one)

    def inverse(name, value):
        return op("Sub",name,one,value)

    def peak(name, value):
        return op("ReduceMax",name,op("Abs",name+"_abs",value),axes=[1],keepdims=1)

    def spread(name, value):
        maximum = op("ReduceMax",name+"_max",value,axes=[1],keepdims=1)
        minimum = op("ReduceMin",name+"_min",value,axes=[1],keepdims=1)
        return op("Sub",name,maximum,minimum), minimum

    def sq(name, value):
        return op("Mul",name,value,value)

    def atan2(name, y, x):
        # The half-angle identity avoids the missing atan2 operator in opset 12.
        mag2 = op("Add",name+"_norm_square",sq(name+"_ys",y),sq(name+"_xs",x))
        denom = op("Add",name+"_denom",op("Sqrt",name+"_norm",mag2),x)
        ratio = op("Div",name+"_ratio",y,op("Max",name+"_safe",denom,epsilon))
        return op("Mul",name,op("Atan",name+"_half",ratio),two)

    obs = "obs"
    x_command = gather("x_command",obs,[6])
    turn_command = gather("turn_command",obs,[7,8])
    reverse = ramp("reverse_steering_weight",x_command,-.26,-.08)
    removed = op("Mul","suppressed_steering",op("Sub","steering_change",reverse,one),steering)
    action = op("Add","reverse_corrected","m1_motion_previous_actions",removed)

    positive = ramp("forward",x_command,.16,.36)
    no_turn = inverse("straight",ramp("turn",peak("turn_peak",turn_command),.005,.05))
    phase = gather("phase",obs,list(range(41,49)))
    quiet = inverse("rolling",ramp("phase_activity",peak("phase_peak",phase),.004,.08))
    straight = op("Mul","forward_rolling",op("Mul","forward_straight",positive,no_turn),quiet)
    gravity_history = gather("gravity_history","obs_history",[i*49+j for i in (2,3,4) for j in (3,4)])
    gravity_xy = gather("gravity_xy",obs,[3,4])
    tilt = peak("tilt",op("Concat","gravity_samples",gravity_xy,gravity_history,axis=1))
    gyr_history = gather("gyro_history","obs_history",[i*49+j for i in (2,3,4) for j in (0,1)])
    gyro = peak("gyro",op("Concat","gyro_samples",gather("gyro_xy",obs,[0,1]),gyr_history,axis=1))
    wheel = gather("wheel_velocity",obs,[21,22,23,24])
    wheel_range,wheel_min = spread("wheel_range",wheel)

    # Sagittal M1 geometry from the asm09 model. The measured wheel centres
    # distinguish flat rolling from the first lifted wheel at a stair.
    hips = np.arange(1,16,4)
    knees = np.arange(2,16,4)
    q = op("Add","hip_angle",gather("hip_obs",obs,[10,13,16,19]),literal("hip_zero",M1_HOME[hips]))
    k = op("Add","knee_angle",gather("knee_obs",obs,[11,14,17,20]),literal("knee_zero",M1_HOME[knees]))
    haa = gather("haa_angle",obs,[9,12,15,18])
    sh,ch = op("Sin","haa_sin",haa),op("Cos","haa_cos",haa)
    two,epsilon = literal("two",2.),literal("epsilon",1e-6)
    l1,l2 = .23848,.36538
    thigh,shank = literal("thigh",l1),literal("shank",l2)
    qk = op("Add","hip_knee",q,k)
    x = op("Neg","wheel_x",op("Add","segment_x",
                              op("Mul","thigh_x",thigh,op("Sin","hip_sin",q)),
                              op("Mul","shank_x",shank,op("Sin","hip_knee_sin",qk))))
    z = op("Neg","wheel_z",op("Add","segment_z",
                              op("Mul","thigh_z",thigh,op("Cos","hip_cos",q)),
                              op("Mul","shank_z",shank,op("Cos","hip_knee_cos",qk))))
    side = literal("lateral_segment",[.21935,.21935,-.21735,-.21935])
    local_x = op("Add","centre_x",x,literal("hip_x",[.28614,-.28586,.28614,-.28586]))
    local_y = op("Add","centre_y",literal("hip_y",[.06,.06,-.06,-.06]),
                 op("Sub","lateral_rotated",op("Mul","local_side",side,ch),op("Mul","tilted_height",z,sh)))
    local_z = op("Add","centre_z",literal("hip_z",.058),
                 op("Add","rotated_height",op("Mul","lateral_rise",side,sh),op("Mul","vertical_rise",z,ch)))
    projected = []
    for i, coord in enumerate((local_x,local_y,local_z)):
        projected.append(op("Mul",f"projected_{i}",coord,gather(f"gravity_{i}",obs,[3+i])))
    height = op("Neg","relative_elevation",op("Sum","wheel_down",*projected))
    level_range,_ = spread("elevation_range",height)
    coplanar = inverse("level_wheels",ramp("wheel_elevation",level_range,.026,.068))
    not_slipping = inverse("contact_wheels",ramp("wheel_jam",wheel_range,.245,.46))
    common = op("Mul","grounded_straight",straight,coplanar)
    roll = op("Mul","roll_weight",common,inverse("level_body",ramp("tilted",tilt,.09,.16)))
    speed_blend = op("Mul","speed_blend",roll,not_slipping)
    leg_blend = common
    for i, value in enumerate((inverse("level_pose",ramp("pose_tilt",tilt,.08,.16)),
                               inverse("quiet_body",ramp("tilt_rate",gyro,.085,.20)),
                               inverse("wheel_agreement",ramp("wheel_spread",wheel_range,.105,.265)),
                               ramp("rolling_speed",wheel_min,.009,.045))):
        leg_blend = op("Mul",f"leg_blend_{i}",leg_blend,value)

    # Shorten a wheel's reach about the hip while preserving its sagittal height.
    corridor = op("Min","stance_cap",op("Max","stance_floor",x,literal("rear_boundary",[-.10,-.095,-.10,-.095])),
                  literal("front_boundary",[.045,.09,.045,.09]))
    difference = op("Sub","stance_error",x,corridor)
    desired = op("Mul","desired_shift",difference,literal("stance_gain",strength))
    shift = op("Clip","bounded_shift",desired,literal("shift_min",-.14),literal("shift_max",.21))
    new_x = op("Sub","shorter_reach",x,op("Mul","applied_shift",leg_blend,shift))
    distance = op("Add","stance_distance_squared",sq("reach_square",new_x),sq("height_square",z))
    cosine = op("Div","knee_cosine",op("Sub","triangle",distance,literal("segment_squares",l1*l1+l2*l2)),
                literal("twice_segment_product",2*l1*l2))
    cosine = op("Clip","safe_knee_cosine",cosine,literal("cosine_min",-.9999),literal("cosine_max",.9999))
    new_k = op("Mul","compact_knee",op("Sign","knee_direction",k),op("Acos","knee_bend",cosine))
    target_angle = atan2("stance_angle",op("Neg","ik_x",new_x),op("Neg","ik_z",z))
    shin_angle = atan2("shin_angle",op("Mul","ik_knee_sin",shank,op("Sin","compact_sin",new_k)),
                       op("Add","ik_segment_sum",thigh,op("Mul","ik_knee_cos",shank,cosine)))
    new_q = op("Sub","compact_hip",target_angle,shin_angle)
    combined = op("Concat","compact_angle_delta",
                  op("Clip","limited_hip",op("Sub","hip_delta",new_q,q),literal("hip_delta_min",-.45),literal("hip_delta_max",.45)),
                  op("Clip","limited_knee",op("Sub","knee_delta",new_k,k),literal("knee_delta_min",-.35),literal("knee_delta_max",.35)),axis=1)
    embed = np.zeros((8,16),np.float32)
    indices = np.r_[hips,knees]
    embed[np.arange(8),indices] = 1/M1_SCALE[indices]
    trim = op("MatMul","leg_action_delta",combined,literal("leg_action_indices",embed))
    enabled = op("Greater","stance_enabled",leg_blend,zero)
    legs = op("Add","compact_actions",action,op("Where","applied_leg_delta",enabled,trim,zero))

    wheel_mask = np.zeros(16,np.float32)
    wheel_mask[WHEELS] = 1.
    wheel_goal = op("Mul","wheel_speed_goal",x_command,literal("wheel_units",.99/(2.*.095*10.)))
    wheel_error = op("Mul","wheel_speed_error",op("Sub","wheel_difference",wheel_goal,legs),literal("wheel_mask",wheel_mask))
    wheel_delta = op("Mul","applied_wheel_delta",wheel_error,speed_blend)

    learned_front_lift = "m1_native_front_correction" in tensors
    if not learned_front_lift:
        # Older checkpoints use a fixed assist on a low, slow right-front wheel.
        left_height = gather("stair_left_front_height",height,[0])
        right_height = gather("stair_right_front_height",height,[2])
        height_gap = op("Sub","stair_front_height_gap",left_height,right_height)
        behind = ramp("stair_height_lag",height_gap,.05,.14)
        right_speed = gather("stair_right_front_speed",wheel,[2])
        slow = inverse("stair_slow_wheel",ramp("stair_speed_recovery",right_speed,.14,.25))
        pitch = op("Neg","stair_nose_up",gather("stair_gravity_x",obs,[3]))
        incline = ramp("stair_pitch",pitch,.095,.24)
        forward = ramp("stair_forward",x_command,.28,.45)
        climbing = op("Mul","stair_straight_climb",op("Mul","stair_straight",no_turn,quiet),
                      op("Mul","stair_uphill",incline,forward))
        assist = op("Mul","stair_assist",op("Mul","stair_lagging_wheel",behind,slow),climbing)
        right_wheel = np.zeros(16,np.float32)
        right_wheel[WHEELS[2]] = .30  # Each action unit represents 10 rad/s at the wheel.
        stair_delta = op("Mul","stair_wheel_delta",assist,literal("stair_wheel_mask",right_wheel))
        wheel_delta = op("Add","combined_wheel_delta",wheel_delta,stair_delta)
    graph.node.append(helper.make_node("Add",[legs,wheel_delta],["actions"],name="m1_motion_actions"))
    stair_variant = "learned-front-lift" if learned_front_lift else "right-stair-assist"
    for key, value in (("m1_motion_base_sha256",sha256(payload).hexdigest()),
                       ("m1_motion_variant",f"rolling-stance-{strength:g}-{stair_variant}")):
        metadata = model.metadata_props.add()
        metadata.key, metadata.value = key,value
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True,exist_ok=True)
    onnx.save(model,output)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--strength",type=float,default=1.1)
    args = p.parse_args()
    export(args.base,args.output,args.strength)
