"""Read-only sensor presentation. Does not feed commands or change the policy."""
import math


def vector(value, size):
    if not isinstance(value, (tuple, list)) or len(value) != size:
        return None
    return [float(x) for x in value] if all(math.isfinite(x) for x in value) else None


def rpy_degrees(quaternion):
    q = vector(quaternion, 4)
    if q is None:
        return None
    norm = math.sqrt(sum(x*x for x in q))
    if not .9 <= norm <= 1.1:
        return None
    w, x, y, z = [v/norm for v in q]
    return [math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
            math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x))))),
            math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))]


def feedback(telemetry, now):
    t = telemetry
    fresh = t.fresh(now)
    q = vector(t.q, 12) if fresh else None
    dq = vector(t.dq, 12) if fresh else None
    quaternion = vector(t.quaternion, 4) if fresh else None
    rpy = rpy_degrees(quaternion)
    gyro = vector(t.gyro, 3) if fresh else None
    independent = t.position_source in ('odom', 'lf_odom') and t.position_fresh(now)
    pos = vector(t.position, 3) if independent else None
    odom_q = vector(t.position_quaternion, 4) if independent else None
    odom_rpy = rpy_degrees(odom_q)
    pose_valid = pos is not None and odom_rpy is not None
    velocity = vector(t.velocity, 3) if independent else None
    native_height = t.body_height if (fresh and t.native_mode is not None and
        t.body_height is not None and math.isfinite(t.body_height) and 0 < t.body_height < 1.5) else None
    yaw_offset = None
    if rpy is not None and odom_rpy is not None:
        yaw_offset = (rpy[2]-odom_rpy[2]+180.) % 360. - 180.
    return dict(
        ground_truth=False, arm_control=False, autonomous_target_control=False,
        locomotion_inputs=dict(source='rt/lowstate', valid=bool(q is not None and dq is not None and rpy is not None and gyro is not None),
            age_ms=round((now-t.received)*1000, 2) if t.received > 0 else None,
            joint_order='SDK: FR, FL, RR, RL; hip, thigh, calf', q_rad=q, dq_rad_s=dq,
            imu_quaternion_wxyz=quaternion if rpy is not None else None, imu_rpy_deg=rpy,
            gyro_body_rad_s=gyro, foot_force_raw=vector(t.foot_force, 4) if fresh else None),
        base_odometry=dict(source=t.position_source if independent else None, valid=pose_valid,
            frame='native_odometry; local estimator origin, not a mapped absolute world',
            position_m=pos if pose_valid else None, quaternion_wxyz=odom_q if pose_valid else None,
            rpy_deg=odom_rpy if pose_valid else None,
            age_ms=round((now-t.position_received)*1000, 2) if independent else None,
            velocity_world_m_s=velocity,
            speed_m_s=math.sqrt(sum(v*v for v in velocity)) if velocity is not None else None,
            imu_minus_odom_yaw_deg=yaw_offset, z_is_ground_clearance=False,
            accuracy_validated=False),
        body_height=dict(value_m=native_height, source='native_controller_body_height' if native_height is not None else None,
            available=native_height is not None, independent_of_native_controller=False),
        note='IMU and encoders feed P4. Odometry is feedback only; no GT, no pose replacement by integration of commands.')
