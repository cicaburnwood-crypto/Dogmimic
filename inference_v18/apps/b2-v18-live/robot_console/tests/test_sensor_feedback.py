"""Sensor-source and staleness checks; no DDS or robot commands."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control import Telemetry
from sensor_feedback import feedback


class SensorFeedbackTests(unittest.TestCase):
    def make(self):
        return Telemetry(received=99.99, tick_changed=99.99, position_received=99.99,
            position_changed=99.99, q=[0., .8, -1.6]*4, dq=[0.]*12,
            quaternion=[1.,0.,0.,0.], gyro=[0.,0.,0.], foot_force=[400.]*4,
            position=[1.,2.,-.03], position_quaternion=[1.,0.,0.,0.], position_source='odom',
            velocity=[.3,.4,0.], body_height=.52, native_mode=9)

    def test_actual_inputs_and_distinct_height(self):
        s=feedback(self.make(),100.)
        self.assertTrue(s['locomotion_inputs']['valid'])
        self.assertTrue(s['base_odometry']['valid'])
        self.assertEqual(s['base_odometry']['position_m'],[1.,2.,-.03])
        self.assertEqual(s['body_height']['value_m'],.52)
        self.assertEqual(s['base_odometry']['speed_m_s'],.5)
        self.assertFalse(s['ground_truth'])

    def test_no_stale_values_or_sport_fallback_as_independent_pose(self):
        t=self.make();t.position_changed=99.
        s=feedback(t,100.);self.assertFalse(s['base_odometry']['valid'])
        self.assertIsNone(s['base_odometry']['position_m'])
        t=self.make();t.position_source='sport'
        self.assertFalse(feedback(t,100.)['base_odometry']['valid'])
        t=self.make();t.tick_changed=99.
        self.assertFalse(feedback(t,100.)['locomotion_inputs']['valid'])
        self.assertIsNone(feedback(t,100.)['locomotion_inputs']['q_rad'])

    def test_odom_keeps_its_orientation_and_survives_native_absence(self):
        t=self.make();t.native_mode=None;t.body_height=0.
        t.position_quaternion=[math.cos(.1),0.,0.,math.sin(.1)]
        s=feedback(t,100.)
        self.assertTrue(s['base_odometry']['valid'])
        self.assertAlmostEqual(s['base_odometry']['rpy_deg'][2],math.degrees(.2))
        self.assertAlmostEqual(s['base_odometry']['imu_minus_odom_yaw_deg'],-math.degrees(.2))
        self.assertIsNone(s['body_height']['value_m'])

    def test_invalid_values_cannot_appear_as_valid_feedback(self):
        for field,value in [('quaternion',[0.]*4),('q',[float('nan')]*12)]:
            t=self.make();setattr(t,field,value)
            self.assertFalse(feedback(t,100.)['locomotion_inputs']['valid'])
        t=self.make();t.position=[float('nan')]*3
        self.assertFalse(feedback(t,100.)['base_odometry']['valid'])


if __name__=='__main__':unittest.main()
