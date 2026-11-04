import sys,threading,time,unittest,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sdk_transport import SDKTransport

class Tests(unittest.TestCase):
 def transport(self):
  t=object.__new__(SDKTransport);t.lock=threading.RLock();t.low=None;t.odom=None
  t.low_tick=None;t.odom_tick=None;t.low_time=0;t.odom_time=0;t.alignment=None;t.publish_error=None;t.fsm=500;t.fsm_time=time.monotonic()
  imu=types.SimpleNamespace(quaternion=[1.,0.,0.,0.],gyroscope=[0.,0.,0.])
  motors=[types.SimpleNamespace(q=0.,dq=0.,tau_est=0.) for _ in range(35)]
  low=types.SimpleNamespace(tick=1,imu_state=imu,motor_state=motors,mode_machine=5)
  odom=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1,nanosec=2),imu_state=imu,position=[1.,2.,.7],velocity=[0.,0.,0.],body_height=.7)
  return t,low,odom
 def test_message_shapes_and_timestamp(self):
  t,low,odom=self.transport();t.on_low(low);t.on_odom(odom)
  s=t.snapshot();self.assertEqual(len(s['q']),29);self.assertEqual(s['odom_stamp'],[1,2]);self.assertEqual(s['mode_machine'],5)
 def test_repeated_stamp_is_not_fresh(self):
  t,low,odom=self.transport();t.on_low(low);t.on_odom(odom);t.odom_time-=1
  t.on_odom(odom)
  with self.assertRaisesRegex(RuntimeError,'stale'):t.snapshot()
 def test_no_data_no_synthetic_position(self):
  t,_,_=self.transport()
  with self.assertRaises(RuntimeError):t.snapshot()
 def test_acquire_rpc_failure_no_publisher(self):
  t,low,odom=self.transport();t.on_low(low);t.on_odom(odom)
  t.switch=types.SimpleNamespace(CheckMode=lambda:(3102,None))
  with self.assertRaisesRegex(RuntimeError,'3102'):t.acquire()
  self.assertFalse(hasattr(t,'publisher'))

if __name__=='__main__':unittest.main()
