import sys,unittest,time,threading
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pd_adapter import policy_output_fault
from backend import SDKBackend
class GuardTest(unittest.TestCase):
 def test_healthy_and_incident(self):
  for flag in [4,5]:self.assertIsNone(policy_output_fault({'bit_flag':flag},.002,.02,32))
  for flag in [36,37]:self.assertIn('事故拒绝位',policy_output_fault({'bit_flag':flag},.002,.02,32))
 def test_stale(self):
  self.assertIsNotNone(policy_output_fault(None,0,0))
  self.assertIsNotNone(policy_output_fault({},.101,0))
  self.assertIsNotNone(policy_output_fault({},0,.101))
  self.assertIsNotNone(policy_output_fault({},0,0,32))
 def test_actual_writer_withdraws_before_pd_or_publish(self):
  b=SDKBackend.__new__(SDKBackend);b.lock=threading.RLock();b.owned=True;b.pub=SimpleNamespace(Write=lambda _:self.fail('must not publish'))
  b.target=[0.]*12;b.target_at=time.monotonic();b.low=(time.monotonic(),dict(bit_flag=36));b.damping=False;b.active_skill='locomotion';b.command_epoch=7;b.last_execution={}
  b.controllers={'locomotion':SimpleNamespace(meta={'skill':'height_locomotion','driver_reject_bits':32})}
  b._publish_once();self.assertFalse(b.owned);self.assertIsNone(b.target);self.assertEqual(b.command_epoch,8)
  b._publish_once();self.assertEqual(b.command_epoch,8)
  with self.assertRaisesRegex(RuntimeError,'控制权'):b._step(None,None,None)
 def test_damping_bypasses_policy_guard(self):
  b=SDKBackend.__new__(SDKBackend);b.lock=threading.RLock();b.owned=True;b.damping=True;b.target=None;b.pub=SimpleNamespace(Write=lambda _:True);b.writer_packets=0
  b.cmd=SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(12)]);b.crc=SimpleNamespace(Crc=lambda _:0)
  b._publish_once();self.assertEqual(b.writer_packets,1);self.assertEqual(b.cmd.motor_cmd[0].kp,0)
if __name__=='__main__':unittest.main()
