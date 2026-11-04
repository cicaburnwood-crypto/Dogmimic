import sys,time,unittest,threading,types,tempfile,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sdk_transport import SDKTransport
from runner import Backend
from test_lifecycle import FakeController,Transport

class Tests(unittest.TestCase):
 def base(self):
  t=object.__new__(SDKTransport);t.publishing=False;t.pubthread=None;t.handover_events=[];t.packet=None
  return t
 def test_busy_retried_and_owner_verified(self):
  t=self.base();owners=iter(['','','','ai']);codes=iter([7002,0]);calls=[]
  t.switch=types.SimpleNamespace(CheckMode=lambda:(0,{'name':next(owners)}),SelectMode=lambda n:(calls.append(n) or next(codes),None))
  with patch('sdk_transport.time.sleep'):t.release({'owner':'ai'})
  self.assertEqual(calls,['ai','ai']);self.assertEqual(t.handover_events[-1]['event'],'native_restored')
 def test_rpc_timeout_can_still_have_restored_owner(self):
  t=self.base();owners=iter(['','ai']);calls=[]
  t.switch=types.SimpleNamespace(CheckMode=lambda:(0,{'name':next(owners)}),SelectMode=lambda n:(calls.append(n) or 3104,None))
  with patch('sdk_transport.time.sleep'):t.release({'owner':'ai'})
  self.assertEqual(calls,['ai']);self.assertEqual(t.handover_events[-1]['event'],'native_restored')
 def test_unexpected_code_not_silently_retried(self):
  t=self.base();t.switch=types.SimpleNamespace(CheckMode=lambda:(0,{'name':''}),SelectMode=lambda n:(7009,None))
  with self.assertRaisesRegex(RuntimeError,'7009'):t.release({'owner':'ai'})
 def test_prepare_before_release_and_hold_before_check(self):
  t=self.base();events=[];owners=iter(['ai',''])
  def check():events.append('check');return 0,{'name':next(owners)}
  t.switch=types.SimpleNamespace(CheckMode=check,ReleaseMode=lambda:(events.append('release') or 0,None))
  t.prepare=lambda:events.append('prepare');t.snapshot=lambda:{'q':[0]*29}
  t.command=lambda *a:events.append('hold')
  t.acquire({});self.assertEqual(events,['check','prepare','release','hold','check'])
 def test_raw_age_not_delayed_decode_age(self):
  t=self.base();now=time.monotonic();t.low_time=now-1;t.odom_time=now-1;t.raw={'low':(b'',now,1)}
  self.assertLess(t.sensor_age('low',now),.001)
 def test_both_errors_persist_in_log(self):
  t=Transport();t.fail=True
  def failrestore(r):raise RuntimeError('restore busy')
  t.release=failrestore
  with tempfile.TemporaryDirectory() as d,patch('runner.Controller',FakeController):
   b=Backend(t,log_dir=d);b.execute('funky_chicken');b.worker.join(2)
   self.assertEqual([e['stage'] for e in b.errors],['action','restore'])
   self.assertIn('lost odometry',b.last_error);self.assertIn('restore busy',b.last_error)
   rows=[json.loads(x) for x in next(Path(d).glob('*.jsonl')).read_text().splitlines()]
   self.assertEqual(len(rows[-1]['errors']),2)

if __name__=='__main__':unittest.main()
