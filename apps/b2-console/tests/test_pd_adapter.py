import os,sys,unittest,math,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pd_adapter import saturated_pd_target
BUNDLE=Path(os.environ.get('B2_TEST_BUNDLE',Path(__file__).resolve().parents[3]/'deployments/b2_192_168_123_164_20261101'))
class PDAdapterTests(unittest.TestCase):
 def test_selected_c1_envelope_matches_training_fixtures(self):
  import json,numpy as np
  folder=BUNDLE/'models/hindstand'
  meta=json.loads((folder/'policy.json').read_text())
  self.assertEqual(meta['checkpoint_sha256'],'52ee319da4b4b97b73467f376a6cf415cbeda7584b0c00e476b0a5b3199aa829')
  with np.load(folder/'envelope_inputs.npz') as data:
   for target,q,dq,oracle in zip(data['target'],data['q'],data['dq'],data['expected_tau']):
    wire,_,tau=saturated_pd_target(target.tolist(),q.tolist(),dq.tolist(),250.,8.,meta['torque_limits'],meta['torque_envelope'])
    np.testing.assert_allclose(tau,oracle,atol=2e-4,rtol=0)
    self.assertLessEqual(sum(abs(t*v) for t,v in zip(tau,dq)),1200.0001)
    np.testing.assert_allclose(250*(np.asarray(wire)-q)-8*dq,tau,atol=1e-4,rtol=0)
 def test_training_torque_equivalence(self):
  r=random.Random(13)
  for _ in range(1000):
   q=[r.uniform(-2,3) for _ in range(12)];dq=[r.uniform(-15,15) for _ in range(12)];target=[r.uniform(-3,4) for _ in range(12)];limits=[200,200,300]*4
   wire,raw,lim=saturated_pd_target(target,q,dq,250,8,limits)
   for i in range(12):
    self.assertAlmostEqual(250*(wire[i]-q[i])-8*dq[i],max(-limits[i],min(limits[i],raw[i])),places=9)
    if abs(raw[i])<=limits[i]:self.assertEqual(wire[i],target[i])
 def test_416nm_trace_case(self):
  wire,raw,lim=saturated_pd_target([1.664]*12,[0]*12,[0]*12,250,8,[200]*12)
  self.assertAlmostEqual(raw[0],416);self.assertEqual(lim[0],200);self.assertEqual(wire[0],.8)
 def test_invalid_input(self):
  with self.assertRaises(ValueError):saturated_pd_target([math.nan]*12,[0]*12,[0]*12,250,8,[200]*12)

class PublisherTests(unittest.TestCase):
 def test_selected_c1_envelope_reaches_published_command(self):
  from test_control import sdk_fixture
  import json,time
  folder=BUNDLE/'models/hindstand'
  meta=json.loads((folder/'policy.json').read_text());b=sdk_fixture()
  b.controllers['hindstand'].meta=meta;b.pub=b.pub_factory();b.owned=True;b.active_skill='hindstand';b.target=[2.]*12
  b.low=(time.monotonic(),dict(q=[0.]*12,dq=[8.]*12));b._publish_once()
  tau=[m.kp*m.q-m.kd*8 for m in b.packets[-1].motor_cmd[:12]]
  self.assertAlmostEqual(sum(abs(t*8) for t in tau),1200.,places=6)
  self.assertTrue(all(abs(t)<160 for t in tau))
 def test_hindstand_wire_limit_and_locomotion_unchanged(self):
  from test_control import sdk_fixture
  import time
  b=sdk_fixture();b.pub=b.pub_factory();b.owned=True;b.active_skill='hindstand';b.target=[2.]*12
  b.low=(time.monotonic(),dict(q=[0.]*12,dq=[0.]*12))
  b.controllers['hindstand'].meta=dict(policy_to_sdk=[3,4,5,0,1,2,9,10,11,6,7,8],torque_limits=[200,200,300]*4)
  b._publish_once()
  self.assertEqual(len(b.packets),1)
  for i,m in enumerate(b.packets[-1].motor_cmd[:12]):self.assertAlmostEqual(m.kp*m.q,[200,200,300][i%3])
  b.active_skill='locomotion';b._publish_once()
  self.assertTrue(all(m.q==2. for m in b.packets[-1].motor_cmd[:12]))
  b.damping=True;b._publish_once()
  self.assertTrue(all(m.kp==0 and m.kd==8 for m in b.packets[-1].motor_cmd[:12]))
 def test_driver_fields_accepted_by_telemetry(self):
  from control import Telemetry
  t=Telemetry(bit_flag=36,tau_est=[0.]*12,motor_lost=[0]*12,power_v=49.,power_a=0.)
  self.assertEqual(t.bit_flag,36)

if __name__=='__main__':unittest.main()
