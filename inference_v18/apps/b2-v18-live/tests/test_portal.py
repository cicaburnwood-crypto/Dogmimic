import asyncio,copy,json,sys,time,unittest,tempfile
from pathlib import Path
from unittest.mock import AsyncMock
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import Portal,make_app
from aiohttp.test_utils import TestServer,TestClient
class PortalTests(unittest.IsolatedAsyncioTestCase):
 def setup_portal(self):
  p=Portal();tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);p.logdir=Path(tmp.name);f=np.load(Path(__file__).resolve().parents[1]/'models/parity.npz');i=0
  s={k:f[k][i].copy() for k in ['pos','rot','v','w','q','qd','aq','aqd']};s.update(gyro=s['rot'].T@s['w'],ee=f['ee'][i],eerot=f['eerot'][i],body=[*Rotation.from_matrix(s['rot']).as_euler('xyz'),s['pos'][2]-.4959882021194876])
  p.state=lambda:s;p.stamp=time.monotonic();p.error='';p.ref={}
  p.arm={'fresh':True,'age_ms':1,'debug':False,'error':0,'active':False,'state':{'servo':1,'fsm_state':'POSITION','task':{'exe_flag':False}},'tool_coordinate':[0,0,0,0,0,0,1]}
  p.dog={'remote_priority':{'blocked':False,'fresh':True,'generation':0,'quiet_remaining_s':0},'active':'native','gate':{'switch_allowed':True},'stationary_reach':{},'joint_manager':{},'policy_context':f['ctx'][i].tolist(),'model_info':{'models':{'locomotion':{'weights_sha256':'e7be320b2e2486f454dcfdb7a292d9349d4e1671c02a240efb4be40754f9462b'}}}}
  async def refresh():p.stamp=time.monotonic()
  p.refresh=refresh;p.armcmd=AsyncMock();p.dogcmd=AsyncMock();return p,s
 async def test_previews_never_move(self):
  p,s=self.setup_portal();await p.preview({'xyz':s['ee'].tolist(),'rpy_deg':Rotation.from_matrix(s['eerot']).as_euler('xyz',degrees=True).tolist()});p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
  await p.preview_prepare();p.armcmd.assert_not_called();p.dogcmd.assert_not_called();self.assertEqual(p.plan['kind'],'prepare')
 async def test_invalid_preview_invalidates_old_plan(self):
  p,s=self.setup_portal();p.plan={'id':'old'}
  with self.assertRaises(ValueError):await p.preview({'xyz':[float('nan')]*3,'rpy_deg':[0]*3})
  self.assertIsNone(p.plan);p.armcmd.assert_not_called();p.dogcmd.assert_not_called()
 async def test_failed_arm_begin_holds_dog(self):
  p,s=self.setup_portal();p.owner='browser-test';p.beats[p.owner]=time.monotonic();p.armcmd.side_effect=[ValueError('stream unavailable'),{}]
  await p.run({'xyz':s['ee'].tolist(),'rpy_deg':[0,0,0]})
  self.assertEqual([c.args[0] for c in p.dogcmd.await_args_list],['begin','hold'])
  self.assertEqual([c.args[0] for c in p.armcmd.await_args_list],['stream_begin','stop'])
  self.assertEqual(p.result['phase'],'stopped')
 async def test_changed_model_and_debug_rejected(self):
  p,s=self.setup_portal();p.arm['debug']=True
  with self.assertRaises(ValueError):p.ready()
  p.arm['debug']=False;p.dog['model_info']['models']['locomotion']['weights_sha256']='wrong'
  with self.assertRaises(ValueError):p.ready()
 async def test_prepare_forbids_trained_mode(self):
  p,s=self.setup_portal();p.dog['active']='trained'
  with self.assertRaises(ValueError):await p.preview_prepare()
  p.armcmd.assert_not_called()
 async def test_prepare_respects_cancel_before_motion(self):
  p,s=self.setup_portal();p.owner='browser-test';p.beats[p.owner]=time.monotonic();p.cancel=True
  await p.run_prepare({'start_pos':s['pos'].tolist(),'start_rot':s['rot'].tolist()})
  self.assertNotIn('move',[c.args[0] for c in p.armcmd.await_args_list])
if __name__=='__main__':unittest.main()
