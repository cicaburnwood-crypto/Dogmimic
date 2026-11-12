import asyncio,time,unittest,sys
from pathlib import Path
from unittest.mock import AsyncMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import Portal
from teleop import KeyboardGoal
class GripperPortalTests(unittest.IsolatedAsyncioTestCase):
 def portal(self):
  p=Portal.__new__(Portal);p.dog={'remote_priority':{'blocked':False,'generation':0}};p.job=None;p.owner='browser-test';p.keyboard=KeyboardGoal();p.result={};p.beats={'browser-test':time.monotonic()};p.armcmd=AsyncMock(return_value={'message':'requested'});p.dogcmd=AsyncMock();return p
 async def call(self,p,browser='browser-test'):
  return await p.action({'op':'gripper_set','client':browser,'id':'command-001','target':'open'})
 async def test_idle_native_gripper_does_not_begin_joint_control(self):
  p=self.portal();await self.call(p);p.armcmd.assert_awaited_once_with('gripper_set',id='command-001',target='open');p.dogcmd.assert_not_called()
 async def test_allowed_during_own_joint_follow(self):
  p=self.portal();p.job=asyncio.get_running_loop().create_future();await self.call(p);p.dogcmd.assert_not_called()
 async def test_other_browser_rejected(self):
  p=self.portal();p.job=asyncio.get_running_loop().create_future()
  with self.assertRaisesRegex(ValueError,'另一页面'):await self.call(p,'another-browser')
  p.armcmd.assert_not_called()
 async def test_standby_rejected(self):
  p=self.portal();p.job=asyncio.get_running_loop().create_future();p.result={'operation':'standby'}
  with self.assertRaisesRegex(ValueError,'等待完成'):await self.call(p)
  p.armcmd.assert_not_called()
 async def test_stale_page_rejected(self):
  p=self.portal();p.beats['browser-test']-=3
  with self.assertRaisesRegex(ValueError,'心跳'):await self.call(p)
  p.armcmd.assert_not_called()
if __name__=='__main__':unittest.main()
