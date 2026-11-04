"""Compare the deployment input adapter with fixtures from the training engine."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from runtime import Controller

class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller=Controller(BASE/'models/jump_60cm')

    def test_selected_checkpoint_and_training_goal_fixtures(self):
        c=self.controller
        self.assertEqual(c.meta['checkpoint_sha256'],'55d47be9ead36fc6a0863d79c64701ea7c86d04b2f66595182eec7432c854a7b')
        with np.load(BASE/'models/jump_60cm/alignment_inputs.npz') as data:
            for row,expected in zip(data['inputs'],data['expected_goal']):
                c.reset()
                np.testing.assert_allclose(c.goal(row[:3],data['rotation'],0,row[3:]),expected,atol=3e-6,rtol=0)

    def test_world_xy_and_heading_do_not_change_policy_goal(self):
        c=self.controller
        p=c.ref['pos'][0].copy();q=c.ref['q'][0].copy();q[0]+=.12
        R=c.ref['rot'][0]
        c.reset();goal=c.goal(p,R,0,q)
        theta=.7;cs,sn=np.cos(theta),np.sin(theta)
        A=np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]],dtype='f')
        c.reset();transformed=c.goal(A@p+np.array([1.2,-.8,0]),A@R,0,q)
        np.testing.assert_allclose(transformed,goal,atol=3e-6,rtol=0)

    def test_height_error_is_preserved_and_reset_realigns(self):
        c=self.controller;p=c.ref['pos'][0].copy();q=c.ref['q'][0];R=c.ref['rot'][0]
        c.reset();goal=c.goal(p,R,0,q).reshape(-1,73)
        c.reset();raised=c.goal(p+[0,0,.03],R,0,q).reshape(-1,73)
        np.testing.assert_allclose(raised[:,:3]-goal[:,:3],np.tile(np.array([0,0,-.03])@R,(c.meta['future'],1)),atol=3e-6,rtol=0)
        c.reset();self.assertIsNone(c.anchor_position)

if __name__=='__main__':unittest.main()
