import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Controller, OfflineBackend

class Backend:
    def __init__(self):
        self.stable = True
        self.calls = []
    def status(self):
        return {'connected': True, 'stable_standing': self.stable}
    def execute(self, action):
        self.calls.append(action)
        self.stable = False
        return 'accepted'

class Tests(unittest.TestCase):
    def test_offline_never_dispatches(self):
        with self.assertRaisesRegex(ValueError, '尚未连接'):
            Controller(OfflineBackend()).execute('jump_60cm')
    def test_both_actions_dispatch_when_stable(self):
        for action in ['jump_60cm', 'funky_chicken']:
            b = Backend()
            self.assertTrue(Controller(b).execute(action)['ok'])
            self.assertEqual(b.calls, [action])
    def test_rechecks_posture_for_every_request(self):
        b = Backend()
        c = Controller(b)
        c.execute('jump_60cm')
        with self.assertRaisesRegex(ValueError, '稳定站立'):
            c.execute('funky_chicken')
        self.assertEqual(len(b.calls), 1)
    def test_unknown_posture_and_unknown_action(self):
        b = Backend()
        b.stable = None
        c = Controller(b)
        for action in ['jump_60cm', 'locomotion']:
            with self.assertRaises(ValueError):
                c.execute(action)
        self.assertFalse(b.calls)

if __name__ == '__main__': unittest.main()
