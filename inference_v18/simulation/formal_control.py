"""Formal-run bookkeeping only; this module does not change physical criteria."""
import hashlib
import json
import math
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checkpoint_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def score(rows):
    """Compare identical suites, prioritizing survival and worst-stratum success."""
    assert rows and all(r['trials'] > 0 for r in rows)
    return (min(r['alive'] / r['trials'] for r in rows),
            min(r['success'] / r['trials'] for r in rows),
            sum(r['success'] for r in rows) / sum(r['trials'] for r in rows),
            -sum(r['final_position_error'] / .01 +
                 r['final_orientation_error_deg'] / 10 for r in rows))


class RegressionGuard:
    """Stop on two consecutive substantial regressions against a verified best.

    Known initial failures remain failures. Their existence does not itself stop
    simulated learning. This changes the short-test stopping policy, never the
    episode safety limits or 1 cm / 10 degree / 1 second success predicate.
    """
    def __init__(self, initial):
        self.reference = initial
        self.streak = 0

    def check(self, rows):
        ref = {r['stage']: r for r in self.reference}
        reasons = []
        for r in rows:
            p = ref[r['stage']]
            assert r['trials'] == p['trials']
            if (p['alive'] - r['alive']) / r['trials'] >= .10:
                reasons.append(f"stage{r['stage']}: survival decreased >=10 percentage points")
            if (p['success'] - r['success']) / r['trials'] > .25:
                reasons.append(f"stage{r['stage']}: success decreased >25 percentage points")
            if r['stage'] == 0 and (
                    r['final_position_error'] > 1.5 * max(.01, p['final_position_error']) or
                    r['final_orientation_error_deg'] > 1.5 * max(10., p['final_orientation_error_deg'])):
                reasons.append('near precision: >1.5 times max(reference, tolerance)')
            if not all(math.isfinite(r[k]) for k in ['final_position_error', 'final_orientation_error_deg']):
                return dict(stop=True, streak=self.streak, reasons=['nonfinite physical evaluation'])
        self.streak = self.streak + 1 if reasons else 0
        if not reasons and score(rows) > score(self.reference):
            self.reference = rows
        return dict(stop=self.streak >= 2, streak=self.streak, reasons=reasons)
