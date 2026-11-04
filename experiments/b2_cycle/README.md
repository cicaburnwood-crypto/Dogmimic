# B2 stable stand and full return cycle

Continues our own B2 iteration-4,000 policy and world model on physical GPU 6.
The critic, replay, optimizer states and training clock are new because the reward
and motion library changed. This uses the same SuperMimic reproduction and BPTT,
with no external trained policy. Original runs and the two G1 tasks are untouched.

The 15.5 s reference is four-legged standing (0–1 s), rising (1–3.5), rear-leg
standing (3.5–9.5), lowering back to four feet (9.5–12.5), and settling (12.5–15.5).
It uses our original procedural trajectory in both directions. Evaluation always
executes the entire sequence from the initial four-legged pose; training additionally
samples phases to learn the previously unseen descent. Episodes permit 20 s, so
the final return and settling are never cut off by the old 15 s timeout.

The previous policy had about 14.0 m/s² whole-body COM acceleration RMS during its
last five seconds of standing, mostly in Z. The old posture-only success metric did
not test stability. Its claimed success does not imply passing the new criteria.

Both physical simulator transitions and differentiable WM transitions use the
same added rewards. World XYZ acceleration is `(velocity_next - velocity_now)/0.02`:

- Whole-body center of mass: mass-weighted inertial positions and velocities,
  including angular velocity cross inertial offset and randomized root COM offset.
- Trunk-center acceleration: all three components of root linear velocity change.
- Isotropic pseudo-Huber acceleration penalties, with useful gradients at high
  acceleration. No axis is omitted; gravity is not added as an accelerometer bias.
- Hold-phase COM speed, trunk speed, angular speed, joint velocity and COM drift
  penalties, plus a near-stillness bonus. Strong hold penalties ramp in over 0.3 s;
  8% acceleration weight remains during rising/lowering. Action smoothing is also
  stronger. Reward terms remain differentiable into the policy.

Acceleration is measured on 50 Hz control transitions without smoothing. Reset
discontinuities are excluded: each terminal successor is compared with its actual
predecessor before the reset occurs. These are control-step mean accelerations;
they do not certify peaks within the 5 ms physics substeps.

Strict evaluation covers a 5.3 s upright window and the final 2.5 s four-foot window.
For both windows, COM acceleration RMS <=1.5 m/s² and peak <=8; trunk acceleration
RMS <=2 and peak <=10; COM speed RMS <=0.10 m/s; trunk speed RMS <=0.12; angular
speed RMS <=0.25 rad/s; COM position standard-deviation norm <=1.5 cm; inclination
standard deviation <=1.5 degrees. Upright pose/contact and final four-foot recovery
must hold for >=95% of their windows, with hind-foot drift <=4 cm in the upright
window. Contacts are assessed from foot sphere center height (2–4.5 cm), alongside
the dynamic stability metrics, rather than a claim of measured foot force.

An automatic pass requires >=90% of 32 trials (including the nominal trial) to pass
all full-cycle and stability checks twice. The other trials have independent small
initial joint perturbations. These are nominal simulation tests, not hardware or
large-disturbance robustness certification. Thresholds and all per-trial metrics
are saved in every evaluation. Best and last models are saved separately.

`audit_cycle.py` verifies the analytic centroid against MuJoCo, acceleration costs
and gradients for positive and negative X/Y/Z, zero acceleration at constant
velocity, real/imagined reward parity, and a complete return reference.

Each job uses 2,048 environments, horizon 32, four WM batches per update, policy/WM
learning rate 1e-4, and exploration noise 0.05. Evaluation and checkpoints occur
every 2,000 iterations; limit 50,000 continuation iterations or three wall-clock
hours, with early stopping only on the new strict criteria.
