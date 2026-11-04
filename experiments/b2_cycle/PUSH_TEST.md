# Light random push evaluation

The trained v2 best policy is frozen; this test does not update weights. The
checkpoint is continuation iteration 4,000 (8,000 policy-training iterations
including v1). The finalized test is `push_test_20261030_v3` under the v2 run.

There are 64 trials per group and 5 pulses per trial. Force magnitude is uniform
8–16 N, direction uniform on the three-dimensional sphere, duration 0.12–0.24 s.
Pulses are applied at the trunk body's center of mass in world coordinates, with
zero additional torque, via `xfrc_applied` throughout the 5 ms physics substeps.
They occur during rise, twice during hind-leg holding, during return, and during
final four-legged settling. The force is about 0.98–1.95% of this model's weight.
Both groups use the original observation settings, bfloat16 MLP inference, and
small independent initial joint perturbations. Initial-state seed is 20261109;
force-schedule seed is 20261110. Force magnitude and seeds were fixed before results.

A free-flight impulse audit checks each XYZ component against F*dt/m. The measured
COM velocity changes differ from prediction by at most 3.4e-6 m/s. Actual applied
forces, the full random schedule, and trajectories are saved.

Controls and push trials have identical initial states, checked exactly, and fresh
independent simulators. GPU contact reduction and bfloat16 quantization prevent
bitwise replay: before the first push, the maximum root-coordinate difference is
about 1.1 mm. All differences are recorded in `evaluation.json`; the comparison is
between matched initial conditions, not a claim of exact deterministic replay.
The v1/v2 test attempts stopped at an overly strict bitwise pre-push trajectory
assertion and are not used as the final evaluation. A short diagnostic confirmed
identical initial state, initial action and joint noise, followed by tiny first-step
physics differences that propagate through quantized inference. The final audit
checks the actual matching requirement (initial states), and records trajectory
roundoff rather than hiding it. No force, seed or success threshold was relaxed.

Both finalized groups passed 64/64 full-cycle and strict stability tests. Under
pushes, mean standing COM acceleration RMS was 0.0448 m/s² and final four-legged
COM acceleration RMS was 0.0482 m/s². This result applies to this defined light-force
test and does not establish robustness to stronger pushes or hardware conditions.

The video uses trial 0, selected before outcomes, with matched control on the left
and pushed policy on the right. Orange arrows show the actual force vector while
applied. Any failed episode would be frozen and labeled, not replaced by a reset
episode. The original policy and task definitions are unchanged by this test.
