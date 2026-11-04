# B2 grounded locomotion retraining

A separate scratch experiment requested after the previous velocity-tracking actor visibly hopped. The original `experiments/b2_locomotion` and `runs/b2_locomotion_fresh_20261031_234839` remain unchanged. Neither old nor official weights are used to initialize this actor, world model, critic, or replay.

## Why the old actor passed

The old physical acceptance tested velocity, upright survival, and stopped feet, but did not test gait while moving. A 752-trial re-evaluation of the old best checkpoint retains 720/720 core velocity/start-stop passes yet passes only 16/720 core cases with the added gait criteria (the standing case). Mean geometric four-foot flight across the whole post-settle sequence was 13.11%, with longest flight 135 ms. Old best SHA256: c97b5949c6a285ed49cc099ad3cabdffd6275aa33f6db189b24b8a72d9d3b924. This diagnostic never initializes training.

## Changes

- Same state-dependent differentiable costs for true-state reward and model-imagined reward: all-four airborne coefficient 4; positive minimum foot clearance coefficient 1 (25 mm scale, 4 mm allowance); insufficient support coefficient 0.5; both-front/both-rear flight coefficient 0.8; vertical-speed coefficient increased from 0.15 to 1.2. No reference trajectory, teacher, or fixed gait clock.
- Smooth contact proxy uses the exact sphere location and radius. Independent acceptance instead reads MuJoCo solver foot-floor normal forces every 5 ms and geometric foot clearance. Non-foot contacts are measured separately, so calf/body support cannot count as feet.
- Full standard-stand, move, stop, opposite-move, stop tests: 47 cases × 16 trials = 752. Core 45 cases (720 trials), extra ±3 rad/s tests reported separately. Existing tracking/stop thresholds are preserved. Each foot must alternate load-bearing and unloaded states in moving cases; continuously dragging planted feet is rejected. New gait criteria are in `gait.py`; they are included in success, curriculum advancement, best-model selection, and two-evaluation early stopping.
- Reproducible nominal trial and randomized dynamics trials; mass, friction, COM, encoder/sensor noise, PD gains, and 0–10 ms delay retained. Actor uses only proprioception/history/actions/commands.
- XY command norm <=2 m/s, |vy|<=0.6 m/s, no extra vx cap, no hard yaw-command clip. Finite tested yaw speeds do not establish unlimited tracking ability.
- Simulator training only. Real-robot configuration is still pending.

## Validation

`audit.py`: forbids loading MotionLibrary, checks random actor/world-model gradients, standard pose and zero initial velocities, delay-consistent replay, and 100,000 feasible command samples.

`gait_audit.py`: penalty ordering (grounded < diagonal swing < front-pair swing < four-foot flight), finite downward airborne gradient, compiled/eager parity, GPU normal forces compared to CPU MuJoCo, total supported weight including separate non-foot contacts, zero support when raised, and rejection of a synthetic 100 ms flight. Zero-action PD eventually settles onto some calf contacts; this is measured, not mislabeled as foot-only support.

A separate 100-iteration DDP smoke run exercises the complete trainer/evaluator/export/supervisor before the formal independent random initialization. Latest and best results are distinct; reward growth alone is never accepted as gait improvement.

Formal budget: 80,000 iterations, 4096 simulated environments across GPUs 4/5 (2048 each), held-out evaluation on GPU 6, checkpoint every 2000 iterations, 6-hour wall guard. Anticipated budget runtime from steady smoke throughput is roughly 95 minutes; this is not a guarantee of gait convergence.
