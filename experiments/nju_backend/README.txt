NJU backend: B2 light-payload hind-leg stand and G1 standing jump

Upstream: https://github.com/NJU-R-L-Group-Embodied-Lab/SuperMimic-Repro-Sim2Real
Pinned commit: f8f60d7bbdba68468c050ef100af02d752ca238e
mjlab: 8bb2bc9e187797689ec82c78c35019afd2fd0376

Scope
- B2: full 15.5 s four-leg -> rise -> hind-leg hold -> lower -> four-leg recovery,
  50% 0 kg / 50% 5 kg; no 10/15 kg. Friction .4-1.3, mass .9-1.1,
  encoder/COM/reset noise, five 8-16 N force pulses retained.
- G1: original 60 cm jump reference, NJU paper-default G1 environment, friction
  .3-1.2, encoder/COM/reset/sensor noise. NJU hardware-specific options remain off.
- Actor/world model/critic/optimizers/replay start from scratch. No old model,
  external teacher, released pretrained policy, or real-robot connection.

Backend identity
The upstream checkout is untouched. build_backend.py materializes a separate
copy with robot dimensions, B2 assets, context and lifecycle hooks. All training
uses NJU's Trainer, WorldModelTrainer and PolicyTrainer. k=8 autoregressive WM,
H=32 BPTT, critic/GAE, return sampling, executed-action replay and paper rewards
are retained. Ten learning/FK/replay files are byte-identical to upstream, recorded
in backend_provenance.json. G1 numerical parity is checked against upstream.
The B2 variant is our task port of NJU, not an upstream-supported B2 result.

B2 context is friction + encoder bias + COM + payload + mass scale (18 values).
As in NJU's optional push event, pushes are exogenous disturbances; the predictor
is not given the prior backend's known future force schedule. The actor has
only proprioception/history plus reference goals, never privileged context.
The all-contact friction adapter gives floor and feet the same sampled value,
so MuJoCo's max mixing does not erase the requested low-friction coefficient.
B2 uses the original XML actuator PD/gains/force and target limits.

Independent evaluation
Evaluation runs in the old isolated venv using unchanged strict evaluator files,
never the old training algorithm. TorchScript carries the NJU network, q0,
action scale and goal features. Inputs use NJU's term-major history and current
reference frame. Outputs are converted to absolute joint targets before mapping
to the strict environment. Converted references retain exact qpos/qvel and fps.
B2: 48 conditions x64 at 0/5 kg, including <=6 rad/s and <=1501 W; G1: 21 x32,
each >=31/32 plus nominal success. Existing selection/holdout requirements remain.
best is selected among trained checkpoints, separately from initial. Full strict
DEV-A plus independent HOLD-A/HOLD-B must pass before accepted is saved.
Task completion alone does not demonstrate actuator compliance or deployment readiness.

Run scope
Test budget: 20,000 environment iterations, at most 6 hours training per robot.
1024 environments per GPU; WM/actor minibatches 512; original network widths;
10 WM minibatches/update; update/data=.4; warmup=40; 2,000-step physical checks.
B2 GPU6; G1 GPU4/5 using NJU's rank pinning and gradient all-reduce.
Final full evaluation is allowed to finish after training. Each job is detached.
No recurring automation is recreated (the user previously cancelled it).

Validation
prepare_reference.py: source hashes, exact trajectory conversion, FK vs CPU MuJoCo.
parity_probe.py: G1 adapted-vs-upstream features, networks and gradients.
validate_adapter.py: environment/analytic reward and observation/goal parity,
executed-action replay, payload sampling and action/joint order.
train.py --smoke: real k8/H32 updates, nonzero changes in WM/actor/critic,
finite parameters, inference export parity, and then an independent evaluation probe.

This is a backend training test, not a promise of convergence or real-world success.
