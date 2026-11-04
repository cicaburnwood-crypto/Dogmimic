# Evaluating SONIC's released checkpoint in our simulator (TODO once BONES-SEED is available)

The paper scores SONIC's released checkpoint on the same 129,514-motion benchmark "with its own
actuator gains and action scale, receiving observations as its training environment builds
them" (App. F). Checkpoint: the GEAR-SONIC release (`gear-sonic/default-deployment/policy/release/`)
(`model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml`). Conventions, read from
`GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`:

- Joint orders: policy uses IsaacLab order. `isaaclab_to_mujoco[i]` = IsaacLab index of MuJoCo
  joint i; `mujoco_to_isaaclab[j]` = MuJoCo index of IsaacLab joint j (policy_parameters.hpp).
- Action: `q_target_mj[i] = default_angles[i] + a[isaaclab_to_mujoco[i]] * scale[i]`,
  scale = 0.25 * effort / stiffness. Gains = mjlab G1 gains except hip pitch uses the 7520_22
  motor (mjlab: 7520_14). `default_angles` = knees-bent pose (-0.312, 0, 0, 0.669, -0.363, 0, ...).
- Decoder input (YAML order): token_state (64, encoder output), then 10-frame histories,
  oldest first: base angular velocity (pelvis IMU gyro, 3), joint positions minus default
  (IsaacLab order, 29), joint velocities (29), last raw actions (IsaacLab order, 29), gravity
  direction in the pelvis frame (3).
- Encoder input: concatenation of all `encoder_observations` in YAML order, zero for those not
  required by the active mode. g1 mode (id 0): encoder_mode_4 = [0, 0, 0, 0];
  motion joint positions and velocities at frames t, t+5, ..., t+45 (IsaacLab order, absolute);
  motion anchor orientation at the same frames = first two columns, row-major, of
  R(conj(q_pelvis) * q_ref_root) after the start-heading alignment (identity when the robot
  starts on the reference).
- Check ONNX input sizes before trusting the YAML comment (it says 436 = 4-frame history).
- Validate on the benchmark: paper reports SONIC normalized return 0.90, 179 failed motions.
