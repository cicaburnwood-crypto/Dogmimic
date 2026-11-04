B2 balance repair

Evidence: paired CPU 15kg experiment with the same own v3 Actor and torque envelope: target slew4 and6rad/s both0/4 completion; near-unslewed4/4. Continued own Actor also0/4 versus4/4 completion. This demonstrates the target slew can suppress balancing; it does not prove actuator compliance. See diagnostics/dual_model_optimization/b2_slew_root_report.json.

Changes: remove artificial absolute-target slew; retain identical physical torque envelope (160/160/240Nm, same motoring derating,1200W requested power). Penalize worst-joint speed rather than joint mean and use enveloped torque in predicted power. BPTT32. Select best by actual all-pass status, worst-load completion, worst strict success, average strict success, speed. No acceptance threshold changes. Revert Actor to our own v3 best; fresh WM, replay, optimizers; frozen own v3 reference. This is continuation, not random initialization.

Preserved: 25% unloaded, remaining5–15kg continuous loads; 8–16N pushes; friction.4–1.3; mass.9–1.1; measured/asymmetric poses; full rise/hold/descend/recovery. Actual physical speed<=6rad/s, power<=1501W and original motion/COM/drift criteria. Full28conditions and96loadedcrossedconditions still required before deployment.

Limit: this loss covers both transition endpoints, not every5ms peak. Independent physical evaluation continues to measure all substeps. The previously unreliable learned peak head is NOT integrated. Implementation smoke passes do not mean physical acceptance passes. Never deploy without full original acceptance.
