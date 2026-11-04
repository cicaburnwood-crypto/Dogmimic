B2 nominal0kg/5kg scope

User explicitly dropped10kg and15kg. Train one own continued policy with Bernoulli50%0kg and50%5kg per reset. No intermediate/heavier nominal payload samples. Retain the existing /15 load-context normalization for inherited WM compatibility; it does not change the actual sampled kilograms.

Start from own power-repair iteration19000 evaluated checkpoint SHA1a88bfd02add3b5b7a7ebbfb567cfee09eef2fbbd42f62cd664e5934ffa6b877, selected by best worst-case0/5kg physical motion success across previous saved evaluations (62/64 and53/64). Not actuator compliant. Continue Actor and WM, fresh replay/optimizers; frozen own source anchor. No official/external policy.

Only payload sampler and evaluation scope change. Motion, reward, physical torque/PD, speed/power budgets,5-pulse8–16N pushes, friction.4–1.3, mass.9–1.1, COM and measured/asymmetric pose randomization retained. Standard full rise/hold/descend/four-foot recovery unchanged. Original6rad/s,1501W, foot drift and COM/acceleration/pose criteria unchanged.

Basic validation two loads x64. Final48 distinct conditions (2loads x4frictions x3masses x2asymmetric starts),64trials each. Same original evaluate_cycle.py. User reduced load scope, not numerical acceptance thresholds. Never report old10/15kg results as new goals or deploy without current full acceptance.

Existing user-deleted periodic automation remains deleted. Run is detached and does not touch G1 or hardware.
