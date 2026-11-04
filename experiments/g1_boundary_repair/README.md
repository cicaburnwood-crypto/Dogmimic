G1 boundary coverage continuation

Observed plateau: contact v10 improved412/672 to474/672 by4000 updates, then remained459–474 through20000. Failures persist at very low friction, native stress, backward pushes and combined conditions. This is not target convergence.

Coverage diagnosis: previous domain sampler mostly combined all randomized dimensions, plus nominal and weak-actuator groups. A100000sample audit produced zero isolatedmu.15/.2/.3 with otherwise nominal dynamics and zero pure-25N backward push. This is a coverage deficiency, not proof that it is the only cause or that every requested boundary is physically feasible.

Repair training mixture:20% nominal,30% original fullcontinuous ranges,5% isolatedweak,5% crossedweak,40% eight targeted boundary groups (5% each:mu.15/.2/.3,backwardpush,nativestress,combinedlow,combinedhigh,forwardpush). For native/combined groups beginning atframe0, use measured joint endpoint and the original unmatched rootorientation, retaining original resetnoise. This explicitly trains the previously failing stress combination, not replacement by an easier pairedpose. Other starts retain previous measured/ref/continuous mixtures and paired/unpaired distribution. Independent RNG and resetnoise remain; no evaluation trajectories or pass labels are loaded for learning.

Reward, physics/control, worldmodel/Actor architecture and optimizer algorithm unchanged; contactslip objective and full304dim delayedcontrol context retained. Original21conditions and allthresholds/evaluate_jump.pySHA333379... unchanged. This changes training distribution, not evaluation. No official policies or externalteachers.

Continue own contactv10 bestActor+WM it20000 SHAc85e440e7c8d2d4e11e6e60be0319287e78283077bca85f4a9bacd41b364e542; frozenownanchor, freshreplay/optimizers.500WM-only prefits,20000budget/6h, GPU4/5training/GPU6evaluation. CPU/CUDA H32real3updates, delayequivalence, contactgradient, airbornezero, ownActor/WMexactload and actualtargetedreset checks required. Guardstill stops two successive evaluations >64/672 below best. Implementationchecks do not prove improved skills.

No deployment. No periodic automation recreation. B2lightpayload training remains independent.
