B2 power-loss repair

Cause: previous analytic power penalty used the torque after a1200W cap and applied a1500W excess threshold. It was identically zero, leaving actual power excursions without this training gradient. Physical evaluator still measured5ms actual torque/post-speed peaks and failed all28conditions.

Fix: reward penalizes uncapped mechanical torque demand and pre-transition enveloped torque paired with predicted next velocity. Keep worst-joint endpoint speed penalty, all original physical dynamics, no artificial target slew, load/mass/friction/asymmetric starts/pushes and full-cycle criteria. No learned peak head. The20ms estimate is not a proof of all5ms peaks, which independent acceptance continues to measure.

Initialization: own balance-repair best Actor and WM (best4000, SHA5f99e59307296f9e234cb16c60a2175b80346b7c274d0f934a31b0607f9aa643); fresh replay/optimizers; frozen same own Actor. No external policy. CPU/CUDA actual3-update smoke and positive violation/nonzero-gradient regression required before launch. Actual6rad/s and1501W acceptance unchanged. Do not deploy without full28 and96crossed original acceptance.
