# Stage 4 input contract

Stage 4 starts exclusively from `outputs/STAGE3_FINAL_RELEASE`.

Authoritative starting state:

- canonical checkpoint: `outputs/stage3_h13_d35_permanent_champion/checkpoints/committed_update_480.pt`
- checkpoint SHA-256: `37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742`
- input scope: 181-point ON-state open-arch pair under `outputs/internal_wiper_moveit_inputs/`
- robot baseline: D40 `routeA_DLS_position_dominant_v4` retained and D41-certified offline replay
- collision contract: `adaptive_discrete_interpolation`; CCD `not_available`; clearance JSON `null`

Stage 4 objectives are system-level offline robotic-motion stability, smoothness,
safety, accuracy, reproducibility, reliability, and robustness. Stage 4 does not
inherit the H1 stretch target as a blocking gate and must not execute D45's
historical D43/D44 shadow candidates as production state.
