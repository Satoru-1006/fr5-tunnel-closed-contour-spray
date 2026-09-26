# D41 final report

TASK_STATUS: PASS
D39_RETENTION: PASS
D40_RETENTION: PASS
COLLISION_CERTIFICATION: PASS
MIN_ROBOT_WORLD_CLEARANCE: 0.08857717027025891 m
MIN_SELF_CLEARANCE: 0.016623586819851727 m
DLS_NUMERICAL_HEALTH: PASS
ROBUSTNESS: 37/37
FULL_OFFLINE_WORKFLOW_SCOPE: 181-point ON-state open-arch only
D41_RETENTION_BASELINE: D41_native_bullet_ccd_clearance_and_joint_limit_aware_dls
FIRST_UNRESOLVED_BLOCKER: NONE

## Findings

The installed MoveIt2 2.12.4 stack exposes native Bullet robot-world two-state collision checking and robot/world and self distance queries in C++. D40’s `ccd=not_available` was a bridge/API exposure gap, not an absent Bullet library. The installed wrapper does not provide a self two-state continuous query, so D41 uses native self checks with motivated adaptive refinement and reports that limitation explicitly.

## Changes and promotion

D41 added a standalone native certification package, DLS numerical-health logging, deterministic robustness generation, targeted tests, and this reproducible artifact assembly. The default D40 DLS route and H1 champion were preserved. D41 additionally enabled an explicit URDF-bound joint-limit-aware projection in the DLS proposal loop; that shadow candidate and the native CCD/clearance layer were promoted only after strict nominal and robustness non-regression gates passed.

## Results

- Native nominal robot-world segments: PASS; native self certification: PASS.
- Adversarial controls: {'endpoint_free_mid_segment_collision': 'PASS', 'free_space_negative_control': 'PASS', 'self_collision_negative_control': 'PASS', 'self_collision_negative_control_detail': None, 'near_miss_positive_clearance': 'PASS', 'near_miss_detail': None, 'comfortable_clearance': 'PASS', 'comfortable_clearance_detail': None}.
- Minimum Jacobian sigma: 6.078443090310153e-11; maximum condition number: 30576767824.35914; trust-region clip events: 39; minimum joint-limit margin: 0.00010000000000021103 rad.
- D40 dynamics utilization remains velocity=0.016421107188169345, acceleration=0.009579396534353725, jerk=0.00023042942821655058; future joint reads remain zero.
- Strict nominal replay executed=True; the regenerated MoveIt2 FK, dynamics, collision, Ruckig, and audit artifacts were available and passed.

## Scope

See `D41_SCOPE_CLOSURE.md`. OFF, approach, retreat, reorientation, closed-contour transition, and multi-segment transition are all outside the current authoritative final offline demonstration scope.

## Reproduction artifacts

Run directory: `D:\robotfucker\outputs\stage3_h13_d41_offline_robot_certification\run_20260827T152326Z`. Native provenance: `D:\robotfucker\outputs\stage3_h13_d41_offline_robot_certification\run_20260827T152326Z\native\nominal\D41_native_provenance.json`. Required handoff files are in `D:\robotfucker\outputs\stage3_h13_d41_offline_robot_certification`.
