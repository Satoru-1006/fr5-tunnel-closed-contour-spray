---
name: robot-experiment-auditor
description: Use when auditing BASELINE, CANDIDATE, CHAMPION, ablation, reproducibility, metric extraction, scientific routing, or PASS/FAIL/UNRESOLVED decisions for robot experiments without fabricating success.
---

# Robot Experiment Auditor

Audit the experiment chain rather than only its final score. Resolve the
baseline, candidate/champion, frozen input, initial condition, seed/config,
checkpoint, stage, output path, and report identity before comparison.

## Closure sequence

`identify baseline -> reproduce baseline -> run candidate -> extract metrics ->
compare -> independent post-closure reproduction -> status`

Check for stale caches, auto0/auto1 routing mix-ups, wrong candidate or input,
and reports belonging to another run. A measurement or pipeline defect is
reproduced, isolated, repaired, known-answer tested, regression-tested, and
affected results recomputed. A real frozen robot weakness is measured and
handed to Stage 4B; it is not optimized away during the audit.

Report applicable metrics such as `TOTAL_VALID`, `OLD_VALID_RETENTION`,
`LOST`, `RECOVERED`, `MAX_JOINT_JUMP`, `MIN_JOINT_MARGIN`, `MIN_CLEARANCE`,
`COLLISION_STATUS`, `CCD_STATUS`, `TRAJECTORY_TIME`, `MAX_VELOCITY`,
`MAX_ACCELERATION`, and `MAX_JERK`. Mark absent evidence `UNAVAILABLE`,
`UNVERIFIED`, `NOT_RUN`, or `UNRESOLVED_THRESHOLD` as appropriate.

`PASS` means the stated measurement/experiment contract was satisfied; it does
not mean every robot case passed, hardware is safe, or a software-shadow
result is deployable. Keep measurement-pipeline status separate from frozen
robot-baseline performance. Never change canonical parameters, official
limits, acceptance sets, difficult cases, thresholds, or trajectories to make
the report green. Pair technical closures with the relevant state, collision,
continuity, or dynamics skill, then use `stage-handoff-packager` at closeout.
