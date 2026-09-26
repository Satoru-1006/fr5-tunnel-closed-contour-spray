---
name: trajectory-dynamics-ruckig
description: Use when validating robot q, dq, ddq, jerk, velocity/acceleration/jerk limits, Ruckig, time parameterization, trajectory duration, synchronization, or post-parameterization behavior.
---

# Trajectory Dynamics Ruckig

Run after state feasibility and IK branch continuity are closed. Confirm joint
ordering, units, timestamps, initial/final state, and the provenance of every
velocity, acceleration, and jerk limit. Missing or unknown limits are
`UNAVAILABLE` or `UNRESOLVED_THRESHOLD`, never safe defaults.

## Required evidence

For an actual Ruckig run, retain the input state, limits, synchronization/time
parameterization settings, duration, output validity, and independent
post-Ruckig checks for finite q/v/a, velocity, acceleration, jerk, continuity,
and terminal conditions. Report only metrics that were actually computed:
`TRAJECTORY_TIME`, `MAX_VELOCITY`, `MAX_ACCELERATION`, `MAX_JERK`, and the
corresponding limit/status fields.

Validate the generated trajectory rather than trusting the Ruckig return code.
Use known-answer cases when repairing a measurement implementation, including
deliberately excessive velocity/jerk and invalid Ruckig input. Preserve any
real violation as a frozen-system finding for `robot-experiment-auditor` and
Stage 4B.

Ruckig is not a branch selector and must not conceal a discontinuity by
smoothing, clipping, changing thresholds, or replacing the path. Pair with
`trajectory-continuity-optimizer` first and `robot-collision-safety` for the
state/segment gate. Hardware, torque, calibrated TCP, and physical safety
claims require separate evidence.
