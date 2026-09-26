---
name: robot-collision-safety
description: Use when checking MoveIt PlanningScene, self collision, environment collision, tool collision, ground collision, ACM, FCL, CCD, clearance, continuous collision, collision meshes, or fail-closed safety for robot states and trajectories.
---

# Robot Collision Safety

Use the actual MoveIt2 PlanningScene and configured collision geometry when
available. Establish robot identity, frames, joint ordering, URDF/SRDF pairing,
environment objects, and ACM provenance before interpreting a result.

## Gates

Check self-collision, environment/ground collision, and tool collision at the
declared discrete states and along the declared interpolation. Record the
backend, sampling/interpolation semantics, resolution, object identity, and
first failing state or segment. A result based on finite samples must be
labelled `adaptive_discrete_interpolation`; it is not strict continuous
collision detection.

Treat an ACM exception as valid only when physically justified and documented.
Never edit SRDF disabled-collision rules merely to make a test pass. Distinguish
FCL observations from hardware certification and from a proof over continuous
time.

## Fail-closed status rules

`NO_COLLISION_FOUND != PROVEN_COLLISION_FREE`

`UNRESOLVED != PASS`; `CCD_UNAVAILABLE != CCD_PASS`;
`CLEARANCE_UNAVAILABLE != CLEARANCE_PASS`; `API_ERROR != SAFE`.

Bullet CCD, strict continuous self-collision, and clearance are available only
when a real backend reports them. Otherwise emit `not_available` and JSON nulls
where required; never infer clearance from a collision-free state. Preserve
errors and missing data instead of replacing them with safe values.

Pair this gate with `urdf`, `srdf`, and `robot-motion` for model and planning
semantics, and with `testing` for a bounded known-answer smoke. Hand the
accepted/rejected state set to `robot-experiment-auditor`; a visible failure is
baseline evidence, not permission to change the model or threshold.
