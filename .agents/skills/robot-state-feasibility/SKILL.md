---
name: robot-state-feasibility
description: Use when auditing robot waypoint state feasibility, IK/FK recovery, multi-solution IK, seed search, official joint-bound validation, roll/task-freedom search, base-layout feasibility, joint-margin improvement, waypoint recovery, 240-point feasibility, or old-valid retention.
---

# Robot State Feasibility

Resolve the stage, robot identity, target set, joint ordering, base transform,
official limits, and collision semantics before searching. Treat the declared
canonical inputs and protected valid set as read-only. A higher raw score never
justifies losing previously valid waypoints.

## Acceptance pipeline

Evaluate candidates in this order:

`TCP target -> IK candidate pool -> exact FK -> official joint bounds ->
collision-state gate -> joint margin -> accepted state`

Use multi-start, structured and deterministic seeds, continuation,
bidirectional continuation, per-waypoint or global roll, and base/layout
probes only when the task explicitly permits them. Keep exploratory roots and
intermediate continuation states separate from accepted states.

For a declared 240-point task, report the waypoint-level valid set and:

`TOTAL_VALID, VALID_SET, OLD_VALID_RETENTION, LOST, RECOVERED, FK_ERROR,
MIN_JOINT_MARGIN, COLLISION_REJECTIONS, ROLL_MAGNITUDE, BASE_TRANSFORM`.

Use `valid_solution_count > 0` or the project's stated equivalent as the
waypoint acceptance test. Do not count a solver row, an approximate root, an
exact-FK-but-collision-invalid state, or a completed replay as a valid point.

## Failure taxonomy

Classify each waypoint as one of `NO_IK_RETURNED`, `APPROXIMATE_ONLY`,
`EXACT_FK_MISMATCH`, `JOINT_BOUND_VIOLATION`, `COLLISION_REJECTION`, or
`VALID_STATE`. Preserve counts and representative evidence for every class;
never collapse all failures into `IK failed`.

## Routing and protection

- Hand accepted states to `robot-collision-safety` for the actual
  PlanningScene/FCL gate; collision labels must say
  `adaptive_discrete_interpolation` when that is the implemented check.
- Hand accepted waypoint graphs to `trajectory-continuity-optimizer` before
  any Ruckig work.
- Use `robot-experiment-auditor` for baseline/candidate comparisons and
  retention claims.
- Do not edit canonical targets, protected floors, official limits, the formal
  robot model, SRDF/ACM, or historical results merely to recover a point.
- `UNRESOLVED`, `NOT_RUN`, unavailable CCD/clearance, and missing backend
  evidence are never acceptance.

If changing search tooling, test known-answer states and rerun the protected
valid set. Report software-shadow scope separately from hardware or safety
claims.
