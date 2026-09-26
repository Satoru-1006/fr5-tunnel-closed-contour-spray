---
name: trajectory-continuity-optimizer
description: Use when resolving IK branch continuity, joint jumps, multi-solution waypoint paths, global branch selection, dynamic programming, graph optimization, or WP-to-WP discontinuity in robot trajectories.
---

# Trajectory Continuity Optimizer

Optimize a path over the whole waypoint sequence, not by independently
choosing `argmin IK(q_i)`. Start from the candidate states that already passed
exact FK, official limits, and the collision-state gate. Keep the canonical
branch and every proposed branch separately reproducible.

## Objective order

Use this lexicographic priority unless the user explicitly changes it:

1. valid states;
2. zero old-valid loss;
3. reduce maximum joint jump;
4. reduce total joint travel;
5. improve joint-limit margin;
6. improve singularity margin;
7. maintain collision validity;
8. improve downstream dynamics.

Useful methods include nearest or bidirectional continuation, dynamic
programming, graph shortest path, beam search, windowed optimization, branch
switch penalties, travel cost, and margin penalties. Choose from evidence and
state the method, candidate pool, edge gates, and tie-breaks. A concrete
symptom such as `WP66 -> WP67` with a roughly 96-degree J6 jump is a continuity
case; it is not proof that every adjacent pair has the same root cause.

Recheck every selected node and edge with exact FK, official bounds, and the
actual collision gate. Send the resulting path to
`trajectory-dynamics-ruckig` only after continuity closure. Ruckig may
parameterize a valid path; it must not be used to hide an IK branch jump.
Do not relax limits, remove difficult waypoints, alter canonical data, or
promote a screen-only improvement.
