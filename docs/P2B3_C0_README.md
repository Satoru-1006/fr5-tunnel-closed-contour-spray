# P2-B3-C0 — Ordered Surface Correspondence Gate

`P2B3_C0_STATUS=BLOCKED_BY_UPSTREAM_GEOMETRY_OR_ORDERING`.

## Finding

The reported wall-station change from sample 15 to 16 is reproducible, but the evidence does not support nearest-polyline ambiguity as its root cause. In the archived R0 FK trace, sample 15 lies within 0.142 mm of open TCP path segment 24; sample 16 lies within 0.014 mm of the indexed target point 16. Their TCP positions are 64.54 mm apart, while authoritative target points 15 and 16 are 4.04 mm apart. The FK TCP sequence's global projection onto the open TCP path also steps backward by 64.70 mm.

The legacy wall projection selects segments 23 then 15 and reports a `-0.13194315439053514 m` station step. At sample 16, the closest forward-ordered wall candidate is segment 24 at 0.271656 m, 11.72 mm farther than the global nearest result. The nearest wall candidate on segment 23 is already 8.45 mm farther than the best point. These measurements show that forcing monotone correspondence would mask a waypoint-order/path reversal in frozen R0 data.

The failure taxonomy should therefore distinguish:

- `GEOMETRIC_PATH_BACKTRACK`: confirmed by the R0 FK TCP sequence and open-path projection.
- `CORRESPONDENCE_PROJECTION_BACKSTEP`: observed in the legacy wall-station output, but not established as the cause.

Details and exact input identities are in [`p2b3_c0_ordered_surface_correspondence.json`](../outputs/p2b3_c0_ordered_surface_correspondence.json).

## Gate status

- Level 1: `NOT_RUN`; the root-cause gate disproved the precondition for a correspondence repair.
- Level 2: `BLOCKED_BY_UPSTREAM_WAYPOINT_ORDERING`; no ordered mapper was applied to make the 181-point sequence appear monotone.
- Level 3: `NOT_TRIGGERED`; no shared evaluator code changed.
- Full P2-B2 replay: `NO`.
- P2-B3-C1 started: `NO`.

This audit read the archived 181-row R0 nominal FK trace, the authoritative 181-point target pose/normal CSV, and published R0 pre/post-Ruckig reference files. It changed no trajectory, waypoint, FK, Ruckig, or P2-B2 artifact. The cached P2-B2 clean R0 reference reported zero pre/post joint and post-Ruckig timestamp deltas. No fresh MoveIt2, FK, Ruckig, robustness, or hardware run is claimed.

## Protected P2-B2 files

The P2-B2 README, landscape JSON, failure taxonomy, remapping implementation, and focused test file remain byte-identical to the P2-B2 source branch. Their SHA-256 values are recorded in the result JSON.

## Next scientific question

Determine why the frozen R0 trajectory state labeled waypoint 15 maps to target path segment 24, followed by waypoint 16 mapping back to segment 16. Reconcile trajectory sample identity and waypoint order upstream before retrying ordered surface correspondence. Do not alter the frozen trajectory in P2-B3-C0.
