# P2-B3-C1 — R0 Ordered Path Fidelity Closure

- **Project:** FAIRINO FR5 tunnel complex-surface spraying
- **Status:** `COMPLETE_WITH_LIMITATIONS`
- **Source base:** `9adfe22d0fa53e3f15052659d5e08af06fbd8733` (P2-B3-C0)
- **R0 execution code:** `cb2464153a101b9deccba7902e6350aea3fc7f62`
- **Selective P2-B2 bracket recheck code:** `05ff5cf853b2203ceba3dfb6c65ca040f0989a3c

## Finding and repair

Legacy R0 rows WP0–15 are the first 16 observed rows copied from D39's `stable_velocity_residual_update.csv`. P2-B1 R0/B0 appended those rows directly to the output joint trajectory, then began DLS at target index 16 using seed row 15. They were therefore present in the emitted, single planned process segment even though they had not been solved for target indices 0–15. The output is a plan artifact; physical controller execution did not occur.

The archived process manifest labels the trajectory scope as ON-state open-arch and reports one process segment. It contains no per-row spray command or physical ON/OFF state. Thus WP0–15 were not proven to be physically sprayed, even though the plan placed them inside the nominal process segment.

That splice explains the C0 backtrack: legacy WP15 projected to open-path segment 24, then WP16 returned to segment 16. The measured TCP step was 64.538 mm versus a 4.045 mm target step; legacy same-index error at WP15 was 68.523 mm. The legacy local time gap was 28.633 s. These are reproduced from the archived 181-row FK trace and R0 trajectory.

C1 adds `P2B3_C1_INITIALIZATION_ONLY=true`, guarded to the frozen R0/B0/16-row/`none` policy. D39 row 0 is used only as the initial seed; DLS solves targets 0–180 and emits no unsolved seed prefix. With the flag unset, the P2-B2 default prefix path is preserved.

## C1 R0 result

- All 181 target rows were solved and emitted in index order.
- Maximum nearest-path distance: **0.001467 mm**. Maximum same-index error: **0.004719 mm**.
- Maximum spray-axis normal error: **1.082°**. Projected station backsteps: **0**; minimum station increment: **3.325 mm**.
- MoveIt2 strict audit and post-Ruckig acceptance summary: `pass`; Ruckig ran. The configured dynamics audit passed. PlanningScene checked 858 sampled states and reported 0 collisions using `adaptive_discrete_interpolation`.
- Of 180 adjacent TCP segments, 179 were within the existing 0.003 m/s ±5% diagnostic band, none were below it, and one was above it: transition WP179→WP180 at **0.004144 m/s**. No zero-distance dwell was measured. This endpoint speed deviation remains visible for Stage 4B.
- Compared with legacy R0, rows 0–16 differ by more than 1 µrad; rows 17–180 remain within 1 µrad. The maximum joint delta is 0.620 rad at the corrected prefix. WP180 differs by about `1.23e-11` rad.

The configured 150 mm tool TCP is still an assumed virtual placeholder. Its production status is failed/waived by the offline scope. Hardware/controller execution, calibrated TCP accuracy, physical spray switching, physical torque, continuous collision detection, and a clearance acceptance threshold are not established here. `NOT_AVAILABLE`, `NOT_RUN`, and `UNRESOLVED_THRESHOLD` remain explicit in the result.

## Historical result disposition

The P2-B2 campaign is **not invalidated as a whole**. Its code, design, acceptance set, and stored results remain intact as measurements of the old R0 trajectory. Legacy nominal geometry, wall-station progression, and local timing claims are replaced for C1 by the new R0 measurements.

Six axis directions were selectively checked at the old `last_pass` and `first_fail` magnitudes: J3 ±, TCP translation X ±, and TCP rotation X ±. All 12 endpoints reproduced their expected PASS/FAIL classification. The fresh Stage4A FK and C1 MoveItPy FK position traces matched exactly. This is a bracket spot-check, not a full margin rescan. TCP translation Y/Z, rotation Y/Z, and complete axis margin scans remain pending before transferring their historical numbers to C1.

`FULL_P2B2_CAMPAIGN_REPLAYED=NO`.

## Validation and artifacts

Tier 1 ran the new C1 scope/ordered/timing tests and existing P2-B1/P2-B2 focused tests: **70 passed**. Tier 2 ran one fresh 181-row C1 R0 MoveIt2 PlanningScene/FK/configured-dynamics/Ruckig validation. Tier 3 evaluated only the 12 selected historical bracket endpoints; the 42-axis campaign was not replayed.

The 181-row lineage and measurements are in `outputs/p2b3_c1_r0_waypoint_lineage.csv`; machine-readable statuses, impact analysis, and selective recheck are in `outputs/p2b3_c1_result.json`. The C0 Drive archive used for the legacy FK trace is identified in that result by its existing Drive ID; the large archive was not copied into this stage.

To reproduce the main R0 run, check out the exact clean C1 execution-code commit on its C1 branch, then run `scripts/run_p2b3_c1_r0.py` with a new external scratch directory and the FAIRINO source checkout at commit `60755d44d521a5ad6bee8494cc19522f8801aa20`. The runner builds the FAIRINO description/config packages and C1 bridge into that scratch area, then runs the strict R0 pipeline. To reproduce the selective Stage 4A checks, use `scripts/run_p2b3_c1_p2b2_bracket_recheck.py` with the generated C1 post-Ruckig CSV, FAIRINO prefix, the recorded Stage4A native/FK binaries, and fresh scratch/output paths.

## Change impact and next research step

The repair materially changes R0 joint states through target 16; after that, the chain returns within 1 µrad of the old R0. That bounds the observed nominal-state change. It does not prove that every stress family is invariant, because robustness perturbations act across the full trajectory. Keep the six matched brackets as targeted evidence and recheck the pending axis families only when C1 margins are needed.

The next useful research step is a C1-based **process-aware stress characterization** that couples trajectory perturbations with local timing and order measurements. Treat the WP179→180 overspeed as a measured boundary case, retain each failed/unavailable case, and rank failure modes by family before Stage 4B changes the robot system. Do not use this C1 closure to claim hardware spraying, continuous collision safety, or physical robustness.

## Protected P2-B2 identity check

The five protected P2-B2 files were rechecked against the C0 hash ledger. Four stored SHA-256 values match. The ledger value for 	ests/test_p2b2_robustness_remapping.py differs from the observed file SHA-256, but its Git blob is identical to the frozen C0 source-base blob and git diff from that base is empty. This is recorded as a C0 hash-record mismatch; the protected file was not changed.
