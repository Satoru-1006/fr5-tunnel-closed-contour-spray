# FR5 C5B1-R2: Threshold attribution and geometric semantics
Date: 2026-10-13. Status: OFFLINE GEOMETRIC ATTRIBUTION; PHYSICAL ACCEPTANCE NOT ASSESSED.

## Provenance

Source repository: Satoru-1006/fr5-tunnel-closed-contour-spray.
Historical scan publication commit: 1d258c90db4a2d320134e4f8d0bccf3984f8ffdb.
Historical native execution code commit: 36324f64a2189a2a5156702f44ee2841e924e142.
C5B1 authoritative Drive JSON: https://drive.google.com/file/d/1FLaONF98i_eBlaIMirNGU1e979MwMPnB/view
C5B1 authoritative Drive CSV: https://drive.google.com/file/d/1OJHj8-A95hphCEqD2WVKuDwEc_GyCtsY/view

Frozen C1 FK: outputs/p2b3_c1_r0_fk_trace.csv, SHA256 ed8cffa87dae845df0dcc3cf50f88225e3f72b1fa425acce6def08e7ee60acee.
Frozen C1 targets and normals: outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv, SHA256 8a7d4a22c9e701bf840da255e1619e81b4a29bac4183528ad76fc918e16fbde7.
Frozen C1 q(t)/timestamps and C4 181-object scene remain unchanged. BASE_LEFT, rotations about base_link origin. No native FCL, IK, Ruckig, collision search or retiming rerun.

## A. What the observed 6 mm crossing means

The historical C5B1 code defines POSITION_GATE_M = 0.006 and classifies a position_diagnostic as FAIL when max_position_error_m > POSITION_GATE_M. For frozen TCP positions and a displaced workpiece target, pure translation yields nearly the same displacement in the position error. Therefore the six translation-ray brackets near 6 mm are predetermined ENGINEERING_THRESHOLD_CROSSING events, not experimentally established coating-failure margins.

An independent analytical C1 FK position replay with hypothetical 3, 6, 10 mm gates moves all six translation-ray crossings to approximately 3, 6, 10 mm. All 12 analytic 6 mm crossings agree with the historical native bracketing data: midpoint discrepancy at most 0.0041 mm on translation rays and 7.6e-6 rad on rotation rays. These are independent mathematical calculations from the earlier frozen C1 FK, not bitwise replication of fresh native C5A FK.

## B. Process-geometric decomposition

The C5A reference-wall convention is: wall = transformed target TCP + 0.260 m * transformed material normal. We intersect the frozen TCP +Z direction with the LOCAL TANGENT PLANE at each correspondingly transformed material point (not an actual curved surface or full spray footprint).

At 6 mm signed workpiece displacement:

| Direction | Max TCP error (mm) | Max abs axial standoff change (mm) | Max absolute local landing offset (mm) | Max same-material normal angle (deg) |
|---|---:|---:|---:|---:|
| tx+ | 6.000 | 6.000 | 7.052 | 0.799 |
| ty+ | 6.000 | 0.025 | 6.047 | 0.799 |
| tz+ | 6.005 | 6.026 | 6.026 | 0.799 |
| tz- | 6.002 | 5.977 | 6.017 | 0.799 |

These results distinguish materially different geometric responses despite the same 6 mm engineering position error. Absolute landing offsets include nominal correspondence mismatch; they are NOT changes relative to a zero-perturbation baseline.

Under pure translation the normal at a FIXED MATERIAL POINT does not rotate; its tool-axis angle is therefore unchanged (max ~0.799 degrees). The original C5A normal diagnostic uses a nearest-polyline-projected normal instead of the fixed-material normal. The original tz+/tz- 10-degree crossings therefore cannot be interpreted as same-material normal-angle crossings. Determining whether nearest-projection indices switch requires additional correspondence instrumentation.

## C. Evidence semantics and limitations

- Legacy 6 mm gate: ENGINEERING_THRESHOLD_CROSSING, retained unchanged as historical regression diagnostic.
- Legacy 10-degree normal gate: ENGINEERING_PROJECTION_DIAGNOSTIC, not material-correspondence angle.
- Historical native FCL zero sampled collisions: ARCHIVAL_SAMPLED_MODEL_ONLY, no CCD claim.
- New geometry: LOCAL_TANGENT_PLANE_PROXY_NOT_MESH. Invalid/facing-away intersections are indeterminate, never PASS.
- Real curved-wall nozzle-axis raycast, spray footprint and coverage: NOT_ASSESSED.
- Coating deposition, thickness, uniformity and validated acceptance tolerance: NOT_ASSESSED.
- Physical registration margin, real physical clearance and continuous collision safety: NOT_AVAILABLE/NOT_ASSESSED.

The original archived C5B1 data is not overwritten or relabeled as a different experiment.

## D. Reproduce

Run from the GitHub repository root, with Python 3.10 or newer:

    python -m src.p2b3_c5b1_r2_attribution \
      --scan-json /path/to/P2B3_C5B1_NATIVE_RESULT.json \
      --c1-fk outputs/p2b3_c1_r0_fk_trace.csv \
      --c1-target outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv \
      --output-json /external/dir/P2B3_C5B1_R2_ATTRIBUTION.json

The script checks both pinned C1 CSV hashes and the published C5B1 JSON SHA256 (20dd1df5a2c4a1239f3a8079d70cb0932c80b9b57b70c94064d3614ea14ee31a), checks the 12-ray native scan structure, prevents overwrites and labels its physical-claim scope. The --allow-hash-mismatch option is exploratory and must never be reported as verified evidence. The authoritative C5B1 JSON remains a separate Drive input, not a generated file in this branch.

Test: python -m unittest discover -s tests -p test_p2b3_c5b1_r2_attribution.py

## Next physical research requirements

1. Instrument C5A polyline nearest-correspondence indices and explain the tz+/tz- projected-normal transitions.
2. Implement raycasting against an independently validated continuous curved-surface model. Distinguish no hit, first hit, occlusion, multiple hits, and curvature effects.
3. Calibrate coating deposition/coverage to real process data, validate held-out thickness before defining process-quality failure and actual process margin.
4. Treat strict continuous collision checking and physical clearance as separate safety evidence.

Context: Chen et al., Journal of Manufacturing Processes 131 (2024), DOI 10.1016/j.jmapro.2024.09.081; Autonomous Trajectory Planning for Spray Painting on Complex Surfaces Based on a Point Cloud Model, PMC10747898. Published process models do not define an FR5-specific acceptance tolerance.

## E. Distribution receipt and code-version caveat (2026-10-13)

- Google Drive offline patch: https://drive.google.com/file/d/1KdC3TbDGRI_o7VqMsJwjTXBLFB4MTypX/view?usp=drivesdk
- Package name: `FR5_C5B1_R2_threshold_attribution_patch.zip`; length: 6064 bytes.
- SHA-256 of locally retained ZIP and Drive raw-file byte readback: `aef7cbfe52817251cc9c11f473c6c27aca50d3f43456aa981301772eb993a03a` (MATCH).
- Verified GitHub branch source blob SHA: `8b17ebdeab99fe1af4f11c05201fea593ea1a44f` (`src/p2b3_c5b1_r2_attribution.py`).
- Verified GitHub branch test blob SHA: `c28b473d49d2b2ec423434fbe90e74695596b675` (`tests/test_p2b3_c5b1_r2_attribution.py`).
- Distribution caveat: the ZIP contains an independently packaged implementation and tests; their file blobs differ from the GitHub branch. This is **not a byte-identical GitHub source snapshot**. For a single source of truth, use the versioned GitHub branch and its documentation; use the ZIP as an independently tested offline patch, not as proof of source-byte parity.
- In-session ZIP-source targeted tests: 5 passed; the GitHub source was inspected, not independently executed in ROS 2 / MoveIt2 here. Do not infer native replay, physical coating validation, CCD, or laboratory acceptance from this receipt.

The historical C5B1 source inputs and original Google Drive native JSON/CSV/report remain unchanged.
