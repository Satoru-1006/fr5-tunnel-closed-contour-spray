# FR5 C5B1 R3-R1 — Projection switch attribution and discretization control

**Date:** 2026-10-13. **Project:** FAIRINO FR5 / open horseshoe contour spray. **Status:** `OFFLINE_C1_REASSOCIATION_DIAGNOSTICS`; neither paint quality nor physical safety assessed.

## Source identity and scientific scope

- Baseline GitHub branch: `codex/fr5-c5b1-r3-mesh-raycast-20261013`; source of original `_project_to_polyline_with_normals()` is `ros2_moveit_bridge/plan_closed_contour_moveit.py` (open polyline, strict `<` for nearest squared distance).
- Frozen C1 FK `outputs/p2b3_c1_r0_fk_trace.csv`: SHA-256 `ed8cffa87dae845df0dcc3cf50f88225e3f72b1fa425acce6def08e7ee60acee`.
- Frozen C1 reference and normals `outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv`: SHA-256 `8a7d4a22c9e701bf840da255e1619e81b4a29bac4183528ad76fc918e16fbde7`.
- C5B1 native JSON from Drive: SHA-256 `20dd1df5a2c4a1239f3a8079d70cb0932c80b9b57b70c94064d3614ea14ee31a`.
- No change to C1 `q(t)`, C4 collision scene, historic 6 mm diagnostic gate, or native C5B1 results. The following results are independent offline recomputations from the connected GitHub C1 CSV contents, not a newly rerun MoveIt/FCL trace. Local Python module's synthetic 8-unit-test suite passed, but a byte-verified *full historical-input CLI run* remains unexecuted in this session.

## Mechanism — two different switches

The historic normal diagnostic is `angle(frozen_tool_Z[i], n_nearest_segment_after_delta[i])` **not** the fixed-material normal angle. Pure `tz` registration does not rotate a fixed material normal. It changes which material position the FK point projects to on the moved reference polyline. If two distinct interior feet have equal or near-equal distances, their interpolated normals may disagree; strict first-minimum selection can switch abruptly. A separate global maximum over 181 waypoints may also change waypoint.

At **`tz−` waypoint 11**, the competing parent segments are **14 and 15**. From the exact C1 target and FK CSV source fields:

| Positive disturbance magnitude | Winning segment | Winner angle | Winner-vs-runner distance gap | Candidate-normal separation |
|---|---:|---:|---:|---:|
| 14.599609375 mm | 14 | 9.742472° | 0.296288 µm | 2.057055° |
| 14.604492188 mm | 14 | 9.743278° | 0.078509 µm | 2.057027° |
| 14.609375000 mm | 15 | 11.801082° | 0.139271 µm | 2.056998° |

**Pair-equal-distance root (signed base `tz`): `−0.0146062524253 m`.** Evaluating at ±1 nanometer either side of that root changes the winning segment **15→14** in signed-`tz` increasing order. The angle's one-sided jump magnitude is **about 2.057°**. The critical max-angle waypoint switches from 12 to 11 at the historical boundary; the *primary abruptness* comes from the within-waypoint 11 nearest-segment reassociation.

At **`tz+` waypoint 16**, segments **12 and 11** compete; equal-distance signed root **`+0.0155978721799 m`**, candidate-normal difference about **2.01426°**; the winner switches from segment 12 (~9.709°) to 11 (~11.723°) at that root. This accounts for the positive-direction diagnostic's abrupt 10° crossing mechanism.

The near-tie gap threshold of `1e-6 m` in the new code is an **investigatory diagnostic choice** only. Its scale is not a validated numerical error bound or FR5 registration tolerance. The exact root is determined by the historical discrete reference model, **not a physical failure margin**. Note: C5A original project passed FK from its native evaluator, whereas this study consumes archived C1 FK. Agreement is numerical/mechanistic, not bit-identical native replay.

## Discretization control and limitations

At waypoint 11, `tz−` historical endpoints:

| Geometry representation | Angle at 14.609375 mm | Angle at 14.599609 mm | Interpretation |
|---|---:|---:|---|
| Original 181 vertices | 11.8011° | 9.7425° | Historical discrete reference |
| Straight subdivision ×2/×4/×8 | Same, to numerical roundoff | Same | Merely splitting same straight edges does **not** change the geometric polyline and is *not* smooth-curve convergence evidence |
| Coarsened stride 2 | 10.7335° | 10.7319° | Different reference polygon; no corresponding jump in this bracket |
| Coarsened stride 3 | 7.5655° | 7.5639° | 10° classification differs |
| Coarsened stride 4 | 8.5657° | 8.5641° | 10° classification differs |

At waypoint 16, `tz+` historical endpoints:

| Representation | Before (15.595703 mm) | After (15.605469 mm) |
|---|---:|---:|
| Original / straight split ×2/×4/×8 | 9.7086° | 11.7244° |
| Coarsened stride 2 | 8.6568° | 8.6584° |
| Coarsened stride 3 | 7.6110° | 7.6126° |
| Coarsened stride 4 | 6.6314° | 6.6330° |

**Conclusion:** the *historical discrete-polyline* 10° crossing is strongly reference-discretization-dependent in these controlled coarsening experiments. Coarsening changes the geometry; it does not prove that a smooth fitted curve would lack any instability. Testing a separately validated smooth/scan/CAD surface is still necessary before asserting process or physical robustness.

## Reproduce with pinned authoritative artifacts

```bash
python -m unittest discover -s tests -p 'test_p2b3_c5b1_r3r1_projection.py'
python -m src.p2b3_c5b1_r3r1_projection \
  --c1-fk outputs/p2b3_c1_r0_fk_trace.csv \
  --c1-target outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv \
  --c5b1-scan /path/to/P2B3_C5B1_NATIVE_RESULT.json \
  --output /separate/output/P2B3_C5B1_R3_R1_REPLAY.json
```

The script validates three SHA-256 identities unless explicitly downgraded by `--allow-hash-mismatch`, rejects overwrites and emits raw candidate gap/normal-angle values and separate sensitivity configurations. No large simulation logs or new native scans are required. For a clean room, include the original C5B1 JSON as a separately pinned input; this minimal supplement does not redistribute the legacy inputs.

**Acceptance status:** fixture tests **8/8 PASS**; independent C1 numerical replay **COMPLETED** using a separate JS implementation; new Python CLI native-data full replay **NOT_RUN**; process coating acceptance **NOT_ASSESSED**; physical clearance and CCD **NOT_ASSESSED**. Do not conflate those gates.

## External research context

Geometrically, multiple equally near points constitute non-unique nearest projection / medial-axis behavior rather than a unique smooth material association (see *Local Criteria for Triangulating General Manifolds*, DOI `10.1007/s00454-022-00431-7`). This literature supports the general mechanism only, not the FR5 numerical results above. FR5 numerical results come from the frozen project CSVs.
