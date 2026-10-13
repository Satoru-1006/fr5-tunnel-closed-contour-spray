# FR5 C5B1-R3-R2 — Bounded raycast semantics and frozen-authority replay

Date: 2026-10-13. Repository: `Satoru-1006/fr5-tunnel-closed-contour-spray`.
Project identity: **FAIRINO FR5**, not ESTUN. This release repairs C1 reference-surface raycast semantics, **not** physical process or safety margins.

## A. Source and baseline authority

Frozen inputs (all bytes verified as SHA-256):

| Input | SHA-256 |
|---|---|
| C1 `p2b3_c1_r0_fk_trace.csv` (181 FK samples) | `ed8cffa87dae845df0dcc3cf50f88225e3f72b1fa425acce6def08e7ee60acee` |
| C1 `open_arch_tcp_poses_base_link.csv` (181 targets/normals) | `8a7d4a22c9e701bf840da255e1619e81b4a29bac4183528ad76fc918e16fbde7` |
| C5B1 `P2B3_C5B1_NATIVE_RESULT.json` (12-ray native scan archive) | `20dd1df5a2c4a1239f3a8079d70cb0932c80b9b57b70c94064d3614ea14ee31a` |

**File-transport precision:** the two GitHub C1 CSVs have LF display text; the frozen SHA-256 identities refer to CRLF byte serialization. Reconstituting the same text with CRLF recovers the original checksums exactly. No numerical values were changed. This must be documented in future reproductions; hashing GitHub's normalized LF content directly is **not** proof of C1 data changes.

Baseline R3 `src/p2b3_c5b1_r3_mesh.py` was executed twice, without `--allow-hash-mismatch`, on the three byte-verified inputs. Both runs produced JSON SHA-256 `fd035d10987fb438fbebdee741aae800894d8c5c023c6d34180fecbe532f04ab`. Original 8/8 synthetic tests passed. This establishes the historical baseline replay, not its raycasting validity.

## B. Confirmed bugs and R3-R2 repair

### B1. Shared-edge position/normal semantics

Previous R3 treated `HIT_FACET_EDGE_AMBIGUOUS_NORMAL` as a complete metric, even when two adjacent reference facets could have distinct geometric normals. R3-R2 groups duplicate coincident hits **only** for adjacent facets at the shared cross-section endpoint; each group retains **all** geometric and material-normal candidates and their angle interval. It reports two independent statuses:

- `position_certainty`: `COMPLETE` only if all 181 reference-surface intersections have a determinate forward first hit;
- `normal_certainty`: `COMPLETE` only if all those hits also have a unique, determinate **facet geometric normal** under declared numerical tolerances.

Thus a shared-edge point can be position-determinate but normal-ambiguous. A synthetic kinked 181-waypoint fixture confirms this exact distinction. The facet geometric normal and C1 interpolated material normal are **different outputs**.

### B2. Irrelevant supporting-plane coplanarity

Previous R3 set a global `degenerate_parallel` flag when a ray lay in any facet's **infinite supporting plane**; an irrelevant far-away facet could suppress an earlier valid first hit. R3-R2 instead clips the ray to each **bounded segment × finite Y slab** to obtain the *actual coplanar ray--surface overlap interval*. An overlap wholly behind the ray or strictly beyond an already established closer point hit does not invalidate that first hit. Overlap at/before a candidate first hit remains `INDETERMINATE_COPLANAR_FINITE_OVERLAP` and cannot silently pass.

### B3. Topology-aware hit identity and tolerances

A plain `|t_i-t_j| < epsilon` is insufficient to merge arbitrary near hits. R3-R2 merges adjacent-edge duplicates only when their facet segment indexes are adjacent, both are at cross-section endpoints and ray distances agree within a declared numerical tolerance. Nonadjacent near hits remain distinct and are explicitly reported. Extra groups beyond the first are retained as an occlusion count and distances. Declared tolerances are purely for **floating-point geometry classification**, not physical FR5 precision or spray tolerance. This is a project-specific analytic solver; it is **not claimed to implement** Woop et al.'s full watertight ray/triangle algorithm.

## C. Formal R3-R2 offline authority replay

Two independently invoked CLI executions with strict hash gates generated identical JSON bytes:

`b6078b5c62617fbbe7e989d2507ab03a968f39e8f3d808b6f1de967a12747def` (SHA-256).

| Probe | Hit positions / 181 | Position completeness | Normal completeness | Unhit waypoint IDs |
|---|---:|---|---|---|
| baseline | 180 | PARTIAL_OR_INDETERMINATE | PARTIAL_OR_AMBIGUOUS | `[0]` |
| `tz−` 14.599609375 mm | 181 | COMPLETE | COMPLETE | `[]` |
| `tz−` 14.609375 mm | 181 | COMPLETE | COMPLETE | `[]` |
| `tz+` 15.595703125 mm | 179 | PARTIAL_OR_INDETERMINATE | PARTIAL_OR_AMBIGUOUS | `[0,180]` |
| `tz+` 15.60546875 mm | 179 | PARTIAL_OR_INDETERMINATE | PARTIAL_OR_AMBIGUOUS | `[0,180]` |

For these actual frozen-data probes, the new code identifies **no adjacent-facet ambiguous normal hits**, but the *synthetic shared-edge stress cases* do exercise and reject the previous semantic defect. A baseline absence of ambiguity is not proof that all possible uncertainties have been exhausted. Upper/lower normal metrics distinguish polygon facet normals versus C1 material-normal interpolation, and maxima exclude unhit waypoints.

**Do not read `COMPLETE` as a physical coating pass or a continuous motion safety pass.** It denotes only the coverage of discrete forward ray intersections on the frozen polygonal C1 reference strip at one studied offset.

## D. Verification and remaining issues

- **Baseline R3 tests:** 8 PASS; **new R3-R2 adversarial geometry tests:** 15 PASS; **R2 prior tests:** 5 PASS, total **28 PASS**.
- Three authority-input hashes exact; R3 baseline double-replay and repaired R3-R2 double-replay byte identical.
- No IK, Ruckig, trajectory, C4 collision shape or historical C5B1 scanner inputs modified; no new native MoveIt/FCL run.
- R3-R2 models **180 finite extruded reference facets**, not measured as-built tunnel geometry, CAD NURBS, or true coating footprint/thickness. `PHYSICAL_COATING_QUALITY=NOT_ASSESSED`; `CCD=NOT_ASSESSED`.
- Numerical tolerances are declared and tested on edge/coplanar/small-angle adversarial fixtures, **not independently certified watertight**. More exhaustive mesh-scale, grazing and adversarial property testing is a valid follow-up.
- The R3-R1 projection switching explanation is a *different* nearest-point diagnostic; R3-R2 does not rewrite it or alter its archival gate.
- The recovered CRLF file bytes are packaged solely for reproducibility; unmodified original inputs remain the authority.

## E. Clean-room reproduction

Extract the self-contained `FR5_C5B1_R3_R2_minimal_repro.zip`, then from its root run:

```bash
python -m unittest discover -s tests -p 'test_p2b3_c5b1_*.py'
python -m src.p2b3_c5b1_r3r2_robust_raycast \
  --c1-fk inputs/p2b3_c1_r0_fk_trace.csv \
  --c1-target inputs/open_arch_tcp_poses_base_link.csv \
  --c5b1-scan inputs/P2B3_C5B1_NATIVE_RESULT.json \
  --output-json /tmp/fr5_r3r2_output_1.json
python -m src.p2b3_c5b1_r3r2_robust_raycast \
  --c1-fk inputs/p2b3_c1_r0_fk_trace.csv \
  --c1-target inputs/open_arch_tcp_poses_base_link.csv \
  --c5b1-scan inputs/P2B3_C5B1_NATIVE_RESULT.json \
  --output-json /tmp/fr5_r3r2_output_2.json
sha256sum /tmp/fr5_r3r2_output_?.json
```

Results refuse output overwrites and never have a hash bypass switch. `R3R2_RESULT.json` in the release is the verified first replay, duplicated checksums prove a deterministic run in this environment. For native ROS 2 / MoveIt2 acceptance, a separate explicitly named execution remains required.

## F. External literature context

Woop, Benthin & Wald (2013), *Watertight Ray/Triangle Intersection*, JCGT, https://jcgt.org/published/0002/01/05/ — motivates careful treatment of shared mesh edges and vertices, **not** a certification of the project's independent quad implementation.

CGAL Linear Geometry Kernel `intersection()` reference: https://doc.cgal.org/latest/Kernel_23/group__intersection__linear__grp.html — differentiates point versus segment-valued geometric intersections. It motivates reporting actual coplanar overlap intervals, rather than treating any infinite-plane coincidence as one invalid point.
