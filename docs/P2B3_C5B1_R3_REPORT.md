# FR5 C5B1-R3 — Nearest-projection mechanism & finite reference-surface raycasting

Date: 2026-10-13. Project: **FAIRINO FR5 tunnel spray**, not ESTUN.

## Source authority and no-mutation contract

The C1 frozen FK (181 points) is `outputs/p2b3_c1_r0_fk_trace.csv`, SHA-256
`ed8cffa87dae845df0dcc3cf50f88225e3f72b1fa425acce6def08e7ee60acee`.
The frozen target normals are `outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv`,
SHA-256 `8a7d4a22c9e701bf840da255e1619e81b4a29bac4183528ad76fc918e16fbde7`.
The C5B1 measured JSON SHA-256 is
`20dd1df5a2c4a1239f3a8079d70cb0932c80b9b57b70c94064d3614ea14ee31a`
(12 rays; 220 native evaluations). Original C1/C4 and archived C5B1 results are immutable.

## 1. Why tz+ / tz- gave the 10° nearest-projection diagnostic boundaries

C5A's `_project_to_polyline_with_normals` loops over consecutive open-wall
polyline segments, computes the nearest Cartesian point for each frozen TCP,
linearly interpolates its endpoint normals, and chooses the minimum squared
distance (strict `<` tie breaking). This is a *spatially re-associated normal*.

The original diagnostic is `angle(tool_Z[i], nearest_interpolated_normal[i, ΔT])`.
It is NOT `angle(tool_Z[i], R(ΔT) @ material_normal[i])`.
For pure translation, the latter is unchanged, while the former can change
because the closest arc parameter and, sometimes, segment change.

Independent full-181-point re-execution of C5A's formula on GitHub's frozen C1
FK and target data confirmed:
- zero displacement max projected angle: 1.081958° at waypoint 13;
- tz- at -14.60 mm: about 9.755° at waypoint 12, nearest segment 15;
- tz- at -14.61 mm: about 11.801° at waypoint 11, nearest segment 15;
- tz+ near +15.60 mm: max projected angle > 10°, near waypoint 16 and
  a nearest segment around 11.

The sharp `tz-` change is largely a **switch of the waypoint attaining the
global maximum** (12→11), not a sudden physical rotation of the spray gun or
workpiece. Within a single waypoint, changing nearest polyline association
moves the evaluated material normal along a curved reference arch.
At -14.60 mm, 153/181 waypoints no longer select their zero-offset nearest
segment. Reproducing C5A's definition explains its **engineering projection
gate** without promoting it to physical spray-quality failure.

The independent calculation uses *archived C1 FK*, not C5B1's fresh native FK:
comparison is semantic/numerical, not a byte-identical native replay.

## 2. Upgrade: actual ray intersection against a finite surface representation

The project currently contains C4 *181 oriented collision boxes*, not a
continuous measured tunnel mesh. R3 therefore reconstructs a different and
more geometrically relevant reference surface from the frozen C1 design:

`wall[i] = C1_target_TCP[i] + 0.260 * C1_material_normal[i]`, i=0..180.
Join neighboring points in the open cross-section by 180 finite straight
segments; extrude along baseline Y by **1.10 m** (C4 authoritative wall
longitudinal extent). Each resulting face is a finite planar quadrilateral;
the union is a continuous piecewise-planar ruled surface. Do **not** close
the arch with a spurious seam or include the C4 floor.

For a BASE_LEFT transform, invert the *ray* into this frozen model's local
coordinates, then calculate all forward intersections with the 180 bounded
surface faces, choose the first positive ray distance, and report the face
(segment) index, fractional surface coordinate, landing point and
normal-interpolation proxy. Unlike C5B1-R2's local tangent-plane intersection,
these intersections respect the **finite cross-section and Y extent**.
Parallel/coincident, zero-hit, multiple-hit and facet-edge cases are
explicitly distinguished, and missing data never becomes PASS.

This is a true intersection with the **C1 polygonal reference surface**, but
NOT a true continuous smooth CAD/NURBS surface, scanned as-built wall, or
physically validated material/paint deposition model. Facet normal estimates
are interpolated C1 material normals, not measured physical normals.

## 3. Independent finite-surface numerical replay (frozen 181-point sources)

| Probe | Forward hits | No hit | Multiple hits | Max abs axial stand-off deviation | Max landing offset | Max ray-to-interpolated-hit-normal angle |
|---|---:|---:|---:|---:|---:|---:|
| zero | 180 | 1 | 0 | 0.025 mm | 3.626 mm | 0.799° |
| tz- 14.599609 mm | 181 | 0 | 0 | 14.581 mm | 14.617 mm | 2.392° |
| tz- 14.609375 mm | 181 | 0 | 0 | 14.590 mm | 14.627 mm | 2.393° |
| tz+ 15.595703 mm | 179 | 2 | 0 | 15.621 mm | 15.621 mm | 2.588° |
| tz+ 15.605469 mm | 179 | 2 | 0 | 15.631 mm | 15.631 mm | 2.590° |

This *independent* replay was performed via a JS implementation consuming
the actual GitHub CSV content, not by running the new Python R3 CLI with its
three authoritative files. Do not claim native or Python historical-input
replay verification based on these numbers.

The contrast is decisive: the original projected-polyline normal diagnostic
crosses 10°, whereas the corresponding first-hit surface-intersection normal
proxy is only around 2.4°/2.6° at these `tz±` boundaries. These are different
geometric correspondences. The zero-hit cases arise from finite model
boundaries and must remain **INDETERMINATE**, not PASS.

The lateral offset in this report is absolute offset from the frozen
same-material reference point, not drift relative to zero perturbation.

## 4. Reproduce on a checkout of the GitHub R3 branch

From the repo root (Python >=3.10; pure standard-library implementation):

```bash
python -m unittest discover -s tests -p 'test_p2b3_c5b1_r3_mesh.py'
python -m src.p2b3_c5b1_r3_mesh \
  --c1-fk outputs/p2b3_c1_r0_fk_trace.csv \
  --c1-target outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv \
  --c5b1-scan /path/to/P2B3_C5B1_NATIVE_RESULT.json \
  --output-json /external/new/path/P2B3_C5B1_R3_FINITE_SURFACE_RESULT.json
```

Exact source hashes are mandatory by default; `--allow-hash-mismatch`
downgrades identity to `UNVERIFIED_EXPLORATORY`. The CLI refuses an existing
output path. The legacy 6 mm position diagnostic stays unchanged.

R3 targeted synthetic/unit tests: **8 PASS** in isolated local Python.
Protected C5B1/C1/C4 suite **not rerun**; native ROS2/FCL **not rerun**;
full CLI **not executed with authoritative C1 CSV files in this session**.
No continuous-collision or physical spraying acceptance certificate.

## 5. Scientific conclusion and next gates

1. The `tz±` 10° crossover is an algorithm-specific nearest-polyline
   association/maximizer effect, not a fixed-material orientation failure.
2. The C1 finite ruled surface offers **real bounded intersection geometry**
   rather than the R2 infinite tangent-plane proxy.
3. The reference mesh exposes edge/no-hit cases that a tangent plane hides.
4. It is still NOT the measured tunnel surface. Acquiring a surveyed mesh,
   physical TCP calibration and paint thickness data is prerequisite to
   experimentally validated coating-quality failure margins.
5. The C4 boxes remain collision proxies; surface intersection alone does
   not establish FCL collision or strict CCD safety.

Literature grounding: Möller & Trumbore (1997), DOI
10.1080/10867651.1997.10487468, ray-triangle intersection;
Tang & Chen (2015), PLOS ONE 10:e0127139, workpiece surface models
and spray-path planning; MoveIt concepts describe collision meshes and
primitive collision shapes as separate geometry representations.
