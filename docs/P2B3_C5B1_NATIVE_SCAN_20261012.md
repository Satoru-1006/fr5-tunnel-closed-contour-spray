# P2-B3-C5B1 native 12-direction scan

Date: 2026-10-12 (Asia/Shanghai)

## Delivery identity

- Publish branch: `codex/fr5-c5b1-native-scan-20261012`
- Source head before execution: `54f60c2403fd7e1c806f4329b5c20d4269a72405`
- Execution code commit: `36324f64a2189a2a5156702f44ee2841e924e142`
- Frozen authority: C1 181-point q(t)/timestamps plus C4 181-object open-arch scene
- Registration: `BASE_LEFT`
- Collision label: `adaptive_discrete_interpolation`

## Native result

The native MoveIt2/FCL scan completed all 12 signed directions with 220 full-scan evaluations. Each evaluation used 858 frozen q(t) samples and passed the native FR5/FCL validity gate. The measurement pipeline is `PASS`; the frozen robot baseline is `FINDINGS_REPORTED`.

- Robot-world sampled collision brackets: 0
- Self-collision sampled brackets: 0
- Position-diagnostic brackets: 12/12 rays
- Normal-diagnostic brackets: 2/12 rays (`tz-`, `tz+`)
- Unresolved intervals: 0

The diagnostic findings are retained as Stage 4B targets. They were not removed or optimized away. Strict continuous collision detection, Bullet CCD, physical clearance, and global-minimum failure proofs remain unavailable/not proven.

## Verification receipts

- C5B1 tests: 20 passed
- Protected 80-test regression: 80 passed
- Frozen-input gate: 8 pinned files, C1 181 waypoints, C4 181 unique base_link objects, `BASE_LEFT` registration
- Native known-answer FCL smoke: passed for separated and overlapping geometry
- Native FR5 smoke: 7 collision-geometry links, 858 samples, valid robot-world/self distance samples

## External artifacts

The minimal result set is generated outside the repository:

- `D:/c5b1_native_scan_20261012_final/P2B3_C5B1_NATIVE_RESULT.json` — SHA-256 `20dd1df5a2c4a1239f3a8079d70cb0932c80b9b57b70c94064d3614ea14ee31a`
- `D:/c5b1_native_scan_20261012_final/P2B3_C5B1_12_AXIS_MARGIN_PROFILE.csv` — SHA-256 `6571c1d53b21bec7618dbbad65123bf36e19534d0f006bf267d47c62565fdbf3`
- `D:/c5b1_native_scan_20261012_final/P2B3_C5B1_NATIVE_REPORT.md` — SHA-256 `5c5c0d9745d1053470fac64e657117abd42236884904a230da6e20feb42bd6c4`

Reproduction used WSL `Ubuntu-24.04-D`, ROS 2 Jazzy/MoveIt2, the authenticated FAIRINO vendor runtime, and the native binary built in `D:/c5b1_native_build_20261012`. Four explicit ray shards were used for the full scan; scratch and result directories remained outside the repository.
