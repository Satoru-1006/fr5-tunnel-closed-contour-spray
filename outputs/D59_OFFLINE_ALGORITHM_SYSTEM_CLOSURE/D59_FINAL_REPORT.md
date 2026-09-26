# D59 — Offline algorithm/system closure

## Decision

- `D59_TASK_STATUS`: `PASS_FOR_MEASURED_OFFLINE_DOMAINS_NO_PROMOTION`
- `D59_SCOPE`: `OFFLINE_ALGORITHM_ONLY`
- `PROMOTION`: `NO_PROMOTION`
- `CURRENT_CANONICAL`: `outputs/D47_STAGE4B_ROLLING_CHAMPION_V1/PROMOTED/B3`
- `CURRENT_PROTECTED_FLOOR`: `c4_clearance_adversarial0101_amp005_consistent_0025`
- `FINAL_VERIFIED_CHAMPION`: `c4_clearance_adversarial0101_amp005_consistent_0025`

D59 found and corrected a real D57→D58 scientific routing mismatch: D58 labelled its second candidate `adversarial_0101` while consuming the `auto_1_adversarial_0100.csv` trajectory. The actual `auto_1_adversarial_0101.csv` path was rerun through native MoveIt2 and all affected measurements were recomputed. No protected D56/D57/D58 scientific artifact was overwritten.

Claim fence: the D59 zero-collision/zero-unresolved observation below is a
historical, scoped result for two correctly routed candidates under the
conservative FK-aware software model. It is not a complete exact external
articulated `FK(q(t))` self-CCD proof; that backend remained unavailable and
the later D61-D64 lineage supersedes D59 as the current continuous-collision
authority.

## Closure results

- Correctly routed auto0: native PASS, 7005 states, duration `294.019376723` s.
- Correctly routed auto1: native PASS, 7030 states, duration `257.193538084` s.
- D58 state-consistent `.01` reference for correctly routed auto0: `561.177833886` s. D59 auto0 `.025` is `294.019376723` s, a `47.607%` reduction. The D58 auto1 timing value is not used as an apples-to-apples comparator because D58 consumed the misrouted auto0 source under the auto1 label; D59 actual auto1 is `257.193538084` s and matches its own D57 source duration.
- Profile.j: auto0 `7004/7004`, auto1 `7029/7029`, analytic jerk maximum `8 rad/s³`, no invalid-input or other errors.
- FK-aware articulated certificate (historical scoped model result): both candidates have `0` collision regions and `0` unresolved regions over all required intervals; minimum conservative lower bounds are `0.004214082` m and `0.006444079` m. This does not establish exact external articulated `FK(q(t))` self-CCD.
- Independent FCL Route D: all intervals completed, 0 continuous contacts, 0 API errors for both candidates. This remains a rigid endpoint-sweep cross-check, not a claim of exact nonlinear articulated CCD.
- Pinocchio RNEA/ABA: PASS, finite model, correct `j1..j6` order, nominal candidate peak model torque approximately `50.925584` N·m; no hardware torque claim.
- Full-chain robustness remeasurement: `12/12` cases passed current native/FK/geometry/task hard gates. The protected D56 12-case conservative certificate is retained; D59 independently recertified the two corrected finalists. Perturbation findings are labelled `SOFTWARE_ROBUSTNESS_STRESS_TEST`.
- Deterministic clean replay: byte-identical native CSV for both D59 finalists.
- Focused regression: 12 tests passed; new D59 tools compile successfully.

## Decision table

The machine-readable full table is in `D59_FINAL_STATUS.json`. Cartesian values are reported as `TASK_SPACE_PATH_FIDELITY`, not physical TCP accuracy.

| Candidate | Native/q path | Duration (s) | Model self-clearance (m) | Conservative articulated self-CCD | Robot-world CCD | Path fidelity max/P95 (mm) | Singularity | Model torque / slew (N·m / N·m·s⁻¹) | Robustness | Promotion |
|---|---|---:|---:|---|---|---:|---|---:|---|---|
| Protected floor auto0 | PASS / preserved | 294.020666 | 0.016623745 | PASS model certificate | PASS | 4.712/3.636 | sigma 0.00119 | 50.932 / 2.521 | 12/12 | retain |
| Protected floor auto1 | PASS / preserved | 257.194080 | 0.016622086 | PASS model certificate | PASS | 6.359/5.347 | sigma 0.0022 | 50.773 / 2.314 | 12/12 | retain |
| D59 `.025` auto0 | PASS / preserved | 294.019377 | 0.016623745 | SCOPED PASS (model), 0 unresolved | PASS | 4.712/3.636 | sigma 0.00124, cond 1.5e+03 | 50.926 / 2.521 | 12/12 | no promotion |
| D59 `.025` auto1 corrected | PASS / preserved | 257.193538 | 0.016622086 | SCOPED PASS (model), 0 unresolved | PASS | 6.359/5.347 | sigma 0.00224, cond 830 | 50.773 / 2.314 | 12/12 | no promotion |

## Remaining boundary

D59 closes the remaining software-measurable D58 gaps for the accepted conservative model chain, but does not relabel a conservative proof as a theorem-level exact external articulated CCD implementation. Tesseract Robotics/BulletCast was investigated and is unavailable in this environment. Hardware calibration, physical clearance, hardware tracking, actuator current/thermal limits, and universal task/singularity thresholds remain out of scope for this offline project.
