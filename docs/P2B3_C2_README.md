# P2-B3-C2 — C1 Robustness Transfer Closure

- **Project:** FAIRINO FR5 tunnel complex-surface spraying
- **Stage status:** `P2B3_C2_AXISWISE_RESCAN_COMPLETE_WITH_MEASURED_BASELINE_FINDINGS`
- **Measurement pipeline:** `PASS`
- **Frozen robot baseline performance:** `MEASURED_FAILURES_PRESENT` under measured perturbations
- **Native measurement run:** 2026-09-28, 3,103.969 s, no failed batches
- **C2 execution code:** `bb86d8f86a0bdb04948fee73155e78c47908c95a`
- **C2 taxonomy postprocessor and regression repair:** `001843d3828efd86755267110f751e683e95fb62`
- **FAIRINO source:** `60755d44d521a5ad6bee8494cc19522f8801aa20`

## Scope and campaign

C2 rescanned the 181-point C1 post-Ruckig open-arch nominal, with the existing ON-state Stage 3 definition. It measured the 24 evaluator-supported directions: 12 joint-state directions, 6 TCP translations, and 6 TCP rotations. The Stage 3 release, checkpoint, canonical configuration, C1 trajectory, acceptance thresholds, and tested cases were held fixed. No trajectory or robot parameter was tuned to improve a result.

All 24 supported directions completed with zero `UNKNOWN` directions. The scan evaluated 1,368 coarse candidate requests and retained 1,606 candidate observations after refinement. There were 921 fresh native cases, 653 exact within-campaign q-state reuses, 32 direct joint-limit failures, 26 native evaluator batches, 32 callback chunks, and zero failed batches. The campaign used a 64-batch maximum and a 12-hour wall-time cap; no unbounded retry was used. Failure-mode counts below overlap because one candidate can have multiple failure modes.

The existing C1 relaxed geometry gate accepted all 181 targets; 173 also met the tight solver tolerances. Fresh MoveIt2 FK matched the saved C1 trace with maximum position delta `0 m`, maximum tool-axis delta `8.92e-16 rad`, and maximum projected surface-normal delta `5.08e-15`. The maximum TCP path deviation in this cross-check was `1.467e-6 m` against the existing `0.006 m` path gate.

## Robustness transfer

The first refined failure frontiers were:

| Family | Controlling direction | First failure | Location | Failure mode |
|---|---|---:|---:|---|
| Joint state | `joint:j3:positive` | `0.00721255 rad` | WP180 | `TERMINAL_POSITION_ERROR` |
| TCP translation | `tcp_tcp_translation:x:positive` | `0.00399609 m` | WP180 | `TERMINAL_POSITION_ERROR` |
| TCP rotation | `tcp_tcp_rotation:x:positive` | `0.15746029 rad` | WP172 | `SPRAY_AXIS_NORMAL_ERROR` |

Compared with the old P2-B2 landscape, 23 of 24 directions were consistent within the configured refinement tolerance. The `tcp_tcp_rotation:y:positive` direction was classified as both `MARGIN_SHIFTED` and `CRITICAL_LOCATION_CHANGED`: its measured first-failure margin changed from `0.16513019 rad` at WP16 to `0.16001692 rad` at WP1, with the same `SPRAY_AXIS_NORMAL_ERROR` mode. No axis showed a measured non-monotonic region or multiple failure regions. WP0–16 did not become a family controller.

The 181-row nominal still carries the C1 WP179→WP180 local timing anomaly: `0.00414384 m/s` against the existing `0.003 m/s ±5%` diagnostic band. C2 counted 1,547 out-of-band stress candidates. There is no validated deposition model, so this is not a coating-quality finding.

## Stage 4B failure taxonomy

The 1,606 candidate observations form the denominator for each category; category counts are not mutually exclusive. The ranked findings are measured on perturbed candidates and do not mean that the unperturbed C1 nominal failed:

| Rank | Finding | Cases | Worst category-specific measurement | Representative case |
|---:|---|---:|---|---|
| 1 | `ENVIRONMENT_COLLISION` | 109 / 1,606 | minimum reported environment clearance `-0.583012 m` | `joint_j2_positive__coarse__0060__2.7158522971386492`, WP90 |
| 2 | `ROBOT_WORLD_TRANSITION_COLLISION` | 109 / 1,606 | 180 endpoint-pair collision reports | `joint_j2_positive__coarse__0061__2.7611165020909603`, WP90 |
| 3 | `SELF_COLLISION` | 96 / 1,606 | minimum reported self-clearance `-0.040854 m` | `joint_j3_negative__coarse__0063__5.0518773059087829`, WP118 |

The transition finding is a bounded native two-state endpoint-pair diagnostic. Collision labels use `adaptive_discrete_interpolation`; this campaign did not establish strict continuous collision detection. The result taxonomy records affected candidate IDs, families, axis IDs, critical locations, severity, reproducibility, safety-impact boundary, likely subsystem, and measurement confidence for Stage 4B.

### Measurement reporting repair

The first report aggregator selected the numerically largest number from all failure-margin fields, mixing units and sometimes attributing a different category’s measurement to the finding. The postprocessor now selects only the metric assigned to that category, with explicit units and a defined min/max rule. A synthetic known-answer regression checks environment clearance and terminal position error against deliberately larger unrelated values. The existing 1,606 candidate records were reprocessed from the saved campaign result; no native measurement was rerun. The native execution commit remains `bb86d8f...`; postprocessing code is recorded separately as `001843d...` in the result.

## Stage 4B targets and limits

Stage 4B should start with the ranked environment-contact, transition endpoint-pair, and self-contact sensitivities while keeping the C1 nominal and the measured acceptance set as its regression barrier. The robot weaknesses stay unmodified in C2. The 18 base-transform and process perturbation axes remain `NOT_AVAILABLE`: the evaluator does not propagate base/workpiece transforms or validated surface-offset and deposition physics. They need an evaluator extension before those domains can be measured.

The available evidence is a software-shadow measurement of the frozen FR5 model and evaluator. `STRICT_CONTINUOUS_CCD=NOT_AVAILABLE`, hardware validation is `NOT_RUN`, and coating quality is `NO`. Candidate Ruckig retiming was `NOT_RUN`; derivatives were audited at the frozen C1 timestamps. Physical singularity criteria, a clearance acceptance threshold, vendor-certified dynamic limits, and calibrated TCP uncertainty remain unresolved. These gaps are not passes.

The D46 Stage 3 release and checkpoint were identity-checked after the campaign: release-manifest SHA-256 `83f8a659df86f57165955bba4b962569e624a70bc903990a8d712e1a2eb276fe`; canonical checkpoint SHA-256 `37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742`. The full identity record, model and binary hashes, coverage gaps, thresholds, and candidate-level evidence are in the result JSON.

## Artifacts and reproduction

- `outputs/p2b3_c2_result.json` contains the complete axis results, all candidate observations, transfer classifications, corrected Stage 4B taxonomy, and run identity.
- `outputs/p2b3_c2_axis_margin_profile.csv` contains the 24 direction-level margin rows.
- `outputs/p2b3_c2_transfer_comparison.csv` contains the 24 old-to-C1 transfer comparisons.

Run the targeted regression suite from the C2 worktree:

```powershell
python scripts/run_p2b3_c2_robustness.py --mode test
```

The final suite passed **114 tests**. The native campaign was executed from a clean checkout at the exact C2 execution commit. For a full campaign replay, check out `bb86d8f86a0bdb04948fee73155e78c47908c95a` on the C2 branch and supply a new, nonexistent scratch directory and campaign-result path. The original runtime inputs were:

```powershell
python scripts/run_p2b3_c2_robustness.py --mode campaign `
  --scratch 'D:\fr5-p2b3-c2-20260928\c2_campaign_replay' `
  --campaign-output 'D:\fr5-p2b3-c2-20260928\c2_campaign_replay_result.json' `
  --underlay-install 'D:\fr5-p2b3-c2-20260928\c1_r0_replay\fairino_install' `
  --overlay-install 'D:\fr5-p2b1-scratch-20260926\install_final3' `
  --native-binary 'D:\fr5-p2b1-scratch-20260926\install_final3\lib\stage3_h13_d41_native\stage3_h13_d41_native' `
  --fk-binary 'D:\fr5-p2b1-scratch-20260926\install_final3\lib\stage4a_fk\stage4a_fk' `
  --fairino-source-checkout 'D:\fr5-p2b3-c2-20260928\fairino_source' `
  --d46-root 'D:\robotfucker\outputs\D46_STAGE4A_SYSTEM_BASELINE_V1' `
  --native-timeout-seconds 7200 --max-native-batches 64 `
  --max-campaign-hours 12 --distro Ubuntu-24.04-D
```

To regenerate the canonical summaries from saved candidate evidence, use the postprocessing commit and the saved campaign-result JSON:

```powershell
python scripts/run_p2b3_c2_robustness.py --mode publish `
  --campaign-result 'D:\fr5-p2b3-c2-20260928\c2_campaign_r2\campaign_result.json'
```
