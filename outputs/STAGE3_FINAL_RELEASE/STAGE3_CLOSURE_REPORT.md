# Stage 3 D45 closure report

TASK_STATUS: PASS
STAGE_3_FINAL_CLOSURE: PASS
STAGE_3_STATUS: FROZEN_AND_CLOSED

## Final canonical

- Checkpoint: `outputs/stage3_h13_d35_permanent_champion/checkpoints/committed_update_480.pt`
- SHA-256: `37645eea5aef8ce2fe7084c83f35429a7a13fbe3e9612eb84417ccce6cba9742`
- Update: `480`
- H1: `7.0893288390132589e-05`
- H32: `0.018491006193811731`
- H32 limit: `0.01856902565856056`; margin: `7.8019464748828316e-05`
- D44 canonical promotions: `0`; canonical mutation: `NO`

## Required gates

| Gate | Result |
|---|---|
| A authoritative chain | PASS |
| B end-to-end execution | PASS |
| C checkpoint/model correctness | PASS |
| D H1-H32 metrics | PASS; deterministic repeated replay |
| E canonical identity | PASS |
| F robot-motion chain | PASS; D41 strict replay and 37/37 robustness |
| G material correctness defects | PASS; all discovered release gaps fixed |
| H accepted-repair regressions | 0 |
| I clean replay | PASS |
| J single release | PASS |

## Accepted D45 repairs

1. `D36 summary missing after successful evaluation` → the D36 runner returned PASS but did not persist its result summary → added the versioned `D36_SUMMARY.json` write at the existing evaluation boundary → D36 reporting only; no model, metric, or optimizer behavior changed → D36 rerun completed with 5 checkpoints, 230400 ledger episodes, 225 summary rows, and focused tests remained green.
2. `D45 robot gate counted the nominal control inside the robustness denominator` → D41's native robustness file includes one nominal control plus 37 robustness cases → filtered the explicitly named nominal row before applying the 37/37 gate → release validation only; no robot artifact or result changed → the fresh native run remains 37/37 PASS.
3. `D45 release path lacked one executable closure entrypoint` → historical evidence was distributed across D35/D40/D41/D44 → added this fail-closed release gate and `scripts/run_stage3_final_release.py` → release packaging and replay validation only → canonical SHA/H1/H32 unchanged, D41 fresh replay PASS, and adversarial checks PASS.

## Historical accounting

- PROVEN AND RETAINED: update-480 canonical checkpoint; D39 stable prior as retained input; D40 locked task-space baseline; D41 native offline certification.
- SUPERSEDED: D37/D38 incomplete robot routes and D39 pre-D40 partial system status, superseded by the locked D40/D41 path while their evidence remains preserved.
- RESEARCH ASSET: D43/D44 shadows, response surfaces, unsafe candidates, and non-promoted ledgers.
- INVALIDATED: NONE.
- DEFERRED TO STAGE 4: system-level stability/smoothness/accuracy robustness expansion and the H1 stretch target.

## Scope and collision semantics

The release covers only the 181-point Stage 0/1 ON-state open-arch pair. OFF states, reorientation, approach, retreat, GNN, PPO, LSTM, Transformer, and closed-contour transitions are excluded. Collision results are labelled `adaptive_discrete_interpolation`; Bullet CCD and clearance are `not_available`/JSON `null` in this release contract. Candidate-specific D43/D44 robot validity is not inferred.

## Final report fields

TASK_STATUS: `PASS`
STAGE_3_FINAL_CLOSURE: `PASS`
FINAL_CANONICAL_CHECKPOINT: `outputs/stage3_h13_d35_permanent_champion/checkpoints/committed_update_480.pt`
FINAL_H1: `7.0893288390132589e-05`
FINAL_H32: `0.018491006193811731`
FINAL_ROBOT_CERTIFICATION_STATUS: `PASS`
AUTHORITATIVE_ENTRYPOINT: `scripts/run_stage3_final_release.py`
END_TO_END_REPLAY_STATUS: `PASS`
CLEAN_REPLAY_STATUS: `PASS`
NUMBER_OF_MATERIAL_DEFECTS_FOUND: `3 release-completeness defects`
NUMBER_OF_MATERIAL_DEFECTS_FIXED: `3`
UNRESOLVED_CORRECTNESS_BLOCKERS: `NONE`
REGRESSIONS_INTRODUCED_BY_ACCEPTED_REPAIRS: `0`
HISTORICAL_RESULTS_INVALIDATED: `NONE`
STAGE3_OPEN_TECHNICAL_DEBT: see `OPEN_TECHNICAL_DEBT.md`
STAGE4_INPUT_READY: `YES`

ONE_SENTENCE_VERDICT: Stage 3 is now a single, correct, deterministic, regression-safe offline release ready as the sole Stage 4 starting baseline.
