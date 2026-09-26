# STAGE3_FINAL_RELEASE

Status: **FROZEN_AND_CLOSED**

This is the single Stage 3 production release. Canonical update `480` has H1 `7.0893288390132589e-05` and H32 `0.018491006193811731`. The protected checkpoint is `outputs/stage3_h13_d35_permanent_champion/checkpoints/committed_update_480.pt`.

Run the clean metric/release replay with:

```text
python scripts/run_stage3_final_release.py --replay
```

For a fresh full native robot replay (slow, offline, no hardware):

```text
python scripts/run_stage3_final_release.py --replay --run-robot
```

The release uses the 181-point ON-state open-arch pair only. See
`STAGE3_CLOSURE_REPORT.md`, `OPEN_TECHNICAL_DEBT.md`, and
`STAGE4_INPUT_CONTRACT.md` for the final gates and handoff boundary.
