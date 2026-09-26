# GitHub project scope and provenance

This branch packages the public FAIRINO FR5 tunnel-spray research project from the original planner release through the Stage 3/D41 and Stage 4/D46 evidence, P1 process-aware stress model, and P2-A axis-wise margin characterization. The authoritative Stage 0/1 input remains the 181-point ON-state open-arch pair in `outputs/internal_wiper_moveit_inputs/`.

## Included

- FR5 planner, MoveIt2/ROS2 bridge, stage tooling, tests, and project-specific execution guidance.
- Frozen Stage 3 release metadata and its canonical checkpoint.
- The D41 run referenced by D46 authentication, its strict replay, and the final D41 summary.
- D46 final scorecards, acceptance set, case result tables, taxonomy, ranked bottlenecks, and the complete captured raw D46 measurement tree in `D46_RAW_MEASUREMENTS_V1.zip`.
- The selected Stage 4B through Stage 5 closure records, plus the P1/P2-A source, tests, and recorded P2-A result.

To restore the raw D46 measurement tree at the paths used by the tools, run from the repository root:

```powershell
Expand-Archive -LiteralPath outputs/D46_STAGE4A_SYSTEM_BASELINE_V1/D46_RAW_MEASUREMENTS_V1.zip -DestinationPath .
```

The archive contains the captured D46 cases, FK trace, native outputs, and native replays. The regular Git tree keeps the small D46 identity and scorecard files at their expected paths.

## Excluded

No ESTUN/iER8 source, configuration, result, 720-point data, unrelated repository, local scratch/temp tree, duplicate handoff archive, build product, cache, or vendor source is included. The project keeps `adaptive_discrete_interpolation` distinct from strict continuous collision detection. A missing CCD, calibrated uncertainty, coating-physics model, or hardware run is not promoted to a pass.

## P2-A provenance boundary

`outputs/p2a_axiswise_robustness_margin.json` is preserved exactly as recorded. It identifies `FAIRINO_FR5`, the 181-point scope, the authenticated Stage 3/D41/D46 inputs, and `working_tree_dirty_at_execution = true`; its `source_commit` is `d67333be9dca7489f8e8c029e385f491570d53df`, the local D8.2-B workspace base. That commit does not identify the FR5 source snapshot published on this branch. The publication commit therefore does not retroactively replace the run's source identity, and this package makes no claim that the recorded P2-A execution used a clean committed FR5 revision.

The P2-A result is a bounded software measurement. Collision transitions are reported as `adaptive_discrete_interpolation`; the result is not a global robustness proof, strict CCD certificate, coating-quality certification, or hardware validation.

## GitHub base

The project branch was built from the verified public repository `Satoru-1006/fr5-tunnel-closed-contour-spray`, starting at `main` commit `8564ecd9101669b8d5568102e34cda8dfbfe4f54`. The publication branch preserves that base; it does not rewrite or update `main`.

## Reproduction boundary

The P2-A module entry point is `python -m src.p2a_axiswise_robustness`. Its native measurements require the project's Windows/WSL ROS 2 Jazzy, MoveIt 2, and FCL environment; this publication step did not rerun the experiment or its tests. The preserved D41 summary records the original Windows worktree path, so its runtime inputs are laid out under the same repository-relative directory here, but a fresh run from a different checkout path may need a narrow path-portability adjustment outside the frozen evidence files.
