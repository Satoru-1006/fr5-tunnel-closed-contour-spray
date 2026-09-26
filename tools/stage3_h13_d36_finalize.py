"""Create the compact D36 analysis/report bundle from executed matched results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage3_h13_d36_system_evaluation"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def text(name: str, value: str) -> None:
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def main() -> int:
    rows = list(csv.DictReader((OUT / "D36_CAPABILITY_RESULTS.csv").open(encoding="utf-8")))
    panel = json.loads((OUT / "D36_CHECKPOINT_PANEL.json").read_text(encoding="utf-8"))["checkpoints"]
    updates = [int(item["update"]) for item in panel]
    h1 = {int(item["update"]): float(item["H1"]) for item in panel}

    inventory = {
        "scope": "Stage 0/1 ON-state 181-point open-arch only",
        "capabilities": {
            "frozen_group_split_trajectory_dataset": "AVAILABLE_NOW",
            "GRU_joint_trajectory_predictor": "AVAILABLE_NOW",
            "matched_offline_checkpoint_inference": "AVAILABLE_NOW",
            "held_out_validation_test_generalization_families": "AVAILABLE_NOW",
            "joint_position_sensor_noise_perturbation": "REQUIRES_SMALL_NEW_IMPLEMENTATION",
            "FR5_MoveIt2_PlanningScene_FK_dynamics_post_Ruckig_stack": "AVAILABLE_NOW_BUT_NOT_CONNECTED_TO_CHECKPOINT_OUTPUT_IN_D36",
            "end_to_end_robot_task_simulator": "NOT_AVAILABLE",
            "fine_manipulation_objects_grasp_contact": "NOT_AVAILABLE",
            "camera_or_visual_observations": "NOT_AVAILABLE",
            "failure_recovery_policy": "NOT_AVAILABLE",
            "real_robot_interface_and_measurements": "NOT_AVAILABLE",
            "sim_to_real_evidence": "NOT_AVAILABLE",
            "Bullet_CCD": "NOT_AVAILABLE",
            "clearance_backend": "NOT_AVAILABLE"
        },
        "collision_label_if_executed": "adaptive_discrete_interpolation",
        "clearance_m": None,
        "real_world_evaluation": "NOT_AVAILABLE"
    }
    dump("D36_CAPABILITY_INVENTORY.json", inventory)

    def metric(update: int, split: str, sigma: float, horizon: int) -> float:
        row = next(x for x in rows if int(x["update"]) == update and x["split"] == split and float(x["perturbation_sigma_rad"]) == sigma and int(x["horizon"]) == horizon)
        return float(row["joint_position_rmse_rad"])

    capability = []
    for u in updates:
        capability.append({
            "update": u, "H1": h1[u],
            "nominal_H1_test_rmse_rad": metric(u, "TEST", 0.0, 1),
            "nominal_H32_test_rmse_rad": metric(u, "TEST", 0.0, 32),
            "generalization_H1_rmse_rad": metric(u, "GENERALIZATION", 0.0, 1),
            "generalization_H32_rmse_rad": metric(u, "GENERALIZATION", 0.0, 32),
            "noise_0p01_generalization_H32_rmse_rad": metric(u, "GENERALIZATION", 0.01, 32),
            "fine_manipulation": None, "task_success": None, "recovery": None,
            "collision_safety": None, "real_robot": None
        })
    dump("D36_CAPABILITY_VECTOR.json", {"null_means": "not_available", "checkpoints": capability})

    # Every checkpoint is non-dominated on nominal plus high-noise generalization horizons.
    vectors = {u: [metric(u, "GENERALIZATION", s, h) for s in (0.0, 0.01) for h in (1, 4, 8, 16, 32)] for u in updates}
    nondominated = [u for u in updates if not any(v != u and all(a <= b for a, b in zip(vectors[v], vectors[u])) and any(a < b for a, b in zip(vectors[v], vectors[u])) for v in updates)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for horizon in (1, 8, 16, 32):
        axes[0].plot([h1[u] for u in updates], [metric(u, "GENERALIZATION", 0.0, horizon) for u in updates], "o-", label=f"H{horizon}")
    axes[0].invert_xaxis(); axes[0].set_xlabel("Historical H1 (lower is right)"); axes[0].set_ylabel("Held-out generalization RMSE (rad)"); axes[0].set_title("H1 gains do not transfer monotonically"); axes[0].legend()
    for u in updates:
        axes[1].plot((0.0, 0.002, 0.01), [metric(u, "GENERALIZATION", s, 32) for s in (0.0, 0.002, 0.01)], "o-", label=str(u))
    axes[1].set_xlabel("Joint-position observation noise σ (rad)"); axes[1].set_ylabel("Generalization H32 RMSE (rad)"); axes[1].set_title("Perturbation robustness"); axes[1].legend(title="update")
    fig.savefig(OUT / "D36_H1_CAPABILITY_AND_ROBUSTNESS.png", dpi=180)
    plt.close(fig)

    text("D36_EVALUATION_PROTOCOL.md", f"""
# D36 evaluation protocol

The five-checkpoint panel (updates 413, 424, 450, 469, 480) was serialized before any new D36 outcomes were computed. Each checkpoint received the same 1,024 deterministic family-balanced windows from each frozen H10 `VALIDATION`, `TEST`, and `GENERALIZATION` role, the same perturbation realizations (σ = 0, 0.002, 0.01 rad), and horizons 1, 4, 8, 16, 32.

The measured endpoint quantity is six-joint position RMSE in radians under causal free-running rollout. The perturbation modifies measured history joint positions and the corresponding normalized position channels. A paired episode bootstrap (2,000 resamples) compares update 480 against each historical checkpoint.

This is model-level offline evidence, not end-to-end robot execution. No result is labelled collision-free or robot-task-successful because D36 did not run checkpoint outputs through PlanningScene, FK, dynamics, and post-Ruckig gates. Bullet CCD and clearance are `not_available`; clearance is null.
""")
    text("D36_PARETO_ANALYSIS.md", f"""
# D36 Pareto analysis

All five checkpoints are non-dominated across nominal and σ=0.01 held-out-generalization errors at horizons 1, 4, 8, 16, and 32: {', '.join('update '+str(u) for u in nondominated)}.

Update 480 is best at nominal generalization H1 ({metric(480,'GENERALIZATION',0,1):.6g} rad), but update 413 is best at nominal generalization H32 ({metric(413,'GENERALIZATION',0,32):.6g} rad), and update 424 is best at σ=0.01 generalization H32 ({metric(424,'GENERALIZATION',0.01,32):.6g} rad). A scalar System Champion would hide these reversals, so the system-level selection remains unresolved.
""")
    text("D36_H1_PROXY_ANALYSIS.md", f"""
# D36 H1 proxy analysis

Historical H1 reduction is productive for one-step held-out precision: generalization H1 RMSE falls from {metric(413,'GENERALIZATION',0,1):.6g} rad at update 413 to {metric(480,'GENERALIZATION',0,1):.6g} rad at update 480. The relationship reverses at long horizon: generalization H32 is {metric(413,'GENERALIZATION',0,32):.6g} rad at update 413 versus {metric(480,'GENERALIZATION',0,32):.6g} rad at update 480. Under σ=0.01 noise, update 424 beats update 480 at H32 ({metric(424,'GENERALIZATION',0.01,32):.6g} versus {metric(480,'GENERALIZATION',0.01,32):.6g} rad).

This is evidence of proxy specialization within model-level metrics, but it cannot establish robot-level saturation or Goodhart behavior because task execution, collisions, recovery, and hardware behavior are unavailable. Therefore `H1_PROXY_STATUS = INCONCLUSIVE`, with model-level diminishing returns and reversals observed.
""")
    text("D36_SYSTEM_CHAMPION_DECISION.md", f"""
# D36 System Champion decision

- H1 Champion: update 480.
- System Champion: `PARETO_SET / UNRESOLVED`.
- Same checkpoint: `UNRESOLVED`.
- Pareto set: {', '.join('update '+str(u) for u in nondominated)}.

No single checkpoint can be declared the strongest robot system. The evidence covers only offline trajectory prediction; safety-critical and real-task dimensions are unavailable.
""")
    text("D36_NEXT_BOTTLENECK_RANKING.md", """
# D36 next bottleneck ranking

| Rank | Bottleneck | Evidence | Severity | Confidence | Tractability / cost |
|---:|---|---|---|---|---|
| 1 | Valid end-to-end execution and evaluation bridge | Checkpoint predictions were not passed through MoveIt2 PlanningScene, FK, dynamics, and post-Ruckig; prior certification produced 0/20 valid final trajectories | Critical | High | Medium-high engineering, CPU/ROS runtime |
| 2 | Real task and hardware evidence | No physical robot data, tracking, contact, deposition, or sim-to-real results | Critical | High | High hardware/data cost |
| 3 | Long-horizon stability | Lower H1 does not monotonically reduce held-out H32 error | High | High for model proxy | Medium modeling/data effort |
| 4 | Perturbation robustness | Historical ordering changes under sensor noise | Medium | High for tested perturbation | Medium |
| 5 | Perception/task diversity | Inputs contain no visual/object observations and one open-arch task family | High | High | Major new data/architecture |

The next authorized stage should connect frozen predictors to a valid Stage 0/1 execution/certification evaluation before additional H1 optimization.
""")
    text("D36_RESEARCH_NOTES.md", """
# D36 research notes

The execution used existing frozen grouped splits and introduced one small, deterministic sensor-noise evaluation. No optimizer was constructed, no checkpoint was written, and D35 state was not mutated. The panel lacks a verified ~9e-5 checkpoint; no checkpoint was fabricated to fill that requested region. External literature research was not necessary to interpret this repository-specific blocker.
""")
    text("D36_FINAL_REPORT.md", f"""
# D36 final report

## Decision

D36 completed a matched offline model-capability screen but cannot decide robot capability. H1 improvement transfers to one-step precision, while long-horizon and noisy-condition rankings reverse. Because no evaluated checkpoint completed the required MoveIt2 PlanningScene + FK + dynamics + post-Ruckig execution chain, the only defensible project-level classification is `INCONCLUSIVE`.

## Required summary

TASK_STATUS: COMPLETED_WITH_DECISION_CRITICAL_CAPABILITY_GAP

D35_FROZEN_CHAMPION: update_480  
D35_FROZEN_H1: 7.089328839013259e-05  
D35_FROZEN_H32: 0.01849100619381173  
D35_FROZEN_H32_MARGIN: 7.801946474882832e-05

CHECKPOINTS_EVALUATED: 5 (413, 424, 450, 469, 480)  
TOTAL_EVALUATION_EPISODES: 46,080 matched checkpoint/scenario episodes; 230,400 horizon-level ledger rows  
ROBOT_CAPABILITIES_EVALUATED: offline joint-trajectory prediction precision, held-out family generalization, observation-noise robustness, long-horizon stability  
ROBOT_CAPABILITIES_NOT_AVAILABLE: end-to-end task success, fine manipulation, collision-certified execution, recovery, visual OOD, real hardware, sim-to-real, Bullet CCD, clearance

H1_PROXY_STATUS: INCONCLUSIVE  
H1_CHAMPION: update_480  
SYSTEM_CHAMPION: PARETO_SET / UNRESOLVED  
SAME_CHECKPOINT: UNRESOLVED  
PARETO_NONDOMINATED_CHECKPOINTS: {', '.join('update_'+str(u) for u in nondominated)}

KEY_SYSTEM_LEVEL_FINDING: lower H1 improves local prediction but is insufficient evidence of a better robot and does not monotonically improve long-horizon or perturbed prediction.  
EVIDENCE_OF_DIMINISHING_RETURNS: YES at the model-proxy level; robot level INCONCLUSIVE  
EVIDENCE_OF_CAPABILITY_REVERSAL: YES at the model-proxy level; robot level INCONCLUSIVE  
MOST_IMPORTANT_REVERSAL_IF_ANY: update_413 has better nominal generalization H32 than update_480 despite worse H1.

SHOULD_PROJECT_RESUME_H1_TO_5E-05: NOT_YET_KNOWN  
EXPECTED_VALUE_OF_FURTHER_H1_OPTIMIZATION: uncertain; local gains coexist with long-horizon reversals  
NEXT_SYSTEM_BOTTLENECK: valid end-to-end execution/certification evaluation bridge  
NEXT_STAGE: EXPAND_EVALUATION_BEFORE_DECIDING

SOL_LOW_PRIMARY_USED: YES (runtime identity reported by task configuration, not independently attested)  
SOL_PRIMARY_AGENT_COUNT: 1  
LUNA_MAX_SUBAGENTS_USED: YES  
TOTAL_LUNA_MAX_SUBAGENTS: 3  
MAX_CONCURRENT_LUNA_MAX: 3  
LUNA_MAX_WORKSTREAMS_COMPLETED: 3  
AGENT_ARCHITECTURE_COMPLIANCE: PARTIAL (requested model labels were configured; independent runtime attestation unavailable)  
HASHING_POLICY: MINIMAL / CRITICAL_IDENTITY_ONLY  
REAL_WORLD_EVALUATION: NOT_AVAILABLE  
FINAL_CONFIDENCE: HIGH that the robot-level decision is presently inconclusive; moderate on model-level tradeoff shape  
FIRST_REMAINING_UNCERTAINTY: whether any frozen checkpoint can produce a fully valid 181-point trajectory through PlanningScene, FK, dynamics, and post-Ruckig checks under matched perturbations.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
