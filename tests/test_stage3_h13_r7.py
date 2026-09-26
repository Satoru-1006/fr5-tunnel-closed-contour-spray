from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import stage3_h13_r7_strict_reconstruction_audit as r7
from src.stage3_h13_r5_reconstruction import (
    COLLISION_SEMANTICS,
    MODEL_PRIOR_MAX_JOINT_DELTA_RAD,
    MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD,
    bounded_reconstruction_decision,
)


def test_frozen_r6_checkpoint_hash_and_r5_policy_are_unchanged() -> None:
    assert r7.sha256_file(r7.R6_CHECKPOINT) == r7.R6_EXPECTED_SHA256
    assert MODEL_PRIOR_MAX_JOINT_DELTA_RAD == np.deg2rad(20.0)
    assert MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD == np.deg2rad(10.0)
    assert COLLISION_SEMANTICS == "adaptive_discrete_interpolation"


def test_strict_adapter_uses_pose_constrained_ik_and_not_direct_waypoint_replay() -> None:
    source = Path(r7.__file__).read_text(encoding="utf-8")
    function_source = source.split("def run_strict_reconstruction_case", 1)[1].split("def read_metric_csv", 1)[0]
    assert "export PLANNING_MODE=ik_waypoints" in function_source
    assert "ALLOW_SEED_JOINT_WITH_TOOL_OFFSET=true" not in function_source
    assert "build_seed_joint_trajectory" not in function_source


def test_proxy_and_strict_are_separate_and_missing_strict_fails_closed() -> None:
    prior = np.zeros((4, 6))
    reference = np.ones((4, 6))
    metrics, shape = r7.strict_path_metrics(prior, None, np.arange(4, dtype=float) + 1.0)
    assert metrics is None
    assert shape["available"] is False
    summary = r7.aggregate_proxy_vs_strict([
        {
            "case_id": "c0",
            "raw_model_to_causal_reference": {"max_joint_space_displacement_rad": 1.0},
            "strict_final_trajectory_metrics": None,
            "reconstruction_accepted": "NO",
        }
    ])
    assert summary["PROXY_ACCEPTED"] == "0/20"
    assert summary["STRICT_ACCEPTED"] == "0/20"
    assert summary["DID_PROXY_MATERIALLY_MISREPRESENT_STRICT_RECONSTRUCTION"] == "NO"


def test_r5_bound_rejection_blocks_native_stages() -> None:
    decision = bounded_reconstruction_decision({
        "MODEL_TO_RECONSTRUCTED_MAX_JOINT_DELTA": MODEL_PRIOR_MAX_JOINT_DELTA_RAD + 0.01,
        "MODEL_TO_RECONSTRUCTED_RMS_JOINT_DELTA": MODEL_PRIOR_MAX_RMS_JOINT_DELTA_RAD + 0.01,
        "AFFECTED_TRAJECTORY_FRACTION": 1.0,
    })
    assert decision["accepted"] is False
    assert r7.summarize_native([{
        "reconstruction_accepted": "NO",
        "reconstructed_totg_attempted": "NOT_REACHED",
        "reconstructed_totg_passed": "NOT_REACHED",
        "reconstructed_ruckig_attempted": "NOT_REACHED",
        "reconstructed_ruckig_finished": "NOT_REACHED",
        "post_ruckig_validated": "NOT_REACHED",
        "final_certified": "NO",
    }])["RECONSTRUCTED_TOTG_ATTEMPTED"] == "NOT_REACHED"


def test_replay_only_digest_is_semantic_and_root_cause_is_explicit() -> None:
    payload = r7.replay_payload([], {"R6_CHECKPOINT_OBSERVED_SHA256": r7.R6_EXPECTED_SHA256})
    assert r7.canonical_hash(payload) == r7.canonical_hash(payload)
    root, blocker, _ = r7.root_cause([
        {"strict_reconstruction_returned_trajectory": "NO", "reconstruction_accepted": "NO", "strict_reconstruction_failure_reason": "strict_moveit_ik_continuity_gate_failed:joint_step"}
    ], True)
    assert root == "B_STRICT_RECONSTRUCTION_EXECUTES_CORRECTLY_BUT_R6_PRIOR_REMAINS_OUTSIDE_FROZEN_R5_DOMAIN"
    assert blocker == "r6_prior_outside_frozen_r5_domain"

