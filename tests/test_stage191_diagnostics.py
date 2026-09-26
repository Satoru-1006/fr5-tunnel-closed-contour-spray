from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.deterministic_ik_candidates import canonical_json, explicit_seed_templates
from src.frozen_replay_diagnostics import (
    OUTPUT_ROOT,
    build_frozen_corpus,
    classify_first_layer,
    compare_replays,
    frozen_task_layers,
    protected_hashes,
    protected_hashes_unchanged,
    quantized_hash,
    raw_hash,
)
from src.planning_scene_snapshot import build_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_stage191_config_exists() -> None:
    assert (ROOT / "config/stage191_determinism_diagnostics.yaml").exists()


def test_stage191_output_is_separate() -> None:
    assert "stage191" in str(OUTPUT_ROOT)


def test_authoritative_input_has_181_waypoints() -> None:
    layers, seeds, _ = frozen_task_layers(ROOT / "config/stage191_determinism_diagnostics.yaml")
    assert len(layers) == len(seeds) == 181


def test_task_pose_candidates_are_nonempty() -> None:
    layers, _, _ = frozen_task_layers(ROOT / "config/stage191_determinism_diagnostics.yaml")
    assert sum(len(layer) for layer in layers) >= 181


def test_frozen_task_pose_order_is_stable() -> None:
    left, _, _ = frozen_task_layers(ROOT / "config/stage191_determinism_diagnostics.yaml")
    right, _, _ = frozen_task_layers(ROOT / "config/stage191_determinism_diagnostics.yaml")
    assert [[row["task_pose_stable_id"] for row in layer] for layer in left] == [[row["task_pose_stable_id"] for row in layer] for layer in right]


def test_seed_templates_have_explicit_order() -> None:
    templates = explicit_seed_templates(0, [0.0] * 6, None, config={"seed_templates": {"include_nominal_seed": True}, "seed_offsets_rad": {}})
    assert [item.seed_order_index for item in templates] == list(range(len(templates)))
    assert templates[0].seed_template_family == "nominal"


def test_seed_templates_do_not_use_random_module() -> None:
    templates = explicit_seed_templates(1, [0.1] * 6, [0.2] * 6, config={"seed_templates": {"include_nominal_seed": True, "include_previous_waypoint_seeds": True, "include_fixed_shoulder_templates": False, "include_fixed_elbow_templates": False, "include_fixed_wrist_templates": False, "include_fixed_combination_templates": False}, "seed_offsets_rad": {}})
    assert len(templates) == 2
    assert templates[1].seed_template_family == "previous_waypoint"


def test_seed_template_ids_are_reproducible() -> None:
    kwargs = {"seed_templates": {"include_nominal_seed": True}, "seed_offsets_rad": {}}
    assert explicit_seed_templates(0, [0.0] * 6, None, config=kwargs)[0].seed_template_id == explicit_seed_templates(0, [0.0] * 6, None, config=kwargs)[0].seed_template_id


def test_canonical_json_sorts_mapping_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_raw_hash_is_order_stable_for_mapping() -> None:
    assert raw_hash({"b": 1, "a": 2}) == raw_hash({"a": 2, "b": 1})


def test_quantized_hash_ignores_sub_step_noise() -> None:
    assert quantized_hash([1.0], 1.0e-3) == quantized_hash([1.0 + 1.0e-8], 1.0e-3)


def test_quantized_hash_detects_step_change() -> None:
    assert quantized_hash([1.0], 1.0e-3) != quantized_hash([1.01], 1.0e-3)


def test_compare_replays_detects_input_change() -> None:
    result = compare_replays({"a": [{"call_id": "c", "target_pose_raw_sha256": "1", "seed_raw_sha256": "1", "initial_robot_state_raw_sha256": "1", "solver_options_sha256": "1", "solution_raw_sha256": "1", "solution_quantized_sha256": "1", "solver_success": True}], "b": [{"call_id": "c", "target_pose_raw_sha256": "2", "seed_raw_sha256": "1", "initial_robot_state_raw_sha256": "1", "solver_options_sha256": "1", "solution_raw_sha256": "1", "solution_quantized_sha256": "1", "solver_success": True}]})
    assert result["classifications"]["c"] == "input_changed"


def test_compare_replays_detects_raw_noise() -> None:
    base = {"call_id": "c", "target_pose_raw_sha256": "1", "seed_raw_sha256": "1", "initial_robot_state_raw_sha256": "1", "solver_options_sha256": "1", "solution_raw_sha256": "1", "solution_quantized_sha256": "q", "solver_success": True}
    other = dict(base, solution_raw_sha256="2")
    result = compare_replays({"a": [base], "b": [other]})
    assert result["classifications"]["c"] == "raw_output_changed_quantized_equal"


def test_compare_replays_detects_quantized_change() -> None:
    base = {"call_id": "c", "target_pose_raw_sha256": "1", "seed_raw_sha256": "1", "initial_robot_state_raw_sha256": "1", "solver_options_sha256": "1", "solution_raw_sha256": "1", "solution_quantized_sha256": "q", "solver_success": True}
    other = dict(base, solution_quantized_sha256="z")
    result = compare_replays({"a": [base], "b": [other]})
    assert result["classifications"]["c"] == "quantized_output_changed"


def test_compare_replays_accepts_full_determinism() -> None:
    row = {"call_id": "c", "target_pose_raw_sha256": "1", "seed_raw_sha256": "1", "initial_robot_state_raw_sha256": "1", "solver_options_sha256": "1", "solution_raw_sha256": "1", "solution_quantized_sha256": "q", "solver_success": True}
    result = compare_replays({"a": [row], "b": [dict(row)]})
    assert result["all_fully_deterministic"]


def test_first_layer_prefers_ik_input() -> None:
    assert classify_first_layer(ik_comparison={"all_inputs_same": False, "first_differing_record_id": "c", "classifications": {"c": "input_changed"}}, node_hashes=[], edge_hashes=[], dp_hashes=[], ruckig_hashes=[], post_hashes=[]) == "ik_call_input_generation"


def test_first_layer_identifies_edge() -> None:
    assert classify_first_layer(ik_comparison={"all_inputs_same": True, "first_differing_record_id": None, "classifications": {}}, node_hashes=["n"], edge_hashes=["a", "b"], dp_hashes=[], ruckig_hashes=[], post_hashes=[]) == "edge_construction_or_collision_validation"


def test_first_layer_identifies_dp() -> None:
    assert classify_first_layer(ik_comparison={"all_inputs_same": True, "first_differing_record_id": None, "classifications": {}}, node_hashes=["n"], edge_hashes=["e"], dp_hashes=["a", "b"], ruckig_hashes=[], post_hashes=[]) == "dp_search"


def test_first_layer_identifies_ruckig() -> None:
    assert classify_first_layer(ik_comparison={"all_inputs_same": True, "first_differing_record_id": None, "classifications": {}}, node_hashes=["n"], edge_hashes=["e"], dp_hashes=["d"], ruckig_hashes=["a", "b"], post_hashes=[]) == "ruckig"


def test_first_layer_identifies_post_validation() -> None:
    assert classify_first_layer(ik_comparison={"all_inputs_same": True, "first_differing_record_id": None, "classifications": {}}, node_hashes=["n"], edge_hashes=["e"], dp_hashes=["d"], ruckig_hashes=["r"], post_hashes=["a", "b"]) == "post_validation_or_planning_scene"


def test_first_layer_is_none_when_all_stable() -> None:
    assert classify_first_layer(ik_comparison={"all_inputs_same": True, "first_differing_record_id": None, "classifications": {}}, node_hashes=["n"], edge_hashes=["e"], dp_hashes=["d"], ruckig_hashes=["r"], post_hashes=["p"]) is None


def test_scene_snapshot_has_required_statuses() -> None:
    snapshot = build_snapshot(robot_model_hash="r", urdf_hash="u", srdf_hash="s", kinematics_hash="k", joint_limits_hash="l", collision_detector_type="test", planning_frame="base_link", group_name="g", objects=[], allowed_collision_matrix={}, robot_state={"joint_positions": []}, interpolation_step_deg=0.5)
    assert snapshot["collision_method"] == "adaptive_discrete_interpolation"
    assert snapshot["ccd_status"] == "not_available"
    assert snapshot["clearance_status"] == "not_available"
    assert snapshot["planning_scene_semantic_hash"]


def test_scene_snapshot_hash_changes_with_object_count() -> None:
    one = build_snapshot(robot_model_hash="r", urdf_hash="u", srdf_hash="s", kinematics_hash="k", joint_limits_hash="l", collision_detector_type="test", planning_frame="base_link", group_name="g", objects=[], allowed_collision_matrix={}, robot_state={"joint_positions": []}, interpolation_step_deg=0.5)
    two = dict(one, collision_object_count=1)
    assert one["planning_scene_semantic_hash"] != raw_hash(two)


def test_protected_hash_comparison_is_exact() -> None:
    sample = {"files": {"a": {"sha256": "x"}}}
    assert protected_hashes_unchanged(sample, sample)


def test_protected_hash_comparison_detects_change() -> None:
    assert not protected_hashes_unchanged({"files": {"a": {"sha256": "x"}}}, {"files": {"a": {"sha256": "y"}}})


def test_protected_hashes_returns_file_map() -> None:
    result = protected_hashes(ROOT)
    assert "files" in result and "groups" in result


def test_config_forbids_dynamic_seed_feedback() -> None:
    text = (ROOT / "config/stage191_determinism_diagnostics.yaml").read_text(encoding="utf-8")
    assert "forbid_dynamic_seed_feedback: true" in text


def test_config_keeps_ccd_unavailable() -> None:
    text = (ROOT / "config/stage191_determinism_diagnostics.yaml").read_text(encoding="utf-8")
    assert "ccd_status: \"not_available\"" in text


def test_config_keeps_clearance_unavailable() -> None:
    text = (ROOT / "config/stage191_determinism_diagnostics.yaml").read_text(encoding="utf-8")
    assert "clearance_status: \"not_available\"" in text


def test_config_uses_adaptive_discrete_interpolation() -> None:
    text = (ROOT / "config/stage191_determinism_diagnostics.yaml").read_text(encoding="utf-8")
    assert "adaptive_discrete_interpolation" in text


def test_stage2_is_not_implemented_in_diagnostic_tools() -> None:
    for name in ("tools/run_stage191_dp_replay.py", "tools/run_stage191_ik_replay.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "OFF" not in text and "GNN" not in text


def test_existing_stage19_algorithm_source_is_not_imported_as_a_solver_fix() -> None:
    text = (ROOT / "tools/run_stage191_ik_replay.py").read_text(encoding="utf-8")
    assert "run_stage19_deterministic_ik" not in text


def test_required_new_tools_exist() -> None:
    for name in ("tools/build_frozen_ik_call_corpus.py", "tools/run_stage191_ik_replay.py", "tools/run_stage191_dp_replay.py", "tools/run_stage191_ruckig_replay.py", "tools/run_stage191_post_validation_replay.py", "tools/audit_stage191_layers.py"):
        assert (ROOT / name).exists()


@pytest.mark.skipif(not (OUTPUT_ROOT / "determinism_layer_report.json").exists(), reason="real Stage 1.9.1 replay artifacts not present")
def test_real_report_keeps_stage2_blocked_until_fix() -> None:
    report = json.loads((OUTPUT_ROOT / "determinism_layer_report.json").read_text(encoding="utf-8"))
    assert report["stage2_status"] == "blocked"
    assert report["deterministic_fix_complete"] is False


@pytest.mark.skipif(not (OUTPUT_ROOT / "edge_replay/comparison.json").exists(), reason="real edge replay artifacts not present")
def test_real_edge_replay_is_threefold() -> None:
    comparison = json.loads((OUTPUT_ROOT / "edge_replay/comparison.json").read_text(encoding="utf-8"))
    assert comparison["repeat_count"] == 3


@pytest.mark.skipif(not (OUTPUT_ROOT / "dp_replay/comparison.json").exists(), reason="real DP artifacts not present")
def test_real_dp_replay_is_100_fold() -> None:
    comparison = json.loads((OUTPUT_ROOT / "dp_replay/comparison.json").read_text(encoding="utf-8"))
    assert comparison["repeat_count"] == 100
