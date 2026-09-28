from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_p2b2_robustness_remapping import _project_target_normals_to_fk_samples
from scripts.run_p2b3_c2_robustness import (
    declared_input_sha256,
    frozen_project_root,
    resolve_artifact_commit,
    static_candidate_budget,
    verify_c1_identity,
)


def _valid_lineage(path: Path) -> None:
    fields = ["index", "baseline_semantic_classification", "c1_semantic_classification",
              "c1_target_attempted", "c1_target_state_emitted", "c1_tight_converged",
              "c1_final_position_residual_m", "c1_final_normal_residual_rad", "c1_iterations_used",
              "c1_termination_reason", "c1_final_relaxed_geometry_gate"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(181):
            writer.writerow({
                "index": index,
                "baseline_semantic_classification": "OBSERVED_D39_ROW_EMITTED_AS_UNSOLVED_PREFIX" if index < 16 else "DLS_TARGET_STATE",
                "c1_semantic_classification": "DLS_TARGET_STATE_SEEDED_FROM_D39_ROW0" if index == 0 else "DLS_TARGET_STATE",
                "c1_target_attempted": "True", "c1_target_state_emitted": "True",
                "c1_tight_converged": "True" if index < 173 else "False",
                "c1_final_position_residual_m": 1e-5 if index < 173 else 1e-3,
                "c1_final_normal_residual_rad": 1e-5 if index < 173 else 0.01,
                "c1_iterations_used": 4, "c1_termination_reason": "SYNTHETIC_MEASURED_TERMINATION",
                "c1_final_relaxed_geometry_gate": "PASS_4MM_5DEG",
            })


def _c1_result(nominal: Path) -> dict:
    return {
        "PROJECT": "FAIRINO_FR5", "STAGE": "P2-B3-C1", "P2B3_C1_STATUS": "COMPLETE_WITH_LIMITATIONS",
        "C1_FATAL_GATES": {"strict": "PASS"},
        "C1_POST_RUCKIG_NOMINAL": {"sha256": __import__("hashlib").sha256(nominal.read_bytes()).hexdigest()},
        "c1_measurement": {"waypoint_count": 181, "tight_converged_count": 173},
    }


def test_c2_nominal_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    nominal = tmp_path / "nominal.csv"
    nominal.write_text("trajectory", encoding="utf-8")
    c1 = _c1_result(nominal)
    c1["C1_POST_RUCKIG_NOMINAL"]["sha256"] = "0" * 64
    lineage = tmp_path / "lineage.csv"
    _valid_lineage(lineage)
    with pytest.raises(RuntimeError, match="nominal_sha256_mismatch"):
        verify_c1_identity(c1, nominal, lineage)


def test_c2_c1_lineage_semantics_are_distinct_and_tight_count_is_measured(tmp_path: Path) -> None:
    nominal = tmp_path / "nominal.csv"
    nominal.write_text("trajectory", encoding="utf-8")
    lineage = tmp_path / "lineage.csv"
    _valid_lineage(lineage)
    digest, summary = verify_c1_identity(_c1_result(nominal), nominal, lineage)
    assert digest == _c1_result(nominal)["C1_POST_RUCKIG_NOMINAL"]["sha256"]
    assert summary == {"waypoint_count": 181, "tight_converged_count": 173, "accepted_target_count": 181}


def test_artifact_publish_commit_uses_explicit_second_phase_resolution(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"C2_EXECUTION_CODE_COMMIT": "a" * 40}), encoding="utf-8")
    result = resolve_artifact_commit(type("Args", (), {
        "result": result_path,
        "artifact_publish_commit": "b" * 40,
    })())
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["artifact_publish_commit"] == "b" * 40
    assert written["C2_ARTIFACT_PUBLISH_COMMIT"]["phase_2_first_canonical_artifact_commit"] == "b" * 40
    assert written["C2_ARTIFACT_PUBLISH_COMMIT"]["status"] == "RESOLVED_IN_SECOND_PHASE"


def test_declared_input_identity_normalizes_windows_path_separators() -> None:
    digest = "a" * 64
    assert declared_input_sha256({r"outputs\p2b2\nominal.csv": digest}, "outputs/p2b2/nominal.csv") == digest


def test_declared_input_identity_rejects_conflicting_normalized_duplicates() -> None:
    with pytest.raises(RuntimeError, match="ambiguous_declared_input_identity"):
        declared_input_sha256({r"outputs\p2b2\nominal.csv": "a" * 64,
                               "outputs/p2b2/nominal.csv": "b" * 64},
                              "outputs/p2b2/nominal.csv")


def test_frozen_project_root_resolves_release_from_stage3_source_repository(tmp_path: Path) -> None:
    d46 = tmp_path / "protected_repo" / "outputs" / "D46_STAGE4A_SYSTEM_BASELINE_V1"
    d46.mkdir(parents=True)
    assert frozen_project_root(d46) == tmp_path / "protected_repo"


def test_static_candidate_budget_records_explicit_native_batch_cap() -> None:
    budget = static_candidate_budget([], None, max_native_batches=7)  # type: ignore[arg-type]
    assert budget["max_native_batches"] == 7


def test_c1_fk_wall_normals_match_projected_authoritative_reference() -> None:
    root = Path(__file__).resolve().parents[1]
    target_path = root / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
    fk_path = root / "outputs/p2b3_c1_r0_fk_trace.csv"
    benchmark_path = root / "outputs/D46_STAGE4A_SYSTEM_BASELINE_V1/STAGE4_SYSTEM_BENCHMARK_V1.json"
    target_rows = list(csv.DictReader(target_path.open(encoding="utf-8-sig", newline="")))
    fk_rows = list(csv.DictReader(fk_path.open(encoding="utf-8-sig", newline="")))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    target_positions = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in target_rows])
    target_normals = np.asarray([[float(row[key]) for key in ("nx", "ny", "nz")] for row in target_rows])
    fk_positions = np.asarray([[float(row[key]) for key in ("actual_tcp_x", "actual_tcp_y", "actual_tcp_z")] for row in fk_rows])
    fk_normals = np.asarray([[float(row[key]) for key in ("wall_normal_x", "wall_normal_y", "wall_normal_z")] for row in fk_rows])
    projected_normals, _mapping = _project_target_normals_to_fk_samples(
        target_positions, target_normals, fk_positions,
        max_path_deviation_m=float(benchmark["diagnostic_thresholds"]["tcp_trajectory_error_m"]["value"]),
    )
    assert projected_normals.shape == (181, 3)
    assert float(np.max(np.linalg.norm(projected_normals - fk_normals, axis=1))) <= 1e-9
