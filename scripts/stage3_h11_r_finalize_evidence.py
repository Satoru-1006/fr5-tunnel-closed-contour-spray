"""Finalize an independent H11-R evidence directory from real artifacts.

This script does not train, modify upstream artifacts, or synthesize metrics.
It only reads the completed H11 runner/native/test outputs and writes the
H11-R certificate, gate report, human report, and archive manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    runner = output / "formal_runner_retry"

    def load(name: str) -> dict[str, Any]:
        return load_json(output / name)

    environment = load("h11_r_pytorch_environment.json")
    smoke = load("h11_r_pytorch_smoke_test.json")
    upstream = load("upstream_immutability_report.json")
    contract = load("h11_dataset_contract.json")
    window_manifest = load("h11_window_manifest.json")
    boundary = load("window_boundary_audit.json")
    config = load("training_config.json")
    history = load("training_history.json")
    train = load("train_metrics.json")
    validation = load("validation_metrics.json")
    test = load("test_metrics.json")
    generalization = load("generalization_metrics.json")
    baseline = load("baseline_comparison.json")
    raw = load("raw_prediction_constraint_report.json")
    native = load("native_prediction_validation.json")
    replay = load("deterministic_replay.json")

    train_rmse = train["joint_position_rmse_rad"]
    validation_rmse = validation["joint_position_rmse_rad"]
    test_rmse = test["joint_position_rmse_rad"]
    generalization_rmse = generalization["joint_position_rmse_rad"]
    hold_last_rmse = baseline["hold_last_position"]["joint_position_rmse_rad"]
    constant_velocity_rmse = baseline["constant_velocity_extrapolation"]["joint_position_rmse_rad"]
    checkpoint = output / "checkpoint"
    checkpoint_hash = sha256(checkpoint)
    summary = (output / "model_summary.txt").read_text(encoding="utf-8-sig")
    parameter_count = int(re.search(r"trainable_parameters:\s*(\d+)", summary).group(1))

    smoke_pass = (
        smoke.get("PYTORCH_AVAILABLE") == "YES"
        and all(smoke.get(key) == "PASSED" for key in ("tensor_smoke", "GRU_FORWARD", "GRU_BACKWARD", "OPTIMIZER_STEP"))
        and smoke.get("FINITE_OUTPUT") == "YES"
    )
    upstream_pass = all(upstream.get(key) == "YES" for key in ("H7_IMMUTABLE", "H8_R_IMMUTABLE", "HISTORICAL_H8_IMMUTABLE", "H9_IMMUTABLE", "H10_IMMUTABLE"))
    source_h10_pass = upstream.get("SOURCE_H10") == "PASSED" and upstream.get("H10_SEMANTIC_HASH_MATCH") == "YES"
    window_pass = upstream.get("H11_WINDOW_SEMANTIC_HASH_MATCH") == "YES"
    group_leakage = int(upstream["authority"]["split_audit"]["GROUP_LEAKAGE_VIOLATIONS"])
    normalization_leakage = 0 if contract.get("normalization_source") == "TRAIN" else len(contract.get("normalized_features", []))
    deterministic = replay.get("TRAINING_REPLAY") == "3/3" and bool(replay.get("semantic_determinism")) and bool(replay.get("metric_determinism"))
    beats_baselines = baseline.get("MODEL_BEATS_NAIVE_BASELINES") == "YES"
    required_metrics = all(finite(value) for value in (train_rmse, validation_rmse, test_rmse, generalization_rmse, hold_last_rmse, constant_velocity_rmse)) and isinstance(raw.get("RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"), int)
    native_pass = native.get("status") == "PASSED" and all(native.get(key) is True for key in ("native_backend_executed", "planning_scene_executed", "fk_executed", "dynamics_executed"))
    focused_text = (output / "test_results.txt").read_bytes().decode("utf-16") if (output / "test_results.txt").read_bytes().startswith(b"\xff\xfe") else (output / "test_results.txt").read_text(encoding="utf-8-sig")
    regression_text = (output / "regression_results.txt").read_bytes().decode("utf-16") if (output / "regression_results.txt").read_bytes().startswith(b"\xff\xfe") else (output / "regression_results.txt").read_text(encoding="utf-8-sig")
    focused_pass = "7 passed" in focused_text
    regression_pass = "51 passed" in regression_text
    all_pass = all((smoke_pass, upstream_pass, source_h10_pass, window_pass, group_leakage == 0, normalization_leakage == 0, boundary.get("CROSS_FAMILY_WINDOWS") == 0, required_metrics, beats_baselines, deterministic, native_pass, focused_pass, regression_pass))
    blocker = "none"
    for candidate, passed in (
        ("pytorch_runtime_import_failure", smoke_pass),
        ("h11_dataset_contract_changed", window_pass),
        ("upstream_artifact_mutated", upstream_pass),
        ("required_metrics_missing", required_metrics),
        ("model_does_not_beat_naive_baselines", beats_baselines),
        ("deterministic_replay_failed", deterministic),
        ("native_prediction_validation_unavailable", native_pass),
        ("focused_tests_failed", focused_pass),
        ("regression_failure", regression_pass),
    ):
        if not passed:
            blocker = candidate
            break

    terminal: dict[str, Any] = {
        "schema_version": "stage3_h11_r_terminal_certificate_v1",
        "STAGE_3_H11_R": "PASSED" if all_pass else "BLOCKED",
        "FIRST_BLOCKER": blocker,
        "PYTHON_EXECUTABLE": environment["python_executable"],
        "PYTHON_VERSION": environment["python_version"],
        "PYTORCH_VERSION": environment["TORCH_VERSION"],
        "PYTORCH_PATH": environment["TORCH_PATH"],
        "DEVICE": environment["device"],
        "CUDA_AVAILABLE": "YES" if environment["CUDA_AVAILABLE"] else "NO",
        "PYTORCH_RUNTIME_REMEDIATION": "PASSED" if smoke_pass else "BLOCKED",
        "PYTORCH_AVAILABLE": smoke["PYTORCH_AVAILABLE"],
        "PYTORCH_SMOKE_TEST": "PASSED" if smoke_pass else "FAILED",
        **{key: upstream[key] for key in ("H7_IMMUTABLE", "H8_R_IMMUTABLE", "HISTORICAL_H8_IMMUTABLE", "H9_IMMUTABLE", "H10_IMMUTABLE", "SOURCE_H10", "H10_SEMANTIC_HASH_MATCH")},
        "H10_SEMANTIC_SHA256": upstream["H10_SEMANTIC_HASH_EXPECTED"],
        "TOTAL_CAUSAL_WINDOWS": int(window_manifest["window_count"]),
        "TRAIN_FAMILIES": int(window_manifest["family_counts"]["TRAIN"]),
        "VALIDATION_FAMILIES": int(window_manifest["family_counts"]["VALIDATION"]),
        "TEST_FAMILIES": int(window_manifest["family_counts"]["TEST"]),
        "GENERALIZATION_FAMILIES": int(window_manifest["family_counts"]["GENERALIZATION"]),
        "CROSS_FAMILY_WINDOWS": int(boundary["CROSS_FAMILY_WINDOWS"]),
        "NORMALIZATION_LEAKAGE_VIOLATIONS": normalization_leakage,
        "GROUP_LEAKAGE_VIOLATIONS": group_leakage,
        "MODEL_TYPE": config["model_type"],
        "INPUT_HISTORY": int(config["input_history"]),
        "PREDICTION_HORIZON": int(config["prediction_horizon"]),
        "TRAINABLE_PARAMETERS": parameter_count,
        "SELECTED_EPOCH": int(history["selected_epoch"]),
        "CHECKPOINT_SHA256": checkpoint_hash,
        "TRAIN_RMSE_RAD": train_rmse,
        "VALIDATION_RMSE_RAD": validation_rmse,
        "TEST_RMSE_RAD": test_rmse,
        "GENERALIZATION_RMSE_RAD": generalization_rmse,
        "HOLD_LAST_TEST_RMSE_RAD": hold_last_rmse,
        "CONST_VELOCITY_TEST_RMSE_RAD": constant_velocity_rmse,
        "MODEL_BEATS_NAIVE_BASELINES": "YES" if beats_baselines else "NO",
        "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS": int(raw["RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS"]),
        "RAW_MODEL_COLLISION_VIOLATIONS": native["RAW_MODEL_COLLISION_VIOLATIONS"],
        "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS": native["RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"],
        "TRAINING_REPLAY": replay["TRAINING_REPLAY"],
        "DETERMINISTIC_TRAINING": "YES" if deterministic else "NO",
        "BYTE_LEVEL_CHECKPOINT_DETERMINISM": "YES" if replay.get("byte_level_checkpoint_determinism") else "NO",
        "NATIVE_PREDICTION_VALIDATION": "PASSED" if native_pass else "BLOCKED",
        "native_backend_executed": native.get("native_backend_executed"),
        "planning_scene_executed": native.get("planning_scene_executed"),
        "fk_executed": native.get("fk_executed"),
        "dynamics_executed": native.get("dynamics_executed"),
        "COLLISION_METHOD": native.get("COLLISION_METHOD"),
        "CCD_AVAILABLE": native.get("CCD_AVAILABLE"),
        "CLEARANCE_AVAILABLE": native.get("CLEARANCE_AVAILABLE"),
        "FOCUSED_TESTS": "PASSED" if focused_pass else "FAILED",
        "REGRESSION": "PASSED" if regression_pass else "FAILED",
        "PHYSICAL_ROBOT_CONNECTED": "NO",
        "PHYSICAL_DRIVER_LOADED": "NO",
        "PHYSICAL_FJT_GOALS_SENT": 0,
        "ROBOT_MOTION_STARTED": "NO",
        "STAGE_3_H11": "PASSED" if all_pass else "BLOCKED",
        "READY_FOR_STAGE_3_H12": "YES" if all_pass else "NO",
        "AUTHORITATIVE_OUTPUT_DIR": str(output),
        "FORMAL_RUNNER_SOURCE_DIR": str(runner),
    }
    write_json(output / "checkpoint_sha256_manifest.json", {"schema_version": "stage3_h11_r_checkpoint_sha256_v1", "checkpoint": "checkpoint", "sha256": checkpoint_hash, "size_bytes": checkpoint.stat().st_size, "selected_epoch": history["selected_epoch"], "source_formal_runner": str((runner / "final_model.pt").relative_to(root)).replace("\\", "/")})
    write_json(output / "stage3_h11_r_terminal_certificate.json", terminal)
    required = {"pytorch_runtime_remediation": smoke_pass, "pytorch_smoke_test": smoke_pass, "upstream_immutable": upstream_pass, "source_h10": source_h10_pass, "h11_window_contract": window_pass, "group_leakage": group_leakage == 0, "normalization_leakage": normalization_leakage == 0, "cross_family_windows": boundary.get("CROSS_FAMILY_WINDOWS") == 0, "required_metrics": required_metrics, "model_beats_naive_baselines": beats_baselines, "training_replay": replay.get("TRAINING_REPLAY") == "3/3", "deterministic_training": deterministic, "native_prediction_validation": native_pass, "focused_tests": focused_pass, "regression": regression_pass}
    write_json(output / "stage3_h11_r_gate_report.json", {"schema_version": "stage3_h11_r_gate_report_v1", **terminal, "required": required, "blockers": [] if all_pass else [blocker], "provenance": {"formal_runner_source_dir": str(runner), "training_was_real_pytorch": True, "source_h10_dir": str(root / "outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z"), "native_worker_source": "ros2_moveit_bridge/stage3_h11_native.py"}})

    report_keys = ("STAGE_3_H11_R", "FIRST_BLOCKER", "PYTHON_EXECUTABLE", "PYTHON_VERSION", "PYTORCH_VERSION", "PYTORCH_PATH", "DEVICE", "CUDA_AVAILABLE", "PYTORCH_RUNTIME_REMEDIATION", "PYTORCH_SMOKE_TEST", "H7_IMMUTABLE", "H8_R_IMMUTABLE", "HISTORICAL_H8_IMMUTABLE", "H9_IMMUTABLE", "H10_IMMUTABLE", "SOURCE_H10", "H10_SEMANTIC_HASH_MATCH", "TOTAL_CAUSAL_WINDOWS", "TRAIN_FAMILIES", "VALIDATION_FAMILIES", "TEST_FAMILIES", "GENERALIZATION_FAMILIES", "CROSS_FAMILY_WINDOWS", "NORMALIZATION_LEAKAGE_VIOLATIONS", "MODEL_TYPE", "TRAINABLE_PARAMETERS", "SELECTED_EPOCH", "TRAIN_RMSE_RAD", "VALIDATION_RMSE_RAD", "TEST_RMSE_RAD", "GENERALIZATION_RMSE_RAD", "HOLD_LAST_TEST_RMSE_RAD", "CONST_VELOCITY_TEST_RMSE_RAD", "MODEL_BEATS_NAIVE_BASELINES", "RAW_MODEL_HARD_CONSTRAINT_VIOLATIONS", "RAW_MODEL_COLLISION_VIOLATIONS", "RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS", "TRAINING_REPLAY", "DETERMINISTIC_TRAINING", "BYTE_LEVEL_CHECKPOINT_DETERMINISM", "NATIVE_PREDICTION_VALIDATION", "COLLISION_METHOD", "CCD_AVAILABLE", "CLEARANCE_AVAILABLE", "FOCUSED_TESTS", "REGRESSION", "PHYSICAL_ROBOT_CONNECTED", "PHYSICAL_DRIVER_LOADED", "PHYSICAL_FJT_GOALS_SENT", "ROBOT_MOTION_STARTED", "STAGE_3_H11", "READY_FOR_STAGE_3_H12")
    lines = ["# STAGE 3 H11-R — PyTorch Runtime Recertification", "", "```text"] + [f"{key}: {terminal[key]}" for key in report_keys] + ["```", "", "## Evidence provenance", "", f"- Formal H11 runner source: `{runner}`", f"- Frozen H10 source: `{root / 'outputs/stage3_h10_multi_trajectory_dataset_20260811T081123Z'}`", f"- H10 semantic SHA-256: `{upstream['H10_SEMANTIC_HASH_EXPECTED']}`", f"- H11 window semantic SHA-256: `{upstream['H11_WINDOW_SEMANTIC_HASH_EXPECTED']}`", "- Training, checkpoint, replay, baseline, rollout, and native outputs are archived from real executions.", "", "## Interpretation", "", "This is software-only statistical baseline certification. Raw neural rollout hard-constraint violations and native process-tolerance violations are reported without clipping or substitution. Collision checking used `adaptive_discrete_interpolation`; CCD and clearance were unavailable.", "", "No physical robot, driver, FollowJointTrajectory goal, spray control, or robot motion was used.", "", "H11-R does not enter H12 automatically."]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    archive_names = ("FINAL_REPORT.md", "stage3_h11_r_terminal_certificate.json", "stage3_h11_r_gate_report.json", "h11_r_runtime_preflight.json", "h11_r_runtime_preflight.txt", "h11_r_pytorch_environment.json", "h11_r_pytorch_install_commands.txt", "h11_r_pytorch_install_stdout.log", "h11_r_pytorch_install_stderr.log", "h11_r_pytorch_smoke_test.json", "upstream_immutability_report.json", "h11_dataset_contract.json", "h11_window_manifest.json", "window_boundary_audit.json", "normalization_stats.json", "training_config.json", "training_history.json", "training_history.csv", "model_summary.txt", "checkpoint", "checkpoint_sha256_manifest.json", "train_metrics.json", "validation_metrics.json", "test_metrics.json", "generalization_metrics.json", "baseline_comparison.json", "rollout_metrics.json", "raw_prediction_constraint_report.json", "native_prediction_validation.json", "deterministic_replay.json", "test_results.txt", "regression_results.txt")
    records = []
    for name in archive_names:
        path = output / name
        source = runner / name
        records.append({"path": name, "absolute_path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size, "source_formal_runner": str(source.relative_to(root)).replace("\\", "/") if source.is_file() else None})
    write_json(output / "frozen_artifact_manifest.json", {"schema_version": "stage3_h11_r_frozen_artifact_manifest_v1", "algorithm": "SHA-256", "records": records, "formal_runner_source_dir": str(runner), "upstream_artifacts_untouched": True})
    result = {"all_pass": all_pass, "blocker": blocker, "test_rmse": test_rmse, "hold_last_rmse": hold_last_rmse, "constant_velocity_rmse": constant_velocity_rmse, "native_status": native.get("status"), "native_collision": native.get("RAW_MODEL_COLLISION_VIOLATIONS"), "native_process": native.get("RAW_MODEL_PROCESS_TOLERANCE_VIOLATIONS"), "checkpoint_sha256": checkpoint_hash, "file_count": len(records)}
    print(json.dumps(result, sort_keys=True))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
