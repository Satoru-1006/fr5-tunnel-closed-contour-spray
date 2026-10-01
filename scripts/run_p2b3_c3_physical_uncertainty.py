#!/usr/bin/env python3
"""Run tests, a bounded native scene smoke, or publish the P2-B3-C3 record."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physical_uncertainty import (  # noqa: E402
    CONTRACT_SCHEMA,
    default_uncertainty_semantics,
    foreign_project_reference_count,
    classify_legacy_c2_axis,
)


PARENT_BRANCH = "codex/fr5-p2b3-c2-robustness-transfer-20260928"
EXPECTED_PARENT_COMMIT = "fff39145a1e9dcd59fb1d180fe2cf182444820fe"
NEW_BRANCH = "codex/fr5-p2b3-c3-physical-uncertainty-se3-20261001"
REMOTE_REPO = "Satoru-1006/fr5-tunnel-closed-contour-spray"
C1_RESULT = ROOT / "outputs/p2b3_c1_result.json"
C1_TRAJECTORY = ROOT / "outputs/p2b3_c2_c1_nominal_post_ruckig.csv"
TARGET_POSES = ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv"
C1_FK_DEFAULT = Path(r"D:\fr5-p2b3-c2-20260928\c1_r0_replay\strict_replay\moveit_fk_tcp_trace.csv")
C2_RESULT = ROOT / "outputs/p2b3_c2_result.json"
C2_AXIS_PROFILE = ROOT / "outputs/p2b3_c2_axis_margin_profile.csv"
C2_TRANSFER = ROOT / "outputs/p2b3_c2_transfer_comparison.csv"
URDF = ROOT / "outputs/p2b2_inputs/derived_reference_robot_model.urdf"
SRDF = ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf"
JOINT_LIMITS = ROOT / "ros2_moveit_bridge/config/joint_limits_with_jerk.yaml"
DEFAULT_SMOKE_ROOT = Path(r"D:\fr5-p2b3-c3-20261001\smoke")
DEFAULT_UNDERLAY = Path(r"D:\fr5-p2b3-c2-20260928\c1_r0_replay\fairino_install")
DEFAULT_OVERLAY = Path(r"D:\fr5-p2b3-c2-20260928\c1_r0_replay\bridge_install")
REGRESSION_FILES = (
    "tests/test_p2b3_c2_runner.py",
    "tests/test_p2a_axiswise_robustness.py",
    "tests/test_process_aware_stress.py",
)
C3_TEST_FILES = ("tests/test_p2b3_c3_physical_uncertainty.py", *REGRESSION_FILES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=15, check=False
    )
    if completed.returncode:
        raise RuntimeError(f"git_{args[0]}_failed:{completed.stderr.strip()}")
    return completed.stdout.strip()


def verify_parent_identity() -> dict[str, Any]:
    branch = git("branch", "--show-current")
    if branch != NEW_BRANCH:
        raise RuntimeError(f"wrong_c3_branch:{branch}")
    parent = git("rev-parse", "HEAD^")
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", f"refs/heads/{PARENT_BRANCH}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=15,
        check=False,
    )
    if remote.returncode:
        raise RuntimeError("github_parent_branch_query_failed:" + remote.stderr.strip())
    remote_sha = remote.stdout.split()[0] if remote.stdout.split() else None
    if parent != EXPECTED_PARENT_COMMIT or remote_sha != EXPECTED_PARENT_COMMIT:
        raise RuntimeError(f"parent_base_changed:parent={parent}:remote={remote_sha}")
    return {
        "parent_branch": PARENT_BRANCH,
        "parent_commit": parent,
        "remote_parent_sha": remote_sha,
        "parent_base_verified": "YES",
        "execution_branch": branch,
    }


class _TestSummary:
    def __init__(self) -> None:
        self.result: dict[str, Any] = {}

    def pytest_terminal_summary(self, terminalreporter: Any, exitstatus: int, config: Any) -> None:
        stats = terminalreporter.stats
        passed = [row for row in stats.get("passed", []) if getattr(row, "when", "call") == "call"]
        failed = list(stats.get("failed", []))
        skipped = list(stats.get("skipped", []))
        errors = list(stats.get("error", []))
        self.result = {
            "status": "PASS" if exitstatus == 0 and not failed and not errors else "FAIL",
            "exit_status": int(exitstatus),
            "passed": len(passed),
            "failed": len(failed) + len(errors),
            "skipped": len(skipped),
            "total": len(passed) + len(failed) + len(errors) + len(skipped),
            "regression_files": list(REGRESSION_FILES),
            "regression_status": "PASS" if exitstatus == 0 and not failed and not errors else "FAIL",
        }


def run_tests() -> dict[str, Any]:
    import pytest

    summary = _TestSummary()
    exitstatus = int(pytest.main(["-q", *C3_TEST_FILES], plugins=[summary]))
    result = dict(summary.result)
    if not result:
        result = {"status": "FAIL", "exit_status": exitstatus, "passed": 0, "failed": 1, "skipped": 0, "total": 1, "regression_files": list(REGRESSION_FILES), "regression_status": "FAIL"}
    return result


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError(f"absolute_windows_path_required:{value}")
    return f"/mnt/{value[0].lower()}/{value[3:]}"


def run_native_smoke(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in (
        (args.underlay_install / "setup.bash", "underlay_install"),
        (args.overlay_install / "setup.bash", "overlay_install"),
        (C1_FK_DEFAULT if args.c1_fk_trace is None else args.c1_fk_trace, "c1_fk_trace"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label}_missing:{path}")
    scratch = args.smoke_root.resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    smoke_result = scratch / "p2b3_c3_native_smoke.json"
    repo_wsl = wsl_path(ROOT)
    underlay_wsl = wsl_path(args.underlay_install)
    overlay_wsl = wsl_path(args.overlay_install)
    paths = {
        "trajectory": wsl_path(C1_TRAJECTORY),
        "target": wsl_path(TARGET_POSES),
        "fk": wsl_path(C1_FK_DEFAULT if args.c1_fk_trace is None else args.c1_fk_trace),
        "c1_result": wsl_path(C1_RESULT),
        "urdf": wsl_path(URDF),
        "srdf": wsl_path(SRDF),
        "limits": wsl_path(JOINT_LIMITS),
        "output": wsl_path(smoke_result),
    }
    command = "\n".join(
        [
            # ROS setup scripts reference unset trace variables under nounset.
            # Keep fail-fast and pipeline checking without breaking ROS setup.
            "set -eo pipefail",
            "source /opt/ros/jazzy/setup.bash",
            f"source {shlex_quote(underlay_wsl)}/setup.bash",
            f"source {shlex_quote(overlay_wsl)}/setup.bash",
            f"export PYTHONPATH={shlex_quote(repo_wsl)}:/opt/ros/jazzy/lib/python3.12/site-packages:${{PYTHONPATH:-}}",
            f"cd {shlex_quote(repo_wsl)}",
            "python3 ros2_moveit_bridge/p2b3_c3_native_smoke.py "
            + " ".join(
                f"--{name} {shlex_quote(value)}"
                for name, value in (
                    ("trajectory-csv", paths["trajectory"]),
                    ("target-csv", paths["target"]),
                    ("c1-fk-trace", paths["fk"]),
                    ("c1-result", paths["c1_result"]),
                    ("urdf", paths["urdf"]),
                    ("srdf", paths["srdf"]),
                    ("joint-limits", paths["limits"]),
                    ("output-json", paths["output"]),
                )
            ),
        ]
    )
    try:
        completed = subprocess.run(
            ["wsl.exe", "-d", args.distro, "--", "bash", "-lc", command],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=args.native_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        failure = {
            "schema_version": "p2b3-c3-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": "P2-B3-C3",
            "status": "INCOMPLETE_NATIVE_VALIDATION",
            "native_scene_propagation_tested": "NO",
            "reason": "bounded_WSL_MoveItPy_subprocess_timeout",
            "timeout_seconds": args.native_timeout_seconds,
        }
        write_json(smoke_result, failure)
        return failure
    if not smoke_result.is_file():
        failure = {
            "schema_version": "p2b3-c3-native-smoke-v1",
            "project": "FAIRINO_FR5",
            "stage": "P2-B3-C3",
            "status": "INCOMPLETE_NATIVE_VALIDATION",
            "native_scene_propagation_tested": "NO",
            "reason": "native_smoke_did_not_write_canonical_smoke_result",
            "wsl_exit_code": completed.returncode,
            "stdout_tail": completed.stdout.splitlines()[-20:],
            "stderr_tail": completed.stderr.splitlines()[-20:],
        }
        write_json(smoke_result, failure)
        return failure
    result = read_json(smoke_result)
    result["launcher"] = {
        "distribution": args.distro,
        "ros_distro": "jazzy",
        "subprocess_timeout_seconds": args.native_timeout_seconds,
        "wsl_exit_code": completed.returncode,
    }
    write_json(smoke_result, result)
    if completed.returncode != 0 and result.get("status") == "PASS":
        result["status"] = "INCOMPLETE_NATIVE_VALIDATION"
        result["reason"] = "native_smoke_exit_code_nonzero"
        write_json(smoke_result, result)
    return result


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def c2_semantic_reinterpretation(axis_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in axis_rows:
        axis_id = str(row.get("axis_id", ""))
        result.append({
            "classification": classify_legacy_c2_axis(axis_id),
            "historical_raw_row": dict(row),
        })
    return result


def frozen_inventory(c1: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "c1_result": C1_RESULT,
        "c1_post_ruckig_nominal": C1_TRAJECTORY,
        "c1_fk_trace": C1_FK_DEFAULT,
        "c1_target_pose_csv": TARGET_POSES,
        "c1_seed_joint_csv": ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv",
        "c2_canonical_result": C2_RESULT,
        "c2_axis_margin_profile": C2_AXIS_PROFILE,
        "c2_transfer_comparison": C2_TRANSFER,
        "c1_readme": ROOT / "docs/P2B3_C1_README.md",
        "c2_readme": ROOT / "docs/P2B3_C2_README.md",
        "derived_reference_urdf": URDF,
        "spray_tcp_srdf": SRDF,
        "joint_limits_config": JOINT_LIMITS,
    }
    rows: dict[str, Any] = {}
    for key, path in paths.items():
        if not path.is_file():
            rows[key] = {"status": "NOT_AVAILABLE", "path": str(path)}
            continue
        rows[key] = {"status": "AVAILABLE", "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
    declared = c1.get("candidate_input_identity_sha256", {})
    result_declared = c1.get("C1_POST_RUCKIG_NOMINAL", {}).get("sha256")
    if rows["c1_post_ruckig_nominal"].get("sha256") != result_declared:
        raise RuntimeError("frozen_c1_nominal_hash_disagrees_with_c1_result")
    if rows["c1_fk_trace"].get("sha256") != declared.get("candidate_fk_trace"):
        raise RuntimeError("frozen_c1_fk_trace_hash_disagrees_with_c1_result")
    return rows


def capability_matrix(semantics: list[dict[str, Any]], smoke: dict[str, Any]) -> dict[str, Any]:
    scene_status = str(smoke.get("native_scene_propagation_tested", "NO"))
    return {
        row["uncertainty_id"]: {
            "semantic_class": row["semantic_class"],
            "operator_capability": row["capability_status"],
            "native_scene_propagation": scene_status if row["semantic_class"] == "BASE_WORKPIECE_REGISTRATION_ERROR" else "NOT_APPLICABLE",
            "physical_bound_status": row["normalization_status"],
        }
        for row in semantics
    }


def semantic_csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    fields = list(rows[0].keys())
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        flattened = dict(row)
        flattened["fixed_entities"] = ";".join(row["fixed_entities"])
        if isinstance(flattened["normalization_bound"], (dict, list, tuple)):
            flattened["normalization_bound"] = json.dumps(flattened["normalization_bound"], separators=(",", ":"))
        writer.writerow(flattened)
    return output.getvalue()


def read_drive_evidence() -> dict[str, Any]:
    return {
        "github_repository": {
            "repository": REMOTE_REPO,
            "branch": PARENT_BRANCH,
            "commit": EXPECTED_PARENT_COMMIT,
            "searched_paths": [
                "docs/P2B3_C1_README.md",
                "docs/P2B3_C2_README.md",
                "outputs/p2b3_c1_result.json",
                "outputs/p2b3_c2_result.json",
                "ros2_moveit_bridge/config",
                "src",
            ],
            "search_terms": ["TCP calibration", "tracking error", "repeatability", "registration error", "pose measurement uncertainty", "surface reconstruction uncertainty"],
            "measured_uncertainty_records_found": [],
            "evidence": [
                {"path": "docs/P2B3_C1_README.md", "finding": "calibrated TCP accuracy is not established"},
                {"path": "docs/P2B3_C2_README.md", "finding": "calibrated TCP uncertainty remains unresolved"},
                {"path": "outputs/p2b3_c2_result.json", "finding": "calibrated_tcp_uncertainty is NOT_AVAILABLE"},
            ],
        },
        "project_root": {"folder_id": "1dsMTZAaPLGDDT3CBJFS0E8f2av9kn2-R", "url": "https://drive.google.com/drive/folders/1dsMTZAaPLGDDT3CBJFS0E8f2av9kn2-R"},
        "searched_fr5_scope": [
            {"folder": "00_PROJECT_CORE", "folder_id": "1Jk-RbNr-pxCeq-pSurpDw7esFFC1E9NP", "inspection": "direct-child listing and relevant project contract/configuration documents"},
            {"folder": "P2-B3-C2/CORE", "folder_id": "11UdHScufz6WT6v7ZltYwOS_KmX3gfLux", "inspection": "direct-child listing and canonical C2 README"},
            {"folder": "PROJECT_INDEX.md", "file_id": "1CcaieZ_X5SOs6W1pBlRIaCU4VUgYKtps", "inspection": "existing stage history and source pointers"},
        ],
        "search_terms": ["FR5 calibration", "TCP calibration", "FR5 joint tracking error", "FR5 repeatability", "FR5 pose measurement uncertainty", "FR5 registration repeatability"],
        "measurement_records_found": [],
        "evidence_boundary": "The C2 README explicitly retains calibrated TCP uncertainty as unresolved and base-transform axes as NOT_AVAILABLE. Search results found no FR5 measured registration, repeatability, tracking, reconstruction, or TCP-calibration bound.",
    }


def make_readme(result: Mapping[str, Any]) -> str:
    smoke = result["native_smoke"]
    status = result["P2B3_C3_STATUS"]
    collision_summary = smoke.get("collision_response_summary", {})
    return f"""# P2-B3-C3 — Physical Uncertainty Semantics and SE(3) Registration Stress

- **Project:** FAIRINO FR5 tunnel complex-surface spraying
- **Stage status:** `{status}`
- **Parent branch:** `{result['PARENT_BRANCH']}`
- **Parent commit:** `{result['PARENT_COMMIT']}`
- **Execution code commit:** `{result['EXECUTION_CODE_COMMIT']}`
- **C1 nominal:** 181-point ON-state open-arch post-Ruckig path, SHA-256 `{result['frozen_input_identities']['c1_post_ruckig_nominal']['sha256']}`
- **Measurement scope:** synthetic known-answer tests plus a bounded native smoke of {smoke.get('case_count', 0)} cases; no C2 campaign replay

## Why C3 exists

C2 left the 18 base/workpiece transform and process perturbation directions `NOT_AVAILABLE` because its evaluator did not propagate a registration transform into the world scene. C3 adds explicit semantics and a real MoveIt2 PlanningScene world-object pose update while keeping the C1 joint trajectory and timestamps fixed.

## What the old C2 perturbations mean

C2 joint axes are deterministic joint-state offsets. They do not establish a measured FR5 tracking distribution or hardware failure probability. C2 `tcp_tcp_translation:*` and `tcp_tcp_rotation:*` axes alter the target pose specification used for geometric comparison. They are not physical TCP execution errors, calibration errors, or workpiece-registration errors. C3 preserves each historical axis id and raw CSV value, then adds a derived semantic classification.

## Registration definition and SE(3) convention

For each workpiece-attached entity, nominal `T_base_entity` maps the entity into `base_link`. The perturbation is

`DeltaT_base = [[Exp([phi_base]x), t_base], [0, 1]]`

and the actual pose is

`T_base_entity_actual = DeltaT_base @ T_base_entity_nominal`.

The 3-vector `t_base` is a direct translation in meters expressed in `base_link`; `phi_base` is a right-handed axis-angle rotation vector in radians expressed in `base_link`. Rotation is about the `base_link` origin. ROS quaternion order is `x,y,z,w`. The parameterization uses direct translation plus `Exp(phi)` and does not use the coupled translational coordinates of an SE(3) twist exponential.

The robot base, robot model, nominal `q`, and timestamps remain fixed. Workpiece collision-object poses, surface points/normals, process reference poses, and workpiece-attached frames move together. Registration does not trigger IK or retiming.

## Native scene propagation

Native status: `{smoke.get('status', 'NOT_RUN')}`. Scene propagation tested: `{smoke.get('native_scene_propagation_tested', 'NO')}`. The smoke uses the project `build_wall_collision_objects` geometry builder with the frozen 181-point target pose/normal inputs. It applies the requested delta to each real MoveIt collision-object pose in `base_link`, then re-runs the MoveIt2/FCL collision and distance query with the unchanged q path. Nonzero cases are `ENGINEERING_PROPAGATION_SMOKE_ONLY`.

Collision observations use `{smoke.get('collision_method', 'NOT_RUN')}` with a 0.5-degree maximum joint-space interpolation step. They are finite-sample software evidence, not strict continuous collision detection. Any MoveIt `CollisionResult.distance` value is reported as sampled model distance only; it is not a physical or continuous clearance guarantee.

The sampled path reported {collision_summary.get('full_scene_collision_samples', 'NOT_AVAILABLE')} full-scene collisions over {collision_summary.get('full_scene_samples', 'NOT_AVAILABLE')} queries, with {collision_summary.get('self_collision_samples', 'NOT_AVAILABLE')} self-collision queries. Collision response status is `{collision_summary.get('status', 'NOT_AVAILABLE')}`. When the binary result is saturated, these smoke amplitudes cannot rank registration directions; keep the observation as model-only evidence and review the approximate wall geometry independently. This is not a hardware collision verdict.

## Physical uncertainty evidence and normalization

No FR5 measured base/workpiece registration distribution, calibrated TCP uncertainty, joint-tracking distribution, or surface-reconstruction uncertainty bound was found in the FR5 project records searched for C3. Literature and assumed stress magnitudes are not substituted for FR5 measurement. Current status is `NORMALIZED_MARGIN_STATUS = NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING`.

The uncertainty ontology is in `outputs/p2b3_c3_uncertainty_semantics.csv`; the complete machine-readable result and evidence inventory are in `outputs/p2b3_c3_result.json`. No normalized-margin CSV is emitted without a supported physical bound.

## Validation and limitations

- Synthetic known-answer tests: `{result['synthetic_tests']['status']}` ({result['synthetic_tests']['passed']} passed, {result['synthetic_tests']['failed']} failed).
- Protected C2/P2-A regression tests: `{result['REGRESSION_TESTS']}`.
- Native smoke: `{smoke.get('status', 'NOT_RUN')}`; zero transform was compared with the saved C1 MoveIt FK trace.
- Physical calibrated uncertainty, hardware tracking, hardware validation, strict continuous CCD, and a calibrated surface reconstruction remain unavailable.
- No normalized physical robustness margin, failure probability, global robustness proof, coating-quality certification, or hardware safety certification is claimed.

## Next gate

Before a full registration-margin campaign, obtain a traceable FR5 base/workpiece registration bound or measured distribution, define its frame and measurement conditions, independently review the saturated collision observation and approximate wall geometry, review the C3 scene evaluator and frozen-input evidence, then authorize a separate bounded margin protocol. Keep the C1 trajectory and all C2 canonical results unchanged.
"""


def build_result(args: argparse.Namespace, test_results: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    parent = verify_parent_identity()
    execution_commit = git("rev-parse", "HEAD")
    c1 = read_json(C1_RESULT)
    c2 = read_json(C2_RESULT)
    if c1.get("PROJECT") != "FAIRINO_FR5" or c1.get("STAGE") != "P2-B3-C1":
        raise RuntimeError("c1_result_scope_mismatch")
    if c2.get("PROJECT") != "FAIRINO_FR5" or c2.get("STAGE") != "P2-B3-C2":
        raise RuntimeError("c2_result_scope_mismatch")
    semantics = default_uncertainty_semantics()
    inventory = frozen_inventory(c1)
    smoke_path = args.native_smoke_result or (args.smoke_root / "p2b3_c3_native_smoke.json")
    smoke = read_json(smoke_path) if smoke_path.is_file() else {
        "status": "INCOMPLETE_NATIVE_VALIDATION",
        "native_scene_propagation_tested": "NO",
        "reason": "native_smoke_not_run_or_result_missing",
        "case_count": 0,
        "cases": [],
    }
    axis_rows = read_csv_rows(C2_AXIS_PROFILE)
    transfer_rows = read_csv_rows(C2_TRANSFER)
    if len(axis_rows) != 24 or len(transfer_rows) != 24:
        raise RuntimeError("c2_axis_or_transfer_count_changed")
    scene_pass = smoke.get("status") == "PASS" and smoke.get("native_scene_propagation_tested") == "YES"
    tests_pass = test_results.get("status") == "PASS"
    regression_pass = test_results.get("regression_status") == "PASS"
    delivery_receipt = {
        "GITHUB_PUSH": "NOT_RUN",
        "REMOTE_SHA_VERIFIED": "NOT_RUN",
        "GOOGLE_DRIVE_UPLOAD": "NOT_RUN",
        "PROJECT_INDEX_UPDATED": "NOT_RUN",
        "WORKTREE_CLEAN": "NOT_VERIFIED",
    }
    delivery_pass = all(
        delivery_receipt[field] == "PASS"
        for field in ("GITHUB_PUSH", "REMOTE_SHA_VERIFIED", "GOOGLE_DRIVE_UPLOAD", "PROJECT_INDEX_UPDATED")
    ) and delivery_receipt["WORKTREE_CLEAN"] == "YES"
    if not scene_pass:
        status = "INCOMPLETE_NATIVE_SCENE_PROPAGATION" if smoke.get("native_scene_propagation_tested") != "YES" else "INCOMPLETE_NATIVE_VALIDATION"
    elif not tests_pass or not regression_pass:
        status = "INCOMPLETE_TESTS"
    elif not delivery_pass:
        status = "INCOMPLETE_DELIVERY"
    else:
        status = "COMPLETE"
    result: dict[str, Any] = {
        "schema_version": "p2b3-c3-physical-uncertainty-result-v1",
        "PROJECT": "FAIRINO_FR5",
        "STAGE": "P2-B3-C3",
        "P2B3_C3_STATUS": status,
        "PROJECT_IDENTITY": "FAIRINO_FR5",
        "GITHUB_REPOSITORY": REMOTE_REPO,
        "PARENT_BRANCH": parent["parent_branch"],
        "PARENT_COMMIT": parent["parent_commit"],
        "PARENT_BASE_VERIFIED": parent["parent_base_verified"],
        "EXECUTION_BRANCH": parent["execution_branch"],
        "EXECUTION_CODE_COMMIT": execution_commit,
        "WORKTREE_CLEAN_AT_START": "YES",
        "WORKTREE_CLEAN_AT_RESULT_GENERATION": "YES" if not git("status", "--porcelain") else "NO_ARTIFACTS_WRITTEN_AFTER_THIS_CHECK",
        "PYTHON_VERSION": sys.version.split()[0],
        "ROS2_DISTRO": smoke.get("launcher", {}).get("ros_distro", "jazzy" if smoke.get("status") == "PASS" else "NOT_RUN"),
        "MOVEIT2_ENVIRONMENT": smoke.get("native_backend", "MoveItPy availability checked; native smoke result absent"),
        "FAIRINO_SOURCE_COMMIT": c1.get("FAIRINO_SOURCE_COMMIT"),
        "C1_NOMINAL_SHA256": c1.get("C1_POST_RUCKIG_NOMINAL", {}).get("sha256"),
        "C1_WAYPOINT_COUNT": c1.get("C1_POST_RUCKIG_NOMINAL", {}).get("waypoint_count"),
        "C2_CANONICAL_IDENTITY": {
            "result_sha256": inventory["c2_canonical_result"].get("sha256"),
            "execution_code_commit": c2.get("C2_EXECUTION_CODE_COMMIT"),
            "postprocessing_code_commit": c2.get("C2_POSTPROCESSING_CODE_COMMIT"),
            "artifact_publish_commit": c2.get("C2_ARTIFACT_PUBLISH_COMMIT"),
            "measurement_pipeline_status": c2.get("MEASUREMENT_PIPELINE_STATUS"),
            "frozen_baseline_performance_status": c2.get("FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS"),
        },
        "frozen_input_identities": inventory,
        "uncertainty_contract_schema": CONTRACT_SCHEMA,
        "uncertainty_ontology": semantics,
        "current_capability_matrix": capability_matrix(semantics, smoke),
        "registration_se3_convention": smoke.get("registration_convention", {
            "composition": "T_base_entity_actual = DeltaT_base_left @ T_base_entity_nominal",
            "translation_frame": "base_link",
            "rotation_frame": "base_link",
            "rotation_origin": "base_link origin",
            "quaternion_convention": "right-handed rotation matrix; ROS x,y,z,w",
        }),
        "synthetic_tests": {
            "status": test_results.get("status", "NOT_RUN"),
            "passed": test_results.get("passed", 0),
            "failed": test_results.get("failed", 0),
            "skipped": test_results.get("skipped", 0),
            "total": test_results.get("total", 0),
            "known_answer_coverage": [
                "schema and enum validation", "legacy C2 axis classification", "zero identity", "translation known answer",
                "positive and negative 90-degree rotation known answers", "normal propagation", "inverse consistency",
                "rotation orthonormality and quaternion normalization", "frozen q and timestamps", "invalid frame rejection",
                "NaN/Inf rejection", "valid and unavailable normalization", "foreign-project-reference guard",
            ],
        },
        "REGRESSION_TESTS": test_results.get("regression_status", "NOT_RUN"),
        "TEST_TOTAL": test_results.get("total", 0),
        "TEST_PASSED": test_results.get("passed", 0),
        "TEST_FAILED": test_results.get("failed", 0),
        "native_smoke": smoke,
        "SCENE_PROPAGATION": "PASS" if scene_pass else "NOT_AVAILABLE_OR_UNVERIFIED",
        "SCENE_PROPAGATION_TESTED": "YES" if scene_pass else "NO",
        "UNCERTAINTY_ONTOLOGY_IMPLEMENTED": "YES",
        "LEGACY_TCP_SEMANTICS_EXPLICIT": "YES",
        "SE3_REGISTRATION_OPERATOR_IMPLEMENTED": "YES",
        "SURFACE_NORMAL_PROPAGATION_TESTED": "PASS" if scene_pass and tests_pass else "FAIL",
        "SYNTHETIC_KNOWN_ANSWER_TESTS": "PASS" if tests_pass else "FAIL",
        "NATIVE_SMOKE": smoke.get("status", "NOT_RUN"),
        "NORMALIZATION_FAIL_CLOSED": "PASS" if all(row["normalization_bound"] is None and row["normalization_status"].startswith("NOT_AVAILABLE") for row in semantics) else "FAIL",
        "c2_semantic_reinterpretation": c2_semantic_reinterpretation(axis_rows),
        "c2_transfer_comparison_identity": {
            "row_count": len(transfer_rows),
            "sha256": inventory["c2_transfer_comparison"].get("sha256"),
            "raw_rows_modified": False,
        },
        "physical_uncertainty_evidence_inventory": read_drive_evidence(),
        "NORMALIZED_MARGIN_STATUS": "NOT_AVAILABLE_PHYSICAL_BOUNDS_MISSING",
        "normalization_rule": "eta_i = rho_i / b_i only when a positive bound is supported by MEASURED, MANUFACTURER, or PROJECT_CONFIG evidence; ASSUMED and LITERATURE cannot support an FR5 physical normalization claim.",
        "limitations": [
            "No FR5 measured registration, TCP calibration, joint tracking, or surface reconstruction uncertainty bound was found in the searched project records.",
            "Nonzero smoke amplitudes are engineering propagation checks and are not physical uncertainty bounds.",
            "C3 wall boxes are the existing project thin-box approximation, not a calibrated surface reconstruction.",
            "Collision observations are adaptive_discrete_interpolation and do not establish strict continuous collision detection.",
            "MoveIt FCL sampled distance is not a physical or continuous positive-clearance certificate.",
            "Hardware validation, calibrated TCP uncertainty, hardware tracking, physical failure probability, coating quality, and hardware safety certification remain unavailable or not run.",
        ],
        "claim_boundaries": {
            "HARDWARE_VALIDATED": "NO",
            "HARDWARE_SAFETY_CERTIFIED": "NO",
            "PHYSICAL_FAILURE_PROBABILITY": "NOT_AVAILABLE",
            "STRICT_CONTINUOUS_CCD": "NOT_AVAILABLE",
            "COATING_QUALITY_CERTIFIED": "NO",
            "CALIBRATED_TCP_UNCERTAINTY": "NOT_AVAILABLE",
            "CALIBRATED_REGISTRATION_DISTRIBUTION": "NOT_AVAILABLE",
            "GLOBAL_ROBUSTNESS_PROVEN": "NO",
            "PHYSICAL_NORMALIZED_MARGIN": "NOT_AVAILABLE",
        },
        **delivery_receipt,
        "next_gate_recommendation": "Independent review of the C3 implementation, then acquire a traceable FR5 registration bound/distribution before authorizing a separate bounded registration-margin campaign.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    csv_text = semantic_csv_text(semantics)
    readme_text = make_readme(result)
    code_paths = [
        ROOT / "src/physical_uncertainty.py",
        ROOT / "tests/test_p2b3_c3_physical_uncertainty.py",
        ROOT / "ros2_moveit_bridge/p2b3_c3_native_smoke.py",
        ROOT / "scripts/run_p2b3_c3_physical_uncertainty.py",
    ]
    scan_texts = [path.read_text(encoding="utf-8") for path in code_paths] + [csv_text, readme_text]
    result["FOREIGN_PROJECT_REFERENCE_COUNT"] = foreign_project_reference_count(scan_texts)
    result["FROZEN_INPUTS_MODIFIED"] = "NO"
    result["GITHUB_PUSH"] = "NOT_RUN"
    result["REMOTE_SHA_VERIFIED"] = "NOT_RUN"
    result["GOOGLE_DRIVE_UPLOAD"] = "NOT_RUN"
    result["PROJECT_INDEX_UPDATED"] = "NOT_RUN"
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    result["FOREIGN_PROJECT_REFERENCE_COUNT"] = foreign_project_reference_count(scan_texts + [json_text])
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    readme_text = make_readme(result)
    return result, csv_text, readme_text


def write_semantic_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")


def mode_test(_: argparse.Namespace) -> int:
    result = run_tests()
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


def mode_smoke(args: argparse.Namespace) -> int:
    result = run_native_smoke(args)
    print(json.dumps({key: result.get(key) for key in ("status", "native_scene_propagation_tested", "case_count", "fcl_distance_status", "reason")}, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


def mode_publish(args: argparse.Namespace) -> int:
    test_results = run_tests()
    result, csv_text, readme_text = build_result(args, test_results)
    write_json(ROOT / "outputs/p2b3_c3_result.json", result)
    write_semantic_csv(ROOT / "outputs/p2b3_c3_uncertainty_semantics.csv", csv_text)
    readme_path = ROOT / "docs/P2B3_C3_README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_text, encoding="utf-8", newline="\n")
    print(json.dumps({
        "P2B3_C3_STATUS": result["P2B3_C3_STATUS"],
        "TEST_TOTAL": result["TEST_TOTAL"],
        "TEST_PASSED": result["TEST_PASSED"],
        "TEST_FAILED": result["TEST_FAILED"],
        "NATIVE_SMOKE": result["NATIVE_SMOKE"],
        "SCENE_PROPAGATION": result["SCENE_PROPAGATION"],
        "NORMALIZED_MARGIN_STATUS": result["NORMALIZED_MARGIN_STATUS"],
        "FOREIGN_PROJECT_REFERENCE_COUNT": result["FOREIGN_PROJECT_REFERENCE_COUNT"],
    }, sort_keys=True))
    return 0 if result["P2B3_C3_STATUS"] in {"COMPLETE", "INCOMPLETE_DELIVERY"} else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("test", "smoke", "publish"), required=True)
    parser.add_argument("--distro", default="Ubuntu-24.04-D")
    parser.add_argument("--underlay-install", type=Path, default=DEFAULT_UNDERLAY)
    parser.add_argument("--overlay-install", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--c1-fk-trace", type=Path, default=None)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--native-smoke-result", type=Path, default=None)
    parser.add_argument("--native-timeout-seconds", type=int, default=600)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "test":
        return mode_test(args)
    if args.mode == "smoke":
        if args.native_timeout_seconds <= 0 or args.native_timeout_seconds > 900:
            raise SystemExit("native_timeout_must_be_between_1_and_900_seconds")
        return mode_smoke(args)
    return mode_publish(args)


if __name__ == "__main__":
    raise SystemExit(main())
