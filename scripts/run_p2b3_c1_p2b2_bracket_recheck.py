#!/usr/bin/env python3
"""Re-evaluate selected old P2-B2 R0 last-pass/first-fail brackets on C1 R0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_p2b1_solver_policy_ablation as p2b1  # noqa: E402
import scripts.run_p2b2_robustness_remapping as p2b2  # noqa: E402
import src.p2a_axiswise_robustness as p2a  # noqa: E402


SELECTED_AXIS_IDS = (
    "joint:j3:positive", "joint:j3:negative",
    "tcp_tcp_translation:x:positive", "tcp_tcp_translation:x:negative",
    "tcp_tcp_rotation:x:positive", "tcp_tcp_rotation:x:negative",
)
NATIVE_SHA256 = "7ccd4514cf7519f9ff652bc2ced785db681e4c47defbc4ab166a6c476c9d7b8a"
FK_SHA256 = "78488b248c15569970d9690fe7976f172bc90d4c71d6628cc4f5e34c28625015"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-post-ruckig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New external scratch output JSON; must not exist")
    parser.add_argument("--scratch", type=Path, required=True, help="New external scratch directory; must not exist")
    parser.add_argument("--fairino-install", type=Path, required=True, help="Verified C1 execution's FAIRINO package prefix")
    parser.add_argument("--runtime-overlay", type=Path, required=True, help="Prior Stage4A native/FK binary overlay")
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--fk-binary", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=ROOT / "outputs/p2b2_inputs/derived_reference_robot_model.urdf")
    parser.add_argument("--distro", default="Ubuntu-24.04-D")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()

    candidate_path, output, scratch, fairino_install, runtime_overlay, native_binary, fk_binary, urdf = (
        path.resolve() for path in (args.candidate_post_ruckig, args.output, args.scratch, args.fairino_install,
                                    args.runtime_overlay, args.native_binary, args.fk_binary, args.urdf)
    )
    if output.exists() or scratch.exists():
        raise RuntimeError("recheck_output_and_scratch_must_be_new_paths")
    for path in (candidate_path, fairino_install / "setup.bash", runtime_overlay / "setup.bash", native_binary, fk_binary, urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    native_sha, fk_sha = file_sha256(native_binary), file_sha256(fk_binary)
    if native_sha != NATIVE_SHA256 or fk_sha != FK_SHA256:
        raise RuntimeError("Stage4A_native_or_FK_binary_identity_mismatch")

    import tools.audit_stage17_reproducibility as d17
    import tools.stage4a_system_benchmark as d46
    from src.process_aware_stress import ProcessTrajectory

    d46.URDF = urdf
    lower, upper, dynamic_limits = d46.load_limits()
    timestamps, q, _velocity, _acceleration, _jerk = d46.load_post_ruckig(candidate_path)
    if q.shape != (181, 6) or timestamps.shape != (181,):
        raise ValueError("C1_post_Ruckig_input_must_be_181_by_6")

    scratch.mkdir(parents=True)
    fk_root = scratch / "nominal_fk"
    fk_root.mkdir()
    p2a._run_existing_fk = lambda batch_root, native_root, module: p2b1._run_fresh_fk(
        batch_root, native_root, module, fk_binary, runtime_overlay, fairino_install, urdf, args.distro,
        timeout_s=args.timeout_seconds,
    )
    d46.run_native = lambda run_dir, cases, native_name="native": p2b1._run_native_with_fresh_build(
        run_dir, cases, native_name, d46, native_binary, runtime_overlay, fairino_install, urdf, args.distro,
        timeout_s=args.timeout_seconds,
    )
    case_id = "P2B3_C1_R0_NOMINAL"
    _trace, q_by_case = p2a._run_fk_only_cases(fk_root, [(case_id, q)], d46)
    fk_cases = p2a._read_fk_trace(fk_root / "fk" / "STAGE4A_FK_TRACE.csv")
    if case_id not in q_by_case or case_id not in fk_cases:
        raise RuntimeError("fresh_nominal_fk_missing")
    fk = fk_cases[case_id]
    poses = np.column_stack((fk["position"], fk["quaternion"]))
    target_positions, _target_quaternions, target_normals = d46.read_targets()
    thresholds = d46.read_json(d46.OUT / "STAGE4_SYSTEM_BENCHMARK_V1.json")["diagnostic_thresholds"]
    aligned_normals, normal_mapping = p2b2._project_target_normals_to_fk_samples(
        target_positions, target_normals, poses[:, :3],
        max_path_deviation_m=float(thresholds["tcp_trajectory_error_m"]["value"]),
    )
    nominal = ProcessTrajectory(
        tcp_poses=poses, joint_states=q, timestamps_s=timestamps, surface_normals=aligned_normals,
        joint_lower_rad=lower, joint_upper_rad=upper,
        metadata={"robot": "FAIRINO_FR5", "scope": "181-point ON-state open-arch only",
                  "candidate": "P2B3_C1_R0", "nominal_source": "fresh Stage4A MoveIt2 FK of C1 post-Ruckig q"},
    )

    axes_by_id = {spec.axis_id: spec for spec in p2a._build_axis_specs(q, lower, upper)}
    landscape = json.loads((ROOT / "outputs/p2b2_robustness_landscape.json").read_text(encoding="utf-8"))
    old_rows = {row["axis_id"]: row for row in landscape["B0_REFERENCE_AXES"]}
    candidates = []
    selected_reference: dict[str, Any] = {}
    for axis_id in SELECTED_AXIS_IDS:
        if axis_id not in axes_by_id or axis_id not in old_rows:
            raise KeyError(f"selected_axis_missing:{axis_id}")
        spec = axes_by_id[axis_id]
        old = old_rows[axis_id]
        if not old.get("last_pass") or not old.get("first_fail"):
            raise ValueError(f"historical_axis_has_no_adjacent_bracket:{axis_id}")
        selected_reference[axis_id] = {
            "historical_last_pass_magnitude": float(old["last_pass"]),
            "historical_first_fail_magnitude": float(old["first_fail"]),
            "historical_failure_mode": old.get("dominant_failure_mode"),
            "units": old.get("units", spec.units),
        }
        for label, magnitude in (("historical_last_pass", float(old["last_pass"])),
                                 ("historical_first_fail", float(old["first_fail"]))):
            candidates.append(p2a._make_candidate(nominal, spec, magnitude, "refinement", len(candidates)))

    scan_root = scratch / "selected_bracket_native"
    scan_root.mkdir()
    evaluator = p2a._D41BatchEvaluator(
        nominal, timestamps, lower, upper, dynamic_limits,
        float(thresholds["tcp_trajectory_error_m"]["value"]),
        float(thresholds["terminal_position_error_m"]["value"]),
        np.radians(float(d17.FORMAL_NORMAL_DEG)),
        float(thresholds["joint_step_rad"]["value"]), scan_root, d46,
    )
    measured = evaluator.evaluate_batch(candidates)
    observations: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        value = measured[candidate.candidate_id]
        label = "historical_last_pass" if candidate.magnitude == selected_reference[candidate.spec.axis_id]["historical_last_pass_magnitude"] else "historical_first_fail"
        observations.setdefault(candidate.spec.axis_id, {})[label] = {
            "magnitude": candidate.magnitude,
            "expected_historical_status": "PASS" if candidate.magnitude == selected_reference[candidate.spec.axis_id]["historical_last_pass_magnitude"] else "FAIL",
            "measured": value.to_record(),
        }
    for axis_id, values in observations.items():
        observations[axis_id] = {"historical_last_pass": values["historical_last_pass"],
                                 "historical_first_fail": values["historical_first_fail"]}

    fresh_xyz = np.asarray(fk["position"], dtype=np.float64)
    c1_fk_rows = list(csv.DictReader((candidate_path.parent / "moveit_fk_tcp_trace.csv").open(encoding="utf-8-sig", newline="")))
    bridge_xyz = np.asarray([[float(row[f"actual_tcp_{axis}"]) for axis in ("x", "y", "z")] for row in c1_fk_rows])
    crosscheck = float(np.max(np.linalg.norm(fresh_xyz - bridge_xyz, axis=1)))
    result = {
        "schema": "p2b3-c1-p2b2-selected-bracket-recheck-v1",
        "project": "FAIRINO_FR5",
        "candidate": "P2B3_C1_R0",
        "runtime": "fresh Stage4A MoveIt2/FK evaluation on selected old P2-B2 R0 bracket cases",
        "historical_axis_source": "outputs/p2b2_robustness_landscape.json:B0_REFERENCE_AXES",
        "recheck_code_commit": git_head(),
        "selected_axis_count": len(SELECTED_AXIS_IDS),
        "selected_case_count": len(candidates),
        "full_p2b2_campaign_replayed": False,
        "selected_axes": SELECTED_AXIS_IDS,
        "historical_brackets": selected_reference,
        "measured_bracket_endpoints": observations,
        "fresh_FK_crosscheck_max_position_delta_m": crosscheck,
        "fresh_FK_normal_reference_mapping": normal_mapping,
        "native_binary_sha256": native_sha,
        "fk_binary_sha256": fk_sha,
        "native_batches": evaluator._batch_number,
        "capability_limits": {"collision_method": "adaptive_discrete_interpolation", "strict_self_ccd": "NOT_AVAILABLE",
                              "clearance_acceptance_threshold": "UNRESOLVED_THRESHOLD", "hardware_validation": "NOT_RUN"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_axis_count": len(SELECTED_AXIS_IDS), "selected_case_count": len(candidates),
                      "output": str(output), "fk_crosscheck_max_delta_m": crosscheck}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
