"""D51 shadow campaign driver.

The driver owns only ``outputs/D51_STAGE4_SHADOW``.  It prepares native
trajectory manifests, freezes the offline clearance policy after the protected
baseline is measured, and produces compact promotion evidence.  Protected
D47/D48/D49/Stage-3 paths are input-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D51_STAGE4_SHADOW"
CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)
FAMILIES = {
    "REGRESSION": {"regression_0000", "regression_0001"},
    "NORMAL": {"normal_0000", "normal_0100"},
    "BOUNDARY": {"boundary_0000", "boundary_0100"},
    "COLLISION_SENSITIVE": {"collision_sensitive_0000", "collision_sensitive_0051"},
    "ADVERSARIAL": {"adversarial_0100", "adversarial_0101"},
    "PERTURBATION": {"perturbation_0000", "perturbation_0100"},
}


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def family(case_id: str) -> str:
    for name, ids in FAMILIES.items():
        if case_id in ids:
            return name
    raise ValueError(f"unknown_case:{case_id}")


def d50_trajectory(root: Path, case_id: str) -> Path:
    path = root / "phase_b_jerk_truth" / "baseline_005" / "native_postprocess" / "trajectories" / f"{case_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def candidate_trajectory(root: Path, case_id: str) -> Path:
    return method_trajectory(root, case_id, "global_0075")


def method_trajectory(root: Path, case_id: str, method: str) -> Path:
    path = root / "phase_b_jerk_truth" / method / "native_postprocess" / "trajectories" / f"{case_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_manifest(path: Path, trajectories: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for case_id in CASES:
            writer.writerow((case_id, wsl_path(trajectories[case_id]), family(case_id)))


def prepare(args: argparse.Namespace) -> int:
    root = args.candidate_root.resolve()
    method = "baseline_005" if args.kind == "baseline" else str(args.method)
    trajectories = {case_id: method_trajectory(root, case_id, method) for case_id in CASES}
    output = args.output.resolve()
    write_manifest(output / "manifest.csv", trajectories)
    (output / "trajectory_sources.json").write_text(
        json.dumps({"method": method, "kind": args.kind, "trajectories": {case_id: str(path) for case_id, path in trajectories.items()}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_policy(args: argparse.Namespace) -> int:
    baseline = load_json(args.baseline.resolve())
    cases = [json.loads(line) for line in args.case_summary.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len(cases) != len(CASES):
        raise RuntimeError("baseline_clearance_case_count_mismatch")
    certified = [float(item["minimum_certified_clearance_m"]) for item in cases if item.get("certified")]
    if len(certified) != len(CASES):
        raise RuntimeError("cannot_freeze_policy_from_uncertified_baseline")
    protected_self = float(args.protected_self)
    protected_world = float(args.protected_world)
    model_margin = float(args.model_margin)
    policy = {
        "schema_version": "d51-offline-model-space-clearance-policy-v1",
        "status": "FROZEN_BEFORE_CANDIDATE",
        "scope": "181-point ON-state open-arch Stage 0/1 only; offline model space",
        "hardware_clearance_certification": "not_available_without_physical_robot_evidence",
        "protected_baseline_source": str(args.baseline.resolve()),
        "baseline_certificate_source": str(args.case_summary.resolve()),
        "protected_baseline_sampled_self_clearance_m": protected_self,
        "protected_baseline_sampled_world_clearance_m": protected_world,
        "baseline_minimum_certified_clearance_m": min(certified),
        "protected_baseline_continuous_certificate_floor_m": min(certified),
        "model_uncertainty_margin_m": model_margin,
        "continuous_certificate_threshold_m": 0.0,
        "safety_thresholds_m": {
            "self": max(0.0, protected_self - model_margin),
            "world": max(0.0, protected_world - model_margin),
        },
        "threshold_selection_rule": "protected baseline measured first; subtract predeclared model uncertainty margin; frozen before candidate evaluation",
        "continuous_acceptance_rule": "every interval must have a positive lower bound above the frozen certificate threshold; candidate minimum certificate may not materially regress below the protected baseline floor",
        "hardware_clearance_rule": "physical clearance remains not_available; no model-space certificate is promoted as a hardware guarantee",
        "engineering_equivalence_bands": {
            "clearance_repeatability_m": 1.0e-9,
            "speed_relative": 0.01,
            "torque_slew_relative": 0.01,
            "rationale": "small predeclared numerical bands only; any hard collision or loss of certificate vetoes",
        },
        "equivalence_band_policy": "only numerical repeatability differences inside the predeclared measurement band are engineering-equivalent; hard collision/penetration is never tradeable",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def clearance_command(manifest: Path, output: Path, case_id: str, stride: int, jerk_bound: float, threshold: float) -> str:
    binary = ROOT / "tmp" / "stage4e_install" / "stage4e_continuous_clearance" / "lib" / "stage4e_continuous_clearance" / "stage4e_continuous_clearance"
    urdf = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
    srdf = ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf"
    poses = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
    library_path = ":".join([
        wsl_path(binary.parent), wsl_path(ROOT / "tmp" / "d41_install" / "lib"),
        wsl_path(ROOT / "install" / "lib"), "/opt/ros/jazzy/lib",
        "/opt/ros/jazzy/lib/x86_64-linux-gnu", "/opt/ros/jazzy/opt/sdformat_vendor/lib",
        "/opt/ros/jazzy/opt/gz_math_vendor/lib", "/opt/ros/jazzy/opt/gz_utils_vendor/lib",
        "/opt/ros/jazzy/opt/gz_tools_vendor/lib", "/usr/lib/x86_64-linux-gnu",
    ])
    arguments = [
        wsl_path(binary), "--poses", wsl_path(poses), "--manifest", wsl_path(manifest),
        "--urdf", wsl_path(urdf), "--srdf", wsl_path(srdf), "--output", wsl_path(output),
        "--jerk-bound", str(jerk_bound), "--threshold", str(threshold), "--stride", str(stride),
        "--case", case_id,
    ]
    return "; ".join([
        "source /opt/ros/jazzy/setup.bash",
        f"source {shlex.quote(wsl_path(ROOT / 'install' / 'setup.bash'))}",
        f"export LD_LIBRARY_PATH={shlex.quote(library_path)}",
        f"mkdir -p {shlex.quote(wsl_path(output))}",
        "exec " + " ".join(shlex.quote(arg) for arg in arguments),
    ])


def run_one_clearance(manifest: Path, output_root: Path, case_id: str, stride: int, jerk_bound: float, threshold: float) -> dict[str, Any]:
    output = output_root / case_id
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "continuous_clearance_case_summary.jsonl"
    if summary_path.is_file() and summary_path.stat().st_size > 0:
        lines = [line for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            return {"case_id": case_id, "returncode": 0, "reused": True, "summary": json.loads(lines[0])}
    command = ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", clearance_command(manifest, output, case_id, stride, jerk_bound, threshold)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    (output / "launch.log").write_text((completed.stdout or "") + "\n--- STDERR ---\n" + (completed.stderr or ""), encoding="utf-8", errors="replace")
    if not summary_path.is_file() or summary_path.stat().st_size == 0:
        raise RuntimeError(f"clearance_case_missing_summary:{case_id}:returncode={completed.returncode}")
    lines = [line for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"case_id": case_id, "returncode": completed.returncode, "reused": False, "summary": json.loads(lines[0])}


def run_clearance(args: argparse.Namespace) -> int:
    manifest = args.manifest.resolve()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    case_ids = [str(row["case_id"]) for row in rows]
    if args.case:
        case_ids = [args.case]
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_one_clearance, manifest, output_root, case_id, args.stride, args.jerk_bound, args.threshold): case_id
            for case_id in case_ids
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: case_ids.index(item["case_id"]))
    summaries = [item["summary"] for item in results]
    with (output_root / "continuous_clearance_case_summary.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for summary in summaries:
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
    aggregate = {
        "schema_version": "d51-continuous-clearance-campaign-v1",
        "kind": args.kind,
        "case_count": len(summaries),
        "certified_case_count": sum(bool(item.get("certified")) for item in summaries),
        "all_cases_certified": bool(summaries) and all(bool(item.get("certified")) for item in summaries),
        "minimum_certified_clearance_m": min((float(item["minimum_certified_clearance_m"]) for item in summaries), default=None),
        "backend": "MoveIt2 CollisionEnvFCL DistanceRequest SINGLE",
        "collision_method": "adaptive_discrete_interpolation",
        "trajectory_stride": args.stride,
        "cases": results,
    }
    (output_root / "continuous_clearance_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if aggregate["all_cases_certified"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare-manifest")
    prepare_parser.add_argument("--candidate-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--kind", choices=("baseline", "candidate"), required=True)
    prepare_parser.add_argument("--method", choices=("global_0075", "asymmetric_005v_010a", "local_0075_midpoints"), default="global_0075")
    prepare_parser.set_defaults(function=prepare)
    policy_parser = sub.add_parser("freeze-policy")
    policy_parser.add_argument("--baseline", type=Path, required=True)
    policy_parser.add_argument("--case-summary", type=Path, required=True)
    policy_parser.add_argument("--protected-self", type=float, required=True)
    policy_parser.add_argument("--protected-world", type=float, required=True)
    policy_parser.add_argument("--model-margin", type=float, default=0.001)
    policy_parser.add_argument("--output", type=Path, required=True)
    policy_parser.set_defaults(function=freeze_policy)
    clearance_parser = sub.add_parser("run-clearance")
    clearance_parser.add_argument("--manifest", type=Path, required=True)
    clearance_parser.add_argument("--output", type=Path, required=True)
    clearance_parser.add_argument("--kind", choices=("baseline", "candidate"), required=True)
    clearance_parser.add_argument("--workers", type=int, default=2)
    clearance_parser.add_argument("--stride", type=int, default=100)
    clearance_parser.add_argument("--jerk-bound", type=float, default=8.0)
    clearance_parser.add_argument("--threshold", type=float, default=0.0)
    clearance_parser.add_argument("--case", default=None)
    clearance_parser.set_defaults(function=run_clearance)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
