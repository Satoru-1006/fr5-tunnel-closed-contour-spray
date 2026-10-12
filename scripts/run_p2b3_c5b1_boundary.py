#!/usr/bin/env python3
"""C5B1 signed BASE_LEFT registration-ray scan orchestration.

Default backend is REAL native C5A FCL (requires an existing authenticated
ROS2 Jazzy / MoveIt2 runtime). Fixture backend is TEST-ONLY and MUST NEVER
be reported as a measured FR5 margin. Writes one JSON outside repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p2b3_c5b1_boundary import DEFAULT_LIMITS, scan_all, ray_transform
from scripts.verify_p2b3_c5b1_frozen_baseline import check as check_frozen

PILOT = (("tx", -1), ("tx", 1), ("rx", -1), ("rx", 1))
RAY_LABELS = {f"{axis}{'+' if sign > 0 else '-'}": (axis, sign) for axis, sign in (
    (axis, sign) for axis in ("tx", "ty", "tz", "rx", "ry", "rz") for sign in (-1, 1)
)}


def native_distance_coverage(raw: Mapping[str, Any], domain: str, expected_samples: int = 858) -> bool:
    """Return coverage from the native C++ evaluator's plural sample-count fields."""
    field = {
        "robot_world": "robot_world_distance_valid_samples",
        "self": "self_distance_valid_samples",
    }.get(domain)
    if field is None:
        raise ValueError(f"unsupported_native_distance_domain:{domain}")
    return raw.get(field) == expected_samples


def parse_ray_selection(value: str | None) -> tuple[tuple[str, int], ...] | None:
    if value is None:
        return None
    labels = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not labels or len(set(labels)) != len(labels) or any(item not in RAY_LABELS for item in labels):
        raise ValueError("invalid_ray_selection")
    return tuple(RAY_LABELS[item] for item in labels)


def fixture_evaluator(axis: str, sign: int, magnitude: float) -> dict[str, Any]:
    """Synthetic deterministic validation cases, NOT FCL, NOT measured FR5."""
    direction = (1.1 if sign < 0 else 1.0)
    if axis[0] == "t":
        robot_world = 0.080 - 5.0 * magnitude * direction
        position = 3.5 * magnitude
        normal_deg = 1.0 + 150.0 * magnitude
    else:
        robot_world = 0.080 - 2.7 * magnitude * direction
        position = 0.25 * magnitude
        normal_deg = 1.0 + 350.0 * magnitude
    return {
        "backend_status": "SYNTHETIC_KNOWN_ANSWER_ONLY",
        "robot_world_collision_coverage": True,
        "self_collision_coverage": True,
        "robot_world_distance_m": robot_world,
        "robot_world_collision_samples": int(robot_world <= 0.0),
        "self_distance_m": 0.016,
        "self_collision_samples": 0,
        "max_position_error_m": position,
        "max_normal_error_deg": normal_deg,
        "critical_waypoint": 86,
        "critical_segment": "85->86",
        "failure_mode": "SYNTHETIC_THRESHOLD",
    }


class NativeEvaluator:
    """One existing C5A native FCL case per ray sample, frozen q/time every time."""

    def __init__(self, scratch: Path, native_binary: Path, timeout: int):
        import numpy as np
        from scripts.run_p2b3_c5a_registration_metrics import (
            C1_TRAJECTORY, C1_TARGET, C1_FK, C4_MANIFEST, URDF, SRDF,
            read_csv, read_json, write_q_only, transform_to_csv, grouped_fk, c1_polyline_projector
        )
        from src.p2b3_c5a_metrics import process_metrics, scene_rows
        from src.physical_uncertainty import registration_transform

        if not native_binary.is_file() or not os.access(native_binary, os.X_OK):
            raise RuntimeError("NATIVE_EXECUTABLE_NOT_AVAILABLE")
        if not URDF.is_file() or not SRDF.is_file():
            raise RuntimeError("FROZEN_FR5_URDF_OR_SRDF_MISSING")
        if scratch.exists():
            raise RuntimeError("NEW_EXTERNAL_SCRATCH_REQUIRED")
        if scratch == ROOT or ROOT in scratch.parents:
            raise RuntimeError("SCRATCH_MUST_BE_OUTSIDE_REPOSITORY")
        scratch.mkdir(parents=True)
        self.np = np
        self.scratch = scratch
        self.binary, self.timeout = native_binary, timeout
        self.urdf, self.srdf = URDF, SRDF
        self.reg = registration_transform
        self.scene_rows, self.process_metrics = scene_rows, process_metrics
        self.transform_to_csv, self.grouped_fk = transform_to_csv, grouped_fk
        self.projector = c1_polyline_projector()
        self.manifest = read_json(C4_MANIFEST)
        self.qfile = scratch / "frozen_q_only.csv"
        rows = read_csv(C1_TRAJECTORY)
        self.q, self.times = write_q_only(self.qfile, rows)
        if self.q.shape != (181, 6):
            raise RuntimeError("C1_FROZEN_Q_SHAPE_WRONG")
        self.target_rows = read_csv(C1_TARGET)
        if len(self.target_rows) != 181:
            raise RuntimeError("FROZEN_TARGET_181_ROWS_REQUIRED")
        self.target = np.asarray([[float(r[k]) for k in ("x","y","z","qx","qy","qz","qw")] for r in self.target_rows])
        self.normals = np.asarray([[float(r[k]) for k in ("nx","ny","nz")] for r in self.target_rows])
        # C4 authenticated stand-off is 0.260m. This is geometry, not coating acceptance.
        self.stand_off = 0.260
        self.calls = 0

    def __call__(self, axis: str, sign: int, magnitude: float) -> dict[str, Any]:
        # Exceptions are mapped by the boundary kernel to UNKNOWN, never PASS.
        self.calls += 1
        transform = ray_transform(axis, sign, magnitude)
        delta = self.reg(transform["translation_base_m"], transform["rotation_vector_base_rad"])
        self.transform_to_csv(self.scratch / "active_scene.csv", self.scene_rows(self.manifest, delta))
        with (self.scratch / "active_cases.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("case_id", "q_csv", "scene_csv"))
            writer.writerow(("c5b1_ray", str(self.qfile), str(self.scratch / "active_scene.csv")))
        output = self.scratch / "native_output"
        output.mkdir(exist_ok=True)
        for name in ("p2b3_c5a_native_cases.jsonl", "p2b3_c5a_fresh_fk.csv"):
            old = output / name
            if old.exists():
                old.unlink()
        completed = subprocess.run([
            str(self.binary), "--cases", str(self.scratch / "active_cases.csv"),
            "--urdf", str(self.urdf), "--srdf", str(self.srdf),
            "--output", str(output), "--group", "fairino5_v6_group",
            "--tip", "spray_tcp_link",
        ], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=self.timeout)
        if completed.returncode:
            raise RuntimeError(f"NATIVE_EXIT_{completed.returncode}:{completed.stdout[-250:]}")
        raw_path = output / "p2b3_c5a_native_cases.jsonl"
        fk_path = output / "p2b3_c5a_fresh_fk.csv"
        raw_rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        if len(raw_rows) != 1 or raw_rows[0].get("case_id") != "c5b1_ray":
            raise RuntimeError("NATIVE_CASE_ID_MISMATCH")
        raw = raw_rows[0]
        with fk_path.open(newline="", encoding="utf-8") as f:
            fresh_rows = list(csv.DictReader(f))
        if len(fresh_rows) != 181 or raw.get("sample_count") != 858:
            raise RuntimeError("NATIVE_FK_OR_SAMPLE_COUNT_MISMATCH")
        fk = self.grouped_fk(fresh_rows)["c5b1_ray"]
        if not self.np.array_equal(fk["time"], self.times):
            raise RuntimeError("FROZEN_TIMESTAMPS_CHANGED")
        metrics = self.process_metrics(
            actual_position_xyz=fk["position"], actual_rotation_matrices=fk["rotation"],
            target_pose_xyz_quat_xyzw=self.target, target_normals=self.normals,
            delta_transform=delta, stand_off_m=self.stand_off,
            project_to_polyline=self.projector,
        )
        def distance(name: str) -> float | None:
            obj = raw.get(name)
            return float(obj["distance_m"]) if isinstance(obj, dict) and obj.get("status") == "AVAILABLE" and obj.get("distance_m") is not None else None
        world = raw.get("minimum_robot_world_clearance") or {}
        return {
            "backend_status": "REAL_FR5_C5A_NATIVE_FCL_SAMPLED",
            "robot_world_collision_coverage": native_distance_coverage(raw, "robot_world"),
            "self_collision_coverage": native_distance_coverage(raw, "self"),
            "robot_world_distance_m": distance("minimum_robot_world_clearance"),
            "self_distance_m": distance("minimum_self_clearance"),
            "robot_world_collision_samples": raw.get("robot_world_collision_samples"),
            "self_collision_samples": raw.get("self_collision_samples"),
            "max_position_error_m": metrics["max_position_error_m"],
            "max_normal_error_deg": math.degrees(metrics["max_normal_error_rad"]),
            "critical_waypoint": metrics["critical_position_waypoint"],
            "critical_segment": world.get("critical_sample", {}).get("segment"),
            "critical_locations": {
                "robot_world_collision": {
                    "waypoint": world.get("critical_sample", {}).get("waypoint_index"),
                    "segment": world.get("critical_sample", {}).get("segment"),
                    "sample": world.get("critical_sample", {}).get("sample_index"),
                },
                "self_collision": {
                    "waypoint": (raw.get("minimum_self_clearance") or {}).get("critical_sample", {}).get("waypoint_index"),
                    "segment": (raw.get("minimum_self_clearance") or {}).get("critical_sample", {}).get("segment"),
                    "sample": (raw.get("minimum_self_clearance") or {}).get("critical_sample", {}).get("sample_index"),
                },
                "position_diagnostic": {"waypoint": metrics["critical_position_waypoint"], "segment": None},
                "normal_diagnostic": {"waypoint": metrics["critical_normal_waypoint"], "segment": None},
            },
            "failure_mode": "MODEL_COLLISION_OR_ENGINEERING_GEOMETRY_GATE",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("native", "fixture"), default="native")
    ap.add_argument("--mode", choices=("pilot", "all", "staged"), default="staged")
    ap.add_argument("--native-binary", type=Path, help="existing built p2b3_c5a_native executable")
    ap.add_argument("--scratch", type=Path, help="new external scratch directory (native)")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--coarse-steps", type=int, default=8)
    ap.add_argument("--max-evaluations-per-ray", type=int, default=80)
    ap.add_argument("--native-timeout", type=int, default=300)
    ap.add_argument("--rays", help="comma-separated signed ray labels for bounded parallel shards, e.g. tx-,tx+,rx-,rx+")
    args = ap.parse_args()
    if args.output_json.resolve().is_relative_to(ROOT):
        ap.error("--output-json must be outside repository")
    if args.output_json.exists():
        ap.error("--output-json must not overwrite prior results")
    # Never proceed with an identity violation, including in fixture mode.
    stages = {}
    try:
        selection = parse_ray_selection(args.rays)
        check_frozen(ROOT)
        if args.backend == "native":
            if args.scratch is None or args.native_binary is None:
                raise ValueError("native backend requires --scratch and --native-binary")
            backend = NativeEvaluator(args.scratch.resolve(), args.native_binary.resolve(), args.native_timeout)
        else:
            backend = fixture_evaluator
        if args.mode in ("pilot", "staged"):
            pilot = scan_all(backend, limits=DEFAULT_LIMITS["pilot"],
                             coarse_steps=args.coarse_steps,
                             max_evaluations_per_ray=args.max_evaluations_per_ray,
                             selection=PILOT)
            stages["pilot"] = pilot
            if args.backend == "native":
                blocked = any(
                    p["status"] in ("UNKNOWN_BASELINE", "INDETERMINATE_WITHIN_DOMAIN")
                    for ray in pilot["rays"] for p in ray["predicates"].values()
                )
                if blocked:
                    raise RuntimeError("PILOT_UNKNOWN_BLOCKS_NATIVE_12_RAY_EXPANSION")
        if args.mode in ("all", "staged"):
            stages["all"] = scan_all(backend, limits=DEFAULT_LIMITS["all"],
                                      coarse_steps=args.coarse_steps,
                                      max_evaluations_per_ray=args.max_evaluations_per_ray,
                                      selection=selection)
        result = {
            "schema": "p2b3-c5b1-signed-boundary-v1",
            "project": "FAIRINO_FR5", "backend": args.backend,
            "measurement_type": "SYNTHETIC_TEST_NOT_FR5" if args.backend == "fixture" else "NATIVE_FR5_FCL_SAMPLED",
            "task_status": "COMPLETE",
            "execution_status": "COMPLETE",
            "source_hotfix_commit": "ad842009562351c8b4a587fd9cd42dbfb3d03cb0",
            "c1_nominal_semantic_sha256": "be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3",
            "c4_manifest_semantic_sha256": "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669",
            "registration": "BASE_LEFT", "q_and_timestamps_mutated": False,
            "selected_rays": args.rays,
            "stages": stages, "strict_self_ccd": "NOT_AVAILABLE",
            "physical_registration_bound": "NOT_AVAILABLE",
            "coating_quality_failure_predicate": "NOT_AVAILABLE",
            "global_minimum_failure_proven": False,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")
        print(json.dumps({"backend": args.backend, "completed_stages": list(stages),
                          "rays": {k:v["ray_count"] for k,v in stages.items()},
                          "result_json": str(args.output_json)}))
        return 0
    except Exception as exc:
        if stages and not args.output_json.exists():
            partial = {
                "schema": "p2b3-c5b1-signed-boundary-v1",
                "project": "FAIRINO_FR5", "backend": args.backend,
                "measurement_type": "SYNTHETIC_TEST_NOT_FR5" if args.backend == "fixture" else "NATIVE_FR5_FCL_SAMPLED",
                "task_status": "PARTIAL",
                "execution_status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}"[:800],
                "source_hotfix_commit": "ad842009562351c8b4a587fd9cd42dbfb3d03cb0",
                "c1_nominal_semantic_sha256": "be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3",
                "c4_manifest_semantic_sha256": "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669",
                "registration": "BASE_LEFT", "q_and_timestamps_mutated": False,
                "selected_rays": args.rays,
                "stages": stages, "strict_self_ccd": "NOT_AVAILABLE",
                "physical_registration_bound": "NOT_AVAILABLE",
                "coating_quality_failure_predicate": "NOT_AVAILABLE",
                "global_minimum_failure_proven": False,
            }
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(partial, indent=2, allow_nan=False, sort_keys=True) + "\n")
            print(f"PARTIAL_RESULT_JSON={args.output_json}")
        print(f"C5B1_SCAN_BLOCKED:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
