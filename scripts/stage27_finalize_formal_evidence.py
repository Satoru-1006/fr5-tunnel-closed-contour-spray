"""Finalize a stopped Stage 2.7 formal run without changing frozen inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.stage27_formal_native_certification import (
    R2_TRAJECTORY,
    build_determinism,
    final_gate,
    load_limits,
    read_json,
    read_trajectory,
    source_integrity_after,
    write_json,
    write_sums,
)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def finalize(output: Path) -> dict:
    if not output.is_dir():
        raise FileNotFoundError(output)

    trajectory = read_trajectory()
    limits = load_limits()
    exact = read_json(output / "stage27_exact_spline_dynamics.json")
    runtime_record = read_json(output / "stage27_runtime_identity.json")
    runtime = runtime_record.get("formal_probe", runtime_record)
    goal = read_json(output / "stage27_follow_joint_trajectory_goal.json")
    action_record = read_json(output / "stage27_action_execution.json")
    action = action_record.get("native_execution", action_record)
    geometry = read_json(output / "stage27_process_geometry_validation.json")
    source_identity = runtime_record.get("source_identity", {})

    query = action.get("native_query_state_vs_reconstruction", {})
    controller = action.get("controller_state_vs_reconstruction", {})
    query_summary = {
        "schema_version": "stage27-native-vs-reconstruction-v1",
        "native_query_state_verified": bool(query.get("reconstruction_matches_native_runtime"))
        and int(action.get("query_state_samples_received", 0))
        == int(action.get("query_state_samples_requested", 0))
        and int(action.get("query_state_schedule_skipped", 0)) == 0,
        "reconstruction_matches_native": bool(query.get("reconstruction_matches_native_runtime")),
        "query_state_samples_received": int(action.get("query_state_samples_received", 0)),
        "query_state_samples_requested": int(action.get("query_state_samples_requested", 0)),
        "query_state_schedule_skipped": int(action.get("query_state_schedule_skipped", 0)),
        "query_state_schedule_mode": action.get("query_state_schedule_mode"),
        "native_query_state_vs_reconstruction": query,
        "controller_state_vs_reconstruction": controller,
        "interpretation": "Native query_state coverage and controller_state reference crosscheck did not establish the required full-trajectory spline equivalence; downstream formal gates are not authoritative.",
    }
    write_json(output / "stage27_native_vs_reconstruction.json", query_summary)
    write_json(
        output / "stage27_spline_semantics.json",
        {
            "schema_version": "stage27-spline-semantics-v1",
            "runtime_interpolation_method": runtime.get("interpolation_method"),
            "runtime_verified": bool(runtime.get("runtime_verified")),
            "reconstruction_method": "quintic physical-time segment polynomial from positions, velocities, and accelerations",
            "native_source_identity": source_identity,
            "native_query_state_crosscheck": query_summary,
            "formal_authority": "native JTC query_state plus controller_state crosscheck; full equivalence was not established in this run",
        },
    )

    expected = count_csv_rows(output / "stage27_bullet_intervals.csv") if (output / "stage27_bullet_intervals.csv").exists() else 0
    bullet = {
        "schema_version": "stage27-bullet-dense-validation-v1",
        "status": "blocked_by_primary_semantics_mismatch",
        "validation_complete": False,
        "checked_intervals": 0,
        "expected_intervals": expected,
        "collisions": None,
        "criterion": "Native Bullet validation was intentionally stopped after the first formal blocker; no partial result is promoted to a collision pass.",
        "authoritative_backend": "native Bullet stage26 runner with certified FR5 shape mode",
    }
    write_json(output / "stage27_bullet_dense_validation.json", bullet)

    determinism = build_determinism(output, trajectory, limits)
    before = read_json(output / "stage27_frozen_integrity_before.json")
    integrity = source_integrity_after(output, before)
    reconstruction = {"summary": exact}
    gate = final_gate(output, runtime, goal, action, reconstruction, geometry, bullet, determinism, integrity, trajectory)

    # Keep the requested source identity explicit in the manifest-facing record.
    write_json(
        output / "stage27_action_execution.json",
        action_record,
    )
    write_sums(output)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gate = finalize(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "Stage_2_7": gate["Stage_2_7"], "primary_blocker": gate["primary_blocker"]}, ensure_ascii=False))
    return 0 if gate["Stage_2_7"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
