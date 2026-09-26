"""Rerun the complete protected D48 0.05 execution-form gate suite in D49."""

from __future__ import annotations

import json
from pathlib import Path

from tools.stage4c_execution_form import (
    ACCEPTANCE,
    D47,
    case_metric,
    compare_native_replay,
    load_targets,
    parse_fk,
    parse_geometry,
    run_fk,
    run_geometry,
    run_native_postprocess,
    summarize,
    windows_path_from_wsl,
    write_manifest,
    write_q_only_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D49_STAGE4D_SHADOW" / "protected_revalidation"


def family_for(case_id: str) -> str:
    prefix = case_id.rsplit("_", 1)[0].upper()
    return "COLLISION_SENSITIVE" if prefix == "COLLISION" else prefix


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [(case_id, D47 / "cases" / f"{case_id}.csv", family_for(case_id)) for case_id in ACCEPTANCE]
    input_manifest = OUT / "input_manifest.csv"
    write_manifest(input_manifest, items)
    native = run_native_postprocess(input_manifest, OUT / "native_postprocess", 0.05, 0.05)
    replay_native = run_native_postprocess(input_manifest, OUT / "native_postprocess_replay", 0.05, 0.05)
    replay = compare_native_replay(native, replay_native, OUT)

    q_items = []
    for item in native["cases"]:
        case_id = str(item["case_id"])
        post = windows_path_from_wsl(str(item["trajectory_csv"]))
        q_path = OUT / "geometry_inputs" / f"{case_id}.csv"
        write_q_only_trajectory(post, q_path)
        q_items.append((case_id, q_path, family_for(case_id)))
    post_manifest = OUT / "post_manifest.csv"
    write_manifest(post_manifest, q_items)
    geometry_out = OUT / "geometry"
    run_geometry(post_manifest, geometry_out)
    fk_path = run_fk(post_manifest, OUT)
    geometry = parse_geometry(geometry_out)
    fk = parse_fk(fk_path)
    targets = load_targets()
    metrics = []
    for case_id, _, family in items:
        post = windows_path_from_wsl(str(next(item["trajectory_csv"] for item in native["cases"] if str(item["case_id"]) == case_id)))
        metrics.append(case_metric(case_id, family, D47 / "cases" / f"{case_id}.csv", post, geometry[case_id], fk[case_id], targets))
    result = summarize(metrics, native, OUT)
    result["replay"] = replay
    result["hard_gates"]["deterministic_replay"] = replay.get("status") == "PASS"
    result["status"] = "PASS" if result["case_count"] == len(ACCEPTANCE) and all(result["hard_gates"].values()) else "BLOCKED"
    dump(OUT / "metrics.json", result)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
