"""Evaluate an already-generated native post-Ruckig shadow candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage4c_execution_form import (  # noqa: E402
    D47,
    case_metric,
    load_targets,
    parse_fk,
    parse_geometry,
    run_fk,
    run_geometry,
    summarize,
    windows_path_from_wsl,
    write_manifest,
    write_q_only_trajectory,
)


def main() -> int:
    native_dir = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    summary = json.loads((native_dir / "execution_form_summary.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    actual_items = [(str(item["case_id"]), windows_path_from_wsl(str(item["trajectory_csv"])), str(item["family"])) for item in summary["cases"] if item.get("status") == "PASS"]
    geometry_items = []
    for case_id, actual, family in actual_items:
        q_only = output / "geometry_inputs" / f"{case_id}.csv"
        write_q_only_trajectory(actual, q_only)
        geometry_items.append((case_id, q_only, family))
    manifest = output / "post_manifest.csv"
    write_manifest(manifest, geometry_items)
    run_geometry(manifest, output / "geometry")
    fk_path = run_fk(manifest, output)
    geometry = parse_geometry(output / "geometry")
    fk = parse_fk(fk_path)
    source_by_id = {case_id: D47 / "cases" / f"{case_id}.csv" for case_id, _, _ in actual_items}
    targets = load_targets()
    metrics = [case_metric(case_id, family, source_by_id[case_id], actual, geometry[case_id], fk[case_id], targets) for case_id, actual, family in actual_items]
    result = summarize(metrics, summary, output)
    result["candidate_safety_status"] = "PASS" if all(result["hard_gates"].values()) else "BLOCKED"
    result["promotion_status"] = "PENDING_DETERMINISTIC_REPLAY_AND_C2"
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["candidate_safety_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
