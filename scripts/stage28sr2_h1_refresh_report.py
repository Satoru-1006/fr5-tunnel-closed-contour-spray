"""Refresh H1 report artifacts from an already completed non-formal bag."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage28sr2_h1_capture import TOPICS, finalize, frozen_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    audit_path = out / "stage28sr2_bag_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    qos = yaml.safe_load((out / "stage28sr2_qos_runtime_snapshot.yaml").read_text(encoding="utf-8"))
    geometry = json.loads((out / "stage28sr2_geometry_regression.json").read_text(encoding="utf-8"))
    metadata = audit.setdefault("rosbag_metadata", {})
    if metadata.get("starting_time_ns") is not None:
        metadata["record_start_time"] = datetime.fromtimestamp(metadata["starting_time_ns"] / 1_000_000_000.0, timezone.utc).isoformat()
    if metadata.get("ending_time_ns") is not None:
        metadata["record_end_time"] = datetime.fromtimestamp(metadata["ending_time_ns"] / 1_000_000_000.0, timezone.utc).isoformat()
    metadata["publisher_count"] = {topic: qos["topics"][topic].get("publisher_count") for topic in TOPICS}
    metadata["subscription_qos"] = {topic: qos["topics"][topic].get("recorder_qos_override") for topic in TOPICS}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state = {
        "formal_fjt_goals_sent": 0,
        "graph_probe_during_capture": 0,
        "recorder_process_independent": True,
        "audit": audit,
        "qos": qos,
        "geometry": geometry,
        "frozen_before": frozen_snapshot(),
    }
    report = finalize(out, state)
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    return 0 if report["Stage_2_8S_R2_H1"]["Stage_2_8S_R2_H1"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
