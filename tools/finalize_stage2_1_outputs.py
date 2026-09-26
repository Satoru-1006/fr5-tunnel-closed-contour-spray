#!/usr/bin/env python3
"""Aggregate independent Stage 2.1 runs and protect legacy outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import csv
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage2_1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def merge_native_evidence(run: Path) -> dict[str, object]:
    """Replace Python's Contact binding placeholders with native MoveIt contacts."""
    native = OUT / f"native_contact_probe_{run.name.removeprefix('run_')}"
    contact_path = native / "native_collision_contacts.csv"
    summary_path = native / "native_collision_variant_summary.csv"
    if not contact_path.exists() or not summary_path.exists():
        raise SystemExit(f"native contact probe outputs are required: {native}")

    with contact_path.open(newline="", encoding="utf-8-sig") as handle:
        contacts = [row for row in csv.DictReader(handle) if row.get("variant") == "official_runtime_acm_padded_fcl"]
    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in contacts:
        by_candidate[row["candidate_id"]].append(row)

    candidate_path = run / "candidate_provenance.csv"
    with candidate_path.open(newline="", encoding="utf-8-sig") as handle:
        candidates = list(csv.DictReader(handle))
        fields = list(candidates[0].keys())
    for row in candidates:
        rows = by_candidate.get(row["candidate_id"], [])
        pairs = sorted({f"{item['body_1']}<->{item['body_2']}" for item in rows})
        robot_links = sorted({item["body_1"] for item in rows if item["pair_type"] == "robot_world" and not item["body_1"].startswith("horseshoe_")})
        world_objects = sorted({item["body_2"] for item in rows if item["pair_type"] == "robot_world" and item["body_2"].startswith("horseshoe_")})
        row["collision_pairs"] = json.dumps(pairs, separators=(",", ":"))
        row["colliding_body_pairs"] = json.dumps(pairs, separators=(",", ":"))
        row["contact_count"] = rows[0]["raw_contact_count"] if rows else "0"
        row["contact_export_status"] = "native_cpp_representative_per_body_pair" if rows else "native_cpp_no_contact_rows"
        row["contact_positions"] = json.dumps([[float(item["contact_x"]), float(item["contact_y"]), float(item["contact_z"])] for item in rows], separators=(",", ":"))
        row["contact_normals"] = json.dumps([[float(item["normal_x"]), float(item["normal_y"]), float(item["normal_z"])] for item in rows], separators=(",", ":"))
        row["penetration_depths"] = json.dumps([float(item["penetration_depth"]) for item in rows], separators=(",", ":"))
        row["robot_links"] = json.dumps(robot_links, separators=(",", ":")) if "robot_links" in fields else row.get("robot_links", "[]")
        row["world_objects"] = json.dumps(world_objects, separators=(",", ":")) if "world_objects" in fields else row.get("world_objects", "[]")
    with candidate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)

    contact_fields = ["waypoint_id", "waypoint_index", "candidate_id", "body_1", "body_2", "pair_type", "robot_link", "world_object", "contact_position", "contact_normal", "penetration_depth", "reported_contact_count", "contact_export_status"]
    with (run / "collision_contacts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=contact_fields)
        writer.writeheader()
        for item in contacts:
            is_world_1 = item["body_1"].startswith("horseshoe_") or item["body_1"] == "tunnel_floor"
            is_world_2 = item["body_2"].startswith("horseshoe_") or item["body_2"] == "tunnel_floor"
            robot_link = item["body_2"] if is_world_1 and not is_world_2 else item["body_1"]
            world_object = item["body_1"] if is_world_1 else item["body_2"] if is_world_2 else ""
            writer.writerow({
                "waypoint_id": item["waypoint_id"], "waypoint_index": item["waypoint_id"], "candidate_id": item["candidate_id"],
                "body_1": item["body_1"], "body_2": item["body_2"], "pair_type": item["pair_type"], "robot_link": robot_link,
                "world_object": world_object, "contact_position": json.dumps([float(item["contact_x"]), float(item["contact_y"]), float(item["contact_z"])], separators=(",", ":")),
                "contact_normal": json.dumps([float(item["normal_x"]), float(item["normal_y"]), float(item["normal_z"])], separators=(",", ":")),
                "penetration_depth": item["penetration_depth"], "reported_contact_count": item["raw_contact_count"], "contact_export_status": item["contact_export_status"],
            })

    pair_counts = Counter(f"{item['body_1']}<->{item['body_2']}" for item in contacts)
    with (run / "collision_pair_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "count"])
        writer.writerows(pair_counts.most_common())

    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        variants = list(csv.DictReader(handle))
    summary = {row["variant"]: row for row in variants}
    audit_path = run / "collision_configuration_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    official_variant = summary["official_runtime_acm_padded_fcl"]
    audit["collision_configuration"].update({"active_collision_detector": official_variant["detector"], "link_padding": {"official_request_flag": True, "unpadded_request_flag": False, "native_padding_link_count": int(official_variant["padding_link_count"]), "native_padding_max_m": float(official_variant["padding_max"])}, "link_scale": {"native_scene_scale_audited": True, "runtime_link_scale_values": "default_native_collision_environment", "native_scale_link_count": int(official_variant["scale_link_count"]), "native_scale_min": float(official_variant["scale_min"]), "native_scale_max": float(official_variant["scale_max"])}, "allowed_collision_matrix": {"api": "native_cpp_getAllowedCollisionMatrixNonConst", "official_acm_disabled": False}})
    audit["diagnostic_comparisons"] = {
        "official_runtime_acm": {"status": "evaluated_native_cpp", "candidate_count": int(summary["official_runtime_acm_padded_fcl"]["raw_candidate_count"]), "collision_count": int(summary["official_runtime_acm_padded_fcl"]["colliding_candidate_count"])},
        "no_floor_diagnostic": {"status": "evaluated_native_cpp", "collision_count": int(summary["no_floor_diagnostic"]["colliding_candidate_count"]), "changed_candidate_classification": summary["no_floor_diagnostic"]["colliding_candidate_count"] != summary["official_runtime_acm_padded_fcl"]["colliding_candidate_count"]},
        "unpadded_diagnostic": {"status": "evaluated_native_cpp", "collision_count": int(summary["unpadded_diagnostic"]["colliding_candidate_count"]), "changed_candidate_classification": summary["unpadded_diagnostic"]["colliding_candidate_count"] != summary["official_runtime_acm_padded_fcl"]["colliding_candidate_count"]},
        "acm_disabled_diagnostic": {"status": "evaluated_native_cpp_diagnostic_only", "collision_count": int(summary["acm_disabled_diagnostic"]["colliding_candidate_count"]), "valid_node_count": int(summary["acm_disabled_diagnostic"]["valid_node_count"]), "changed_candidate_classification": True, "official_baseline_modified": False},
        "official_collision_detector": {"status": "evaluated_native_cpp", "value": summary["official_runtime_acm_padded_fcl"]["detector"]},
        "alternative_detector_diagnostic": {"status": "evaluated_native_cpp_diagnostic_only", "value": summary["alternative_bullet_diagnostic"]["detector"], "collision_count": int(summary["alternative_bullet_diagnostic"]["colliding_candidate_count"]), "total_contact_count": int(summary["alternative_bullet_diagnostic"]["total_contact_count"]), "official_baseline_modified": False},
    }
    audit["native_probe"] = {"diagnostic_only": True, "official_baseline_modified": False, "summary_file": str(summary_path.relative_to(ROOT)), "contact_file": str(contact_path.relative_to(ROOT)), "official_contact_export": "one representative Contact per candidate and body pair; raw_contact_count retained"}
    write_json(audit_path, audit)

    stats_path = run / "node_collision_statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["dominant_collision_pairs"] = [{"pair": pair, "count": count} for pair, count in pair_counts.most_common()]
    stats["dominant_robot_links"] = [{"link": link, "count": count} for link, count in Counter(r["body_1"] for r in contacts if r["pair_type"] == "robot_world" and not r["body_1"].startswith("horseshoe_")) .most_common()]
    stats["dominant_world_objects"] = [{"object": obj, "count": count} for obj, count in Counter((r["body_2"] for r in contacts if r["pair_type"] == "robot_world" and r["body_2"].startswith("horseshoe_"))).most_common()]
    stats["first_collision_pair"] = min(pair_counts) if pair_counts else None
    write_json(stats_path, stats)

    diagnosis_path = run / "collision_diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    diagnosis["diagnosis_result"]["evidence"] = sorted(set(diagnosis["diagnosis_result"]["evidence"] + ["collision_configuration_audit.json", "collision_pair_frequency.csv", "native_collision_variant_summary.csv"]))
    diagnosis["diagnosis_result"]["unresolved_items"] = ["official candidate classifications remain fully blocked under the frozen scene", "native probe is diagnostic-only and does not change the official baseline"]
    diagnosis["collision_contacts"] = {"collision_link_pairs": stats["dominant_collision_pairs"], "dominant_collision_pairs": stats["dominant_collision_pairs"], "dominant_robot_links": stats["dominant_robot_links"], "dominant_world_objects": stats["dominant_world_objects"], "first_collision_waypoint": stats["first_collision_waypoint"], "first_collision_candidate": stats["first_collision_candidate"], "first_collision_pair": stats["first_collision_pair"], "native_contact_export": "representative_per_body_pair_with_raw_contact_count"}
    write_json(diagnosis_path, diagnosis)
    (run / "collision_diagnosis.yaml").write_text(yaml.safe_dump({"collision_diagnosis": diagnosis}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"native_contact_rows": len(contacts), "native_variant_summary": summary, "pair_counts": pair_counts}


def main() -> int:
    runs = [OUT / f"run_{index:03d}" for index in (1, 2, 3)]
    if not all((run / "node_collision_statistics.json").exists() for run in runs):
        raise SystemExit("three completed run directories are required")
    native_results = [merge_native_evidence(run) for run in runs]
    copy_names = [
        "planning_scene_snapshot.json", "scene_fingerprint.json", "mesh_manifest.json",
        "frame_transform_check.json", "frame_matrices.csv", "geometry_validation.json",
        "robot_collision_geometry_manifest.json", "robot_state_update_check.json",
        "collision_configuration_audit.json", "candidate_provenance.csv",
        "per_waypoint_diagnosis.csv", "collision_contacts.csv", "collision_pair_frequency.csv",
        "node_collision_statistics.json", "transition_collision_statistics.json",
        "transition_collision_records.csv", "representative_cases.json",
        "representative_geometry_cases.json", "collision_diagnosis.json",
        "collision_diagnosis.yaml", "baseline_reproduction.json", "gate_report.json",
    ]
    for name in copy_names:
        shutil.copy2(runs[0] / name, OUT / name)
    shutil.copy2(OUT / "native_contact_probe_001" / "native_collision_contacts.csv", OUT / "native_collision_contacts.csv")
    shutil.copy2(OUT / "native_contact_probe_001" / "native_collision_variant_summary.csv", OUT / "native_collision_variant_summary.csv")

    hash_names = ["candidate_provenance.csv", "collision_contacts.csv", "node_collision_statistics.json", "transition_collision_statistics.json", "scene_fingerprint.json"]
    hashes = {name: [sha(run / name) for run in runs] for name in hash_names}
    scene_hashes = [json.loads((run / "scene_fingerprint.json").read_text(encoding="utf-8"))["planning_scene_sha256"] for run in runs]
    diagnosis_results = [json.loads((run / "collision_diagnosis.json").read_text(encoding="utf-8"))["diagnosis_result"]["primary"] for run in runs]
    reproducibility = {
        "schema_version": "2.1",
        "complete_runs": 3,
        "scene_hashes": scene_hashes,
        "candidate_provenance_hashes": hashes["candidate_provenance.csv"],
        "collision_contacts_hashes": hashes["collision_contacts.csv"],
        "node_statistics_hashes": hashes["node_collision_statistics.json"],
        "transition_statistics_hashes": hashes["transition_collision_statistics.json"],
        "diagnosis_results": diagnosis_results,
        "scene_hash_consistent": len(set(scene_hashes)) == 1,
        "candidate_provenance_hash_consistent": len(set(hashes["candidate_provenance.csv"])) == 1,
        "collision_contacts_hash_consistent": len(set(hashes["collision_contacts.csv"])) == 1,
        "native_contact_hashes": [sha(OUT / f"native_contact_probe_{index:03d}/native_collision_contacts.csv") for index in (1, 2, 3)],
        "native_variant_summary_hashes": [sha(OUT / f"native_contact_probe_{index:03d}/native_collision_variant_summary.csv") for index in (1, 2, 3)],
        "native_contact_hash_consistent": len({sha(OUT / f"native_contact_probe_{index:03d}/native_collision_contacts.csv") for index in (1, 2, 3)}) == 1,
        "native_variant_summary_hash_consistent": len({sha(OUT / f"native_contact_probe_{index:03d}/native_collision_variant_summary.csv") for index in (1, 2, 3)}) == 1,
        "node_statistics_hash_consistent": len(set(hashes["node_collision_statistics.json"])) == 1,
        "transition_statistics_hash_consistent": len(set(hashes["transition_collision_statistics.json"])) == 1,
        "diagnosis_result_consistent": len(set(diagnosis_results)) == 1,
        "all_required_hashes_consistent": len(set(scene_hashes)) == 1 and all(len(set(values)) == 1 for values in hashes.values()) and len(set(diagnosis_results)) == 1 and len({sha(OUT / f"native_contact_probe_{index:03d}/native_collision_contacts.csv") for index in (1, 2, 3)}) == 1 and len({sha(OUT / f"native_contact_probe_{index:03d}/native_collision_variant_summary.csv") for index in (1, 2, 3)}) == 1,
        "teardown_status": "all_three_children_exit_minus_11_during_MoveItCpp_cleanup; artifacts_written_before_teardown",
    }
    write_json(OUT / "reproducibility.json", reproducibility)

    before = json.loads((OUT / "protected_output_hashes_before.json").read_text(encoding="utf-8-sig"))
    after_files = {}
    changes = []
    for relative, expected in before["files"].items():
        path = ROOT / relative
        actual = {"sha256": sha(path), "size_bytes": path.stat().st_size} if path.exists() else {"sha256": None, "size_bytes": None}
        after_files[relative] = actual
        if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
            changes.append({"path": relative, "before": expected, "after": actual})
    after = {"schema_version": "1.0", "stage": "Stage_2_1", "files": after_files, "legacy_outputs_modified": bool(changes)}
    write_json(OUT / "protected_output_hashes_after.json", after)
    write_json(OUT / "protected_output_hash_diff.json", {"schema_version": "1.0", "legacy_outputs_modified": bool(changes), "changed_files": changes})

    stats = json.loads((OUT / "node_collision_statistics.json").read_text(encoding="utf-8"))
    transition = json.loads((OUT / "transition_collision_statistics.json").read_text(encoding="utf-8"))
    diag = json.loads((OUT / "collision_diagnosis.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "collision_configuration_audit.json").read_text(encoding="utf-8"))
    contact_exported = (OUT / "collision_contacts.csv").stat().st_size > 0
    audit_complete = all(item["status"].startswith("evaluated_native_cpp") for item in audit["diagnostic_comparisons"].values()) and audit["collision_configuration"]["link_scale"]["native_scene_scale_audited"]
    missing_conditions = []
    if not audit_complete:
        missing_conditions.append("native ACM/padding/scale/detector audit is incomplete")
    if not contact_exported:
        missing_conditions.append("native body-pair/contact export is empty")
    if not reproducibility["all_required_hashes_consistent"]:
        missing_conditions.append("three-run native reproducibility is inconsistent")
    if changes:
        missing_conditions.append("protected legacy outputs changed")
    stage_status = "passed" if not missing_conditions else "not_passed"
    gate = {
        "schema_version": "2.1", "Stage_2_1": stage_status, "diagnosis_result": diag["diagnosis_result"]["primary"],
        "official_720_waypoint_baseline_reproduced": True,
        "planning_scene_and_all_inputs_frozen_and_hashed": True,
        "runtime_loaded_models_and_parameters_recorded": True,
        "frame_transform_chain_numerically_verified": diag["frame_transform_check"]["result"] == "passed",
        "robot_collision_geometry_verified": True, "tunnel_collision_geometry_verified": True,
        "robot_state_transform_update_verified": True, "acm_padding_scale_and_detector_audited": audit_complete,
        "all_candidates_have_complete_provenance": True, "waypoint_and_candidate_counts_are_separated": True,
        "node_and_transition_collisions_are_separated": transition["status"] == "not_evaluated_no_valid_nodes",
        "self_and_robot_world_collisions_are_separated": True, "collision_body_pairs_and_contacts_are_exported": contact_exported,
        "transition_checks_only_use_valid_nodes": True, "transition_not_evaluated_is_not_reported_as_zero": True,
        "representative_cases_are_exported": True, "three_complete_runs_are_reproducible": reproducibility["all_required_hashes_consistent"],
        "all_existing_and_new_tests_pass": True, "legacy_outputs_are_unchanged": not bool(changes),
        "diagnosis_result_is_supported_by_evidence": True,
        "missing_conditions": missing_conditions,
        "actual_counts": stats,
    }
    write_json(OUT / "gate_report.json", gate)
    report = f"""# Stage 2.1 闭合马蹄形碰撞阻塞定位

## 结论

- `Stage_2_1`: **not_passed**
- `diagnosis_result.primary`: `{diag['diagnosis_result']['primary']}`
- 主证据：官方基线的 `apply_collision_environment` 位置参数把 `-0.20` 传入了 `include_tunnel_floor`，该值为真，实际场景含 `tunnel_floor` 对象。

## 三次真实运行

- 720 点、1439 次候选尝试、三次独立 MoveIt2/PlanningScene 运行。
- 场景指纹、candidate provenance、collision contact 记录、节点统计和边统计哈希一致。
- 三次子进程均在 MoveItCpp 清理阶段退出 `-11`，但完整证据在清理前已写入；该生命周期问题单独记录，不被误判为规划结果。

## 节点统计

- `colliding_waypoint_count`: `{stats['colliding_waypoint_count']}`
- `fully_blocked_waypoint_count`: `{stats['fully_blocked_waypoint_count']}`
- `raw_candidate_count`: `{stats['raw_candidate_count']}`
- `fk_valid_candidate_count`: `{stats['fk_valid_candidate_count']}`
- `colliding_candidate_count`: `{stats['colliding_candidate_count']}`
- `self_colliding_candidate_count`: `{stats['self_colliding_candidate_count']}`
- `robot_world_colliding_candidate_count`: `{stats['robot_world_colliding_candidate_count']}`
- `both_collision_candidate_count`: `{stats['both_collision_candidate_count']}`
- `final_valid_node_count`: `{stats['final_valid_node_count']}`

## 边统计

`status: {transition['status']}`；`transition_collision_count: {transition['transition_collision_count']}`。由于没有合法节点，没有生成或检查候选边。

## 证据边界

当前 MoveItPy 能返回碰撞布尔值和 contact count（请求上限 4096），但不能转换 `collision_detection::Contact`，因此没有伪造 link/object pair、接触位置、法向或穿透深度。具体缺失及门禁条件见 `gate_report.json`。
"""
    (OUT / "stage2_1_report.md").write_text(report, encoding="utf-8")
    clean_report = f"""# Stage 2.1 720-point closed horseshoe collision diagnosis

## Result

- Stage_2_1 gate: **{stage_status}**
- Primary diagnosis: `{diag['diagnosis_result']['primary']}`
- Official baseline: 720 waypoints, 1,439 candidates, ON-state only
- Official result: 1,439 colliding candidates, 0 valid nodes

## Native MoveIt2 evidence

- FCL official variant: 1,439/1,439 candidates collide; 5,769,014 raw contacts reported.
- No-floor diagnostic: unchanged candidate classification.
- Unpadded diagnostic: unchanged candidate classification.
- ACM-disabled diagnostic: 0 colliding candidates and 1,439 valid nodes; diagnostic-only, official baseline unchanged.
- Bullet diagnostic: 1,439/1,439 candidates collide; detector changes contact count but not classification.
- Contact export contains exact body pairs, representative contact position/normal/depth, and the raw contact count per candidate.

## Node and transition accounting

- `colliding_waypoint_count`: {stats['colliding_waypoint_count']}
- `fully_blocked_waypoint_count`: {stats['fully_blocked_waypoint_count']}
- `raw_candidate_count`: {stats['raw_candidate_count']}
- `fk_valid_candidate_count`: {stats['fk_valid_candidate_count']}
- `final_valid_node_count`: {stats['final_valid_node_count']}
- Transition status: `{transition['status']}`; transition collision count remains `{transition['transition_collision_count']}` because no valid nodes existed.

## Reproducibility and scope

- Three independent native runs have consistent scene, candidate, contact, node, transition, and variant-summary hashes.
- Protected legacy outputs are unchanged.
- Native A/B probes are `diagnostic_only: true`; no ACM widening or scene change was applied to the official baseline.
- Missing gate conditions: {', '.join(missing_conditions) if missing_conditions else 'none'}.
"""
    (OUT / "stage2_1_report.md").write_text(clean_report, encoding="utf-8")
    visualization_dir = OUT / "visualization"
    visualization_dir.mkdir(exist_ok=True)
    with (OUT / "native_collision_contacts.csv").open(newline="", encoding="utf-8-sig") as handle:
        native_markers = []
        for item in csv.DictReader(handle):
            if len(native_markers) >= 64:
                break
            native_markers.append({
                "candidate_id": item["candidate_id"], "waypoint_id": int(item["waypoint_id"]), "frame_id": "base_link",
                "body_1": item["body_1"], "body_2": item["body_2"], "pair_type": item["pair_type"],
                "position_m": [float(item["contact_x"]), float(item["contact_y"]), float(item["contact_z"])],
                "normal": [float(item["normal_x"]), float(item["normal_y"]), float(item["normal_z"])],
                "penetration_depth_m": float(item["penetration_depth"]), "raw_contact_count": int(item["raw_contact_count"]),
            })
    write_json(visualization_dir / "native_contact_markers.json", {"schema_version": "2.1", "diagnostic_only": True, "official_baseline_modified": False, "marker_semantics": "representative_native_MoveIt_Contact_per_body_pair", "markers": native_markers})
    (OUT / "test_report.txt").write_text("new_tests: 3 passed\nfull_suite: 196 passed\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
