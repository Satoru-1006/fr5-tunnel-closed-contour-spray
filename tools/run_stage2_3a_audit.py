#!/usr/bin/env python3
"""Assemble the Stage 2.3A evidence bundle without changing earlier outputs.

The already completed Stage 2.2 native probe is the authoritative fallback when
the strengthened full-contact export cannot finish.  This script never turns
representative contacts into full contact points: the resulting gate remains
``blocked_environment`` until a complete native export is available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "ik_graph_stage2_2"
STAGE21 = ROOT / "outputs" / "ik_graph_stage2_1"
OUT = ROOT / "outputs" / "ik_graph_stage23a"
METHOD = "adaptive_discrete_interpolation"
UNAVAILABLE = "not_available"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_required(source_name: str, output_name: str) -> None:
    shutil.copy2(SOURCE / source_name, OUT / output_name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(STAGE21 / "run_001" / "candidate_provenance.csv")
    poses = read_csv(ROOT / "outputs" / "tcp_poses_base_link.csv")
    source_contacts = SOURCE / "native_probe_run_002" / "native_collision_contacts.csv"
    source_variants = SOURCE / "native_probe_run_002" / "native_collision_variant_summary.csv"
    if len(candidates) != 1439 or len(poses) != 720:
        raise SystemExit(f"frozen count mismatch: candidates={len(candidates)} waypoints={len(poses)}")

    # Keep all native rows already persisted, including the explicit export-status
    # marker.  Do not call this a complete point-level export.
    rows = read_csv(source_contacts)
    fields = list(rows[0]) if rows else []
    write_csv(OUT / "collision_contacts_all.csv", rows, fields)

    copy_map = {
        "collision_pair_summary.csv": "collision_pair_full_summary.csv",
        "candidate_collision_summary.csv": "candidate_collision_summary.csv",
        "waypoint_collision_summary.csv": "waypoint_collision_summary.csv",
        "collision_type_summary.csv": "collision_type_summary.csv",
        "collision_depth_summary.csv": "collision_depth_summary.csv",
        "collision_spatial_clusters.csv": "collision_spatial_clusters.csv",
        "expected_contact_classification.csv": "expected_contact_classification.csv",
        "case_A_self_collision.json": "case_A_self_collision.json",
        "case_B_robot_world_collision.json": "case_B_robot_world_collision.json",
        "case_C_official_baseline.json": "case_C_official_baseline.json",
        "case_D_single_pair_diagnostic.json": "case_D_single_pair_diagnostic.json",
        "robot_state_update_check.json": "robot_state_update_check.json",
    }
    for source_name, output_name in copy_map.items():
        copy_required(source_name, output_name)

    frozen_files = {
        "candidate_set": STAGE21 / "run_001" / "candidate_provenance.csv",
        "waypoint_input": ROOT / "outputs" / "tcp_poses_base_link.csv",
        "planning_scene": STAGE21 / "planning_scene_snapshot.json",
        "urdf": ROOT / "outputs" / "ik_graph_stage193" / "expanded_runtime_urdf.urdf",
        "srdf": ROOT / "ros2_moveit_bridge" / "config" / "fairino5_v6_spray_tcp.srdf",
        "tunnel_geometry": SOURCE / "tunnel_geometry.json",
        "acm": SOURCE / "acm_snapshot.json",
        "tcp_transform": SOURCE / "frame_transform_check.json",
        "padding_scale": SOURCE / "detector_configuration.json",
    }
    manifest = {
        "schema_version": "2.3A",
        "candidate_count": len(candidates),
        "waypoint_count": len(poses),
        "candidate_set_hash": sha256(frozen_files["candidate_set"]),
        "waypoint_input_hash": sha256(frozen_files["waypoint_input"]),
        "planning_scene_hash": sha256(frozen_files["planning_scene"]),
        "urdf_hash": sha256(frozen_files["urdf"]),
        "srdf_hash": sha256(frozen_files["srdf"]),
        "tunnel_geometry_hash": sha256(frozen_files["tunnel_geometry"]),
        "acm_hash": sha256(frozen_files["acm"]),
        "tcp_transform_hash": sha256(frozen_files["tcp_transform"]),
        "padding_hash": sha256(frozen_files["padding_scale"]),
        "scale_hash": sha256(frozen_files["padding_scale"]),
        "candidate_source": str(frozen_files["candidate_set"]),
        "waypoint_source": str(frozen_files["waypoint_input"]),
        "reused_frozen_candidates": True,
        "ik_recomputed": False,
    }
    write_json(OUT / "frozen_input_manifest.json", manifest)

    geometry = read_json(STAGE21 / "geometry_validation.json")
    scene = read_json(SOURCE / "scene_snapshot.json")
    mesh = read_json(SOURCE / "frozen_inputs" / "mesh_manifest.json")
    robot_geometry = read_json(SOURCE / "frozen_inputs" / "robot_collision_geometry_manifest.json")
    frame = read_json(SOURCE / "frame_transform_check.json")
    geometry_audit = {
        "stage": "Stage_2_3A",
        "robot_base_pose": frame.get("world_T_robot_base"),
        "tunnel_pose": frame.get("world_T_tunnel"),
        "tunnel_dimensions_expected": {"wall_thickness_m": 0.04, "span_y_m": 1.10, "floor_z_m": -0.220},
        "tunnel_dimensions_loaded": read_json(SOURCE / "tunnel_geometry.json"),
        "tunnel_units": "meters",
        "tunnel_mesh_bounds": "not_applicable_primitive_boxes",
        "tunnel_duplicate_object_count": len(scene.get("duplicate_object_ids", [])),
        "tunnel_primitive_mesh_overlap": "not_applicable_no_mesh",
        "shoulder_mesh_path": robot_geometry.get("links", {}).get("shoulder_link", {}).get("mesh_path", "not_exposed"),
        "shoulder_mesh_scale": robot_geometry.get("links", {}).get("shoulder_link", {}).get("scale", [1.0, 1.0, 1.0]),
        "shoulder_collision_origin": robot_geometry.get("links", {}).get("shoulder_link", {}).get("origin", "not_exposed"),
        "flange_geometry_loaded": robot_geometry.get("links", {}).get("flange_link", "not_exposed"),
        "wrist3_geometry_loaded": robot_geometry.get("links", {}).get("wrist3_link", "not_exposed"),
        "nozzle_geometry_loaded": robot_geometry.get("links", {}).get("spray_tcp_link", "no_collision_geometry_in_urdf"),
        "duplicate_tool_geometry": "not_found_in_frozen_scene",
        "tcp_origin": frame.get("wrist3_T_tcp"),
        "tcp_axis": "wrist3_link +Z, 0.150 m offset",
        "wrist3_tcp_alignment": frame.get("transform_direction_check"),
        "source_geometry_validation": geometry,
        "source_mesh_manifest_hash": sha256(SOURCE / "frozen_inputs" / "mesh_manifest.json"),
        "audit_limitations": ["visual_check_not_verified", "mesh_watertightness_not_verified", "primitive_scene_not_mesh_scene"],
    }
    write_json(OUT / "geometry_audit.json", geometry_audit)
    write_json(OUT / "frame_transform_check.json", frame)

    variants = read_csv(source_variants)
    by_variant = {row["variant"]: row for row in variants}
    write_csv(OUT / "variant_comparison.csv", variants, list(variants[0]) if variants else [])
    pair_rows = read_csv(SOURCE / "collision_pair_summary.csv")
    pair_rows_sorted = sorted(pair_rows, key=lambda r: (-float(r.get("candidate_coverage_rate") or 0.0), -int(float(r.get("contact_count") or 0))))
    top_pair = pair_rows_sorted[0] if pair_rows_sorted else {}
    all_candidate_pairs = sum(float(r.get("candidate_coverage_rate") or 0.0) >= 1.0 for r in pair_rows)
    collision_types = sorted({r.get("collision_type", "unknown") for r in pair_rows})
    detector_matrix = {
        "collision_method": METHOD,
        "ccd_status": UNAVAILABLE,
        "clearance_status": UNAVAILABLE,
        "fcl_padded": {"status": "executed_native_moveit2", "variant": "case_C_official_baseline", "candidate_count": 1439, "valid_node_count": int(by_variant["case_C_official_baseline"]["valid_node_count"]), "classification_hash": sha256(source_variants)},
        "fcl_unpadded": {"status": "executed_native_moveit2", "variant": "unpadded_diagnostic", "candidate_count": 1439, "valid_node_count": int(by_variant["unpadded_diagnostic"]["valid_node_count"]), "classification_hash": sha256(source_variants)},
        "bullet_padded": {"status": "executed_native_moveit2", "variant": "alternative_bullet_diagnostic", "candidate_count": 1439, "valid_node_count": int(by_variant["alternative_bullet_diagnostic"]["valid_node_count"]), "classification_hash": sha256(source_variants)},
        "bullet_unpadded": {"status": UNAVAILABLE, "reason": "no native Bullet-unpadded run in frozen evidence; not inferred"},
        "interpretation": "FCL padded/unpadded and Bullet padded both classify all frozen candidates as robot-world colliding; this does not prove physical-world collision beyond the modeled scene.",
    }
    write_json(OUT / "detector_matrix.json", detector_matrix)

    # The existing Case D was a single-pair diagnostic.  Preserve it and make
    # the missing iterative peel explicit instead of silently promoting it.
    old_d = read_json(SOURCE / "case_D_single_pair_diagnostic.json")
    write_json(OUT / "acm_peeling_report.json", {
        "status": "incomplete_single_pair_peeling",
        "temporary_only": True,
        "official_acm_modified": False,
        "steps": [{"pair": "shoulder_link<->horseshoe_wall_184", "source": old_d, "executed": True}],
        "next_step_blocked": "full native contact export did not complete; no further ACM pair was masked",
        "broad_acm_disable_used": False,
    })

    old_repro = read_json(SOURCE / "three_run_reproducibility.json")
    write_json(OUT / "three_run_reproducibility.json", {
        "status": "blocked_environment",
        "native_full_contact_export_runs_completed": 0,
        "attempted_run": "native_probe_run_001",
        "attempted_run_output_status": "truncated_or_unfinalized_on_wsl_mounted_filesystem",
        "stage2_2_reproducibility_evidence": old_repro,
        "candidate_set_hash_equal": True,
        "collision_pair_hash_equal": True,
        "waypoint_collision_hash_equal": True,
        "valid_node_count_equal": True,
        "scene_hash_equal": True,
        "output_hash_equal": False,
        "determinism_gate": "not_passed",
    })

    protected = {}
    for path in [
        STAGE21 / "stage2_1_report.md",
        STAGE21 / "gate_report.json",
        SOURCE / "stage2_2_report.md",
        SOURCE / "stage2_2_status.yaml",
        SOURCE / "stage2_2_gate_report.json",
    ]:
        protected[str(path.relative_to(ROOT))] = sha256(path)
    write_json(OUT / "protected_outputs_hash.json", {
        "protected_paths": protected,
        "stage23a_writes_outside_protected_paths": True,
        "legacy_outputs_modified": False,
        "verification_scope": "hashes captured after Stage 2.3A assembly; compare against Stage 2.2 protected hashes when available",
    })

    status = {
        "Stage_2_3A": {
            "status": "blocked_environment",
            "geometry_audit_complete": True,
            "transform_audit_complete": True,
            "full_collision_pair_enumeration_complete": True,
            "full_contact_point_export_complete": False,
            "dominant_pair_masking_eliminated": False,
            "geometry_consistent_with_frozen_stage2_2": True,
            "candidate_count": 1439,
            "waypoint_count": 720,
            "valid_node_count": 0,
            "collision_method": METHOD,
            "ccd_status": UNAVAILABLE,
            "clearance_status": UNAVAILABLE,
            "native_full_contact_runs": 0,
            "reason": "The strengthened native full-contact export blocked on WSL mounted-filesystem I/O before finalization. Existing Stage 2.2 evidence remains auditable but is representative_per_body_pair, not full contact-point persistence.",
            "next": "repeat native full-contact export on a responsive native filesystem; do not enter Stage 2.3B",
        },
        "graph_edges": {"status": "not_evaluated_no_valid_nodes"},
        "transition_evaluation": {"status": "not_evaluated_no_valid_nodes"},
        "ruckig": {"status": "not_evaluated_no_valid_trajectory"},
    }
    (OUT / "stage23a_status.yaml").write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(OUT / "stage23a_gate_report.json", {
        "Stage_2_3A_gate": {
            "geometry_audit_complete": True,
            "transform_audit_complete": True,
            "full_collision_pair_enumeration_complete": True,
            "full_contact_point_export_complete": False,
            "dominant_pair_masking_eliminated": False,
            "correction_physical_basis_documented": False,
            "correction_single_variable_traceable": False,
            "valid_waypoint_coverage": "0/720",
            "prohibited_node_collision_count": 1439,
            "native_runs": 0,
            "candidate_hash_consistent": True,
            "collision_hash_consistent": True,
            "status": "blocked_environment",
            "next": "repeat native full-contact export; remain in Stage 2.3A",
        }
    })

    report = f"""# Stage 2.3A：冻结候选的几何、变换与碰撞接触审计

## 状态

- Stage 2.3A：`blocked_environment`
- 冻结输入：1,439 个 IK 候选、720 个 waypoint；未重新求 IK。
- 当前合法节点：0；图边、过渡碰撞和 Ruckig 均保持未评估。
- 碰撞方法：`{METHOD}`；CCD 与 clearance：`{UNAVAILABLE}`。

## 已确认的冻结证据

Stage 2.2 的真实 MoveIt2 native probe 将 1,439/1,439 候选判为 robot-world collision，self collision 为 0；Case A 去除世界物体后自碰撞为 0，Case B 机器人-世界隔离仍为 1,439/1,439。已枚举的 body-pair 汇总有 {len(pair_rows)} 个 robot-world 对，其中 {all_candidate_pairs} 个对在代表性导出中覆盖全部候选；没有 self pair。`shoulder_link ↔ horseshoe_wall_184` 是最高覆盖的主导对（候选覆盖 {top_pair.get('candidate_coverage_rate', 'not_available')}、waypoint 覆盖 {top_pair.get('waypoint_coverage_rate', 'not_available')}），但不能称为 sole blocker：Case D 临时允许该对后仍有碰撞。

冻结的碰撞行已复制到 `collision_contacts_all.csv`，但每行保留原始 `representative_per_body_pair` 标记。它不是完整 contact-point 持久化；本阶段尝试的增强 native export 在 WSL 挂载盘 I/O 阻塞并未正常终结，因此没有把截断文件纳入证据。

## 几何与变换结论

隧道对象是 `base_link` 下的 oriented box primitive，尺寸、单位、padding/scale、机器人基座和 `wrist3_link → spray_tcp_link` 变换均沿用已冻结的真实审计。当前 URDF 的 `spray_tcp_link` 无 collision geometry，已有接触应解释为机器人连杆—隧道对象碰撞，不能当作喷头工艺接触。视觉对齐、mesh watertightness 和 Bullet unpadded 对照仍不可用。

## 门禁结论

不能进入 Stage 2.3B。原因是完整 contact-point 导出和逐主导碰撞对 Case D 尚未完成；不能通过扩大 ACM、关闭 robot-wall collision 或伪造合法节点来推进。也不能据此声称真实物理世界绝对不可行，因为结论只覆盖冻结候选集合和冻结几何语义。

## 下游状态

```yaml
graph_edges: not_evaluated_no_valid_nodes
transition_evaluation: not_evaluated_no_valid_nodes
ruckig: not_evaluated_no_valid_trajectory
```

详细证据见本目录的 CSV/JSON/YAML 文件；Stage 2.1/2.2 正式输出未被覆盖。

## 必答项

1. 1439 个候选的已持久化碰撞对是本目录 `collision_pair_full_summary.csv` 中的 {len(pair_rows)} 个 `{', '.join(collision_types)}` 对；contact 行为 {len(rows)} 条代表性 body-pair 记录。
2. `shoulder_link ↔ horseshoe_wall_184` 是主导对，但不是唯一拒绝原因；Case D 后仍为 0 个合法节点。
3. 共同拒绝候选的是其它 shoulder/upperarm/forearm 等机器人连杆与 `horseshoe_wall_*` 的 robot-world 对，完整列表见 CSV；其中 {all_candidate_pairs} 个代表性对覆盖全部候选。
4. 当前拒绝类型是 robot-world collision；自碰撞隔离为 0。
5. 接触位置、法向和 penetration depth 见 `collision_contacts_all.csv` 与 `collision_spatial_clusters.csv`；独立 clearance 不可用。
6. 冻结 PlanningScene 未发现重复隧道 object id；primitive/mesh overlap 对当前 primitive 场景不适用。
7. 当前冻结审计支持米制、scale=1、padding max=0；mesh origin 的运行时细节仍标为 not_exposed，不能补充推断。
8. robot base 与 tunnel frame 均在 `base_link`，变换闭合误差沿用冻结审计且通过。
9. TCP 使用 `wrist3_T_tcp`，沿 wrist3 +Z 偏移 0.150 m；`spray_tcp_link` 在 URDF 中无 collision geometry，喷头未作为 attached object 重复加载。
10. frozen FCL padded 与 unpadded 都是 1439/1439 碰撞、0 合法节点；这不支持“padding 单独造成阻塞”。
11. FCL padded 与已有 Bullet padded 分类一致；Bullet unpadded 未运行，已明确标记 not_available。
12. 未恢复合法节点；完整 contact-point native 导出未完成，不能推进 ACM peeling 或恢复节点。
13. 当前应留在 Stage 2.3A，先在可响应 native filesystem 上完成 full contact export；不能进入 Stage 2.3B。
14. 不能声称物理世界绝对不可行，因为证据只覆盖冻结候选集合、冻结几何语义和当前可用检测器，不覆盖全局 IK/姿态/工具设计空间。
"""
    (OUT / "stage23a_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
