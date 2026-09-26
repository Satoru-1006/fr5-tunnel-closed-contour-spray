"""Scientific regression checks for the D41 offline certification handoff."""

import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D41 = ROOT / "outputs" / "stage3_h13_d41_offline_robot_certification"


def _summary():
    return json.loads((D41 / "D41_SUMMARY.json").read_text(encoding="utf-8"))


def _urdf_bounds():
    urdf = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "derived_robot_model.urdf"
    root = ET.parse(urdf).getroot()
    bounds = {}
    for joint in root.findall("joint"):
        if joint.get("name") in {f"j{i}" for i in range(1, 7)}:
            limit = joint.find("limit")
            bounds[joint.get("name")] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    return np.asarray([bounds[f"j{i}"][0] for i in range(1, 7)]), np.asarray([bounds[f"j{i}"][1] for i in range(1, 7)])


def test_d41_retains_d39_d40_and_causal_authority():
    summary = _summary()
    identity = json.loads((Path(summary["D41_RUN_DIRECTORY"]) / "D41_AUTHORITATIVE_IDENTITY.json").read_text(encoding="utf-8"))
    assert summary["TASK_STATUS"] == "PASS"
    assert summary["D39_RETENTION"] == "PASS"
    assert summary["D40_RETENTION"] == "PASS"
    assert identity["point_count"] == 181
    assert identity["future_joint_reads"] == 0
    assert identity["D40_candidate"] == "routeA_DLS_position_dominant_v4"


def test_d41_native_controls_and_clearances_are_live():
    summary = _summary()
    cert = json.loads((D41 / "D41_COLLISION_CERTIFICATION.json").read_text(encoding="utf-8"))
    controls = {row["control"]: row for row in cert["controls"]}
    assert all(row["status"] == "PASS" for row in controls.values() if "detail" not in row["control"])
    assert controls["endpoint_free_mid_segment_collision"]["native_continuous_collision"] is True
    assert controls["near_miss_detail"]["distance_m"] > 0.0
    assert controls["comfortable_clearance_detail"]["distance_m"] > controls["near_miss_detail"]["distance_m"]
    clearance = list(csv.DictReader((D41 / "D41_CLEARANCE_REPORT.csv").open(newline="", encoding="utf-8")))
    world = [float(row["robot_world_distance_m"]) for row in clearance if row["robot_world_distance_m"] not in ("", "null")]
    self_distance = [float(row["self_distance_m"]) for row in clearance if row["self_distance_m"] not in ("", "null")]
    assert min(world) == summary["MIN_ROBOT_WORLD_CLEARANCE"]
    assert min(self_distance) == summary["MIN_SELF_CLEARANCE"]
    assert min(world) > 0.0 and min(self_distance) > 0.0


def test_d41_matrix_covers_structured_families_without_failures():
    rows = list(csv.DictReader((D41 / "D41_ROBUSTNESS_MATRIX.csv").open(newline="", encoding="utf-8")))
    families = {row["family"] for row in rows}
    assert families == {"initial_state", "cartesian_translation", "surface_normal", "waypoint_density", "curvature_shape", "near_singularity", "near_joint_limit"}
    assert len(rows) == 37
    assert all(row["robustness_status"] == "PASS" and row["future_joint_reads"] == "0" for row in rows)


def test_d41_candidate_trajectory_respects_native_urdf_position_limits():
    summary = _summary()
    candidate = Path(summary["D41_RUN_DIRECTORY"]) / "strict_replay" / "moveit_waypoint_joint_trajectory.csv"
    rows = list(csv.DictReader(candidate.open(newline="", encoding="utf-8")))
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows])
    lower, upper = _urdf_bounds()
    margin = np.minimum(q - lower, upper - q)
    assert np.isfinite(q).all()
    assert float(margin.min()) > 0.0
    assert math.isclose(float(margin.min()), summary["MIN_JOINT_LIMIT_MARGIN"], rel_tol=0.0, abs_tol=1.0e-12)
    assert summary["D40_LOCKED_MIN_JOINT_LIMIT_MARGIN"] < 0.0


def test_d41_native_provenance_distinguishes_world_ccd_from_self_refinement():
    summary = _summary()
    provenance = json.loads((Path(summary["D41_RUN_DIRECTORY"]) / "native" / "nominal" / "D41_native_provenance.json").read_text(encoding="utf-8"))
    assert provenance["collision_detector"] == "Bullet"
    assert "checkRobotCollision(req,result,state1,state2,acm)" in provenance["continuous_robot_world_api"]
    assert provenance["self_continuous_api"].startswith("not_available")
    assert summary["native_d41_candidate_case"]["native_continuous_segment_collision_count"] == 0
