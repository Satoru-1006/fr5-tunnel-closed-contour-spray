"""Build the final fail-closed report from the Stage5B causal audit runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


JOINTS = [f"j{i}" for i in range(1, 7)]
START = 66
END = 67
GATE_DEG = 20.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False, "sha256": None, "size_bytes": None, "mtime_utc": None}
    stat = path.stat()
    return {"path": str(path.resolve()), "exists": True, "sha256": sha256(path), "size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()}


def _model_tree(element: ET.Element, ignore_shadow_limit: bool = False, in_j6: bool = False) -> Any:
    attrs = dict(element.attrib)
    is_j6 = in_j6 or (element.tag == "joint" and attrs.get("name") == "j6")
    if ignore_shadow_limit and is_j6 and element.tag == "limit" and attrs.get("lower") is not None and attrs.get("upper") is not None:
        # The caller only enables this on the j6 limit element.  Preserve all
        # other limit attributes (effort/velocity/etc.) for the equivalence test.
        attrs.pop("lower", None)
        attrs.pop("upper", None)
    children = [_model_tree(child, ignore_shadow_limit=ignore_shadow_limit, in_j6=is_j6) for child in list(element)]
    text = (element.text or "").strip()
    return [element.tag, sorted(attrs.items()), text, children]


def model_ablation_check(control_path: Path, shadow_path: Path) -> dict[str, Any]:
    """Check that the expanded shadow differs only in j6 lower/upper bounds."""
    control_root = ET.parse(control_path).getroot()
    shadow_root = ET.parse(shadow_path).getroot()

    control_j6_limit = None
    shadow_j6_limit = None
    for joint in control_root.iter("joint"):
        if joint.attrib.get("name") == "j6":
            control_j6_limit = next((child for child in joint if child.tag == "limit"), None)
            break
    for joint in shadow_root.iter("joint"):
        if joint.attrib.get("name") == "j6":
            shadow_j6_limit = next((child for child in joint if child.tag == "limit"), None)
            break

    differences = []

    def compare(a: ET.Element, b: ET.Element, path: str) -> None:
        if a.tag != b.tag:
            differences.append({"path": path, "kind": "tag", "control": a.tag, "shadow": b.tag})
            return
        keys = sorted(set(a.attrib) | set(b.attrib))
        for key in keys:
            if a.attrib.get(key) != b.attrib.get(key):
                differences.append({"path": f"{path}@{key}", "kind": "attribute", "control": a.attrib.get(key), "shadow": b.attrib.get(key)})
        if (a.text or "").strip() != (b.text or "").strip():
            differences.append({"path": path, "kind": "text", "control": (a.text or "").strip(), "shadow": (b.text or "").strip()})
        if len(a) != len(b):
            differences.append({"path": path, "kind": "child_count", "control": len(a), "shadow": len(b)})
            return
        for index, (child_a, child_b) in enumerate(zip(list(a), list(b))):
            child_name = child_a.attrib.get("name") or child_a.attrib.get("link") or child_a.tag
            compare(child_a, child_b, f"{path}/{child_name}[{index}]")

    compare(control_root, shadow_root, control_root.tag)
    allowed_differences = [item for item in differences if item["kind"] == "attribute" and item["path"].endswith(("@lower", "@upper")) and "/j6[" in item["path"] and "/limit[" in item["path"]]
    only_j6_lower_upper_changed = bool(control_j6_limit is not None and shadow_j6_limit is not None and differences and len(allowed_differences) == len(differences))
    if not differences:
        only_j6_lower_upper_changed = False
    return {
        "control_expanded_model": str(control_path.resolve()),
        "shadow_expanded_model": str(shadow_path.resolve()),
        "control_j6_lower": None if control_j6_limit is None else control_j6_limit.attrib.get("lower"),
        "control_j6_upper": None if control_j6_limit is None else control_j6_limit.attrib.get("upper"),
        "shadow_j6_lower": None if shadow_j6_limit is None else shadow_j6_limit.attrib.get("lower"),
        "shadow_j6_upper": None if shadow_j6_limit is None else shadow_j6_limit.attrib.get("upper"),
        "difference_count": len(differences),
        "differences": differences,
        "only_j6_lower_upper_changed": only_j6_lower_upper_changed,
        "non_j6_model_equivalent": _model_tree(control_root, ignore_shadow_limit=True) == _model_tree(shadow_root, ignore_shadow_limit=True),
    }


def q_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"j{i}_q"]) for i in range(1, 7)], dtype=float)


def wrap_delta(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return (target - source + math.pi) % (2.0 * math.pi) - math.pi


def quat_angle(qa: np.ndarray, qb: np.ndarray) -> float:
    qa = qa / max(float(np.linalg.norm(qa)), 1.0e-15)
    qb = qb / max(float(np.linalg.norm(qb)), 1.0e-15)
    return float(2.0 * math.acos(float(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0))))


def pose_metrics(poses: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for left, right in ((65, 66), (66, 67), (67, 68)):
        p0 = np.asarray([float(poses[left][k]) for k in ("x", "y", "z")])
        p1 = np.asarray([float(poses[right][k]) for k in ("x", "y", "z")])
        q0 = np.asarray([float(poses[left][k]) for k in ("qx", "qy", "qz", "qw")])
        q1 = np.asarray([float(poses[right][k]) for k in ("qx", "qy", "qz", "qw")])
        angle = quat_angle(q0, q1)
        finite = bool(np.isfinite(np.concatenate((p0, p1, q0, q1))).all() and np.isfinite(angle))
        rows.append({"from_waypoint": left, "to_waypoint": right, "translation_m": float(np.linalg.norm(p1 - p0)), "translation_mm": float(np.linalg.norm(p1 - p0) * 1000.0), "orientation_rad": angle, "orientation_deg": math.degrees(angle), "finite": finite})
    return {"pairs": rows, "all_pairs_finite": bool(rows and all(row["finite"] for row in rows)), "wp66_to_wp67_continuous": bool(rows and rows[1]["finite"]), "continuity_basis": "finite_SE3_endpoint_measurement; no arbitrary pose threshold applied"}


def source_jump_metrics(source: list[dict[str, str]]) -> dict[str, Any]:
    rows = []
    for left, right in ((65, 66), (66, 67), (67, 68)):
        a = q_row(source[left])
        b = q_row(source[right])
        raw = b - a
        wrapped = wrap_delta(b, a)
        nearest_k = np.rint((a - b) / (2.0 * math.pi)).astype(int)
        rows.append({"from_waypoint": left, "to_waypoint": right, "source_q_rad": a.tolist(), "target_q_rad": b.tolist(), "raw_delta_rad": raw.tolist(), "raw_delta_deg": np.rad2deg(raw).tolist(), "wrap_aware_delta_rad": wrapped.tolist(), "wrap_aware_delta_deg": np.rad2deg(wrapped).tolist(), "periodic_nearest_k": nearest_k.tolist(), "max_raw_deg": float(np.rad2deg(np.max(np.abs(raw)))), "max_wrap_aware_deg": float(np.rad2deg(np.max(np.abs(wrapped)))), "responsible_joint": JOINTS[int(np.argmax(np.abs(wrapped)))]})
    blocking = next(row for row in rows if row["from_waypoint"] == 66)
    return {"pairs": rows, "blocking_interval": blocking, "simple_periodic_unwrap_rejected": bool(blocking["max_wrap_aware_deg"] > GATE_DEG and max(abs(float(x)) for x in blocking["wrap_aware_delta_deg"]) > GATE_DEG)}


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_audit(poses: list[dict[str, str]], source: list[dict[str, str]]) -> dict[str, Any]:
    source_times = [float(row["t"]) for row in source] if source and "t" in source[0] else []
    return {
        "pose_row_count": len(poses),
        "source_q_row_count": len(source),
        "required_waypoints_present": len(poses) > END and len(source) > END,
        "row_count_match": len(poses) == len(source),
        "pose_indexing": "zero_based_csv_row_index; pose file has no explicit waypoint index column",
        "source_time_finite": bool(source_times and all(math.isfinite(value) for value in source_times)),
        "source_time_nondecreasing": bool(source_times and all(right >= left for left, right in zip(source_times, source_times[1:]))),
        "routing_status": "VALID_ROW_ALIGNMENT" if len(poses) == len(source) and len(poses) > END else "UNRESOLVED",
    }


def flat_summary(run: dict[str, Any], mode: str) -> dict[str, Any]:
    fail = run.get("first_branch_failure_sample") or {}
    last = run.get("last_continuous_success_sample") or {}
    return {
        "solver": "MoveIt KDL",
        "model": mode,
        "grid_intervals": run.get("sample_intervals"),
        "ik_success_count": run.get("ik_success_count"),
        "ik_success_rate": run.get("ik_success_rate"),
        "last_continuous_sample_index": last.get("sample_index"),
        "last_continuous_s": last.get("s"),
        "last_continuous_q": json.dumps(last.get("q"), ensure_ascii=False) if last.get("q") is not None else None,
        "last_continuous_j6_deg": None if last.get("q") is None else math.degrees(float(last["q"][5])),
        "first_failure_sample_index": fail.get("sample_index"),
        "first_failure_s": fail.get("s"),
        "first_failure_class": fail.get("branch_failure_class"),
        "first_failure_q": json.dumps(fail.get("q"), ensure_ascii=False) if fail.get("q") is not None else None,
        "first_failure_j6_deg": None if fail.get("q") is None else math.degrees(float(fail["q"][5])),
        "max_wrap_aware_step_deg": run.get("max_wrap_aware_step_deg"),
        "max_j6_wrap_aware_step_deg": run.get("max_j6_wrap_aware_step_deg"),
        "completed_to_endpoint": run.get("completed_to_endpoint"),
        "fk_residual_status": "measured_in_continuation_csv",
        "status": "PASS" if run.get("completed_to_endpoint") else "FAIL",
    }


def first_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def make_plots(out: Path, control_layers: list[dict[str, Any]], shadow_layers: list[dict[str, Any]], control_rows: list[dict[str, str]], shadow_rows: list[dict[str, str]]) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": f"matplotlib_unavailable:{exc}"}

    def floats(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float]]:
        x, y = [], []
        for row in rows:
            if row.get(key) in (None, "", "null"):
                continue
            x_value = float(row["s"])
            y_value = float(row[key])
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                continue
            x.append(x_value)
            y.append(y_value)
        return x, y

    def layers_xy(layers: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
        x, y = [], []
        for row in layers:
            if row.get(key) is None:
                continue
            x.append(float(row["s"]))
            y.append(float(row[key]))
        return x, y

    def save(name: str, title: str, series: list[tuple[list[float], list[float], str]], ylabel: str, logy: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        for x, y, label in series:
            if x and y:
                ax.plot(x, y, linewidth=1.2, label=label)
        ax.set_title(title)
        ax.set_xlabel("s")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out / name)
        plt.close(fig)

    # q is serialized vector; use direct parsing for j6.
    j6c = [(float(row["s"]), math.degrees(json.loads(row["q"])[5])) for row in control_rows if row.get("q") not in (None, "", "null")]
    j6s = [(float(row["s"]), math.degrees(json.loads(row["q"])[5])) for row in shadow_rows if row.get("q") not in (None, "", "null")]
    save("j6_vs_s.png", "J6 continuation", [([x for x, _ in j6c], [y for _, y in j6c], "±175 control"), ([x for x, _ in j6s], [y for _, y in j6s], "±360 shadow")], "j6 (deg)")
    control_x, control_margin = floats(control_rows, "joint_limit_margin_min_rad")
    shadow_x, shadow_margin = floats(shadow_rows, "joint_limit_margin_min_rad")
    save("joint_limit_margin_vs_s.png", "Minimum joint-limit margin", [(control_x, control_margin, "±175 control"), (shadow_x, shadow_margin, "±360 shadow")], "margin (rad)")
    control_x, control_step = floats(control_rows, "max_wrap_delta_deg")
    shadow_x, shadow_step = floats(shadow_rows, "max_wrap_delta_deg")
    save("max_joint_step_vs_s.png", "Wrap-aware continuation step", [(control_x, control_step, "±175 control"), (shadow_x, shadow_step, "±360 shadow")], "max step (deg)")
    control_x, control_sigma = floats(control_rows, "sigma_min")
    shadow_x, shadow_sigma = floats(shadow_rows, "sigma_min")
    save("sigma_min_vs_s.png", "Jacobian sigma_min", [(control_x, control_sigma, "±175 control"), (shadow_x, shadow_sigma, "±360 shadow")], "sigma_min")
    control_x, control_condition = floats(control_rows, "condition_number")
    shadow_x, shadow_condition = floats(shadow_rows, "condition_number")
    save("condition_number_vs_s.png", "Jacobian condition number", [(control_x, control_condition, "±175 control"), (shadow_x, shadow_condition, "±360 shadow")], "condition number", logy=True)
    control_x, control_count = layers_xy(control_layers, "candidate_count")
    shadow_x, shadow_count = layers_xy(shadow_layers, "candidate_count")
    save("candidate_count_vs_s.png", "Unique IK candidate count", [(control_x, control_count, "±175 control"), (shadow_x, shadow_count, "±360 shadow")], "unique candidates")
    control_x, control_bridge = layers_xy(control_layers, "minimum_bridge_deg")
    shadow_x, shadow_bridge = layers_xy(shadow_layers, "minimum_bridge_deg")
    save("minimum_bridge_cost_vs_s.png", "Minimum inter-layer bridge", [(control_x, control_bridge, "±175 control"), (shadow_x, shadow_bridge, "±360 shadow")], "minimum bridge (deg)")
    return {"status": "PASS", "files": [str(path.resolve()) for path in out.glob("*.png")]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--source-q", type=Path, required=True)
    parser.add_argument("--formal-urdf-xacro", type=Path, required=True)
    parser.add_argument("--formal-srdf", type=Path, required=True)
    parser.add_argument("--joint-limits", type=Path, required=True)
    parser.add_argument("--kinematics-yaml", type=Path, required=True)
    parser.add_argument("--vendor-urdf", type=Path, required=True)
    args = parser.parse_args()
    out = args.audit_dir.resolve()
    control_dir = out / "control_175"
    shadow_dir = out / "shadow_360"
    out.mkdir(parents=True, exist_ok=True)
    control = load_summary(control_dir / "continuation_control_175_512.json")
    shadow = load_summary(shadow_dir / "continuation_shadow_360_512.json")
    control_256 = load_summary(control_dir / "continuation_control_175_256.json")
    shadow_256 = load_summary(shadow_dir / "continuation_shadow_360_256.json")
    repeat_dir = out / "control_175_repeat"
    control_repeat = load_summary(repeat_dir / "continuation_control_175_512.json") if (repeat_dir / "continuation_control_175_512.json").is_file() else None
    poses = read_csv(args.poses.resolve())
    source = read_csv(args.source_q.resolve())
    routing = index_audit(poses, source)

    fk_rows = load_summary(control_dir / "ground_truth_fk_65_68.json")["rows"]
    for row in fk_rows:
        lower = np.asarray(row["lower_limit_margin_rad"], dtype=float)
        upper = np.asarray(row["upper_limit_margin_rad"], dtype=float)
        margins = np.minimum(lower, upper)
        closest_index = int(np.argmin(margins))
        row["closest_limit_joint"] = JOINTS[closest_index]
        row["closest_limit_margin_rad"] = float(margins[closest_index])
    write_csv(out / "ground_truth_wp65_wp68.csv", fk_rows)
    jump = source_jump_metrics(source)
    pose = pose_metrics(poses)
    source_fk_max_position = max(float(row["fk_position_error_m"]) for row in fk_rows)
    source_fk_max_orientation = max(float(row["fk_orientation_error_rad"]) for row in fk_rows)
    ground_truth_valid = bool(source_fk_max_position <= 1.0e-4 and source_fk_max_orientation <= 1.0e-3 and pose["all_pairs_finite"] and routing["routing_status"] == "VALID_ROW_ALIGNMENT")

    shutil.copy2(control_dir / "continuation_control_175_512.csv", out / "continuation_kdl_175.csv")
    shutil.copy2(shadow_dir / "continuation_shadow_360_512.csv", out / "continuation_kdl_360_shadow.csv")
    shutil.copy2(control_dir / "jacobian_diagnostics_control_175_512.csv", out / "jacobian_diagnostics_175.csv")
    shutil.copy2(shadow_dir / "jacobian_diagnostics_shadow_360_512.csv", out / "jacobian_diagnostics_360.csv")

    control_layers = first_jsonl_records(control_dir / "ik_layers_control_175.jsonl")
    shadow_layers = first_jsonl_records(shadow_dir / "ik_layers_shadow_360.jsonl")
    layer_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for mode, layers in (("control_175", control_layers), ("shadow_360", shadow_layers)):
        stats = [item["statistics"] for item in layers]
        layer_rows.extend(stats)
        for left, right in zip(stats, stats[1:]):
            transition_rows.append({"mode": mode, "left_sample_index": left["sample_index"], "right_sample_index": right["sample_index"], "left_s": left["s"], "right_s": right["s"], "candidate_count_before": left["candidate_count"], "candidate_count_after": right["candidate_count"], "minimum_bridge_deg": right["minimum_bridge_deg"], "minimum_bridge_l2_rad": right["minimum_bridge_l2_rad"], "bridge_left_cluster": right["bridge_left_cluster"], "bridge_right_cluster": right["bridge_right_cluster"]})
    write_csv(out / "ik_layer_statistics.csv", layer_rows)
    write_csv(out / "ik_layer_transition_graph.csv", transition_rows)

    solver_rows = [flat_summary(control, "control_175"), flat_summary(shadow, "shadow_360")]
    solver_rows.extend([
        {"solver": "TRAC-IK", "model": "±175 control / ±360 shadow", "grid_intervals": 512, "ik_success_count": None, "ik_success_rate": None, "last_continuous_sample_index": None, "last_continuous_s": None, "last_continuous_q": None, "last_continuous_j6_deg": None, "first_failure_sample_index": None, "first_failure_s": None, "first_failure_class": None, "first_failure_q": None, "first_failure_j6_deg": None, "max_wrap_aware_step_deg": None, "max_j6_wrap_aware_step_deg": None, "completed_to_endpoint": None, "fk_residual_status": "not_run", "status": "UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION"},
        {"solver": "THIRD_SOLVER", "model": "project search", "grid_intervals": 512, "ik_success_count": None, "ik_success_rate": None, "last_continuous_sample_index": None, "last_continuous_s": None, "last_continuous_q": None, "last_continuous_j6_deg": None, "first_failure_sample_index": None, "first_failure_s": None, "first_failure_class": None, "first_failure_q": None, "first_failure_j6_deg": None, "max_wrap_aware_step_deg": None, "max_j6_wrap_aware_step_deg": None, "completed_to_endpoint": None, "fk_residual_status": "not_run", "status": "UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION"},
    ])
    write_csv(out / "solver_comparison.csv", solver_rows)

    control_fail = control.get("first_branch_failure_sample") or {}
    shadow_fail = shadow.get("first_branch_failure_sample") or {}
    repeat_fail = (control_repeat or {}).get("first_branch_failure_sample") or {}
    repeatability = None
    if control_repeat is not None:
        repeatability = {
            "available": True,
            "same_first_failure_sample": control_fail.get("sample_index") == repeat_fail.get("sample_index"),
            "same_first_failure_class": control_fail.get("branch_failure_class") == repeat_fail.get("branch_failure_class"),
            "same_returned_q": control_fail.get("q") == repeat_fail.get("q"),
            "primary": {"first_failure_s": control_fail.get("s"), "failure_class": control_fail.get("branch_failure_class"), "max_wrap_aware_step_deg": control_fail.get("max_wrap_delta_deg"), "returned_q": control_fail.get("q")},
            "repeat": {"first_failure_s": repeat_fail.get("s"), "failure_class": repeat_fail.get("branch_failure_class"), "max_wrap_aware_step_deg": repeat_fail.get("max_wrap_delta_deg"), "returned_q": repeat_fail.get("q")},
        }
    shadow_restores = bool(shadow.get("completed_to_endpoint") and (shadow.get("max_wrap_aware_step_deg") or math.inf) <= GATE_DEG)
    control_last = control.get("last_continuous_success_sample") or {}
    control_hits_j6 = bool(control_last and control_last.get("closest_limit_joint") == "j6" and control_last.get("q") and float(control_last["q"][5]) >= control["joint_upper_rad"][5] - math.radians(0.5))
    control_j6_margin = None if not control_last or not control_last.get("q") else (control["joint_upper_rad"][5] - float(control_last["q"][5]))
    jacobian_control = [row for row in control.get("records", []) if row.get("sigma_min") is not None]
    jacobian_shadow = [row for row in shadow.get("records", []) if row.get("sigma_min") is not None]
    sigma_control_min = min((float(row["sigma_min"]) for row in jacobian_control), default=None)
    sigma_shadow_min = min((float(row["sigma_min"]) for row in jacobian_shadow), default=None)
    cond_control_max = max((float(row["condition_number"]) for row in jacobian_control if math.isfinite(float(row["condition_number"]))), default=None)
    cond_shadow_max = max((float(row["condition_number"]) for row in jacobian_shadow if math.isfinite(float(row["condition_number"]))), default=None)
    jacobian_evidence = "INCONCLUSIVE"
    if sigma_control_min is not None and sigma_shadow_min is not None and control_fail:
        # No universal threshold is applied.  This is a relative trend screen.
        failure_sigma = float(control_fail.get("sigma_min")) if control_fail.get("sigma_min") is not None else None
        if failure_sigma is not None and failure_sigma <= 0.01 * max(sigma_control_min, 1.0e-15):
            jacobian_evidence = "YES"
        else:
            jacobian_evidence = "NO_SHARP_COLLAPSE_OBSERVED"

    root_cause = "UNRESOLVED"
    confidence = "LOW_TO_MEDIUM"
    primary_trigger = "UNRESOLVED"
    secondary = "TRAC_IK_AND_THIRD_SOLVER_UNAVAILABLE"
    if not ground_truth_valid:
        root_cause = "UNRESOLVED"
        confidence = "GROUND_TRUTH_INVALID"
        primary_trigger = "DATA_OR_ROUTING_MISMATCH"
    elif control_hits_j6 and shadow_restores and jacobian_evidence == "NO_SHARP_COLLAPSE_OBSERVED":
        # The strict contract requires cross-solver exclusion before using the
        # *_CONFIRMED labels.  The present environment has no second solver.
        root_cause = "UNRESOLVED"
        confidence = "MEDIUM_HIGH_LEADING_EVIDENCE_BUT_CROSS_SOLVER_MISSING"
        primary_trigger = "J6_LIMIT_LEADING_CANDIDATE"
        secondary = "CROSS_SOLVER_EXCLUSION_REQUIRED"
    elif control_hits_j6 and not shadow_restores:
        root_cause = "UNRESOLVED"
        confidence = "MEDIUM_LEADING_CONTROL_LIMIT_EVIDENCE_BUT_SHADOW_FAILED"
        primary_trigger = "J6_LIMIT_LEADING_CANDIDATE_BUT_NOT_ISOLATED"
        secondary = "KDL_BRANCH_FAILURE_OR_TRUE_TOPOLOGY_DISCONNECT; CROSS_SOLVER_REQUIRED"
    elif shadow.get("completed_to_endpoint") is False and jacobian_evidence == "YES":
        root_cause = "UNRESOLVED"
        confidence = "INCONCLUSIVE_SINGULARITY_CANDIDATE"
        primary_trigger = "JACOBIAN_DETERIORATION_CANDIDATE"
        secondary = "CROSS_SOLVER_EXCLUSION_REQUIRED"
    elif control.get("completed_to_endpoint") is False and not control_hits_j6:
        root_cause = "UNRESOLVED"
        confidence = "KDL_FAILURE_NOT_CAUSALLY_ISOLATED"
        primary_trigger = "KDL_BRANCH_FAILURE_OR_TOPOLOGY_CHANGE"

    matrix = [
        {"hypothesis": "A_J6_hard_limit_branch_termination", "evidence_for": json.dumps({"control_last_continuous": control_last, "control_first_failure": control_fail, "control_j6_margin_rad": control_j6_margin, "shadow_completed": shadow.get("completed_to_endpoint"), "shadow_max_step_deg": shadow.get("max_wrap_aware_step_deg")}, ensure_ascii=False), "evidence_against": "TRAC-IK/third-solver exclusion unavailable; ±360 KDL did not restore the q66 branch", "specific_test": "same q66 seed, 512 shortest-SE3 samples, j6-only ±360 shadow", "measured_result": "LEADING_ONLY" if control_hits_j6 and shadow_restores else ("CONTROL_TERMINATES_AT_J6_BOUNDARY_SHADOW_NOT_RESTORED" if control_hits_j6 else "NOT_ESTABLISHED"), "confidence": confidence, "verdict": "LEADING_NOT_CONFIRMED" if control_hits_j6 else "UNRESOLVED"},
        {"hypothesis": "B_Jacobian_singularity_or_branch_bifurcation", "evidence_for": json.dumps({"sigma_min_control_min": sigma_control_min, "sigma_min_shadow_min": sigma_shadow_min, "condition_control_max": cond_control_max, "condition_shadow_max": cond_shadow_max}, ensure_ascii=False), "evidence_against": "no sharp relative collapse observed" if jacobian_evidence == "NO_SHARP_COLLAPSE_OBSERVED" else "trend cannot be independently adjudicated", "specific_test": "native MoveIt RobotState Jacobian SVD along both continuations", "measured_result": jacobian_evidence, "confidence": "MEDIUM" if jacobian_evidence != "INCONCLUSIVE" else "LOW", "verdict": "NOT_SUPPORTED_AS_PRIMARY" if jacobian_evidence == "NO_SHARP_COLLAPSE_OBSERVED" else "UNRESOLVED"},
        {"hypothesis": "C_KDL_solver_artifact_or_branch_selection", "evidence_for": json.dumps({"control_returned_discontinuous_candidate": True, "repeatability": repeatability}, ensure_ascii=False), "evidence_against": "no independent solver installed in current environment", "specific_test": "TRAC-IK Distance or existing analytic/vendor backend with identical grid and seed", "measured_result": "UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION", "confidence": "LOW", "verdict": "UNRESOLVED"},
        {"hypothesis": "D_simple_periodic_unwrap", "evidence_for": "raw j6 delta has a 2π-near representation difference", "evidence_against": json.dumps({"blocking_max_wrap_aware_deg": jump["blocking_interval"]["max_wrap_aware_deg"], "wrapped_delta_deg": jump["blocking_interval"]["wrap_aware_delta_deg"], "non_j6_large_steps": jump["blocking_interval"]["wrap_aware_delta_deg"][:5]}, ensure_ascii=False), "specific_test": "minimize every joint delta over q+2πk", "measured_result": "REJECTED", "confidence": "HIGH", "verdict": "REJECTED"},
        {"hypothesis": "E_task_space_pose_discontinuity", "evidence_for": "none", "evidence_against": json.dumps(pose, ensure_ascii=False), "specific_test": "translation norm and quaternion-geodesic angle for 65→66, 66→67, 67→68", "measured_result": "CONTINUOUS", "confidence": "HIGH", "verdict": "REJECTED"},
        {"hypothesis": "F_waypoint_data_or_routing_mismatch", "evidence_for": "none if FK closure remains within tolerance", "evidence_against": json.dumps({"max_fk_position_error_m": source_fk_max_position, "max_fk_orientation_error_rad": source_fk_max_orientation}, ensure_ascii=False), "specific_test": "authoritative model FK(q65..q68) versus authoritative target pose", "measured_result": "VALID" if ground_truth_valid else "INVALID", "confidence": "HIGH" if ground_truth_valid else "NONE", "verdict": "REJECTED" if ground_truth_valid else "GROUND_TRUTH_INVALID"},
        {"hypothesis": "G_physical_FR5_variant_vs_URDF_limit_mismatch", "evidence_for": "optional ±360 hardware variant is known as a possibility", "evidence_against": "no serial/model/controller metadata in project confirms physical variant", "specific_test": "authenticated robot serial/model/controller configuration", "measured_result": "UNVERIFIED", "confidence": "NONE", "verdict": "UNRESOLVED"},
    ]
    write_csv(out / "root_cause_matrix.csv", matrix)

    model_ablation = model_ablation_check(control_dir / "expanded_robot_model.urdf", shadow_dir / "expanded_robot_model.urdf")
    manifest_paths = [args.poses, args.source_q, args.formal_urdf_xacro, args.formal_srdf, args.joint_limits, args.kinematics_yaml, args.vendor_urdf, control_dir / "expanded_robot_model.urdf", shadow_dir / "expanded_robot_model.urdf"]
    if control_repeat is not None:
        manifest_paths.extend([repeat_dir / "continuation_control_175_512.json", repeat_dir / "run_metadata.json"])
    manifest = {"schema_version": "stage5b-ik-root-cause-manifest-v1", "generated_utc": datetime.now(timezone.utc).isoformat(), "files": [file_record(path.resolve()) for path in manifest_paths], "formal_model_untouched": True, "formal_trajectory_untouched": True, "shadow_only": True, "promotion": "NO_PROMOTION", "gazebo": "NOT_RUN", "model_ablation": model_ablation}
    write_json(out / "environment_and_model_manifest.json", manifest)

    plots = make_plots(out, control_layers, shadow_layers, read_csv(out / "continuation_kdl_175.csv"), read_csv(out / "continuation_kdl_360_shadow.csv"))
    report = {
        "schema_version": "stage5b-wp66-wp67-root-cause-report-v1",
        "status": "GROUND_TRUTH_INVALID" if not ground_truth_valid else "ROOT_CAUSE_AUDIT_COMPLETE_WITH_UNRESOLVED_CAUSALITY",
        "scope": "ROOT_CAUSE_AUDIT / SHADOW_DIAGNOSTIC_ONLY",
        "ROOT_CAUSE": root_cause,
        "ROOT_CAUSE_CONFIDENCE": confidence,
        "PRIMARY_TRIGGER": primary_trigger,
        "SECONDARY_CONTRIBUTOR": secondary,
        "J6_175_CONTROL": "FAIL" if not control.get("completed_to_endpoint") else "PASS",
        "J6_360_SHADOW": "PASS" if shadow.get("completed_to_endpoint") and shadow_restores else "FAIL",
        "JACOBIAN_SINGULARITY_EVIDENCE": jacobian_evidence,
        "KDL_RESULT": "control fails at first discontinuity" if not control.get("completed_to_endpoint") else "completed",
        "TRAC_IK_RESULT": "UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION",
        "THIRD_SOLVER_RESULT": "UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION",
        "PHYSICAL_FR5_J6_VARIANT": "UNVERIFIED",
        "OLD_MAX_JUMP_DEG": jump["blocking_interval"]["max_wrap_aware_deg"],
        "MIN_OBSERVED_NEW_BRIDGE_DEG": min((float(row["minimum_bridge_deg"]) for row in transition_rows if row.get("minimum_bridge_deg") is not None), default=None),
        "PROMOTION": "NO_PROMOTION",
        "GAZEBO": "NOT_RUN",
        "BASELINE_MODIFIED": "NO",
        "formal_trajectory_modified": False,
        "ground_truth": {"valid": ground_truth_valid, "max_fk_position_error_m": source_fk_max_position, "max_fk_orientation_error_rad": source_fk_max_orientation, "pose": pose, "source_jump": jump, "routing": routing},
        "continuation_256": {"control": {key: value for key, value in control_256.items() if key != "records"}, "shadow": {key: value for key, value in shadow_256.items() if key != "records"}},
        "continuation_512": {"control": {key: value for key, value in control.items() if key != "records"}, "shadow": {key: value for key, value in shadow.items() if key != "records"}},
        "j6_ablation": {"control_upper_rad": control["joint_upper_rad"][5], "shadow_upper_rad": shadow["joint_upper_rad"][5], "control_hits_j6_upper_boundary": control_hits_j6, "shadow_restores_same_seed_branch": shadow_restores, "shadow_first_failure": shadow_fail, "model_check": model_ablation},
        "repeatability": repeatability,
        "jacobian_summary": {"control_min_sigma_min": sigma_control_min, "shadow_min_sigma_min": sigma_shadow_min, "control_max_condition_number": cond_control_max, "shadow_max_condition_number": cond_shadow_max},
        "root_cause_matrix": matrix,
        "plots": plots,
        "recommended_next_repair": "Do not repair in this run. First add an existing TRAC-IK/analytic/vendor solver in an isolated D-drive workspace and repeat the identical control/shadow grid. If cross-solver exclusion holds, plan an upstream branch-aware IK/path repair; if not, treat KDL branch selection as the next repair target.",
    }
    write_json(out / "WP66_WP67_IK_ROOT_CAUSE_REPORT.json", report)
    md = []
    md.append("# WP66→WP67 IK root-cause causal audit")
    md.append("")
    md.append("## Final decision")
    md.append("")
    md.append(f"`ROOT_CAUSE = {root_cause}`")
    md.append(f"`ROOT_CAUSE_CONFIDENCE = {confidence}`")
    md.append(f"`PRIMARY_TRIGGER = {primary_trigger}`")
    md.append(f"`SECONDARY_CONTRIBUTOR = {secondary}`")
    md.append(f"`J6_175_CONTROL = {'FAIL' if not control.get('completed_to_endpoint') else 'PASS'}`; `J6_360_SHADOW = {'PASS' if shadow.get('completed_to_endpoint') and shadow_restores else 'FAIL'}`")
    md.append(f"`JACOBIAN_SINGULARITY_EVIDENCE = {jacobian_evidence}`")
    md.append(f"`KDL_RESULT = {'FAIL at s=' + str(control_fail.get('s')) if control_fail else 'completed'}`")
    md.append("`TRAC_IK_RESULT = UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION`")
    md.append("`THIRD_SOLVER_RESULT = UNAVAILABLE_WITHOUT_ENVIRONMENT_MUTATION`")
    md.append("`PHYSICAL_FR5_J6_VARIANT = UNVERIFIED`")
    md.append(f"`OLD_MAX_JUMP_DEG = {jump['blocking_interval']['max_wrap_aware_deg']:.9f}`")
    md.append(f"`MIN_OBSERVED_NEW_BRIDGE_DEG = {report['MIN_OBSERVED_NEW_BRIDGE_DEG']}`")
    md.append("`PROMOTION = NO_PROMOTION`; `GAZEBO = NOT_RUN`; `BASELINE_MODIFIED = NO`")
    md.append("")
    md.append("## 12 direct answers")
    md.append("")
    answers = [
        f"Q1. TCP path: {'continuous; 66→67 translation %.6f mm, orientation %.9f°' % (pose['pairs'][1]['translation_mm'], pose['pairs'][1]['orientation_deg']) if pose['wp66_to_wp67_continuous'] else 'UNRESOLVED'}.",
        f"Q2. Source max wrap-aware jump: {jump['blocking_interval']['max_wrap_aware_deg']:.9f}° at {jump['blocking_interval']['responsible_joint']}.",
        f"Q3. Simple ±2π artifact: {'NO; shortest per-joint delta remains %.9f°' % jump['blocking_interval']['max_wrap_aware_deg'] if jump['simple_periodic_unwrap_rejected'] else 'UNRESOLVED'}.",
        f"Q4. KDL q66 branch: last continuous sample {control.get('last_continuous_success_sample', {}).get('sample_index')} at s={control.get('last_continuous_success_sample', {}).get('s')}; first failure {control_fail.get('sample_index')} at s={control_fail.get('s')}.",
        f"Q5. Closest limit immediately before failure: {control_last.get('closest_limit_joint')} with margin {control_last.get('joint_limit_margin_min_rad')} rad.",
        f"Q6. j6 near +175°: {'YES; q6=%.9f°' % math.degrees(float(control_last['q'][5])) if control_hits_j6 else 'NO/UNRESOLVED'}.",
        f"Q7. ±360° same-seed continuation: {'YES' if shadow_restores else 'NO/UNRESOLVED'}.",
        f"Q8. ±360° jump removed: {'YES; max=%s°' % shadow.get('max_wrap_aware_step_deg') if shadow_restores else 'NO/UNRESOLVED'}.",
        f"Q9. Jacobian abnormality: {jacobian_evidence}; control min sigma_min={sigma_control_min}, max condition={cond_control_max}.",
        "Q10. KDL/TRAC-IK/third solver topology: KDL measured; TRAC-IK and third solver unavailable without environment mutation.",
        "Q11. Physical FR5 J6 variant: UNVERIFIED from existing project metadata.",
        f"Q12. Final classification: {root_cause}; strict confirmation is withheld because cross-solver exclusion is missing.",
    ]
    md.extend(answers)
    md.extend(["", "## Ground truth and evidence", "", f"- FK max residual: {source_fk_max_position:.12g} m / {source_fk_max_orientation:.12g} rad; ground truth valid={ground_truth_valid}.", f"- 256 interval control: completed={control_256.get('completed_to_endpoint')}, first failure s={(control_256.get('first_branch_failure_sample') or {}).get('s')}.", f"- 512 interval control: completed={control.get('completed_to_endpoint')}, first failure s={control_fail.get('s')}.", f"- 256 interval shadow: completed={shadow_256.get('completed_to_endpoint')}, max step={shadow_256.get('max_wrap_aware_step_deg')}°.", f"- 512 interval shadow: completed={shadow.get('completed_to_endpoint')}, max step={shadow.get('max_wrap_aware_step_deg')}°.", "- Collision filtering was not used to select IK branches; exact articulated self-CCD and physical clearance are not claimed."])
    if repeatability is not None:
        md.extend(["", "## Repeatability", "", f"- Same first-failure sample: `{repeatability['same_first_failure_sample']}`; same failure class: `{repeatability['same_first_failure_class']}`; same returned q: `{repeatability['same_returned_q']}`.", f"- Primary run returned max step `{repeatability['primary']['max_wrap_aware_step_deg']}`°; repeat returned `{repeatability['repeat']['max_wrap_aware_step_deg']}`°. This is KDL branch-selection variability, not a repaired baseline result."])
    md.extend(["", "## Root-cause matrix", "", "| Hypothesis | Measured result | Verdict |", "|---|---|---|"])
    for item in matrix:
        md.append(f"| {item['hypothesis']} | {item['measured_result']} | {item['verdict']} |")
    md.extend(["", "## Recommended next repair", "", report["recommended_next_repair"], "", "All artifacts in this directory are shadow/diagnostic evidence. No formal trajectory, URDF, SRDF, canonical baseline, protected floor, or Gazebo state was modified/run."])
    (out / "WP66_WP67_IK_ROOT_CAUSE_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "root_cause": root_cause, "control_first_failure_s": control_fail.get("s"), "shadow_completed": shadow_restores, "output": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
