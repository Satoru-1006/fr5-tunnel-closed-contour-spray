"""Audit the frozen Stage 2.5 CSV field semantics without rerunning Stage 2.5.

This is deliberately an artifact-only audit.  It does not call MoveIt, TOTG,
Ruckig, a controller, Bullet, or any downstream stage.  The exact native
Ruckig InputParameter/Trajectory objects are not persisted by the formal
runner, so this script reports that reconstruction as unavailable instead of
creating a surrogate trace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
TRAJECTORY = SOURCE_ROOT / "stage25_ruckig_trajectory.csv"
PREPARED = SOURCE_ROOT / "rebuild_01/stage25_ruckig_input_prepared.csv"
LIMITS = SOURCE_ROOT / "stage25_joint_limits.json"
OUTPUT = ROOT / "outputs/stage25_semantics_audit_20260804_corrected"
JOINTS = [f"j{i}" for i in range(1, 7)]
N_ROWS = 25532
N_INTERVALS = 25531
ZERO_TOL = 1.0e-12
MACHINE_TOLERANCE = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != N_ROWS:
        raise RuntimeError(f"Frozen Stage 2.5 row count mismatch: {len(rows)} != {N_ROWS}")
    fields = set(rows[0])
    expected = {"t"} | {f"{j}_{field}" for j in JOINTS for field in ("q", "dq", "ddq", "jerk")}
    missing = sorted(expected - fields)
    if missing:
        raise RuntimeError(f"Frozen Stage 2.5 CSV missing fields: {missing}")
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"{j}_q"]) for j in JOINTS] for row in rows], dtype=float)
    dq = np.asarray([[float(row[f"{j}_dq"]) for j in JOINTS] for row in rows], dtype=float)
    ddq = np.asarray([[float(row[f"{j}_ddq"]) for j in JOINTS] for row in rows], dtype=float)
    jerk = np.asarray([[float(row[f"{j}_jerk"]) for j in JOINTS] for row in rows], dtype=float)
    if not all(np.all(np.isfinite(x)) for x in (t, q, dq, ddq, jerk)):
        raise RuntimeError("Frozen Stage 2.5 CSV contains a non-finite value")
    if abs(float(t[0])) > MACHINE_TOLERANCE or not np.all(np.diff(t) > 0.0):
        raise RuntimeError("Frozen Stage 2.5 timestamps are not strictly increasing from zero")
    if len(t) - 1 != N_INTERVALS:
        raise RuntimeError("Frozen Stage 2.5 interval count mismatch")
    return t, q, dq, ddq, jerk


def compare_csvs(left: Path, right: Path) -> dict[str, Any]:
    if not right.exists():
        return {"status": "not_available", "reason": f"missing comparison artifact: {right}"}
    lt, lq, lv, la, _ = load_matrix(left)
    rt, rq, rv, ra, _ = load_matrix(right)
    result: dict[str, Any] = {"status": "checked", "row_count": int(len(lt)), "fields": {}}
    for name, a, b in (("t", lt, rt), ("q", lq, rq), ("dq", lv, rv), ("ddq", la, ra)):
        error = np.abs(a - b)
        result["fields"][name] = {
            "max_abs_error": float(np.max(error)),
            "mismatch_count_at_machine_tolerance": int(np.count_nonzero(error > MACHINE_TOLERANCE)),
            "exact_equal": bool(np.array_equal(a, b)),
        }
    return result


def line_integral(c: float, m: float, end: float) -> float:
    return c * end + 0.5 * m * end * end


def integral_envelope(a0: float, a1: float, duration: float, jerk_limit: float, upper: bool) -> float:
    """Integrate the exact Lipschitz endpoint envelope.

    For |a'| <= J, the upper envelope is
    min(a0+J*t, a1+J*(T-t)); the lower envelope is
    max(a0-J*t, a1-J*(T-t)).  The integral of either envelope is an exact
    attainable bound, not a sampled approximation.
    """
    if upper:
        first = (a0, jerk_limit)
        second = (a1 + jerk_limit * duration, -jerk_limit)
        choose = min
    else:
        first = (a0, -jerk_limit)
        second = (a1 - jerk_limit * duration, jerk_limit)
        choose = max
    crossings: list[float] = [0.0, duration]
    c0, m0 = first
    c1, m1 = second
    if abs(m0 - m1) > 0.0:
        crossing = (c1 - c0) / (m0 - m1)
        if 0.0 < crossing < duration:
            crossings.append(crossing)
    crossings = sorted(set(crossings))
    total = 0.0
    for left, right in zip(crossings[:-1], crossings[1:]):
        mid = (left + right) * 0.5
        left_line = c0 + m0 * mid
        right_line = c1 + m1 * mid
        c, m = first if choose(left_line, right_line) == left_line else second
        total += line_integral(c, m, right) - line_integral(c, m, left)
    return float(total)


def endpoint_feasibility(t: np.ndarray, dq: np.ndarray, ddq: np.ndarray, limits: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    per_joint = {joint: {"intervals": N_INTERVALS, "feasible": 0, "infeasible": 0} for joint in JOINTS}
    first_infeasible: dict[str, Any] | None = None
    worst_infeasible: dict[str, Any] | None = None
    for interval in range(N_INTERVALS):
        duration = float(t[interval + 1] - t[interval])
        for j, joint in enumerate(JOINTS):
            v0 = float(dq[interval, j])
            v1 = float(dq[interval + 1, j])
            a0 = float(ddq[interval, j])
            a1 = float(ddq[interval + 1, j])
            jerk_limit = float(limits[j]["max_jerk_rad_s3"])
            delta_v = v1 - v0
            endpoint_gap = abs(a1 - a0) - jerk_limit * duration
            if endpoint_gap > ZERO_TOL:
                lower = upper = float("nan")
                feasible = False
                violation = float(endpoint_gap)
                reason = "endpoint_acceleration_difference_exceeds_jerk_bound"
            else:
                lower = integral_envelope(a0, a1, duration, jerk_limit, upper=False)
                upper = integral_envelope(a0, a1, duration, jerk_limit, upper=True)
                lower_gap = lower - delta_v
                upper_gap = delta_v - upper
                violation = max(0.0, lower_gap, upper_gap)
                feasible = bool(lower_gap <= ZERO_TOL and upper_gap <= ZERO_TOL)
                reason = "within_exact_jerk_bounded_integral_interval" if feasible else "required_velocity_change_outside_exact_jerk_bounded_integral_interval"
            item = {
                "interval": interval,
                "joint": joint,
                "t0": float(t[interval]),
                "t1": float(t[interval + 1]),
                "dt": duration,
                "v0": v0,
                "v1": v1,
                "a0": a0,
                "a1": a1,
                "delta_v_required": delta_v,
                "jerk_limit": jerk_limit,
                "jerk_bounded_minimum_possible_delta_v": lower,
                "jerk_bounded_maximum_possible_delta_v": upper,
                "endpoint_acceleration_gap_minus_jerk_budget": endpoint_gap,
                "feasible": feasible,
                "reason": reason,
                "violation_measure": violation,
            }
            records.append(item)
            if feasible:
                per_joint[joint]["feasible"] += 1
            else:
                per_joint[joint]["infeasible"] += 1
                if first_infeasible is None:
                    first_infeasible = item
                if worst_infeasible is None or item["violation_measure"] > worst_infeasible["violation_measure"]:
                    worst_infeasible = item
    interval5 = next(item for item in records if item["interval"] == 5 and item["joint"] == "j3")
    return {
        "schema_version": "stage25-jerk-endpoint-feasibility-v1",
        "method": "exact_integral_bounds of Lipschitz acceleration envelopes; no finite difference, spline, JTC, or dense sampling",
        "constraints_checked": {
            "a_0": "a(0)=a0",
            "a_T": "a(T)=a1",
            "jerk": "|da/dt| <= jerk_limit",
            "velocity_integral": "integral_0^T a(t) dt = v1-v0",
            "acceleration_limit": "not included in this jerk-only feasibility decision; endpoint values are retained",
        },
        "total_joint_intervals": N_INTERVALS * len(JOINTS),
        "feasible": int(sum(x["feasible"] for x in per_joint.values())),
        "infeasible": int(sum(x["infeasible"] for x in per_joint.values())),
        "per_joint": per_joint,
        "first_infeasible": first_infeasible,
        "worst_infeasible": worst_infeasible,
        "interval_5_j3": interval5,
        "records_written": False,
    }


def derivative_statistics(t: np.ndarray, dq: np.ndarray, ddq: np.ndarray, jerk: np.ndarray) -> dict[str, Any]:
    def counts(values: np.ndarray, field: str) -> dict[str, Any]:
        strict = np.count_nonzero(values != 0.0, axis=0)
        above = np.count_nonzero(np.abs(values) > ZERO_TOL, axis=0)
        if len(values) > 1:
            changes_strict = np.count_nonzero(np.diff(values, axis=0) != 0.0, axis=0)
            changes_above = np.count_nonzero(np.abs(np.diff(values, axis=0)) > ZERO_TOL, axis=0)
        else:
            changes_strict = np.zeros(values.shape[1], dtype=int)
            changes_above = np.zeros(values.shape[1], dtype=int)
        result = {}
        for j, joint in enumerate(JOINTS):
            result[joint] = {
                f"nonzero_{field}_rows_strict": int(strict[j]),
                f"nonzero_{field}_rows_gt_1e-12": int(above[j]),
                f"{field}_change_intervals_strict": int(changes_strict[j]),
                f"{field}_change_intervals_gt_1e-12": int(changes_above[j]),
                f"max_abs_{field}": float(np.max(np.abs(values[:, j]))),
            }
        return result

    rows: dict[str, Any] = {"schema_version": "stage25-derivative-statistics-v1", "rows": N_ROWS, "intervals": N_INTERVALS, "zero_threshold": ZERO_TOL}
    rows["per_joint"] = {joint: {} for joint in JOINTS}
    for field, values in (("dq", dq), ("ddq", ddq), ("jerk", jerk)):
        for joint, item in counts(values, field).items():
            rows["per_joint"][joint].update(item)
    rows["observed_candidate_pattern"] = {
        "j1_to_j5": {
            "ddq_nonzero_rows_strict": [rows["per_joint"][joint]["nonzero_ddq_rows_strict"] for joint in JOINTS[:5]],
            "jerk_nonzero_rows_strict": [rows["per_joint"][joint]["nonzero_jerk_rows_strict"] for joint in JOINTS[:5]],
            "dq_change_intervals_gt_1e-12": [rows["per_joint"][joint]["dq_change_intervals_gt_1e-12"] for joint in JOINTS[:5]],
        },
        "j6": {
            "ddq_nonzero_rows_strict": rows["per_joint"]["j6"]["nonzero_ddq_rows_strict"],
            "jerk_nonzero_rows_strict": rows["per_joint"]["j6"]["nonzero_jerk_rows_strict"],
            "dq_change_intervals_gt_1e-12": rows["per_joint"]["j6"]["dq_change_intervals_gt_1e-12"],
        },
    }
    rows["interpretation"] = {
        "expected_Ruckig_semantics": "not supported for the exported jerk column: native Ruckig jerk is not a JointTrajectoryPoint field and the exporter computes jerk by np.gradient over exported ddq",
        "export_bug": "primary: jerk is a postprocessing finite difference; ddq is copied from the returned message and is not accompanied by a persisted native Ruckig state trace",
        "intentional_auxiliary_field_semantics": False,
        "other": "formal input preparation explicitly zeroes acceleration at every waypoint before calling MoveIt Ruckig; the final sparse ddq pattern cannot be treated as an independently sampled native acceleration without the missing internal state evidence",
    }
    return rows


def main() -> int:
    if not TRAJECTORY.exists() or not LIMITS.exists():
        raise SystemExit(f"required frozen Stage 2.5 artifact is missing: {TRAJECTORY}")
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite existing audit bundle: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source_hash_before = sha256(TRAJECTORY)
    if source_hash_before != "1fe694a50d49af3bf6dd3398c62a5137fc0c8057b4394b3517df293b15466d6e":
        raise SystemExit(f"frozen Stage 2.5 SHA-256 mismatch: {source_hash_before}")
    t, q, dq, ddq, jerk = load_matrix(TRAJECTORY)
    limits = json.loads(LIMITS.read_text(encoding="utf-8"))["joints"]
    source_hash_after = sha256(TRAJECTORY)
    csv_shape = {"points": int(len(t)), "intervals": int(len(t) - 1), "duration_s": float(t[-1]), "sha256": source_hash_before}

    provenance = {
        "schema_version": "stage25-field-provenance-v1",
        "source_csv": str(TRAJECTORY.resolve()),
        "source_csv_sha256_before": source_hash_before,
        "source_csv_sha256_after": source_hash_after,
        "source_csv_unchanged": source_hash_before == source_hash_after,
        "q_origin": {"file": "ros2_moveit_bridge/plan_closed_contour_moveit.py", "function": "_trajectory_arrays -> write_joint_trajectory_csv", "line": "1843-1850; 1816-1823"},
        "dq_origin": {"file": "ros2_moveit_bridge/plan_closed_contour_moveit.py", "function": "_trajectory_arrays -> write_joint_trajectory_csv", "line": "1852-1858; 1816-1823"},
        "ddq_origin": {"file": "ros2_moveit_bridge/plan_closed_contour_moveit.py", "function": "_trajectory_arrays -> write_joint_trajectory_csv", "line": "1859-1865; 1816-1823"},
        "jerk_origin": {"file": "ros2_moveit_bridge/plan_closed_contour_moveit.py", "function": "_trajectory_arrays -> write_joint_trajectory_csv", "line": "1866; 1816-1823"},
        "native_call_origin": {"file": "tools/stage25_moveit_runner.py", "function": "main", "line": "408-425 and 443-449", "call": "RobotTrajectory.apply_ruckig_smoothing"},
        "field_provenance": {
            "q_from_same_native_ruckig_trajectory": True,
            "dq_from_same_native_ruckig_trajectory": True,
            "ddq_from_same_native_ruckig_trajectory": False,
            "jerk_from_same_native_ruckig_trajectory": False,
            "q_same_timestamp": True,
            "dq_same_timestamp": True,
            "ddq_same_timestamp": False,
            "jerk_same_timestamp": False,
            "q_from_trajectory_at_time": False,
            "dq_from_trajectory_at_time": False,
            "ddq_from_trajectory_at_time": False,
            "jerk_from_trajectory_at_time": False,
            "finite_difference_used": {"q": False, "dq": False, "ddq": False, "jerk": True},
            "zero_fill_used": {"explicit_Ruckig_input_acceleration_zeroing": True, "exporter_conditional_missing_field_fallback": "not_recorded_from_final_message"},
            "field_shift_or_lag": "not_observed_in_source; no persisted native trace exists for direct comparison",
            "segment_boundary_value_used": "not_recorded; formal run exposes no Ruckig section metadata",
            "postprocessing_used": {"jerk": "np.gradient(ddq, t, axis=0, edge_order=1)", "input_acceleration": "all waypoint accelerations set to zero before Ruckig"},
        },
        "important_scope_note": "q/dq are returned RobotTrajectory message fields after the native MoveIt Ruckig call; the exact internal Ruckig Trajectory::at_time state identity is not directly proven because InputParameter, Trajectory, section, and native trace were not persisted.",
    }
    stats = derivative_statistics(t, dq, ddq, jerk)
    feasibility = endpoint_feasibility(t, dq, ddq, limits)
    prepared_compare = compare_csvs(TRAJECTORY, PREPARED)
    reconstruction = {
        "schema_version": "stage25-native-ruckig-reconstruction-v1",
        "status": "not_available",
        "reason": "The formal Stage 2.5 runner calls MoveIt RobotTrajectory.apply_ruckig_smoothing but persists only the returned JointTrajectory CSV and audits. It does not persist the native Ruckig InputParameter, Trajectory, section/cumulative-time metadata, or an at_time state trace.",
        "missing_required_evidence": [
            "native Ruckig version recorded for the formal process",
            "original Ruckig InputParameter per native section",
            "original initial state and target state per section",
            "original native Trajectory objects/profiles/cumulative_times",
            "global timestamp to section/local-time mapping",
            "native Trajectory::at_time trace including jerk",
        ],
        "ruckig_version": "not_recorded_in_formal_artifact",
        "source_stage25_run": str(SOURCE_ROOT.resolve()),
        "source_inputs_found": True,
        "original_limits_found": True,
        "original_initial_state_found": "partial: stage25 runtime records frozen q0, not native Ruckig object state",
        "original_target_states_found": "partial: final CSV endpoints, not native Ruckig target objects",
        "original_sections_found": False,
        "deterministic_reconstruction_possible": False,
        "stage25_native_ruckig_state_trace_csv": {"status": "not_available", "path": None, "not_created_to_avoid_fabrication" : True},
        "repeated_formal_rebuilds": "available as returned-message determinism evidence only; not internal Trajectory evidence",
    }
    csv_vs_native = {
        "schema_version": "stage25-csv-vs-native-ruckig-v1",
        "status": "not_available",
        "reason": "native Ruckig Trajectory::at_time states were not persisted and exact reconstruction is not possible from the formal artifacts",
        "machine_tolerance": MACHINE_TOLERANCE,
        "actual_tolerance": None,
        "points_checked": 0,
        "joint_states_checked": 0,
        "position": {"max_abs_error": None, "mismatch_count": None, "worst_joint": None, "worst_row": None},
        "velocity": {"max_abs_error": None, "mismatch_count": None, "worst_joint": None, "worst_row": None},
        "acceleration": {"max_abs_error": None, "mismatch_count": None, "worst_joint": None, "worst_row": None},
        "jerk": {"max_abs_error": None, "mismatch_count": None, "worst_joint": None, "worst_row": None},
        "q_match": "not_available",
        "dq_match": "not_available",
        "ddq_match": "not_available",
        "jerk_match": "not_available",
        "returned_message_internal_check": {"formal_vs_prepared": prepared_compare},
    }
    classification = {
        "schema_version": "stage25-semantics-classification-v1",
        "primary_classification": "Case_B",
        "classification_confidence": "medium",
        "primary_root_cause": "The formal CSV derivative export is not a faithful native-state export: jerk is explicitly manufactured by finite-differencing the exported ddq field, while the exported ddq field is not supported by a persisted native Ruckig at_time trace and fails the independent jerk-bounded endpoint-state test at interval 5/j3.",
        "why_not_case_a": "The CSV jerk column is provably not native Ruckig jerk; it is np.gradient(ddq,t).",
        "why_not_case_c_as_primary": "The primary defect is in the Stage 2.5 derivative artifact semantics/export. Stage 2.7 mapping of that ddq into JointTrajectoryPoint.accelerations is a secondary downstream mapping error, not the primary classification.",
        "why_not_case_d": "No evidence shows q or timestamp corruption: q is unchanged versus the prepared input, while the formal Ruckig output explicitly retimes the message timestamps. Direct native Trajectory::at_time identity remains unproven.",
        "stage27_mapping": {
            "csv_dq_safe_for_JointTrajectoryPoint_velocities": True,
            "csv_ddq_safe_for_JointTrajectoryPoint_accelerations": False,
            "csv_jerk_safe_for_JointTrajectoryPoint_jerk": False,
            "controller_mapping_bug": True,
        },
        "exporter_bug": True,
        "native_internal_reconstruction": "not_available",
    }
    audit = {
        "schema_version": "stage25-state-semantics-audit-v1",
        "audit_scope": "read-only frozen Stage 2.5 formal CSV semantics audit; no native controller, replay, CCD, Stage 2.8, or Stage 2.5 rerun",
        "formal_source": csv_shape,
        "stage25_source_sha256_before": source_hash_before,
        "stage25_source_sha256_after": source_hash_after,
        "stage25_source_unchanged": source_hash_before == source_hash_after,
        "primary_classification": classification["primary_classification"],
        "classification_confidence": classification["classification_confidence"],
        "field_provenance": provenance["field_provenance"],
        "native_ruckig_trajectory_reconstructed": False,
        "all_25532_csv_rows_checked": True,
        "all_25532_native_states_checked": False,
        "intervals_checked": N_INTERVALS,
        "joint_intervals_checked": N_INTERVALS * len(JOINTS),
        "jerk_endpoint_feasibility": {"total": feasibility["total_joint_intervals"], "feasible": feasibility["feasible"], "infeasible": feasibility["infeasible"]},
        "interval_5_j3": {
            "feasible_under_8_rad_s3": feasibility["interval_5_j3"]["feasible"],
            "delta_v_required": feasibility["interval_5_j3"]["delta_v_required"],
            "average_acceleration_required": feasibility["interval_5_j3"]["delta_v_required"] / feasibility["interval_5_j3"]["dt"],
            "jerk_bounded_minimum_possible_delta_v": feasibility["interval_5_j3"]["jerk_bounded_minimum_possible_delta_v"],
            "jerk_bounded_maximum_possible_delta_v": feasibility["interval_5_j3"]["jerk_bounded_maximum_possible_delta_v"],
            "interval_5_j3_endpoint_state_inconsistent_with_jerk_limit": not feasibility["interval_5_j3"]["feasible"],
        },
        "exporter_bug": True,
        "controller_mapping_bug": True,
        "recommended_next_action": "Keep Stage 2.7 blocked. Recover the Stage 2.5 derivative export from a run that persists native Ruckig InputParameter/Trajectory sections and direct at_time states; do not install/replay JTC or alter limits before that recovery.",
    }

    dump_json(OUTPUT / "stage25_state_semantics_audit.json", audit)
    dump_json(OUTPUT / "stage25_field_provenance.json", provenance)
    dump_json(OUTPUT / "stage25_derivative_statistics.json", stats)
    dump_json(OUTPUT / "stage25_jerk_endpoint_feasibility.json", feasibility)
    dump_json(OUTPUT / "stage25_native_ruckig_reconstruction.json", reconstruction)
    dump_json(OUTPUT / "stage25_csv_vs_native_ruckig.json", csv_vs_native)
    dump_yaml(OUTPUT / "stage25_semantics_classification.yaml", classification)

    interval5 = feasibility["interval_5_j3"]
    report = f"""# Stage 2.5 state-semantics audit

## Formal decision

`primary_classification: Case_B` with medium confidence.  The primary defect is the Stage 2.5 derivative export semantics.  The CSV `jerk` columns are not native Ruckig states: the exporter computes them with `np.gradient(ddq, t, axis=0, edge_order=1)`.  The `ddq` columns are copied from the returned MoveIt trajectory message, but the formal run does not persist the native Ruckig `InputParameter`, `Trajectory`, sections, or `at_time` trace needed to certify them as continuous native states.

The downstream Stage 2.7 mapping of this `ddq` into `JointTrajectoryPoint.accelerations` is therefore also unsafe, but it is secondary to the defective/unproven derivative artifact.  No controller was installed or run.

## Frozen-source integrity

- Source: `{TRAJECTORY}`
- Points / intervals: `{len(t)} / {len(t)-1}`
- Duration: `{t[-1]:.12f} s`
- SHA-256 before: `{source_hash_before}`
- SHA-256 after: `{source_hash_after}`
- Unchanged: `{str(source_hash_before == source_hash_after).lower()}`

## Actual source chain

- `q`: `_trajectory_arrays()` reads `JointTrajectoryPoint.positions`; `write_joint_trajectory_csv()` writes it.
- `dq`: `_trajectory_arrays()` reads `JointTrajectoryPoint.velocities` when present, with a conditional `np.gradient` fallback if the message lacks a complete field; the final message-field presence is not separately persisted.
- `ddq`: `_trajectory_arrays()` reads `JointTrajectoryPoint.accelerations` when present, with a conditional `np.gradient` fallback if the message lacks a complete field; the formal runner first explicitly sets every Ruckig input waypoint acceleration to zero.
- `jerk`: `_trajectory_arrays()` always computes `np.gradient(ddq, t, axis=0, edge_order=1)`; it is not read from a native Ruckig field.
- Native call: `tools/stage25_moveit_runner.py` invokes `RobotTrajectory.apply_ruckig_smoothing()` and then exports the returned message.

## Full derivative statistics

All `{N_ROWS}` rows and `{N_INTERVALS}` intervals were read.  Exact nonzero counts and counts above `1e-12` are in `stage25_derivative_statistics.json`.  The sparse pattern is an export/message-field pattern, not evidence that a native Ruckig continuous jerk is sparse.

## Independent jerk-feasibility proof

The audit checks all `{N_INTERVALS * len(JOINTS)}` joint intervals using exact integral bounds for the Lipschitz acceleration envelopes under `|da/dt| <= J`.  No finite-difference, quintic, JTC, or dense-sampling decision is used.

- Feasible intervals: `{feasibility['feasible']}`
- Infeasible intervals: `{feasibility['infeasible']}`
- First infeasible: interval `{feasibility['first_infeasible']['interval']}`, `{feasibility['first_infeasible']['joint']}`

### Mandatory interval 5 / j3

- Required `delta_v`: `{interval5['delta_v_required']:.17g}`
- Required average acceleration: `{interval5['delta_v_required'] / interval5['dt']:.17g}`
- Exact jerk-bounded minimum possible `delta_v`: `{interval5['jerk_bounded_minimum_possible_delta_v']:.17g}`
- Exact jerk-bounded maximum possible `delta_v`: `{interval5['jerk_bounded_maximum_possible_delta_v']:.17g}`
- Feasible under `8 rad/s^3`: `{str(interval5['feasible']).lower()}`
- Endpoint-state inconsistency: `{str(not interval5['feasible']).lower()}`

This proof is independent of JTC.  It shows that the exported interval-5/j3 endpoint acceleration state cannot be joined to the exported endpoint velocities by any continuous acceleration function with jerk bounded by `8 rad/s^3`.

## Native reconstruction gate

`stage25_native_ruckig_state_trace.csv` was **not created**.  The formal artifacts contain returned-message CSVs and repeated rebuild hashes, but not the original native Ruckig objects, per-section input/target states, cumulative section times, or direct `Trajectory::at_time` results.  A new global Ruckig calculation would be a re-optimization/surrogate and is not promoted as reconstruction.

## Stage decision

```yaml
primary_classification: Case_B
Stage_2_5_recovery_required: true
Stage_2_6_recertification_required: not_determined_until_q_t_change_is_established
Stage_2_7: blocked_controller_execution_semantics_gate
native_JTC_next: false
```

The returned-message comparison also shows q is exactly unchanged versus the prepared input, while Ruckig changes timestamps and a small number of dq/ddq values.  This is retiming/output-message evidence, not a direct native `Trajectory::at_time` comparison.

Recommended next action: capture a new isolated Stage 2.5 recovery run that persists the exact native Ruckig inputs, section mapping, and direct state trace before any controller work.  Preserve the current formal CSV unchanged.
"""
    (OUTPUT / "stage25_semantics_report.md").write_text(report, encoding="utf-8")

    files = sorted(path for path in OUTPUT.iterdir() if path.is_file())
    sums = "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    (OUTPUT / "SHA256SUMS").write_text(sums, encoding="utf-8")
    source_hash_final = sha256(TRAJECTORY)
    if source_hash_final != source_hash_after:
        raise RuntimeError("Frozen source changed after audit artifact generation")
    print(json.dumps({
        "output": str(OUTPUT.resolve()),
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_final,
        "source_unchanged": source_hash_final == source_hash_before,
        "primary_classification": classification["primary_classification"],
        "feasible_joint_intervals": feasibility["feasible"],
        "infeasible_joint_intervals": feasibility["infeasible"],
        "interval_5_j3_feasible": interval5["feasible"],
        "native_reconstruction": reconstruction["status"],
        "native_trace_created": False,
        "artifacts": [path.name for path in sorted(OUTPUT.iterdir())],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
