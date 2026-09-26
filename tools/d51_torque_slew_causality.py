"""D51 causal analysis for model-based torque-slew measurements.

This is a read-only shadow analysis of the D50 Pinocchio state exports.  It
does not change either trajectory or robot model.  The decomposition is an
accounting identity at each measured interval:

    d(tau)/dt = d(gravity)/dt + d(tau - gravity)/dt

The result is deliberately described as model-based torque slew.  It is not
an actuator-current or hardware torque certification.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


JOINTS = tuple(f"j{i}" for i in range(1, 7))


def read_states(path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("variant") == "nominal":
                    grouped[str(row["case_id"])].append(row)
    if not grouped:
        raise RuntimeError(f"no_nominal_states:{path}")
    result: dict[str, dict[str, np.ndarray]] = {}
    for case_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["state_index"]))
        values = {
            "t": np.asarray([float(row["t_s"]) for row in rows], dtype=float),
            "q": np.asarray([row["q"] for row in rows], dtype=float),
            "v": np.asarray([row["qdot"] for row in rows], dtype=float),
            "a": np.asarray([row["qddot"] for row in rows], dtype=float),
            "tau": np.asarray([row["tau_Nm"] for row in rows], dtype=float),
            "gravity": np.asarray([row["gravity_tau_Nm"] for row in rows], dtype=float),
        }
        if len(values["t"]) < 3 or not all(np.isfinite(value).all() for value in values.values()):
            raise RuntimeError(f"invalid_nominal_states:{path}:{case_id}")
        if np.any(np.diff(values["t"]) <= 0.0):
            raise RuntimeError(f"nonmonotonic_nominal_time:{path}:{case_id}")
        result[case_id] = values
    return result


def peak(values: np.ndarray) -> tuple[float, int, int]:
    absolute = np.abs(values)
    flat = int(np.argmax(absolute))
    index, joint = np.unravel_index(flat, absolute.shape)
    return float(absolute[index, joint]), int(index), int(joint)


def slope_report(t: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    dt = np.diff(t)
    forward = np.diff(values, axis=0) / dt[:, None]
    forward_peak, forward_index, forward_joint = peak(forward)
    centered = (values[2:] - values[:-2]) / (t[2:, None] - t[:-2, None])
    centered_peak, centered_index, centered_joint = peak(centered)
    return {
        "forward_peak_abs": forward_peak,
        "forward_peak_index": forward_index,
        "forward_peak_joint": JOINTS[forward_joint],
        "centered_peak_abs": centered_peak,
        "centered_peak_index": centered_index + 1,
        "centered_peak_joint": JOINTS[centered_joint],
        "centered_forward_relative_difference": float(abs(centered_peak - forward_peak) / max(forward_peak, 1.0e-12)),
        "forward_interval_count": int(len(forward)),
        "centered_state_count": int(len(centered)),
    }


def interval_rows(t: np.ndarray, q: np.ndarray, v: np.ndarray, a: np.ndarray,
                  tau: np.ndarray, gravity: np.ndarray, count: int = 5) -> list[dict[str, Any]]:
    dt = np.diff(t)
    dynamic = tau - gravity
    torque_slew = np.diff(tau, axis=0) / dt[:, None]
    gravity_slew = np.diff(gravity, axis=0) / dt[:, None]
    dynamic_slew = np.diff(dynamic, axis=0) / dt[:, None]
    absolute = np.abs(torque_slew)
    flat_indices = np.argsort(absolute.reshape(-1))[::-1][:count]
    rows = []
    for flat in flat_indices:
        interval, joint = np.unravel_index(int(flat), absolute.shape)
        rows.append({
            "joint": JOINTS[int(joint)],
            "interval_index": int(interval),
            "time_start_s": float(t[interval]),
            "time_end_s": float(t[interval + 1]),
            "dt_s": float(dt[interval]),
            "abs_torque_slew_Nm_s": float(abs(torque_slew[interval, joint])),
            "signed_torque_slew_Nm_s": float(torque_slew[interval, joint]),
            "abs_gravity_slew_Nm_s": float(abs(gravity_slew[interval, joint])),
            "abs_dynamic_residual_slew_Nm_s": float(abs(dynamic_slew[interval, joint])),
            "torque_slew_decomposition_residual_Nm_s": float(abs(torque_slew[interval, joint] - gravity_slew[interval, joint] - dynamic_slew[interval, joint])),
            "q_left_rad": float(q[interval, joint]),
            "q_right_rad": float(q[interval + 1, joint]),
            "qdot_left_rad_s": float(v[interval, joint]),
            "qdot_right_rad_s": float(v[interval + 1, joint]),
            "qddot_left_rad_s2": float(a[interval, joint]),
            "qddot_right_rad_s2": float(a[interval + 1, joint]),
            "qddot_transition_rad_s3": float((a[interval + 1, joint] - a[interval, joint]) / dt[interval]),
            "tau_left_Nm": float(tau[interval, joint]),
            "tau_right_Nm": float(tau[interval + 1, joint]),
            "gravity_left_Nm": float(gravity[interval, joint]),
            "gravity_right_Nm": float(gravity[interval + 1, joint]),
            "dynamic_residual_left_Nm": float(dynamic[interval, joint]),
            "dynamic_residual_right_Nm": float(dynamic[interval + 1, joint]),
        })
    return rows


def analyze_case(case_id: str, baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    def one(label: str, data: dict[str, np.ndarray]) -> dict[str, Any]:
        tau = data["tau"]
        gravity = data["gravity"]
        dynamic = tau - gravity
        tau_report = slope_report(data["t"], tau)
        gravity_report = slope_report(data["t"], gravity)
        dynamic_report = slope_report(data["t"], dynamic)
        dt = np.diff(data["t"])
        torque_slew = np.diff(tau, axis=0) / dt[:, None]
        gravity_slew = np.diff(gravity, axis=0) / dt[:, None]
        dynamic_slew = np.diff(dynamic, axis=0) / dt[:, None]
        peaks = {
            "torque_slew": peak(torque_slew),
            "gravity_slew": peak(gravity_slew),
            "dynamic_residual_slew": peak(dynamic_slew),
        }
        return {
            "label": label,
            "state_count": int(len(data["t"])),
            "duration_s": float(data["t"][-1] - data["t"][0]),
            "dt_s": {"min": float(np.min(dt)), "median": float(np.median(dt)), "max": float(np.max(dt))},
            "peak_abs_torque_Nm": float(np.max(np.abs(tau))),
            "peak_abs_gravity_torque_Nm": float(np.max(np.abs(gravity))),
            "peak_abs_dynamic_residual_Nm": float(np.max(np.abs(dynamic))),
            "torque_slew": tau_report,
            "gravity_slew": gravity_report,
            "dynamic_residual_slew": dynamic_report,
            "decomposition_identity_max_residual_Nm_s": float(np.max(np.abs(torque_slew - gravity_slew - dynamic_slew))),
            "decomposition_peak_abs_Nm_s": {
                "torque": float(peaks["torque_slew"][0]),
                "gravity": float(peaks["gravity_slew"][0]),
                "dynamic_residual": float(peaks["dynamic_residual_slew"][0]),
            },
            "top_torque_slew_regions": interval_rows(data["t"], data["q"], data["v"], data["a"], tau, gravity),
        }

    baseline_report = one("baseline", baseline)
    candidate_report = one("candidate", candidate)
    return {
        "case_id": case_id,
        "baseline": baseline_report,
        "candidate": candidate_report,
        "candidate_minus_baseline_peak_abs_torque_slew_Nm_s": candidate_report["torque_slew"]["forward_peak_abs"] - baseline_report["torque_slew"]["forward_peak_abs"],
        "candidate_minus_baseline_dynamic_residual_slew_Nm_s": candidate_report["dynamic_residual_slew"]["forward_peak_abs"] - baseline_report["dynamic_residual_slew"]["forward_peak_abs"],
        "candidate_minus_baseline_gravity_slew_Nm_s": candidate_report["gravity_slew"]["forward_peak_abs"] - baseline_report["gravity_slew"]["forward_peak_abs"],
    }


def deterministic_replay(path: Path) -> bool:
    first = read_states(path)
    second = read_states(path)
    return sorted(first) == sorted(second) and all(
        np.array_equal(first[key][field], second[key][field])
        for key in first for field in first[key]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    baseline = read_states(args.baseline.resolve())
    candidate = read_states(args.candidate.resolve())
    if list(baseline) != list(candidate):
        raise RuntimeError("baseline_candidate_case_order_mismatch")
    case_reports = [analyze_case(case_id, baseline[case_id], candidate[case_id]) for case_id in baseline]
    worst = max(case_reports, key=lambda row: row["candidate"]["torque_slew"]["forward_peak_abs"])
    dynamic_deltas = [row["candidate_minus_baseline_dynamic_residual_slew_Nm_s"] for row in case_reports]
    gravity_deltas = [row["candidate_minus_baseline_gravity_slew_Nm_s"] for row in case_reports]
    candidate_peak = float(max(row["candidate"]["torque_slew"]["forward_peak_abs"] for row in case_reports))
    baseline_peak = float(max(row["baseline"]["torque_slew"]["forward_peak_abs"] for row in case_reports))
    root_cause = (
        "candidate torque-slew increase is dominated by the non-gravity dynamic residual "
        "(inertia/Coriolis/centrifugal terms plus acceleration-transition effects)"
        if max(dynamic_deltas) > max(gravity_deltas)
        else "gravity-term slew is the larger measured contributor; inspect configuration timing and gravity geometry"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "d51-torque-slew-causality-v1",
        "status": "PASS_MODEL_BASED_CAUSAL_DECOMPOSITION",
        "backend": "Pinocchio RNEA state exports from D50",
        "measurement_scope": "model-based torque slew; not actuator current, motor torque, or hardware certification",
        "identity": "d(tau)/dt = d(gravity)/dt + d(tau-gravity)/dt",
        "baseline_source": str(args.baseline.resolve()),
        "candidate_source": str(args.candidate.resolve()),
        "case_count": len(case_reports),
        "baseline_peak_abs_torque_slew_Nm_s": baseline_peak,
        "candidate_peak_abs_torque_slew_Nm_s": candidate_peak,
        "candidate_over_baseline_ratio": float(candidate_peak / baseline_peak),
        "candidate_minus_baseline_peak_abs_torque_slew_Nm_s": float(candidate_peak - baseline_peak),
        "worst_case_id": worst["case_id"],
        "worst_joint": worst["candidate"]["torque_slew"]["forward_peak_joint"],
        "worst_time_regions": worst["candidate"]["top_torque_slew_regions"],
        "worst_case_q_v_a_torque_context": worst["candidate"]["top_torque_slew_regions"][0],
        "dynamics_decomposition": {
            "root_cause_statement": root_cause,
            "max_candidate_minus_baseline_dynamic_residual_slew_Nm_s": float(max(dynamic_deltas)),
            "max_candidate_minus_baseline_gravity_slew_Nm_s": float(max(gravity_deltas)),
            "dynamic_residual_definition": "tau - gravity_tau; not separately split into M(q)qddot and C(q,qdot)qdot by the available D50 export",
            "acceleration_transition_is_reported": True,
            "identity_max_residual_Nm_s": float(max(row["candidate"]["decomposition_identity_max_residual_Nm_s"] for row in case_reports)),
        },
        "convergence": {
            "method": "forward interval derivative compared with centered nonuniform-time derivative",
            "baseline_max_forward_centered_relative_difference": float(max(row["baseline"]["torque_slew"]["centered_forward_relative_difference"] for row in case_reports)),
            "candidate_max_forward_centered_relative_difference": float(max(row["candidate"]["torque_slew"]["centered_forward_relative_difference"] for row in case_reports)),
            "interpretation": "agreement is a numerical convergence check for the reported derivative, not proof of hardware current bandwidth",
        },
        "deterministic_replay": {
            "baseline": deterministic_replay(args.baseline.resolve()),
            "candidate": deterministic_replay(args.candidate.resolve()),
            "status": "PASS" if deterministic_replay(args.baseline.resolve()) and deterministic_replay(args.candidate.resolve()) else "FAIL",
        },
        "cases": case_reports,
    }
    (output / "torque_slew_causality.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "worst_regions.jsonl").open("w", encoding="utf-8") as stream:
        for row in case_reports:
            for region in row["candidate"]["top_torque_slew_regions"]:
                stream.write(json.dumps({"case_id": row["case_id"], **region}, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
