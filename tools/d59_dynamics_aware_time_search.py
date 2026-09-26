"""Bounded, model-dynamics-aware time-parameterization shadow search.

This tool deliberately operates before the final native verification step:
it re-parameterizes the *fixed* D57 q path in time, evaluates Pinocchio RNEA
and torque slew for a small set of candidate time scales, and filters only
against inherited kinematic limits.  Torque values are relative software
model metrics, never hardware limits.  The selected shadow must still pass
the native MoveIt2/Ruckig and geometry chain before it is considered valid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin


def read_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise RuntimeError(f"trajectory_missing:{path}")
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    v = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)
    a = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    if not all(np.isfinite(value).all() for value in (t, q, v, a)) or np.any(np.diff(t) <= 0.0):
        raise RuntimeError(f"trajectory_invalid:{path}")
    return t, q, v, a


def load_limits(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    limits = payload["cases"][0]["consistent_time_repair"]["active_limits"]
    return {
        "velocity": np.asarray(limits["velocity_rad_s"], dtype=float),
        "acceleration": np.asarray(limits["acceleration_rad_s2"], dtype=float),
        "jerk": np.asarray(limits["jerk_rad_s3"], dtype=float),
    }


def model_inventory(model: pin.Model) -> dict[str, Any]:
    return {
        "joint_order": list(model.names[1:]),
        "expected_joint_order": [f"j{i}" for i in range(1, 7)],
        "joint_order_match": list(model.names[1:]) == [f"j{i}" for i in range(1, 7)],
        "inertial_provenance": "DERIVED_PROJECT_URDF_NOT_HARDWARE_CERTIFIED",
        "gravity_m_s2": np.asarray(model.gravity.linear, dtype=float).tolist(),
    }


def evaluate(model: pin.Model, t: np.ndarray, q: np.ndarray, v: np.ndarray, a: np.ndarray,
             scale: float, limits: dict[str, np.ndarray]) -> dict[str, Any]:
    ts = t * scale
    vs = v / scale
    acc = a / (scale * scale)
    data = model.createData()
    tau = np.asarray([pin.rnea(model, data, qi, vi, ai) for qi, vi, ai in zip(q, vs, acc)], dtype=float)
    slew = np.diff(tau, axis=0) / np.diff(ts)[:, None]
    max_v = float(np.max(np.abs(vs)))
    max_a = float(np.max(np.abs(acc)))
    max_tau = float(np.max(np.abs(tau)))
    max_slew = float(np.max(np.abs(slew)))
    velocity_limit = float(np.max(limits["velocity"]))
    acceleration_limit = float(np.max(limits["acceleration"]))
    # The native chain uses a 1.05 safety margin; this is a kinematic filter,
    # not a dynamics or hardware acceptance threshold.
    kinematic_feasible = bool(
        max_v <= velocity_limit / 1.05
        and max_a <= acceleration_limit / 1.05
    )
    return {
        "time_scale": float(scale),
        "duration_s": float(ts[-1] - ts[0]),
        "max_abs_velocity_rad_s": max_v,
        "max_abs_acceleration_rad_s2": max_a,
        "peak_model_torque_Nm": max_tau,
        "peak_model_torque_slew_Nm_s": max_slew,
        "kinematic_feasible_under_inherited_margin": kinematic_feasible,
        "torque_constraint_status": "RELATIVE_METRIC_ONLY_NO_HARDWARE_LIMIT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    t, q, v, a = read_trajectory(args.trajectory.resolve())
    limits = load_limits(args.native_summary.resolve())
    model = pin.buildModelFromUrdf(str(args.urdf.resolve()))
    inventory = model_inventory(model)
    if not inventory["joint_order_match"]:
        raise RuntimeError("joint_order_mismatch")
    scales = (0.75, 0.9, 1.0, 1.25, 1.908649704875276)
    rows = [evaluate(model, t, q, v, a, scale, limits) for scale in scales]
    feasible = [row for row in rows if row["kinematic_feasible_under_inherited_margin"]]
    if not feasible:
        selected = None
        status = "NO_KINEMATICALLY_FEASIBLE_SCALE_IN_BOUNDED_SEARCH"
    else:
        # First minimize duration; use model torque slew only as a transparent
        # tie-break for numerically equivalent durations.
        selected = min(feasible, key=lambda row: (row["duration_s"], row["peak_model_torque_slew_Nm_s"]))
        status = "PASS_BOUNDED_MODEL_DYNAMICS_AWARE_ROUTE"
    payload = {
        "schema_version": "d59-dynamics-aware-time-search-v1",
        "status": status,
        "scope": "OFFLINE_ALGORITHM_ONLY",
        "route": "fixed_D57_q_path_time_reparameterization_then_Pinocchio_RNEA",
        "trajectory": str(args.trajectory.resolve()),
        "native_limit_reference": str(args.native_summary.resolve()),
        "q_path_mutated": False,
        "model": inventory,
        "search_scales": list(scales),
        "candidates": rows,
        "selected_shadow": selected,
        "selection_policy": "minimum duration among inherited-margin kinematically feasible scales; model torque is relative diagnostic only",
        "native_verification_required": True,
        "hardware_torque_certification": "OUT_OF_SCOPE_OFFLINE_PROJECT",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected_shadow": selected}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
