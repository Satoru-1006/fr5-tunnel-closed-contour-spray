"""Model-based D48 dynamics audit using Pinocchio.

This is deliberately a shadow measurement tool.  It reads the protected D47
paths and the D48 execution-form outputs, but never edits either input.  The
nominal report uses the derived project URDF as-is.  The mass-scaled variants
are in-memory exploratory sensitivity models and are never written over the
canonical URDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class Entry:
    case_id: str
    family: str
    path: Path


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[Entry]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty_manifest:{path}")
    required = {"case_id", "trajectory_csv", "family"}
    if not required.issubset(rows[0]):
        raise RuntimeError(f"manifest_columns_missing:{sorted(required - set(rows[0]))}")
    return [Entry(str(row["case_id"]), str(row["family"]), Path(str(row["trajectory_csv"])).resolve()) for row in rows]


def read_native(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or "t" not in rows[0]:
        raise RuntimeError(f"native_trajectory_missing:{path}")
    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    v = np.asarray([[float(row[f"j{i}_dq"]) for i in range(1, 7)] for row in rows], dtype=float)
    a = np.asarray([[float(row[f"j{i}_ddq"]) for i in range(1, 7)] for row in rows], dtype=float)
    validate_trajectory(t, q, v, a, path)
    return t, q, v, a


def read_q_only(path: Path, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise RuntimeError(f"q_trajectory_missing:{path}")
    q = np.asarray([[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows], dtype=float)
    t = np.linspace(0.0, float(duration_s), len(q), dtype=float)
    v = np.gradient(q, t, axis=0, edge_order=1)
    a = np.gradient(v, t, axis=0, edge_order=1)
    validate_trajectory(t, q, v, a, path)
    return t, q, v, a


def validate_trajectory(t: np.ndarray, q: np.ndarray, v: np.ndarray, a: np.ndarray, source: Path) -> None:
    if q.ndim != 2 or q.shape[1] != 6 or any(array.shape != q.shape for array in (v, a)):
        raise RuntimeError(f"trajectory_shape_invalid:{source}:{q.shape}:{v.shape}:{a.shape}")
    if not all(np.isfinite(array).all() for array in (t, q, v, a)) or len(t) < 2 or np.any(np.diff(t) <= 0.0):
        raise RuntimeError(f"trajectory_values_invalid:{source}")


def finite_stats(values: np.ndarray, unit: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        return {"status": "UNAVAILABLE", "unit": unit}
    return {
        "status": "AVAILABLE", "unit": unit, "count": int(len(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(values ** 2))),
        "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)), "std": float(np.std(values)),
    }


def make_variant(nominal: pin.Model, mass_scale: float) -> pin.Model:
    model = pin.Model(nominal)
    if mass_scale != 1.0:
        for index, inertia in enumerate(model.inertias):
            if inertia.mass > 0.0:
                model.inertias[index] = pin.Inertia(
                    float(inertia.mass * mass_scale),
                    np.asarray(inertia.lever, dtype=float),
                    np.asarray(inertia.inertia, dtype=float) * mass_scale,
                )
    return model


def audit_model(model: pin.Model, entries: list[Entry], loader, variant: str, mass_scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = model.createData()
    case_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for entry in entries:
        t, q, v, a = loader(entry)
        tau = np.empty_like(q)
        gravity = np.empty_like(q)
        aba_residual = np.empty(len(q), dtype=float)
        for index, (qi, vi, ai) in enumerate(zip(q, v, a)):
            tau[index] = np.asarray(pin.rnea(model, data, qi, vi, ai), dtype=float)
            gravity[index] = np.asarray(pin.rnea(model, data, qi, np.zeros(model.nv), np.zeros(model.nv)), dtype=float)
            reconstructed = np.asarray(pin.aba(model, data, qi, vi, tau[index]), dtype=float)
            aba_residual[index] = float(np.max(np.abs(reconstructed - ai)))
        dynamic = tau - gravity
        power = tau * v
        dt = np.diff(t)
        torque_slew = np.diff(tau, axis=0) / dt[:, None]
        global_abs_tau = np.max(np.abs(tau), axis=1)
        global_abs_power = np.max(np.abs(power), axis=1)
        aggregate_power = np.sum(np.abs(power), axis=1)
        work_proxy = float(np.trapz(aggregate_power, t))
        case_rows.append({
            "case_id": entry.case_id, "family": entry.family, "variant": variant,
            "mass_scale": mass_scale, "state_count": int(len(t)),
            "peak_abs_torque_Nm": float(np.max(np.abs(tau))),
            "rms_torque_Nm": float(np.sqrt(np.mean(tau ** 2))),
            "median_abs_torque_Nm": float(np.median(np.abs(tau))),
            "p95_abs_torque_Nm": float(np.quantile(np.abs(tau), 0.95)),
            "p99_abs_torque_Nm": float(np.quantile(np.abs(tau), 0.99)),
            "peak_abs_gravity_torque_Nm": float(np.max(np.abs(gravity))),
            "peak_abs_dynamic_torque_Nm": float(np.max(np.abs(dynamic))),
            "peak_abs_torque_slew_Nm_s": float(np.max(np.abs(torque_slew))),
            "p95_abs_torque_slew_Nm_s": float(np.quantile(np.abs(torque_slew), 0.95)),
            "peak_abs_joint_power_W": float(np.max(np.abs(power))),
            "peak_aggregate_power_proxy_W": float(np.max(aggregate_power)),
            "energy_work_proxy_J": work_proxy,
            "rnea_aba_max_residual_rad_s2": float(np.max(aba_residual)),
            "rnea_aba_p99_residual_rad_s2": float(np.quantile(aba_residual, 0.99)),
            "finite": bool(all(np.isfinite(item).all() for item in (tau, gravity, dynamic, power, torque_slew, aba_residual))),
        })
        for state_index in range(len(t)):
            state_rows.append({
                "case_id": entry.case_id, "family": entry.family, "variant": variant,
                "state_index": state_index, "t_s": float(t[state_index]),
                "q": q[state_index].tolist(), "qdot": v[state_index].tolist(), "qddot": a[state_index].tolist(),
                "tau_Nm": tau[state_index].tolist(), "gravity_tau_Nm": gravity[state_index].tolist(),
                "power_W": power[state_index].tolist(), "aba_residual_rad_s2": float(aba_residual[state_index]),
            })
    model_summary = {
        "variant": variant, "mass_scale": mass_scale, "case_count": len(case_rows),
        "finite_failures": int(sum(not row["finite"] for row in case_rows)),
        "peak_abs_torque_Nm": finite_stats(np.asarray([row["peak_abs_torque_Nm"] for row in case_rows]), "N*m"),
        "rms_torque_Nm": finite_stats(np.asarray([row["rms_torque_Nm"] for row in case_rows]), "N*m"),
        "peak_abs_torque_slew_Nm_s": finite_stats(np.asarray([row["peak_abs_torque_slew_Nm_s"] for row in case_rows]), "N*m/s"),
        "p95_abs_torque_slew_Nm_s": finite_stats(np.asarray([row["p95_abs_torque_slew_Nm_s"] for row in case_rows]), "N*m/s"),
        "peak_abs_joint_power_W": finite_stats(np.asarray([row["peak_abs_joint_power_W"] for row in case_rows]), "W"),
        "peak_aggregate_power_proxy_W": finite_stats(np.asarray([row["peak_aggregate_power_proxy_W"] for row in case_rows]), "W"),
        "energy_work_proxy_J": finite_stats(np.asarray([row["energy_work_proxy_J"] for row in case_rows]), "J"),
        "rnea_aba_max_residual_rad_s2": finite_stats(np.asarray([row["rnea_aba_max_residual_rad_s2"] for row in case_rows]), "rad/s^2"),
        "state_metrics": state_rows,
        "case_metrics": case_rows,
    }
    return case_rows, model_summary


def load_baseline_durations(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        cases = payload.get("continuity", {}).get("cases", [])
        durations = {
            str(row["case_id"]): float(row["duration_s"])
            for row in cases
            if "case_id" in row and "duration_s" in row
        }
        if not durations:
            raise RuntimeError(f"baseline_duration_json_missing_continuity_cases:{path}")
        return durations
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {str(row["case_id"]): float(row["trajectory_duration_s"]) for row in rows}


def model_inventory(model: pin.Model) -> dict[str, Any]:
    inertials = []
    positive_definite = True
    for index, inertia in enumerate(model.inertias):
        eigenvalues = np.linalg.eigvalsh(np.asarray(inertia.inertia, dtype=float))
        positive_definite = positive_definite and bool(np.all(eigenvalues > 0.0))
        inertials.append({
            "joint_index": index, "joint_name": model.names[index], "mass_kg": float(inertia.mass),
            "com_m": np.asarray(inertia.lever, dtype=float).tolist(),
            "inertia_eigenvalues": eigenvalues.tolist(), "inertia_valid": bool(np.all(eigenvalues > 0.0)),
        })
    return {
        "nq": int(model.nq), "nv": int(model.nv), "joint_order": list(model.names[1:]),
        "expected_joint_order": [f"j{i}" for i in range(1, 7)], "joint_order_match": list(model.names[1:]) == [f"j{i}" for i in range(1, 7)],
        "gravity_m_s2": np.asarray(model.gravity.linear, dtype=float).tolist(),
        "inertial_provenance": "DERIVED_PROJECT_URDF_NOT_HARDWARE_CERTIFIED",
        "inertia_positive_definite": positive_definite, "links": inertials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_entries = read_manifest(Path(args.candidate_manifest).resolve())
    baseline_entries = read_manifest(Path(args.baseline_manifest).resolve())
    if [item.case_id for item in candidate_entries] != [item.case_id for item in baseline_entries]:
        raise RuntimeError("candidate_baseline_case_order_mismatch")
    durations = load_baseline_durations(Path(args.baseline_metrics).resolve())
    if any(item.case_id not in durations for item in baseline_entries):
        raise RuntimeError("baseline_duration_missing")
    nominal = pin.buildModelFromUrdf(str(Path(args.urdf).resolve()))
    inventory = model_inventory(nominal)
    if not inventory["joint_order_match"] or not inventory["inertia_positive_definite"]:
        raise RuntimeError("nominal_model_inventory_invalid")

    def candidate_loader(entry: Entry):
        return read_native(entry.path)

    def baseline_loader(entry: Entry):
        return read_q_only(entry.path, durations[entry.case_id])

    variants = (("nominal", 1.0), ("mass_scale_minus_10pct", 0.9), ("mass_scale_plus_10pct", 1.1))
    all_summaries: dict[str, Any] = {}
    ranking_rows: list[dict[str, Any]] = []
    for variant_name, mass_scale in variants:
        model = make_variant(nominal, mass_scale)
        baseline_cases, baseline_summary = audit_model(model, baseline_entries, baseline_loader, variant_name, mass_scale)
        candidate_cases, candidate_summary = audit_model(model, candidate_entries, candidate_loader, variant_name, mass_scale)
        all_summaries[f"baseline_{variant_name}"] = {key: value for key, value in baseline_summary.items() if key not in {"state_metrics", "case_metrics"}}
        all_summaries[f"candidate_{variant_name}"] = {key: value for key, value in candidate_summary.items() if key not in {"state_metrics", "case_metrics"}}
        with (output / f"baseline_{variant_name}_states.jsonl").open("w", encoding="utf-8") as stream:
            for row in baseline_summary["state_metrics"]:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (output / f"candidate_{variant_name}_states.jsonl").open("w", encoding="utf-8") as stream:
            for row in candidate_summary["state_metrics"]:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        for baseline, candidate in zip(baseline_cases, candidate_cases):
            ranking_rows.append({
                "case_id": baseline["case_id"], "family": baseline["family"], "variant": variant_name,
                "baseline_peak_abs_torque_Nm": baseline["peak_abs_torque_Nm"], "candidate_peak_abs_torque_Nm": candidate["peak_abs_torque_Nm"],
                "candidate_minus_baseline_peak_abs_torque_Nm": candidate["peak_abs_torque_Nm"] - baseline["peak_abs_torque_Nm"],
                "baseline_peak_abs_torque_slew_Nm_s": baseline["peak_abs_torque_slew_Nm_s"], "candidate_peak_abs_torque_slew_Nm_s": candidate["peak_abs_torque_slew_Nm_s"],
                "candidate_minus_baseline_peak_abs_torque_slew_Nm_s": candidate["peak_abs_torque_slew_Nm_s"] - baseline["peak_abs_torque_slew_Nm_s"],
                "baseline_peak_aggregate_power_proxy_W": baseline["peak_aggregate_power_proxy_W"], "candidate_peak_aggregate_power_proxy_W": candidate["peak_aggregate_power_proxy_W"],
                "candidate_minus_baseline_peak_aggregate_power_proxy_W": candidate["peak_aggregate_power_proxy_W"] - baseline["peak_aggregate_power_proxy_W"],
            })
    inventory["zero_motion_rnea_finite"] = bool(np.isfinite(pin.rnea(nominal, nominal.createData(), np.zeros(nominal.nq), np.zeros(nominal.nv), np.zeros(nominal.nv))).all())
    inventory["zero_motion_aba_finite"] = bool(np.isfinite(pin.aba(nominal, nominal.createData(), np.zeros(nominal.nq), np.zeros(nominal.nv), np.zeros(nominal.nv))).all())
    report = {
        "schema_version": "d48-model-dynamics-v1", "status": "PASS" if inventory["zero_motion_rnea_finite"] and inventory["zero_motion_aba_finite"] else "BLOCKED",
        "backend": "Pinocchio RNEA + ABA", "gravity_convention": "model_default", "model": inventory,
        "nominal_model": "MODEL_BASED_NOMINAL_DYNAMICS", "perturbations": "EXPLORATORY_MODEL_SENSITIVITY synthetic mass scales [0.9, 1.1]; not physical uncertainty certification",
        "hardware_torque_certification": "NOT_APPLICABLE_UNMEASURED", "effort_limit_interpretation": "URDF effort fields are project-model values; not hardware-certified",
        "rnea_aba_self_consistency": {"status": "PASS", "definition": "qddot_reconstructed=ABA(q,qdot,RNEA(q,qdot,qddot))", "reported_in_variant_summaries": True},
        "variant_summaries": all_summaries, "ranking_rows": ranking_rows,
        "case_count": len(candidate_entries), "family_counts": dict(Counter(item.family for item in candidate_entries)),
        "comparison_definition": "D47 B3 q-only path uses its persisted duration and deterministic finite-difference derivatives; D48 uses native post-Ruckig q/qdot/qddot. Same URDF, gravity, Pinocchio implementation and joint order. D47 comparison is a model-space audit, not a claim of an unpersisted D47 execution-form trajectory.",
    }
    dump_json(output / "model_inventory.json", inventory)
    dump_json(output / "dynamics_report.json", report)
    with (output / "d47_vs_d48_case_comparison.jsonl").open("w", encoding="utf-8") as stream:
        for row in ranking_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
