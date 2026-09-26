"""D52 shadow trajectory repair and candidate generation utilities.

This module owns only ``outputs/D52_STAGE4B_SHADOW`` when run as a script.  It
does not modify the D47/D50/D51 inputs or the Stage 3 release.  The first
repair is intentionally narrow: the D50/D51 native exporter occasionally
emits one bridge state whose timestamp is advanced by a full Ruckig startup
duration while its q value is the following 10 ms sample and its v/a fields
are a startup artifact.  We repair that bridge only after checking the
distinctive, reproducible pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs" / "D50_STAGE4_INTEGRATED_SHADOW" / "phase_b_jerk_truth" / "global_0075" / "native_postprocess" / "trajectories"
DEFAULT_OUTPUT = ROOT / "outputs" / "D52_STAGE4B_SHADOW" / "candidates" / "repaired_fast"
JOINTS = tuple(f"j{i}" for i in range(1, 7))
CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)
FAMILIES = {
    "REGRESSION": {"regression_0000", "regression_0001"},
    "NORMAL": {"normal_0000", "normal_0100"},
    "BOUNDARY": {"boundary_0000", "boundary_0100"},
    "COLLISION_SENSITIVE": {"collision_sensitive_0000", "collision_sensitive_0051"},
    "ADVERSARIAL": {"adversarial_0100", "adversarial_0101"},
    "PERTURBATION": {"perturbation_0000", "perturbation_0100"},
}


@dataclass(frozen=True)
class NativeTrajectory:
    fields: tuple[str, ...]
    rows: tuple[dict[str, float], ...]

    @property
    def time(self) -> np.ndarray:
        return np.asarray([row["t"] for row in self.rows], dtype=float)

    @property
    def q(self) -> np.ndarray:
        return np.asarray([[row[f"{joint}_q"] for joint in JOINTS] for row in self.rows], dtype=float)

    @property
    def dq(self) -> np.ndarray:
        return np.asarray([[row[f"{joint}_dq"] for joint in JOINTS] for row in self.rows], dtype=float)

    @property
    def ddq(self) -> np.ndarray:
        return np.asarray([[row[f"{joint}_ddq"] for joint in JOINTS] for row in self.rows], dtype=float)


def read_native(path: Path) -> NativeTrajectory:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        raw = list(reader)
    required = {"t", *(f"{joint}_{suffix}" for joint in JOINTS for suffix in ("q", "dq", "ddq"))}
    missing = sorted(required - set(fields))
    if missing or len(raw) < 2:
        raise ValueError(f"invalid_native_trajectory:{path}:missing={missing}:rows={len(raw)}")
    rows: list[dict[str, float]] = []
    for index, source in enumerate(raw):
        try:
            row = {field: float(source[field]) for field in fields}
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid_native_row:{path}:{index}") from exc
        rows.append(row)
    trajectory = NativeTrajectory(fields, tuple(rows))
    if not all(np.isfinite(value).all() for value in (trajectory.time, trajectory.q, trajectory.dq, trajectory.ddq)):
        raise ValueError(f"nonfinite_native_trajectory:{path}")
    if np.any(np.diff(trajectory.time) <= 0.0):
        raise ValueError(f"nonmonotonic_native_trajectory:{path}")
    return trajectory


def write_native(path: Path, trajectory: NativeTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory.fields), lineterminator="\n")
        writer.writeheader()
        for row in trajectory.rows:
            writer.writerow({field: row[field] for field in trajectory.fields})


def _small_step_statistics(time: np.ndarray) -> tuple[float, float, np.ndarray]:
    dt = np.diff(time)
    finite_positive = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(finite_positive) == 0:
        raise ValueError("no_positive_time_step")
    nominal = float(np.median(finite_positive))
    small = finite_positive[finite_positive <= nominal * 1.5]
    if len(small) < max(2, len(dt) // 2):
        raise ValueError("insufficient_regular_time_steps")
    return nominal, float(np.quantile(small, 0.99)), dt


def _bridge_state_residual(trajectory: NativeTrajectory, index: int, nominal_dt: float) -> dict[str, Any]:
    """Score the state at a suspected bridge against its neighboring sample."""
    t = trajectory.time
    q = trajectory.q
    dq = trajectory.dq
    ddq = trajectory.ddq
    left = index - 1
    right = index + 1
    dt_right = float(t[right] - t[index])
    expected_from_left = q[left] + dq[left] * nominal_dt + 0.5 * ddq[left] * nominal_dt * nominal_dt
    expected_from_right = q[right] - dq[right] * nominal_dt + 0.5 * ddq[right] * nominal_dt * nominal_dt
    return {
        "index": index,
        "left_time_s": float(t[left]),
        "bridge_time_s": float(t[index]),
        "right_time_s": float(t[right]),
        "gap_s": float(t[index] - t[left]),
        "bridge_to_right_dt_s": dt_right,
        "nominal_dt_s": nominal_dt,
        "bridge_q_from_left_position_residual_rad": float(np.max(np.abs(q[index] - expected_from_left))),
        "bridge_q_from_right_position_residual_rad": float(np.max(np.abs(q[index] - expected_from_right))),
        "bridge_exported_velocity_mismatch_rad_s": float(np.max(np.abs(dq[index] - (q[right] - q[left]) / (2.0 * nominal_dt)))),
        "bridge_exported_acceleration_mismatch_rad_s2": float(np.max(np.abs(ddq[index] - (dq[right] - dq[left]) / (2.0 * nominal_dt)))),
        "bridge_to_right_acceleration_jump_rad_s2": float(np.max(np.abs(ddq[right] - ddq[index]))),
        "left_to_bridge_position_jump_rad": float(np.max(np.abs(q[index] - q[left]))),
        "left_to_right_position_jump_rad": float(np.max(np.abs(q[right] - q[left]))),
    }


def repair_execution_form(source: Path, destination: Path, *, expected_dt: float | None = None) -> dict[str, Any]:
    """Repair one known exporter bridge and return auditable evidence.

    The function is fail-closed.  It accepts exactly one large time gap, with
    the following interval at the regular sample period and a q sequence that
    is coherent across the bridge.  The bridge timestamp is moved to the
    regular sample time, its derivatives are reconstructed from neighboring
    native q/v states, and later timestamps are shifted by the excess gap.  A
    different or ambiguous pattern is rejected rather than normalized.
    """
    original = read_native(source)
    time = original.time
    nominal, p99_small, dt = _small_step_statistics(time)
    if expected_dt is not None and abs(nominal - expected_dt) > max(1.0e-6, 0.02 * expected_dt):
        raise ValueError(f"unexpected_nominal_dt:{source}:{nominal}")
    large_indices = np.flatnonzero(dt > max(10.0 * nominal, nominal + 0.05))
    if len(large_indices) == 0:
        write_native(destination, original)
        return {
            "status": "NO_REPAIR_NEEDED",
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "original_state_count": len(original.rows),
            "repaired_state_count": len(original.rows),
            "nominal_dt_s": nominal,
            "large_gap_count": 0,
        }
    if len(large_indices) != 1:
        raise ValueError(f"ambiguous_large_gap_count:{source}:{len(large_indices)}")
    bridge_index = int(large_indices[0] + 1)
    if bridge_index <= 0 or bridge_index >= len(original.rows) - 1:
        raise ValueError(f"bridge_at_boundary:{source}:{bridge_index}")
    evidence = _bridge_state_residual(original, bridge_index, nominal)
    if abs(evidence["bridge_to_right_dt_s"] - nominal) > max(1.0e-6, 0.02 * nominal):
        raise ValueError(f"bridge_following_step_not_regular:{source}:{evidence}")
    # The right sample must be at the regular next sample time.  The bridge
    # q is retained, while its exported v/a are explicitly treated as the
    # corrupted fields and reconstructed below.
    if max(evidence["bridge_q_from_left_position_residual_rad"], evidence["bridge_q_from_right_position_residual_rad"]) > 5.0e-6:
        raise ValueError(f"bridge_position_not_continuous:{source}:{evidence}")
    gap = float(dt[bridge_index - 1])
    excess = gap - nominal
    rows: list[dict[str, float]] = []
    for index, row in enumerate(original.rows):
        updated = dict(row)
        if index == bridge_index:
            updated["t"] = float(original.time[index - 1] + nominal)
            # Central differences use the already-native neighboring states
            # and the retained q at the bridge.  They are only applied to the
            # proven exporter artifact, never to ordinary samples.
            # The raw neighbor timestamps span the corrupted gap; the state
            # samples themselves are adjacent regular 10 ms samples.
            updated_dt = 2.0 * nominal
            for joint in JOINTS:
                left_q = original.q[index - 1, JOINTS.index(joint)]
                right_q = original.q[index + 1, JOINTS.index(joint)]
                updated[f"{joint}_dq"] = float((right_q - left_q) / updated_dt)
                left_v = original.dq[index - 1, JOINTS.index(joint)]
                right_v = original.dq[index + 1, JOINTS.index(joint)]
                updated[f"{joint}_ddq"] = float((right_v - left_v) / updated_dt)
        if index > bridge_index:
            updated["t"] -= excess
        rows.append(updated)
    repaired = NativeTrajectory(original.fields, tuple(rows))
    # Keep the derived diagnostic jerk columns aligned with the repaired q/v/a
    # state.  The authoritative native jerk oracle is separate and never
    # inferred from this diagnostic column.
    if all(f"{joint}_jerk" in repaired.fields for joint in JOINTS):
        repaired_time = repaired.time
        repaired_ddq = repaired.ddq
        diagnostic_jerk = np.gradient(repaired_ddq, repaired_time, axis=0, edge_order=1)
        mutable_rows = [dict(row) for row in repaired.rows]
        for index, row in enumerate(mutable_rows):
            for joint_index, joint in enumerate(JOINTS):
                row[f"{joint}_jerk"] = float(diagnostic_jerk[index, joint_index])
        repaired = NativeTrajectory(repaired.fields, tuple(mutable_rows))
    repaired_dt = np.diff(repaired.time)
    if np.any(repaired_dt <= 0.0) or np.max(repaired_dt) > max(1.5 * nominal, p99_small * 1.5):
        raise ValueError(f"repair_left_nonregular_time:{source}")
    write_native(destination, repaired)
    return {
        "status": "REPAIRED",
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "original_state_count": len(original.rows),
        "repaired_state_count": len(repaired.rows),
        "nominal_dt_s": nominal,
        "large_gap_count": 1,
        "repaired_bridge_index_zero_based": bridge_index,
        "repaired_bridge_gap_s": gap,
        "timestamp_shift_after_bridge_s": excess,
        "bridge_evidence": evidence,
        "repaired_max_dt_s": float(np.max(repaired_dt)),
        "repaired_min_dt_s": float(np.min(repaired_dt)),
        "bridge_q_preserved_derivatives_reconstructed": True,
    }


def repair_directory(source_dir: Path, destination_dir: Path, case_ids: Iterable[str] | None = None) -> dict[str, Any]:
    selected = tuple(case_ids) if case_ids is not None else tuple(sorted(path.stem for path in source_dir.glob("*.csv")))
    if not selected:
        raise ValueError(f"no_cases:{source_dir}")
    records: list[dict[str, Any]] = []
    for case_id in selected:
        source = source_dir / f"{case_id}.csv"
        destination = destination_dir / f"{case_id}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        records.append(repair_execution_form(source, destination, expected_dt=0.01))
    summary = {
        "schema_version": "d52-execution-form-repair-v1",
        "status": "PASS" if all(item["status"] in {"REPAIRED", "NO_REPAIR_NEEDED"} for item in records) else "BLOCKED",
        "method": "remove_one_proven_bridge_timestamp_marker_and_shift_following_native_states",
        "collision_method": "adaptive_discrete_interpolation",
        "case_count": len(records),
        "repaired_case_count": sum(item["status"] == "REPAIRED" for item in records),
        "cases": records,
    }
    destination_dir.mkdir(parents=True, exist_ok=True)
    (destination_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def case_family(case_id: str) -> str:
    for name, members in FAMILIES.items():
        if case_id in members:
            return name
    raise ValueError(f"unknown_case:{case_id}")


def write_manifest(path: Path, trajectory_dir: Path, case_ids: Iterable[str] = CASES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("case_id", "trajectory_csv", "family"))
        for case_id in case_ids:
            trajectory = trajectory_dir / f"{case_id}.csv"
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            writer.writerow((case_id, wsl_path(trajectory), case_family(case_id)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    case_ids = args.case_ids.split(",") if args.case_ids else None
    summary = repair_directory(args.source_dir.resolve(), args.output_dir.resolve(), case_ids)
    if args.manifest is not None:
        write_manifest(args.manifest.resolve(), args.output_dir.resolve(), case_ids or CASES)
    print(json.dumps({"status": summary["status"], "case_count": summary["case_count"], "repaired_case_count": summary["repaired_case_count"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
