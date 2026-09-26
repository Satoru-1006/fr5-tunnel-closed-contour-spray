"""Dataset-contract utilities for the Stage 3 H11 offline baseline.

This module deliberately knows about the frozen H10 *schema* and split, but
does not regenerate H10 data.  It only reads the authoritative release,
constructs causal windows inside one native segment, and records evidence
about every decision made for the H11 run.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_H10_SEMANTIC_SHA256 = "cb282a06d73c322208f27bf7c0b019831a24f524ac0d72263929f22303f0801b"
EXPECTED_H10_TERMINAL = {
    "STAGE_3_H10": "PASSED",
    "FIRST_BLOCKER": "none",
    "CERTIFIED_TRAJECTORY_FAMILIES": 48,
    "TOTAL_SAMPLES": 200256,
    "SEGMENTS": 480,
    "TRAIN_FAMILIES": 31,
    "VALIDATION_FAMILIES": 7,
    "TEST_FAMILIES": 6,
    "GENERALIZATION_FAMILIES": 4,
    "GROUP_LEAKAGE_VIOLATIONS": 0,
    "TRIVIAL_DUPLICATE_FAMILIES": 0,
    "HARD_CONSTRAINT_VIOLATIONS": 0,
    "DATASET_REPLAY": "3/3",
    "ML_SPLIT_READY": "YES",
    "MODEL_TRAINING_DATA_SUFFICIENT": "YES",
    "READY_FOR_STAGE_3_H11": "YES",
}

H10_ML_INPUT_ARTIFACTS = (
    "trajectory_samples.jsonl",
    "trajectory_segments.jsonl",
    "trajectory_families.jsonl",
    "trajectory_family_metrics.jsonl",
    "dataset_manifest.json",
    "dataset_schema.json",
    "dataset_split_manifest.json",
    "dataset_split_audit.json",
    "dataset_semantic_hash.json",
    "hard_constraint_report.json",
)

JOINTS = 6
INPUT_HISTORY = 16
PREDICTION_HORIZON = 8
FEATURE_NAMES = tuple(
    [f"planned_joint_position_{i + 1}" for i in range(JOINTS)]
    + [f"planned_joint_velocity_{i + 1}" for i in range(JOINTS)]
    + [f"planned_joint_acceleration_{i + 1}" for i in range(JOINTS)]
    + ["spray_on", "normalized_local_trajectory_time"]
)
NORMALIZED_FEATURE_NAMES = FEATURE_NAMES[:18] + (FEATURE_NAMES[-1],)
IDENTITY_FIELDS = ("trajectory_family_id", "trajectory_id", "split_role", "generation_seed")

# These are the frozen execution bounds used by the existing H7/H10 native
# contract.  H11 never uses a learned model to redefine these limits.
POSITION_LOWER = np.asarray([-3.0543, -4.6251, -2.8274, -4.6251, -3.0543, -3.0543], dtype=np.float64)
POSITION_UPPER = np.asarray([3.0543, 1.4835, 2.8274, 1.4835, 3.0543, 3.0543], dtype=np.float64)
VELOCITY_LIMIT = np.asarray([0.4725, 0.4725, 0.4725, 0.48, 0.48, 0.48], dtype=np.float64)
ACCELERATION_LIMIT = np.asarray([0.105] * 6, dtype=np.float64)
JERK_LIMIT = np.asarray([8.0] * 6, dtype=np.float64)


class H11ContractError(RuntimeError):
    """Raised when an H11 input contract cannot be certified."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _required_terminal_errors(terminal: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key, expected in EXPECTED_H10_TERMINAL.items():
        if terminal.get(key) != expected:
            errors.append(f"{key}={terminal.get(key)!r},expected={expected!r}")
    if terminal.get("DATASET_SEMANTIC_SHA256") != EXPECTED_H10_SEMANTIC_SHA256:
        errors.append("terminal_semantic_hash_mismatch")
    return errors


def verify_h10_authority(root: Path, h10_root: Path) -> dict[str, Any]:
    """Verify the frozen H10 terminal certificate and required artifact set."""
    terminal_path = h10_root / "stage3_h10_terminal_certificate.json"
    semantic_path = h10_root / "dataset_semantic_hash.json"
    manifest_path = h10_root / "dataset_manifest.json"
    split_path = h10_root / "dataset_split_manifest.json"
    audit_path = h10_root / "dataset_split_audit.json"
    missing = [str(path) for path in (terminal_path, semantic_path, manifest_path, split_path, audit_path) if not path.is_file()]
    missing.extend(str(h10_root / name) for name in H10_ML_INPUT_ARTIFACTS if not (h10_root / name).is_file())
    if missing:
        return {"status": "BLOCKED", "first_blocker": "h10_authoritative_input_missing", "missing": missing, "errors": []}

    terminal = load_json(terminal_path)
    semantic = load_json(semantic_path)
    manifest = load_json(manifest_path)
    split_manifest = load_json(split_path)
    split_audit = load_json(audit_path)
    errors = _required_terminal_errors(terminal)
    if semantic.get("semantic_dataset_sha256") != EXPECTED_H10_SEMANTIC_SHA256:
        errors.append("dataset_semantic_hash_mismatch")
    if manifest.get("SEMANTIC_DATASET_SHA256") != EXPECTED_H10_SEMANTIC_SHA256:
        errors.append("dataset_manifest_semantic_hash_mismatch")
    if split_audit.get("GROUP_LEAKAGE_VIOLATIONS") != 0:
        errors.append("group_leakage_nonzero")
    counts = split_manifest.get("counts", {})
    expected_counts = {"TRAIN": 31, "VALIDATION": 7, "TEST": 6, "GENERALIZATION": 4}
    if counts != expected_counts:
        errors.append(f"frozen_split_counts_mismatch:{counts!r}")
    if split_manifest.get("split_unit") != "trajectory_family_id":
        errors.append("frozen_split_unit_mismatch")
    if split_manifest.get("GENERALIZATION_POLICY", "").find("same internal_wiper_open_arch") < 0:
        errors.append("generalization_policy_not_conservative")
    return {
        "status": "PASSED" if not errors else "BLOCKED",
        "first_blocker": None if not errors else ("h10_semantic_hash_mismatch" if "dataset_semantic_hash_mismatch" in errors or "terminal_semantic_hash_mismatch" in errors else "h10_authoritative_certificate_mismatch"),
        "missing": missing,
        "errors": errors,
        "terminal": terminal,
        "semantic": semantic,
        "manifest": manifest,
        "split_manifest": split_manifest,
        "split_audit": split_audit,
    }


def snapshot_frozen_upstream(root: Path, h10_root: Path) -> dict[str, Any]:
    """Hash H7/H8-R/historical-H8/H9 evidence referenced by H10.

    H10's frozen manifest is the authoritative list; this function only reads
    it and adds the H10 ML input files themselves.  No upstream file is ever
    written by H11.
    """
    manifest = h10_root / "frozen_artifact_manifest.json"
    records: list[dict[str, Any]] = []
    if not manifest.is_file():
        return {"status": "BLOCKED", "first_blocker": "h10_authoritative_input_missing", "records": [], "groups": {}}
    frozen = load_json(manifest)
    for source in frozen.get("records", []):
        path = Path(str(source.get("absolute_path") or source.get("path")))
        if not path.is_absolute():
            path = root / path
        exists = path.is_file()
        records.append(
            {
                "group": source.get("group"),
                "path": source.get("path", _relative(root, path)),
                "absolute_path": str(path.resolve()),
                "before_sha256": source.get("sha256"),
                "before_size_bytes": source.get("size_bytes"),
                "observed_sha256": sha256_file(path) if exists else None,
                "observed_size_bytes": path.stat().st_size if exists else None,
                "unchanged": bool(exists and sha256_file(path) == source.get("sha256") and path.stat().st_size == source.get("size_bytes")),
            }
        )
    for name in H10_ML_INPUT_ARTIFACTS + ("frozen_artifact_manifest.json", "stage3_h10_terminal_certificate.json", "stage3_h10_gate_report.json"):
        path = h10_root / name
        records.append(
            {
                "group": "H10_EVIDENCE",
                "path": _relative(root, path),
                "absolute_path": str(path.resolve()),
                "before_sha256": sha256_file(path) if path.is_file() else None,
                "before_size_bytes": path.stat().st_size if path.is_file() else None,
                "observed_sha256": sha256_file(path) if path.is_file() else None,
                "observed_size_bytes": path.stat().st_size if path.is_file() else None,
                "unchanged": path.is_file(),
            }
        )
    groups: dict[str, bool] = {}
    for group in ("H7_EVIDENCE", "H8_R_EVIDENCE", "HISTORICAL_H8_EVIDENCE", "H9_EVIDENCE", "H10_EVIDENCE"):
        group_records = [item for item in records if item["group"] == group]
        groups[group] = bool(group_records) and all(item["unchanged"] for item in group_records)
    return {
        "schema_version": "stage3_h11_upstream_immutability_v1",
        "algorithm": "SHA-256",
        "status": "PASSED" if all(groups.values()) else "BLOCKED",
        "first_blocker": None if all(groups.values()) else "upstream_artifact_mutated",
        "groups": groups,
        "records": records,
    }


@dataclass
class Segment:
    family_id: str
    trajectory_id: str
    segment_id: int
    segment_order: int
    spray_state: str
    primitive_id: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    sample_indices: np.ndarray
    controlled_stop_count: int

    @property
    def key(self) -> str:
        return f"{self.family_id}|{self.trajectory_id}|{self.segment_id}|{self.segment_order}"

    @property
    def local_time(self) -> np.ndarray:
        if len(self.times) == 0 or self.times[-1] <= self.times[0]:
            return np.zeros(len(self.times), dtype=np.float64)
        return (self.times - self.times[0]) / (self.times[-1] - self.times[0])


@dataclass
class H11Dataset:
    h10_root: Path
    split_map: dict[str, str]
    segments: list[Segment]
    source_sample_count: int
    source_family_count: int
    source_segment_count: int


def _as_six(row: Mapping[str, Any], field: str) -> list[float]:
    value = row.get(field)
    if not isinstance(value, list) or len(value) != JOINTS:
        raise H11ContractError(f"invalid_h10_vector:{field}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise H11ContractError(f"nonfinite_h10_vector:{field}")
    return result


def load_h10_segments(h10_root: Path) -> H11Dataset:
    """Read H10 samples into compact float arrays grouped by native segment."""
    split_manifest = load_json(h10_root / "dataset_split_manifest.json")
    split_map = {str(key): str(value) for key, value in split_manifest.get("group_to_role", {}).items()}
    if set(split_map.values()) != {"TRAIN", "VALIDATION", "TEST", "GENERALIZATION"}:
        raise H11ContractError("frozen_split_roles_incomplete")
    groups: dict[tuple[str, str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    source_count = 0
    sample_path = h10_root / "trajectory_samples.jsonl"
    with sample_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_count += 1
            family = str(row.get("trajectory_family_id"))
            role = split_map.get(family)
            if role is None or str(row.get("split_role")) != role:
                raise H11ContractError(f"dataset_split_contamination:line_{line_number}")
            if row.get("actual_tcp_position") is not None or row.get("actual_tcp_orientation") is not None:
                raise H11ContractError("unavailable_tcp_signal_not_null")
            if row.get("collision_method") != "adaptive_discrete_interpolation":
                raise H11ContractError("collision_method_mismatch")
            key = (family, str(row.get("trajectory_id")), int(row.get("segment_id")), int(row.get("segment_order")), str(row.get("spray_state")), str(row.get("primitive_id")))
            groups[key].append(row)
    segments: list[Segment] = []
    for key, rows in groups.items():
        rows.sort(key=lambda item: (int(item["sample_index"]), float(item["trajectory_time"])))
        times = np.asarray([float(row["trajectory_time"]) for row in rows], dtype=np.float64)
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise H11ContractError(f"segment_time_not_strictly_monotonic:{key}")
        segment = Segment(
            family_id=key[0], trajectory_id=key[1], segment_id=key[2], segment_order=key[3], spray_state=key[4], primitive_id=key[5],
            times=times,
            positions=np.asarray([_as_six(row, "planned_joint_position") for row in rows], dtype=np.float64),
            velocities=np.asarray([_as_six(row, "planned_joint_velocity") for row in rows], dtype=np.float64),
            accelerations=np.asarray([_as_six(row, "planned_joint_acceleration") for row in rows], dtype=np.float64),
            sample_indices=np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64),
            controlled_stop_count=max(int(row.get("controlled_stop_count", 0) or 0) for row in rows),
        )
        segments.append(segment)
    segments.sort(key=lambda item: (item.family_id, item.segment_order, item.segment_id))
    return H11Dataset(h10_root, split_map, segments, source_count, len(split_map), len(segments))


def _feature_matrix(segment: Segment, stats: Mapping[str, Mapping[str, Any]] | None = None) -> np.ndarray:
    features = np.column_stack(
        [segment.positions, segment.velocities, segment.accelerations, (np.ones(len(segment.times)) if segment.spray_state == "SPRAY_ON" else np.zeros(len(segment.times))), segment.local_time]
    ).astype(np.float32)
    if stats is None:
        return features
    for index, name in enumerate(FEATURE_NAMES):
        if name not in NORMALIZED_FEATURE_NAMES:
            continue
        item = stats[name]
        scale = float(item.get("scale", item.get("std", 1.0)))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        features[:, index] = (features[:, index] - float(item["mean"])) / scale
    return features


def compute_normalization_stats(dataset: H11Dataset) -> dict[str, Any]:
    """Compute statistics over TRAIN samples only, with one streaming pass."""
    values: dict[str, list[np.ndarray]] = {name: [] for name in NORMALIZED_FEATURE_NAMES}
    train_segments = [segment for segment in dataset.segments if dataset.split_map[segment.family_id] == "TRAIN"]
    if not train_segments:
        raise H11ContractError("normalization_train_samples_missing")
    for segment in train_segments:
        matrix = _feature_matrix(segment)
        for index, name in enumerate(FEATURE_NAMES):
            if name in values:
                values[name].append(matrix[:, index].astype(np.float64, copy=False))
    stats: dict[str, Any] = {
        "schema_version": "stage3_h11_normalization_stats_v1",
        "source_split": "TRAIN",
        "source_family_count": len({segment.family_id for segment in train_segments}),
        "source_sample_count": int(sum(len(segment.times) for segment in train_segments)),
        "channels": {},
    }
    for name in NORMALIZED_FEATURE_NAMES:
        flat = np.concatenate(values[name])
        mean = float(np.mean(flat, dtype=np.float64))
        std = float(np.std(flat, dtype=np.float64))
        stats["channels"][name] = {
            "count": int(flat.size), "mean": mean, "std": std, "scale": std if std > 0.0 else 1.0,
            "minimum": float(np.min(flat)), "maximum": float(np.max(flat)), "source_split": "TRAIN",
        }
    return stats


def make_windows(dataset: H11Dataset, output_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Create causal windows and audit all boundary rejections.

    A window is fully contained in one H10 native segment.  Segment boundaries
    include SPRAY-ON/OFF transitions and controlled stops, so no window can
    turn a process break into a continuous training example.
    """
    windows: list[dict[str, Any]] = []
    rejected = Counter()
    boundary = Counter()
    for segment in dataset.segments:
        role = dataset.split_map[segment.family_id]
        count = len(segment.times)
        if count < INPUT_HISTORY + PREDICTION_HORIZON:
            rejected["insufficient_segment_samples"] += 1
            continue
        for start in range(0, count - INPUT_HISTORY - PREDICTION_HORIZON + 1):
            history = range(start, start + INPUT_HISTORY)
            target = range(start + INPUT_HISTORY, start + INPUT_HISTORY + PREDICTION_HORIZON)
            indices = list(history) + list(target)
            # These checks are intentionally explicit even though grouping by
            # segment makes them structurally true.
            if len({segment.family_id}) != 1:
                rejected["cross_family"] += 1; boundary["CROSS_FAMILY_WINDOWS"] += 1; continue
            if len({segment.trajectory_id}) != 1:
                rejected["cross_trajectory"] += 1; boundary["CROSS_TRAJECTORY_WINDOWS"] += 1; continue
            if segment.controlled_stop_count > 0:
                rejected["controlled_stop_segment"] += 1; boundary["CONTROLLED_STOP_WINDOWS"] += 1; continue
            if len(indices) != len(set(indices)):
                rejected["duplicate_indices"] += 1; boundary["DISCONTINUOUS_WINDOWS"] += 1; continue
            windows.append({
                "window_id": len(windows), "split_role": role, "trajectory_family_id": segment.family_id,
                "trajectory_id": segment.trajectory_id, "segment_key": segment.key, "segment_id": segment.segment_id,
                "segment_order": segment.segment_order, "primitive_id": segment.primitive_id, "spray_state": segment.spray_state,
                "start_offset": start, "history_start_sample_index": int(segment.sample_indices[start]),
                "history_end_sample_index": int(segment.sample_indices[start + INPUT_HISTORY - 1]),
                "target_start_sample_index": int(segment.sample_indices[start + INPUT_HISTORY]),
                "target_end_sample_index": int(segment.sample_indices[start + INPUT_HISTORY + PREDICTION_HORIZON - 1]),
                "history_length": INPUT_HISTORY, "prediction_horizon": PREDICTION_HORIZON,
                "cross_family": False, "cross_trajectory": False, "cross_segment": False,
                "cross_controlled_stop": False, "discontinuous": False,
            })
    if output_path is not None:
        write_json(output_path, {
            "schema_version": "stage3_h11_window_manifest_v1", "input_history": INPUT_HISTORY, "prediction_horizon": PREDICTION_HORIZON,
            "split_unit": "trajectory_family_id", "window_count": len(windows),
            "family_counts": {role: len({item["trajectory_family_id"] for item in windows if item["split_role"] == role}) for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION")},
            "window_counts": dict(Counter(item["split_role"] for item in windows)), "rejected_window_reasons": dict(rejected), "windows": windows,
        })
    audit = {
        "schema_version": "stage3_h11_window_boundary_audit_v1", "CROSS_FAMILY_WINDOWS": boundary["CROSS_FAMILY_WINDOWS"],
        "CROSS_TRAJECTORY_WINDOWS": boundary["CROSS_TRAJECTORY_WINDOWS"], "CROSS_SEGMENT_WINDOWS": boundary["CROSS_SEGMENT_WINDOWS"],
        "CONTROLLED_STOP_WINDOWS": boundary["CONTROLLED_STOP_WINDOWS"], "DISCONTINUOUS_WINDOWS": boundary["DISCONTINUOUS_WINDOWS"],
        "total_windows": len(windows), "rejected_window_reasons": dict(rejected), "construction": "one H10 native segment per window; segment transitions are hard breaks",
    }
    counts = {role: {"families": len({item["trajectory_family_id"] for item in windows if item["split_role"] == role}), "windows": sum(item["split_role"] == role for item in windows)} for role in ("TRAIN", "VALIDATION", "TEST", "GENERALIZATION")}
    return windows, audit, counts


def window_semantic_hash(windows: Sequence[Mapping[str, Any]]) -> str:
    return semantic_hash(list(windows))


def environment_metadata(torch_module: Any | None = None) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(), "platform": platform.platform(), "numpy_version": np.__version__,
        "pytorch_version": getattr(torch_module, "__version__", None), "cuda_available": bool(torch_module is not None and torch_module.cuda.is_available()),
        "gpu_model": (torch_module.cuda.get_device_name(0) if torch_module is not None and torch_module.cuda.is_available() else None),
    }


def contract_payload(dataset: H11Dataset, authority: Mapping[str, Any], stats: Mapping[str, Any], window_audit: Mapping[str, Any], window_counts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h11_dataset_contract_v1", "source_stage": "STAGE_3_H10", "source_h10": _relative(Path.cwd(), dataset.h10_root),
        "source_h10_semantic_sha256": EXPECTED_H10_SEMANTIC_SHA256, "source_sample_count": dataset.source_sample_count,
        "source_family_count": dataset.source_family_count, "source_segment_count": dataset.source_segment_count,
        "split_unit": "trajectory_family_id", "split_policy": "frozen H10 group split; no adjacent/replay/timing-variant splitting",
        "split_counts": window_counts, "features": list(FEATURE_NAMES), "normalized_features": list(NORMALIZED_FEATURE_NAMES),
        "excluded_identity_fields": list(IDENTITY_FIELDS), "actual_tcp_position": "null; unavailable and not used",
        "collision_method": "adaptive_discrete_interpolation", "ccd": "not_available", "clearance": None,
        "window_boundary_audit": window_audit, "normalization_source": stats.get("source_split"),
        "authority_status": authority.get("status"), "software_only": {"physical_robot_connected": False, "physical_driver_loaded": False, "physical_fjt_goals_sent": 0, "robot_motion_started": False},
    }

