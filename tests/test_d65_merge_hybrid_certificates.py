"""Fail-closed D65 hybrid merge tests."""

from __future__ import annotations

import hashlib
import json
import pytest
from pathlib import Path

from tools.d65_merge_hybrid_certificates import HybridMergeError, merge


TRAJ = "C:/evidence/adversarial_0100.csv"
TRAJ_SHA = "a" * 64
MODEL_SHA = "b" * 64
PAIRS = ["base_link|forearm_link", "shoulder_link|wrist_link"]
BINARY = str(Path(__file__).resolve())
BINARY_SHA = hashlib.sha256(Path(BINARY).read_bytes()).hexdigest()


def evidence(tmp_path: Path | None = None) -> tuple[dict[str, object], dict[str, object]]:
    evidence_dir = tmp_path or Path(__file__).resolve().parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trajectory = evidence_dir / "known_answer_trajectory.csv"
    profile = evidence_dir / "known_answer_profile.json"
    urdf = evidence_dir / "known_answer.urdf"
    srdf = evidence_dir / "known_answer.srdf"
    backend_dir = evidence_dir / "known_answer_backend"
    backend_dir.mkdir(parents=True, exist_ok=True)
    backend_certificate = backend_dir / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
    backend_manifest = backend_dir / "certificate_manifest.csv"
    trajectory.write_text("t,q\n0,0\n1,0\n", encoding="utf-8")
    profile.write_text("{}\n", encoding="utf-8")
    urdf.write_text("<robot name='known'/>\n", encoding="utf-8")
    srdf.write_text("<robot name='known'/>\n", encoding="utf-8")
    backend_manifest.write_text(f"case_id,trajectory_csv,family\nknown_answer,{trajectory.as_posix()},TEST\n", encoding="utf-8")
    raw_backend = {
        "schema_version": "d54-fk-aware-articulated-certificate-v1",
        "status": "PASS",
        "certificate_method": "fk_aware_adaptive_pairwise_interval_bound",
        "collision_method": "adaptive_discrete_interpolation",
        "unresolved_is_not_pass": True,
        "case_count": 1,
        "passing_case_count": 1,
        "worst_case_id": "known_answer",
        "required_pair_count": 2,
        "required_pair_coverage_complete": True,
        "all_required_pairs": PAIRS,
        "collision_region_count": 0,
        "unresolved_region_count": 0,
        "minimum_certified_clearance_m": 0.01,
    }
    backend_certificate.write_text(json.dumps(raw_backend), encoding="utf-8")
    trajectory_sha = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    profile_sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    urdf_sha = hashlib.sha256(urdf.read_bytes()).hexdigest()
    srdf_sha = hashlib.sha256(srdf.read_bytes()).hexdigest()
    interval = {
        "schema_version": "d62-bvh-explicit-fk-qt-interval-certificate-v1",
        "status": "UNRESOLVED",
        "trajectory_id": "known_answer",
        "interval_count": 2,
        "certificate_type": "OUTWARD_INTERVAL_BVH",
        "checked_interval_indices": [0, 1],
        "certified_interval_indices": [0],
        "unresolved_interval_indices": [1],
        "collision_interval_indices": [],
        "coverage": {"coverage_complete": True, "resource_limit_reached": False, "checked_interval_count": 2, "requested_interval_count": 2},
        "input_contract": {
            "trajectory_validation": {"status": "VALID"},
            "trajectory": str(trajectory),
            "trajectory_sha256": trajectory_sha,
            "urdf": str(urdf),
            "srdf": str(srdf),
            "urdf_sha256": urdf_sha,
            "srdf_sha256": srdf_sha,
            "required_pairs": PAIRS,
            "required_pair_count": 2,
        },
        "execution": {"wall_time_s": 3.0},
    }
    fallback = {
        "schema_version": "d65-validated-fk-lipschitz-fallback-v1",
        "status": "PASS",
        "trajectory_id": "known_answer",
        "certificate_type": "QUALIFIED_FK_LIPSCHITZ",
        "interval_count": 2,
        "collision_interval_count": 0,
        "unresolved_interval_count": 0,
        "required_pair_count": 2,
        "required_pairs": PAIRS,
        "minimum_certified_model_clearance_m": 0.01,
        "input_evidence": {
            "status": "VALID",
            "family": "TEST",
            "trajectory_validation": {"status": "VALID"},
            "trajectory": {"path": str(trajectory), "sha256": trajectory_sha},
            "profile_j": {"path": str(profile), "sha256": profile_sha},
            "authorized_binding": {"status": "VALID", "trajectory_id": "known_answer"},
            "model_identity": {"urdf_sha256": urdf_sha, "srdf_sha256": srdf_sha},
        },
        "compiled_method": {"numerical_acceptance_margin_m": 1.0e-9},
        "backend": {
            "runtime_s": 1.0,
            "exit_code": 0,
            "raw_status": "PASS",
            "missing_fields": [],
            "certificate": str(backend_certificate),
            "certificate_sha256": hashlib.sha256(backend_certificate.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(backend_manifest.read_bytes()).hexdigest(),
            "binary": {"path": BINARY, "sha256": BINARY_SHA},
        },
    }
    return interval, fallback


def test_merge_closes_only_with_complete_passing_fallback(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    result = merge(interval, fallback, 10.0)
    assert result["status"] == "PASS"
    assert result["unresolved_interval_count"] == 0
    assert result["performance"]["speedup_vs_d62_reference"] == 2.5


def test_merge_preserves_collision(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    interval["status"] = "COLLISION_FOUND"
    interval["verification_result"] = "COLLISION_FOUND"
    interval["collision_interval_indices"] = [1]
    interval["unresolved_interval_indices"] = []
    result = merge(interval, fallback, 10.0)
    assert result["status"] == "COLLISION_FOUND"


def test_merge_preserves_fallback_unresolved(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    fallback["status"] = "UNRESOLVED"
    fallback["unresolved_interval_count"] = 1
    fallback["backend"]["raw_status"] = "UNRESOLVED"
    raw_path = Path(fallback["backend"]["certificate"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["status"] = "UNRESOLVED"
    raw["passing_case_count"] = 0
    raw["unresolved_region_count"] = 1
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    fallback["backend"]["certificate_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    result = merge(interval, fallback, 10.0)
    assert result["status"] == "UNRESOLVED"
    assert result["unresolved_interval_count"] == 1


def test_merge_rejects_partial_interval_coverage(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    interval["coverage"]["coverage_complete"] = False
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


@pytest.mark.parametrize("field", ["trajectory_id", "required_pairs", "trajectory_sha256"])
def test_merge_rejects_identity_or_partition_gaps(field: str, tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    if field == "trajectory_id":
        fallback["trajectory_id"] = "other"
    elif field == "required_pairs":
        fallback["required_pairs"] = []
    else:
        interval["input_contract"][field] = None
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_rejects_missing_schema_instead_of_defaulting_to_zero(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    del fallback["collision_interval_count"]
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_rejects_overlapping_or_incomplete_partition(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    interval["certified_interval_indices"] = [0, 1]
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_accepts_same_pair_identity_in_different_order(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    interval["input_contract"]["required_pairs"] = [PAIRS[1], PAIRS[0]]
    interval["input_contract"]["required_pair_count"] = 2
    fallback["required_pairs"] = [PAIRS[0], PAIRS[1]]
    fallback["required_pair_count"] = 2
    result = merge(interval, fallback, 10.0)
    assert result["status"] == "PASS"
    assert result["identity"]["required_pairs"] == sorted(PAIRS)


def test_merge_rejects_duplicate_pairs_or_wrong_pair_count(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    interval["input_contract"]["required_pairs"] = [PAIRS[0], PAIRS[0]]
    interval["input_contract"]["required_pair_count"] = 2
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)

    interval, fallback = evidence(tmp_path)
    fallback["required_pair_count"] = 3
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_rejects_recorded_binary_digest_mismatch(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    fallback["backend"]["binary"]["sha256"] = "0" * 64
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_rejects_raw_certificate_or_current_input_tamper(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    fallback["backend"]["certificate_sha256"] = "0" * 64
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_rejects_self_consistent_manifest_and_summary_forgery(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    manifest_path = Path(fallback["backend"]["certificate"]).parent / "certificate_manifest.csv"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(",TEST\n", ",FORGED\n"),
        encoding="utf-8",
    )
    fallback["backend"]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)

    interval, fallback = evidence(tmp_path)
    raw_path = Path(fallback["backend"]["certificate"])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["minimum_certified_clearance_m"] = 123.0
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    fallback["backend"]["certificate_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)


def test_merge_rejects_status_partition_and_invalid_runtime(tmp_path: Path) -> None:
    interval, fallback = evidence(tmp_path)
    interval["status"] = "PASS"
    interval["verification_result"] = "PASS"
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)
    interval, fallback = evidence(tmp_path)
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 0.0)

    interval, fallback = evidence(tmp_path)
    Path(interval["input_contract"]["trajectory"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(HybridMergeError):
        merge(interval, fallback, 10.0)
