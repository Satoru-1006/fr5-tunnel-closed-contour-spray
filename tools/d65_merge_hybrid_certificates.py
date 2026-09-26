"""Fail-closed merge of D65 interval-FK and qualified FK/Lipschitz evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


class HybridMergeError(ValueError):
    """Raised when evidence identity or coverage is insufficient to merge."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridMergeError(f"{name} must be an object")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HybridMergeError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _norm_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise HybridMergeError("identity path is missing")
    # Backend manifests are emitted inside WSL while the merge runs on the
    # host.  Canonicalise /mnt/<drive>/... before comparing it with Windows
    # paths; otherwise a valid manifest would appear to reference a different
    # trajectory and path tampering could not be checked consistently.
    candidate = value.replace("\\", "/")
    if len(candidate) >= 7 and candidate.lower().startswith("/mnt/") and candidate[5].isalpha() and candidate[6] == "/":
        candidate = f"{candidate[5].upper()}:/{candidate[7:]}"
    return os.path.normcase(os.path.abspath(candidate))


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
        raise HybridMergeError(f"{name} must be a 64-character SHA-256 digest")
    return value.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_current_file(path_value: Any, sha_value: Any, name: str) -> Path:
    path = Path(path_value) if isinstance(path_value, Path) else Path(_norm_path(path_value))
    if not path.is_file():
        raise HybridMergeError(f"{name} path is not present")
    expected = _require_sha(sha_value, f"{name} sha256")
    if _sha256_file(path) != expected:
        raise HybridMergeError(f"{name} digest does not match the current bytes")
    return path


def _require_pair_set(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or "|" not in item for item in value):
        raise HybridMergeError(f"{name} must be a non-empty collision-pair array")
    if len(value) != len(set(value)):
        raise HybridMergeError(f"{name} contains duplicate collision pairs")
    return tuple(sorted(value))


def _require_complete_partition(interval: Mapping[str, Any], requested: int) -> tuple[list[int], list[int], list[int], list[int]]:
    arrays = {}
    expected = list(range(requested))
    for name in ("checked_interval_indices", "certified_interval_indices", "unresolved_interval_indices", "collision_interval_indices"):
        value = interval.get(name)
        if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise HybridMergeError(f"{name} must be an integer array")
        if value != sorted(set(value)):
            raise HybridMergeError(f"{name} must be sorted and duplicate-free")
        if any(item < 0 or item >= requested for item in value):
            raise HybridMergeError(f"{name} contains an out-of-range interval index")
        arrays[name] = value
    if arrays["checked_interval_indices"] != expected:
        raise HybridMergeError("D62 checked interval indices are not the complete ordered range")
    buckets = [set(arrays[name]) for name in ("certified_interval_indices", "unresolved_interval_indices", "collision_interval_indices")]
    if any(left & right for index, left in enumerate(buckets) for right in buckets[index + 1:]):
        raise HybridMergeError("D62 interval result buckets overlap")
    if set.union(*buckets) != set(expected):
        raise HybridMergeError("D62 interval result buckets do not cover the checked range")
    coverage = _require_mapping(interval.get("coverage"), "coverage")
    if coverage.get("checked_interval_count") != len(expected) or coverage.get("requested_interval_count") != requested:
        raise HybridMergeError("D62 coverage counts disagree with index arrays")
    if coverage.get("coverage_complete") is not True or coverage.get("resource_limit_reached") is True:
        raise HybridMergeError("D62 interval phase does not cover the complete trajectory")
    status = interval.get("status")
    if status not in {"PASS", "UNRESOLVED", "COLLISION_FOUND"}:
        raise HybridMergeError("D62 interval status is not a recognized completed result")
    if interval.get("verification_result", status) != status:
        raise HybridMergeError("D62 verification_result disagrees with status")
    if status == "PASS" and (arrays["unresolved_interval_indices"] or arrays["collision_interval_indices"]):
        raise HybridMergeError("D62 PASS status disagrees with unresolved/collision interval indices")
    if status == "UNRESOLVED" and (not arrays["unresolved_interval_indices"] or arrays["collision_interval_indices"]):
        raise HybridMergeError("D62 UNRESOLVED status disagrees with interval indices")
    if status == "COLLISION_FOUND" and not arrays["collision_interval_indices"]:
        raise HybridMergeError("D62 COLLISION_FOUND status has no collision interval witness")
    return arrays["checked_interval_indices"], arrays["certified_interval_indices"], arrays["unresolved_interval_indices"], arrays["collision_interval_indices"]


def merge(interval: Mapping[str, Any], fallback: Mapping[str, Any], before_runtime_s: float) -> dict[str, Any]:
    if isinstance(before_runtime_s, bool) or not isinstance(before_runtime_s, (int, float)) or not math.isfinite(float(before_runtime_s)) or float(before_runtime_s) <= 0.0:
        raise HybridMergeError("before_runtime_s must be finite and positive")
    if interval.get("schema_version") != "d62-bvh-explicit-fk-qt-interval-certificate-v1":
        raise HybridMergeError("unexpected D62 certificate schema")
    if fallback.get("schema_version") != "d65-validated-fk-lipschitz-fallback-v1":
        raise HybridMergeError("unexpected D65 fallback certificate schema")
    input_contract = _require_mapping(interval.get("input_contract"), "input_contract")
    interval_validation = _require_mapping(input_contract.get("trajectory_validation"), "trajectory_validation")
    fallback_input = _require_mapping(fallback.get("input_evidence"), "fallback input_evidence")
    fallback_validation = _require_mapping(fallback_input.get("trajectory_validation"), "fallback trajectory_validation")
    if interval_validation.get("status") != "VALID" or fallback_validation.get("status") != "VALID":
        raise HybridMergeError("D62 interval phase lacks a VALID trajectory precondition gate")
    if fallback_input.get("status") != "VALID":
        raise HybridMergeError("FK/Lipschitz fallback lacks a VALID CSV + Profile.j gate")
    coverage_for_count = _require_mapping(interval.get("coverage"), "coverage")
    raw_count = interval.get("interval_count", coverage_for_count.get("requested_interval_count"))
    requested = _require_int(raw_count, "interval_count", minimum=1)
    checked_indices, certified_indices, unresolved_indices, collision_indices = _require_complete_partition(interval, requested)
    fallback_intervals = _require_int(fallback.get("interval_count"), "fallback.interval_count", minimum=1)
    if requested != fallback_intervals:
        raise HybridMergeError("interval phase and fallback interval counts disagree")
    for key in ("collision_interval_count", "unresolved_interval_count", "required_pair_count"):
        if key not in fallback or not isinstance(fallback[key], int) or fallback[key] < 0:
            raise HybridMergeError(f"fallback.{key} is missing or invalid")
    if fallback.get("collision_interval_count") > requested or fallback.get("unresolved_interval_count") > requested:
        raise HybridMergeError("fallback interval counts exceed requested coverage")
    interval_id = interval.get("trajectory_id")
    fallback_id = fallback.get("trajectory_id")
    if not isinstance(interval_id, str) or not interval_id or interval_id != fallback_id:
        raise HybridMergeError("trajectory ids do not match")
    interval_path = _norm_path(input_contract.get("trajectory"))
    fallback_path = _norm_path(_require_mapping(fallback_input.get("trajectory"), "fallback trajectory").get("path"))
    if interval_path != fallback_path:
        raise HybridMergeError("trajectory paths do not match")
    interval_hash = _require_sha(input_contract.get("trajectory_sha256"), "D62 trajectory_sha256")
    fallback_hash = _require_sha(_require_mapping(fallback_input.get("trajectory"), "fallback trajectory").get("sha256"), "fallback trajectory sha256")
    if interval_hash != fallback_hash:
        raise HybridMergeError("trajectory hashes do not match")
    _require_current_file(input_contract.get("trajectory"), interval_hash, "trajectory")
    model_identity = _require_mapping(fallback_input.get("model_identity"), "fallback model_identity")
    for name in ("urdf_sha256", "srdf_sha256"):
        if _require_sha(input_contract.get(name), f"D62 {name}") != _require_sha(model_identity.get(name), f"fallback {name}"):
            raise HybridMergeError(f"model identity mismatch: {name}")
    _require_current_file(input_contract.get("urdf"), input_contract.get("urdf_sha256"), "URDF")
    _require_current_file(input_contract.get("srdf"), input_contract.get("srdf_sha256"), "SRDF")
    profile_identity = _require_mapping(fallback_input.get("profile_j"), "fallback Profile.j")
    _require_current_file(profile_identity.get("path"), profile_identity.get("sha256"), "Profile.j")
    binding = _require_mapping(fallback_input.get("authorized_binding"), "fallback authorized_binding")
    if binding.get("status") != "VALID" or binding.get("trajectory_id") != fallback_id:
        raise HybridMergeError("fallback trajectory/Profile.j authorized binding is not valid")
    # Production D65 finalists have an immutable source binding established by
    # the wrapper.  Re-check those hashes here so a forged fallback cannot make
    # itself self-consistent merely by rewriting its own binding object.
    try:
        from tools.d65_hybrid_certificate import AUTHORIZED_BINDINGS
    except ImportError:  # pragma: no cover - direct script fallback
        AUTHORIZED_BINDINGS = {}
    authorized = AUTHORIZED_BINDINGS.get(fallback_id)
    if authorized is not None:
        if interval_hash != authorized.get("trajectory_sha256") or fallback_hash != authorized.get("trajectory_sha256"):
            raise HybridMergeError("trajectory bytes do not match the authorized D65 source binding")
        if _require_sha(profile_identity.get("sha256"), "Profile.j sha256") != authorized.get("profile_j_sha256"):
            raise HybridMergeError("Profile.j bytes do not match the authorized D65 source binding")
        if requested != authorized.get("interval_count"):
            raise HybridMergeError("interval count does not match the authorized D65 source binding")
    interval_pairs = _require_pair_set(input_contract.get("required_pairs"), "D62 required_pairs")
    fallback_pairs = _require_pair_set(fallback.get("required_pairs"), "fallback required_pairs")
    if interval_pairs != fallback_pairs:
        raise HybridMergeError("required collision-pair identity is missing or disagrees")
    if input_contract.get("required_pair_count") != len(interval_pairs) or fallback.get("required_pair_count") != len(fallback_pairs):
        raise HybridMergeError("required collision-pair counts disagree with pair identities")
    if fallback.get("certificate_type") != "QUALIFIED_FK_LIPSCHITZ":
        raise HybridMergeError("fallback certificate type is not qualified FK/Lipschitz")
    backend = _require_mapping(fallback.get("backend"), "fallback backend")
    if backend.get("exit_code") != 0 or backend.get("raw_status") != fallback.get("status") or backend.get("missing_fields") not in ([], None):
        raise HybridMergeError("fallback backend was not a fresh successful invocation")
    binary = _require_mapping(backend.get("binary"), "fallback backend binary")
    binary_path = Path(str(binary.get("path")))
    if not binary_path.is_file():
        raise HybridMergeError("fallback backend binary path is not present")
    binary_sha = _require_sha(binary.get("sha256"), "fallback backend binary sha256")
    if _sha256_file(binary_path) != binary_sha:
        raise HybridMergeError("fallback backend binary digest does not match the recorded bytes")
    certificate_path = _require_current_file(backend.get("certificate"), backend.get("certificate_sha256"), "fallback backend certificate")
    try:
        raw_backend = json.loads(certificate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HybridMergeError(f"fallback backend certificate is unreadable: {error}") from error
    raw_pairs = _require_pair_set(raw_backend.get("all_required_pairs"), "raw backend required_pairs")
    raw_case_count = raw_backend.get("case_count")
    raw_passing_count = raw_backend.get("passing_case_count")
    raw_clearance = raw_backend.get("minimum_certified_clearance_m")
    if (
        raw_backend.get("schema_version") != "d54-fk-aware-articulated-certificate-v1"
        or raw_backend.get("status") != fallback.get("status")
        or raw_backend.get("certificate_method") != "fk_aware_adaptive_pairwise_interval_bound"
        or raw_backend.get("collision_method") != "adaptive_discrete_interpolation"
        or raw_backend.get("unresolved_is_not_pass") is not True
        or raw_case_count != 1
        or raw_passing_count != (1 if fallback.get("status") == "PASS" else 0)
        or raw_backend.get("worst_case_id") != fallback_id
        or raw_backend.get("required_pair_coverage_complete") is not True
        or raw_backend.get("required_pair_count") != len(raw_pairs)
        or raw_pairs != fallback_pairs
        or raw_backend.get("collision_region_count") != fallback.get("collision_interval_count")
        or raw_backend.get("unresolved_region_count") != fallback.get("unresolved_interval_count")
        or not isinstance(raw_clearance, (int, float))
        or isinstance(raw_clearance, bool)
        or not math.isfinite(float(raw_clearance))
        or raw_clearance != fallback.get("minimum_certified_model_clearance_m")
    ):
        raise HybridMergeError("fallback summary disagrees with its raw backend certificate")
    manifest_path = certificate_path.parent / "certificate_manifest.csv"
    _require_current_file(manifest_path, backend.get("manifest_sha256"), "fallback backend manifest")
    try:
        import csv
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeError) as error:
        raise HybridMergeError(f"fallback backend manifest is unreadable: {error}") from error
    if len(rows) != 1:
        raise HybridMergeError("fallback backend manifest must contain exactly one case row")
    manifest_row = rows[0]
    if manifest_row.get("case_id") != fallback_id or _norm_path(manifest_row.get("trajectory_csv")) != interval_path or manifest_row.get("family") != fallback_input.get("family"):
        raise HybridMergeError("fallback backend manifest identity disagrees with fallback inputs")
    interval_collisions = len(collision_indices)
    fallback_collisions = fallback.get("collision_interval_count")
    fallback_unresolved = fallback.get("unresolved_interval_count")
    fallback_status = fallback.get("status")
    if interval_collisions or fallback_collisions or fallback_status == "COLLISION_FOUND":
        status = "COLLISION_FOUND"
        combined_unresolved = None
    elif fallback_status == "PASS" and fallback_unresolved == 0:
        status = "PASS"
        combined_unresolved = 0
    else:
        status = "UNRESOLVED"
        combined_unresolved = fallback_unresolved
    d62_runtime = float(_require_mapping(interval.get("execution"), "execution").get("wall_time_s", math.nan))
    fallback_runtime = float(backend.get("runtime_s", math.nan))
    if not math.isfinite(d62_runtime) or d62_runtime <= 0 or not math.isfinite(fallback_runtime) or fallback_runtime <= 0:
        raise HybridMergeError("runtime evidence is missing or non-positive")
    combined_runtime = d62_runtime + fallback_runtime
    speedup = before_runtime_s / combined_runtime if math.isfinite(combined_runtime) and combined_runtime > 0.0 else None
    return {
        "schema_version": "d65-hybrid-articulated-self-collision-v1",
        "status": status,
        "verification_result": status,
        "verification_domain": "model_based_articulated_continuous_self_collision",
        "trajectory_id": fallback.get("trajectory_id"),
        "interval_count": requested,
        "collision_interval_count": interval_collisions + fallback_collisions,
        "unresolved_interval_count": combined_unresolved,
        "hardware_or_physical_clearance": None,
        "exact_external_nonlinear_ccd_backend": "NOT_CLAIMED",
        "identity": {
            "trajectory_id": interval_id,
            "trajectory_path": interval_path,
            "trajectory_sha256": interval_hash,
            "urdf_sha256": model_identity["urdf_sha256"],
            "srdf_sha256": model_identity["srdf_sha256"],
            "required_pairs": list(interval_pairs),
            "fallback_binary_sha256": binary_sha,
        },
        "interval_phase": {
            "status": interval.get("status"),
            "checked_interval_count": len(checked_indices),
            "certified_interval_count": len(interval.get("certified_interval_indices", [])),
            "unresolved_interval_count": len(interval.get("unresolved_interval_indices", [])),
            "collision_interval_count": interval_collisions,
            "coverage_complete": True,
            "method": interval.get("certificate_type"),
            "runtime_s": d62_runtime,
        },
        "qualified_fallback": {
            "status": fallback_status,
            "certificate_type": fallback.get("certificate_type"),
            "coverage": "all required pairs and all trajectory intervals",
            "unresolved_pair_region_count": fallback_unresolved,
            "collision_pair_region_count": fallback_collisions,
            "minimum_certified_model_clearance_m": fallback.get("minimum_certified_model_clearance_m"),
            "numerical_acceptance_margin_m": fallback.get("compiled_method", {}).get("numerical_acceptance_margin_m"),
            "runtime_s": fallback_runtime,
        },
        "merge_rule": {
            "D62_PASS": "accepted as outward-rounded interval enclosure proof",
            "D62_UNRESOLVED": "retained unless the complete qualified fallback proves every required pair-region clear",
            "collision": "any collision finding blocks PASS",
            "missing_or_partial_coverage": "BLOCKED; merge raises instead of emitting a certificate",
            "unresolved_is_not_pass": True,
        },
        "performance": {
            "d62_reference_before_runtime_s": before_runtime_s,
            "d65_interval_phase_runtime_s": d62_runtime,
            "d65_fallback_runtime_s": fallback_runtime,
            "d65_combined_runtime_s": combined_runtime,
            "speedup_vs_d62_reference": speedup,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--before-runtime-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        interval = json.loads(args.interval.read_text(encoding="utf-8"))
        fallback = json.loads(args.fallback.read_text(encoding="utf-8"))
        result = merge(interval, fallback, args.before_runtime_s)
    except (OSError, json.JSONDecodeError, HybridMergeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_INVALID_EVIDENCE", "error": str(error)}, sort_keys=True))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
