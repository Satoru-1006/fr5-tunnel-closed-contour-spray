"""D65 fail-closed hybrid articulated self-collision certificate driver.

The Python gate validates the full 6D CSV and binds it to the native Ruckig
``Profile.j`` evidence.  A separately built C++/MoveIt2/FCL backend then proves
pairwise separation using exact FK distance witnesses and conservative
jerk-bounded link-motion radii.  The result is model-based software evidence;
it is neither physical clearance nor an exact external nonlinear CCD query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.d61_fk_interval_certificate import (
        DEFAULT_SRDF,
        DEFAULT_URDF,
        load_model,
        read_trajectory,
        validate_trajectory_contract,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.d61_fk_interval_certificate import (
        DEFAULT_SRDF,
        DEFAULT_URDF,
        load_model,
        read_trajectory,
        validate_trajectory_contract,
    )


ROOT = Path(__file__).resolve().parents[1]
CERTIFIER = ROOT / "tmp" / "d65_install" / "stage4f_articulated_certificate" / "lib" / "stage4f_articulated_certificate" / "stage4f_articulated_certificate"
MODEL_IDENTITY = {
    "urdf_sha256": "f9c53b9da6bca9387d635da7caee4927dab12e8f15e6e76eabe0faa1794620b4",
    "srdf_sha256": "0d6ca01d4667cb8f4772a8a26d5e83e1098c604d1cac0ef3739cb3d495c3e182",
}
# D59 is the frozen source of the two D65 finalist trajectories.  A known
# case id is accepted only with these exact bytes; same-length or re-labelled
# CSVs cannot be paired with the Profile.j summaries by accident.
AUTHORIZED_BINDINGS = {
    "adversarial_0100": {
        "trajectory_sha256": "13f8075296a0defa12f17348cd6bafb913a8c10d22012a651d5614082df0a264",
        "profile_j_sha256": "d13598530d713427971f0a93926514501fdbeb3bc7277ace64c8a5767f5ea51e",
        "state_count": 7005,
        "interval_count": 7004,
    },
    "adversarial_0101": {
        "trajectory_sha256": "dda61e3be2328536f0c688333f3b24861c56dbdb347e807d881ca7d71f617416",
        "profile_j_sha256": "1ebeb2261585340fa5495f18b45d9c0bf44e5434d9862380473d7492bb3a48a1",
        "state_count": 7030,
        "interval_count": 7029,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_authorized_binding(trajectory_id: str, trajectory_sha256: str, profile_sha256: str, state_count: int) -> dict[str, Any]:
    binding = AUTHORIZED_BINDINGS.get(trajectory_id)
    errors: list[dict[str, Any]] = []
    if binding is None:
        errors.append({"code": "UNKNOWN_TRAJECTORY_BINDING", "trajectory_id": trajectory_id})
    else:
        for field, actual in (("trajectory_sha256", trajectory_sha256), ("profile_j_sha256", profile_sha256)):
            if actual != binding[field]:
                errors.append({"code": "AUTHORIZED_IDENTITY_MISMATCH", "field": field, "expected": binding[field], "actual": actual})
        if state_count != binding["state_count"]:
            errors.append({"code": "AUTHORIZED_STATE_COUNT_MISMATCH", "expected": binding["state_count"], "actual": state_count})
    return {"status": "VALID" if not errors else "BLOCKED_INVALID_INPUT", "errors": errors, "trajectory_id": trajectory_id}


def _wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 3 and value[1:3] == ":/":
        return f"/mnt/{value[0].lower()}/{value[3:]}"
    return value


def validate_profile_j(
    profile: Mapping[str, Any],
    *,
    state_count: int,
    jerk_bound: float,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if isinstance(state_count, bool) or not isinstance(state_count, int) or state_count < 2:
        errors.append({"code": "INVALID_STATE_COUNT", "message": "Profile.j state_count must be an integer >= 2", "state_count": state_count})
        interval_count = None
    else:
        interval_count = state_count - 1
    try:
        bound_value = float(jerk_bound)
    except (TypeError, ValueError, OverflowError):
        bound_value = None
    if bound_value is None or not math.isfinite(bound_value) or bound_value < 0.0:
        errors.append({"code": "INVALID_JERK_BOUND", "message": "jerk bound must be finite and nonnegative", "jerk_bound": jerk_bound})
    if not isinstance(profile, Mapping):
        errors.append({"code": "INVALID_PROFILE_SCHEMA", "message": "Profile.j evidence must be a JSON object"})
        return {
            "status": "BLOCKED_INVALID_INPUT",
            "errors": errors,
            "state_count": state_count if isinstance(state_count, int) and not isinstance(state_count, bool) else None,
            "interval_count": interval_count,
            "jerk_bound_rad_s3": bound_value,
        }

    exact_counts = {
        "input_state_count": state_count,
        "segments": interval_count,
        "strict_current_target_valid": interval_count,
        "successful_native_profiles": interval_count,
        "error_invalid_input_count": 0,
        "other_error_count": 0,
        "overshoot_segment_count": 0,
        "extended_segment_count": 0,
    }
    for field, expected in exact_counts.items():
        actual = profile.get(field)
        if isinstance(expected, int) and (isinstance(actual, bool) or not isinstance(actual, int)):
            mismatch = True
        else:
            mismatch = actual != expected
        if mismatch:
            errors.append({"code": "PROFILE_COUNT_MISMATCH", "field": field, "expected": expected, "actual": actual})
    if profile.get("passed") is not True or profile.get("jerk_truth") != "NATIVE_RUCKIG_PROFILE_J":
        errors.append({"code": "PROFILE_NOT_AUTHORITATIVE", "message": "Profile.j evidence must be a passing native Ruckig oracle"})
    maximum = profile.get("max_abs_analytic_jerk_rad_s3")
    maximum_valid = not isinstance(maximum, bool) and isinstance(maximum, (int, float))
    maximum_value = float(maximum) if maximum_valid else math.nan
    maximum_record = maximum_value if math.isfinite(maximum_value) else None
    if not maximum_valid or not math.isfinite(maximum_value) or maximum_value < 0.0 or bound_value is None or maximum_value > bound_value:
        errors.append({"code": "PROFILE_JERK_BOUND_VIOLATION", "maximum": maximum_record, "jerk_bound": bound_value})
    limits = profile.get("limits", {}).get("max_jerk_rad_s3") if isinstance(profile.get("limits"), Mapping) else None
    limits_valid = isinstance(limits, list) and len(limits) == 6
    if limits_valid:
        for value in limits:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                limits_valid = False
                break
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0 or bound_value is None or numeric > bound_value:
                limits_valid = False
                break
    if not limits_valid:
        safe_limits = None
        if isinstance(limits, list):
            safe_limits = []
            for value in limits:
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError):
                    safe_limits.append(None)
                else:
                    safe_limits.append(numeric if math.isfinite(numeric) else None)
        errors.append({"code": "PROFILE_LIMIT_CONTRACT_INVALID", "max_jerk_rad_s3": safe_limits, "jerk_bound": bound_value})
    return {
        "status": "VALID" if not errors else "BLOCKED_INVALID_INPUT",
        "errors": errors,
        "state_count": state_count if isinstance(state_count, int) and not isinstance(state_count, bool) else None,
        "interval_count": interval_count,
        "jerk_bound_rad_s3": bound_value,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _blocked_input_result(
    output_dir: Path,
    *,
    error: str,
    code: str,
    input_evidence: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one machine-readable invalid-input result at the orchestration boundary."""

    result: dict[str, Any] = {
        "schema_version": "d65-validated-fk-lipschitz-fallback-v1",
        "status": "BLOCKED_INVALID_INPUT",
        "verification_result": "BLOCKED_INVALID_INPUT",
        "error": str(error),
        "error_code": str(code),
        "input_evidence": dict(input_evidence or {"status": "BLOCKED_INVALID_INPUT"}),
        "collision_interval_count": None,
        "unresolved_interval_count": None,
    }
    if extra:
        result.update(dict(extra))
    try:
        _write_json(output_dir / "D65_VALIDATED_FK_LIPSCHITZ_CERTIFICATE.json", result)
    except (OSError, TypeError, ValueError):
        # The caller still receives the structured record even when the
        # requested output directory itself is not writable.
        pass
    return result


def run(
    trajectory_path: Path,
    profile_path: Path,
    output_dir: Path,
    trajectory_id: str,
    family: str,
    *,
    jerk_bound: float = 8.0,
    initial_stride: int = 1,
    max_depth: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    trajectory_path = trajectory_path.resolve()
    profile_path = profile_path.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        # Resolve is non-mutating, but failure to create the output location
        # still has to be represented as a fail-closed result when possible.
        return _blocked_input_result(output_dir, error=str(error), code="OUTPUT_DIRECTORY_ERROR")
    try:
        bound_value = float(jerk_bound)
    except (TypeError, ValueError, OverflowError) as error:
        return _blocked_input_result(output_dir, error=str(error), code="INVALID_JERK_BOUND")
    if not math.isfinite(bound_value) or bound_value < 0.0:
        return _blocked_input_result(output_dir, error="jerk bound must be finite and nonnegative", code="INVALID_JERK_BOUND")
    if isinstance(initial_stride, bool) or not isinstance(initial_stride, int) or initial_stride < 1:
        return _blocked_input_result(output_dir, error="initial_stride must be a positive integer", code="INVALID_INITIAL_STRIDE")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        return _blocked_input_result(output_dir, error="max_depth must be a nonnegative integer", code="INVALID_MAX_DEPTH")
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        return _blocked_input_result(output_dir, error="trajectory_id must be a non-empty string", code="INVALID_TRAJECTORY_ID")
    if not isinstance(family, str) or not family.strip():
        return _blocked_input_result(output_dir, error="family must be a non-empty string", code="INVALID_FAMILY")

    try:
        model = load_model(DEFAULT_URDF)
        joint_names = [joint.name for joint in model.joints if joint.joint_type in {"revolute", "continuous"}]
        trajectory = read_trajectory(trajectory_path, joint_names, require_six_dof=True)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_gate = validate_profile_j(profile, state_count=len(trajectory.times), jerk_bound=bound_value)
        # Bind velocity/acceleration limits from the native Profile.j oracle
        # to the URDF position/velocity limits before accepting the CSV.  The
        # merge remains fail-closed if a later artifact omits this gate.
        joint_limits = {name: dict(model.joint_limits.get(name, {})) for name in joint_names}
        profile_limits = profile.get("limits", {}) if isinstance(profile, Mapping) else {}
        if isinstance(profile_limits, Mapping):
            for field, key in (("max_velocity_rad_s", "max_velocity"), ("max_acceleration_rad_s2", "max_acceleration")):
                values = profile_limits.get(field)
                if isinstance(values, list) and len(values) == len(joint_names):
                    for index, name in enumerate(joint_names):
                        joint_limits[name][key] = values[index]
        trajectory_gate = validate_trajectory_contract(
            trajectory,
            bound_value,
            require_six_dof=True,
            joint_limits=joint_limits,
        )
        trajectory_sha256 = _sha256(trajectory_path)
        profile_sha256 = _sha256(profile_path)
    except (OSError, ET.ParseError, TypeError, ValueError, KeyError) as error:
        return _blocked_input_result(output_dir, error=str(error), code=getattr(error, "code", "INPUT_OR_FILE_ERROR"))
    binding_gate = validate_authorized_binding(trajectory_id, trajectory_sha256, profile_sha256, len(trajectory.times))
    input_status = "VALID" if trajectory_gate["status"] == profile_gate["status"] == binding_gate["status"] == "VALID" else "BLOCKED_INVALID_INPUT"
    input_evidence = {
        "status": input_status,
        "family": family,
        "trajectory_validation": trajectory_gate,
        "profile_j_validation": profile_gate,
        "authorized_binding": binding_gate,
        "model_identity": MODEL_IDENTITY,
        "trajectory": {"path": str(trajectory_path), "size_bytes": trajectory_path.stat().st_size, "sha256": trajectory_sha256},
        "profile_j": {"path": str(profile_path), "size_bytes": profile_path.stat().st_size, "sha256": profile_sha256},
    }
    if input_status != "VALID":
        return _blocked_input_result(
            output_dir,
            error="trajectory/Profile.j/authorized binding precondition failed",
            code="INVALID_INPUT_CONTRACT",
            input_evidence=input_evidence,
        )
    if not CERTIFIER.is_file():
        return _blocked_input_result(
            output_dir,
            error=f"D65 compiled certifier is unavailable: {CERTIFIER}",
            code="BACKEND_UNAVAILABLE",
            input_evidence=input_evidence,
        )

    run_nonce = f"run_{time.time_ns()}"
    backend_dir = output_dir / run_nonce
    backend_dir.mkdir(parents=True, exist_ok=False)
    manifest = backend_dir / "certificate_manifest.csv"
    manifest.write_text(
        "case_id,trajectory_csv,family\n" + f"{trajectory_id},{_wsl_path(trajectory_path)},{family}\n",
        encoding="utf-8",
        newline="\n",
    )
    command_args = [
        _wsl_path(CERTIFIER),
        "--manifest", _wsl_path(manifest),
        "--urdf", _wsl_path(DEFAULT_URDF),
        "--srdf", _wsl_path(DEFAULT_SRDF),
        "--output", _wsl_path(backend_dir),
        "--initial-stride", str(initial_stride),
        "--max-depth", str(max_depth),
        "--jerk-bound", str(bound_value),
        "--threshold", "0.0",
    ]
    shell = "; ".join([
        "set -e",
        "source /opt/ros/jazzy/setup.bash",
        "source /mnt/d/robotfucker/install/setup.bash",
        "source /mnt/d/robotfucker/tmp/d65_install/setup.bash",
        "export LD_LIBRARY_PATH=/mnt/d/robotfucker/tmp/d65_install/stage4f_articulated_certificate/lib:/mnt/d/robotfucker/tmp/d41_install/lib:/mnt/d/robotfucker/install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}",
        "exec " + " ".join(shlex.quote(item) for item in command_args),
    ])
    backend_started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=6 * 60 * 60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return _blocked_input_result(
            output_dir,
            error=str(error),
            code="BACKEND_EXECUTION_ERROR",
            input_evidence=input_evidence,
            extra={"backend": {"status": "UNAVAILABLE", "runtime_s": time.perf_counter() - backend_started}},
        )
    backend_runtime = time.perf_counter() - backend_started
    backend_log = bytes(completed.stdout or b"").decode("utf-8", errors="replace")
    try:
        (output_dir / "compiled_backend.log").write_text(backend_log, encoding="utf-8")
    except OSError:
        # The backend result remains authoritative only if its JSON is
        # available; failure to preserve the diagnostic log must not turn a
        # failed invocation into a pass.
        pass
    backend_path = backend_dir / "D54_CONTINUOUS_SELF_COLLISION_CERTIFICATION.json"
    if not backend_path.is_file():
        return _blocked_input_result(
            output_dir,
            error=f"compiled backend produced no certificate (exit {completed.returncode})",
            code="BACKEND_CERTIFICATE_MISSING",
            input_evidence=input_evidence,
            extra={"backend": {"exit_code": completed.returncode, "runtime_s": backend_runtime, "run_nonce": run_nonce}},
        )
    try:
        backend = json.loads(backend_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return _blocked_input_result(
            output_dir,
            error=str(error),
            code="BACKEND_CERTIFICATE_INVALID_JSON",
            input_evidence=input_evidence,
            extra={"backend": {"exit_code": completed.returncode, "runtime_s": backend_runtime, "run_nonce": run_nonce}},
        )
    if not isinstance(backend, Mapping):
        return _blocked_input_result(
            output_dir,
            error="compiled backend certificate must be a JSON object",
            code="BACKEND_CERTIFICATE_INVALID_SCHEMA",
            input_evidence=input_evidence,
            extra={"backend": {"exit_code": completed.returncode, "runtime_s": backend_runtime, "run_nonce": run_nonce}},
        )
    required_backend_fields = ("schema_version", "status", "required_pair_count", "required_pair_coverage_complete", "all_required_pairs", "collision_region_count", "unresolved_region_count")
    missing_backend = [field for field in required_backend_fields if field not in backend]
    backend_status = backend.get("status")
    backend_pairs = backend.get("all_required_pairs")
    backend_pair_count = backend.get("required_pair_count")
    backend_collisions = backend.get("collision_region_count")
    backend_unresolved = backend.get("unresolved_region_count")
    backend_clearance = backend.get("minimum_certified_clearance_m")
    backend_clearance_valid = backend_clearance is None
    if backend_clearance is not None and not isinstance(backend_clearance, bool) and isinstance(backend_clearance, (int, float)):
        backend_clearance_valid = math.isfinite(float(backend_clearance)) and float(backend_clearance) >= 0.0
    backend_contract_valid = bool(
        not missing_backend
        and backend.get("schema_version") == "d54-fk-aware-articulated-certificate-v1"
        and backend.get("required_pair_coverage_complete") is True
        and isinstance(backend_pairs, list)
        and backend_pairs
        and all(isinstance(pair, str) and "|" in pair for pair in backend_pairs)
        and len(backend_pairs) == len(set(backend_pairs))
        and isinstance(backend_pair_count, int)
        and not isinstance(backend_pair_count, bool)
        and backend_pair_count > 0
        and backend_pair_count == len(backend_pairs)
        and isinstance(backend_collisions, int)
        and not isinstance(backend_collisions, bool)
        and backend_collisions >= 0
        and isinstance(backend_unresolved, int)
        and not isinstance(backend_unresolved, bool)
        and backend_unresolved >= 0
        and backend_clearance_valid
        and (backend_status != "PASS" or (backend_collisions == 0 and backend_unresolved == 0))
    )
    if not backend_contract_valid or completed.returncode != 0 or backend_status not in {"PASS", "UNRESOLVED", "COLLISION_FOUND"}:
        status = "BLOCKED_INVALID_INPUT"
    else:
        status = str(backend_status)
    result = {
        "schema_version": "d65-validated-fk-lipschitz-fallback-v1",
        "status": status,
        "verification_result": status,
        "trajectory_id": trajectory_id,
        "verification_domain": "model_based_articulated_continuous_self_collision",
        "input_evidence": input_evidence,
        "collision_interval_count": int(backend["collision_region_count"]) if not missing_backend else None,
        "unresolved_interval_count": int(backend["unresolved_region_count"]) if not missing_backend else None,
        "interval_count": len(trajectory.times) - 1,
        "required_pair_count": backend.get("required_pair_count"),
        "required_pairs": backend.get("all_required_pairs"),
        "minimum_certified_model_clearance_m": float(backend_clearance) if backend_clearance_valid and backend_clearance is not None else None,
        "hardware_or_physical_clearance": None,
        "certificate_type": "QUALIFIED_FK_LIPSCHITZ",
        "compiled_method": {
            "method": "MoveIt2 exact FK state witnesses plus FCL signed distance and configuration-independent jerk-bounded link-motion Lipschitz radius",
            "coverage": "all required pairs and all trajectory intervals",
            "numerical_acceptance_margin_m": 1.0e-9,
            "exact_external_nonlinear_ccd_backend": "NOT_CLAIMED",
            "unresolved_is_not_pass": True,
        },
        "backend": {
            "exit_code": completed.returncode,
            "runtime_s": backend_runtime,
            "certificate": str(backend_path),
            "certificate_sha256": _sha256(backend_path),
            "manifest_sha256": _sha256(manifest),
            "raw_status": backend_status,
            "missing_fields": missing_backend,
            "run_nonce": run_nonce,
            "schema_version": backend.get("schema_version"),
            "binary": {"path": str(CERTIFIER), "size_bytes": CERTIFIER.stat().st_size, "sha256": _sha256(CERTIFIER)},
        },
        "execution": {"total_runtime_s": time.perf_counter() - started},
    }
    _write_json(output_dir / "D65_VALIDATED_FK_LIPSCHITZ_CERTIFICATE.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--profile-j", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--family", default="ADVERSARIAL")
    parser.add_argument("--jerk-bound", type=float, default=8.0)
    parser.add_argument("--initial-stride", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=12)
    args = parser.parse_args()
    try:
        result = run(args.trajectory, args.profile_j, args.output, args.trajectory_id, args.family, jerk_bound=args.jerk_bound, initial_stride=args.initial_stride, max_depth=args.max_depth)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError, ET.ParseError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "BLOCKED_INVALID_INPUT", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": result["status"], "output": str((args.output / "D65_VALIDATED_FK_LIPSCHITZ_CERTIFICATE.json").resolve())}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
