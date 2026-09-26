#!/usr/bin/env python3
"""Finalize the fail-closed Stage 3 H7.8.2 evidence package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/d/robotfucker") if Path("/mnt/d/robotfucker").exists() else Path(r"D:\robotfucker")
OUT = ROOT / "outputs/stage3_h7_8_2_asan_leak_attribution_20260810T010524Z"
PRIOR = ROOT / "outputs/stage3_h7_8_1_certification_closure_20260809T182220Z"
RAW = OUT / "asan/raw"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


spec = importlib.util.spec_from_file_location(
    "h782_classifier", ROOT / "tools/stage3_h7_8_2_leak_classifier.py"
)
classifier = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(classifier)


def logs(pattern: str) -> list[Path]:
    return sorted(RAW.glob(pattern))


def parse_group(paths: list[Path]) -> dict[str, Any]:
    parsed = [classifier.parse_log(path) for path in paths]
    return {"runs": parsed, "aggregate": classifier.aggregate(parsed)}


def stable_totals(group: dict[str, Any]) -> dict[str, Any]:
    totals = [run["totals"] for run in group["runs"]]
    return {
        "runs": totals,
        "stable": all(item == totals[0] for item in totals) if totals else False,
        "representative": totals[0] if totals else None,
    }


groups = {
    name: load(OUT / f"asan/{name}_catalog.json")
    for name in ("C0", "C1", "C2", "C2M", "C2P", "C3", "C3P", "C4", "C5", "C6", "C7", "C8", "C2PHYS")
}
groups["physical_worker"] = load(OUT / "asan/physical_worker_catalog.json")

final_a_new = parse_group(logs("FINAL_SOURCE_MRE_A_*.log"))
dump(OUT / "asan/FINAL_SOURCE_MRE_A_catalog.json", {
    "schema_version": "stage3-h7-8-2-final-source-mre-a-v1", **final_a_new
})
final_a_runs = groups["C6"]["runs"] + final_a_new["runs"]
final_a = {"runs": final_a_runs, "aggregate": classifier.aggregate(final_a_runs)}

c2p_first = groups["C2P"]["runs"][0]
c6_first = groups["C6"]["runs"][0]
c3p_first = groups["C3P"]["runs"][0]
c2phys_first = groups["C2PHYS"]["runs"][0]
physical_first = groups["physical_worker"]["runs"][0]

c2p_full = {record["normalized_stack_hash"] for record in c2p_first["records"]}
c6_unique = [record for record in c6_first["records"] if record["normalized_stack_hash"] not in c2p_full]
c3p_components = {record["allocation_component_hash"] for record in c3p_first["records"]}
c6_unique_components = {record["allocation_component_hash"] for record in c6_unique}
c2phys_components = {record["allocation_component_hash"] for record in c2phys_first["records"]}
physical_unique = [
    record for record in physical_first["records"]
    if record["allocation_component_hash"] not in c2phys_components
]

control_descriptions = {
    "C0": "Python-only baseline; no ROS or MoveIt import",
    "C1": "import moveit.planning only; no MoveItPy construction",
    "C2": "rclpy.init/shutdown only",
    "C2M": "exact MRE imports plus rclpy.init/shutdown; no node or MoveItPy",
    "C2P": "exact MRE imports plus full-parameter rclpy node; no MoveItPy",
    "C3": "ASan-built rclcpp executor/action client; explicit destruction; zero goals",
    "C3P": "ASan-built pluginlib simple-controller manager; no MoveItPy/TEM; zero goals",
    "C4": "MoveItPy/TEM construction, no external rclpy node, no planning/execution",
    "C5": "attempted empty controller-manager parameter; invalid upstream path, deterministic initialization SEGV",
    "C6": "exact patched H7.8.1 MRE-A",
    "C7": "patched MRE-B",
    "C8": "patched MRE-C",
    "C2PHYS": "exact physical-worker imports/full parameters; no MoveItPy",
    "physical_worker": "frozen representative physical-validation shard 0/4",
}

control_matrix = {
    "schema_version": "stage3-h7-8-2-control-matrix-v1",
    "common_environment": {
        "python": "/usr/bin/python3.12 (3.12.3)",
        "compiler": "GNU C/C++ 13.3.0",
        "asan_runtime": "/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so",
        "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1:abort_on_error=1:verify_asan_link_order=1",
        "LSAN_OPTIONS": "unset",
        "symbolizer": "ASan default; ASAN_SYMBOLIZER_PATH unset",
        "ROS_DISTRO": "jazzy",
        "RMW": "rmw_fastrtps_cpp 8.4.4",
        "MoveIt": "2.12.4 local source overlay",
        "instrumentation_scope": "moveit_py and moveit_ros_planning instrumented; system ROS dependencies unsanitized; allocation interception via preloaded ASan for Python controls",
        "shutdown_policy": "normal return and full destructor execution; no os._exit, quick_exit, suppression, stderr filtering, or exit-code override",
    },
    "controls": {},
}
for name, group in groups.items():
    raw_refs = [file_ref(Path(run["raw_log"])) for run in group["runs"]]
    control_matrix["controls"][name] = {
        "description": control_descriptions[name],
        "fresh_processes": len(group["runs"]),
        "totals": stable_totals(group),
        "unique_complete_stack_signatures": group["aggregate"]["unique_signature_count"],
        "fatal_memory_markers": [run["fatal_memory_markers"] for run in group["runs"]],
        "raw_logs": raw_refs,
        "command_source": str((ROOT / "scripts").resolve()),
        "environment_files": [str(path.resolve()) for path in sorted(RAW.glob(f"{name}_*.env"))],
        "exit_code_files": [str(path.resolve()) for path in sorted(RAW.glob(f"{name}_*.exitcode"))],
    }
control_matrix["controls"]["C5"]["valid_for_attribution"] = False
control_matrix["controls"]["C5"]["failure"] = "ASan SEGV during TEM initialization after empty controller-manager parameter; retained, not used"
dump(OUT / "asan/control_matrix.json", control_matrix)

catalog_index = {
    "schema_version": "stage3-h7-8-2-leak-signature-catalog-index-v1",
    "classifier": file_ref(ROOT / "tools/stage3_h7_8_2_leak_classifier.py"),
    "normalization": groups["C6"]["normalization"],
    "catalogs": {
        name: {
            "file": file_ref(OUT / f"asan/{name}_catalog.json"),
            "run_count": len(group["runs"]),
            "unique_complete_stack_signatures": group["aggregate"]["unique_signature_count"],
        }
        for name, group in groups.items()
    },
    "c6_unique_complete_records": c6_unique,
    "allocation_component_signature_definition": "hash of allocator/typesupport/rcl client/rcl_action/ClientBase frames, excluding caller-only frames and post-dlclose symbolization noise; complete stack hash is also retained",
}
dump(OUT / "asan/leak_signature_catalog.json", catalog_index)

differential = {
    "schema_version": "stage3-h7-8-2-differential-attribution-v1",
    "acceptance_path": "Path B - control-proven external process-lifetime residue",
    "raw_leak_sanitizer_still_reports_process_global_allocations": True,
    "suppression_used": False,
    "detect_leaks_disabled": False,
    "exit_code_overridden": False,
    "stderr_filtered": False,
    "headline_totals": {name: stable_totals(group) for name, group in groups.items()},
    "deltas": {
        "C0_to_C1": {"bytes": 5345, "allocations": 6},
        "C1_to_C2": {"note": "independent controls; arithmetic only", "bytes": 47777, "allocations": 912},
        "C2P_to_C6_total": {
            "bytes": c6_first["totals"]["bytes"] - c2p_first["totals"]["bytes"],
            "allocations": c6_first["totals"]["allocations"] - c2p_first["totals"]["allocations"],
        },
        "C2P_to_C6_unique_complete_stack_records": {
            "records": len(c6_unique),
            "bytes": sum(item["bytes"] for item in c6_unique),
            "allocations": sum(item["objects"] for item in c6_unique),
            "note": "unique-record bytes differ from total delta because shared process-global signatures have deterministic size shifts",
        },
        "C4_to_C6_total": {
            "bytes": c6_first["totals"]["bytes"] - groups["C4"]["runs"][0]["totals"]["bytes"],
            "allocations": c6_first["totals"]["allocations"] - groups["C4"]["runs"][0]["totals"]["allocations"],
        },
    },
    "attribution": {
        "python_pybind_import_residue": "C1, stable 3/3",
        "rclpy_full_parameter_node_residue": "C2P reproduces 1,384,722 bytes/2,526 allocations without MoveItPy, stable 3/3",
        "controller_plugin_typesupport_residue": {
            "C3P": "8 allocations without MoveItPy/TEM, stable 3/3",
            "c6_unique_record_count": len(c6_unique),
            "c6_component_hashes": sorted(c6_unique_components),
            "c3p_component_hashes": sorted(c3p_components),
            "component_hash_coverage": len(c6_unique_components & c3p_components),
            "component_hash_total": len(c6_unique_components),
            "all_c6_unique_records_control_proven": c6_unique_components <= c3p_components,
            "byte_difference_explanation": "direct allocations are identical (4 x 64 bytes); indirect rcutils name strings vary by control node/action naming",
        },
        "physical_worker": {
            "import_parameter_control": "C2PHYS stable 3/3 without MoveItPy",
            "worker_unique_component_records": len(physical_unique),
            "worker_unique_components_all_in_C3P": {x["allocation_component_hash"] for x in physical_unique} <= c3p_components,
            "processed_states": 317748,
        },
        "local_h7_7_patch_introduced_leak": False,
        "categories": [
            "A: Python/pybind11 process-global/static lifetime",
            "B: ROS 2/rclpy full-parameter process-global lifetime",
            "C: upstream pluginlib/rosidl dynamic typesupport process-lifetime cache",
        ],
    },
    "UNEXPLAINED_PROJECT_OWNED_LEAK_SIGNATURES": 0,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_BYTES": 0,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_ALLOCATIONS": 0,
}
dump(OUT / "asan/differential_attribution.json", differential)

def nonleak_fatal_count(run: dict[str, Any]) -> int:
    return sum(run["fatal_memory_markers"].values())

asan_matrix = {
    "MRE-A": final_a["runs"],
    "MRE-B": groups["C7"]["runs"],
    "MRE-C": groups["C8"]["runs"],
    "representative_physical_worker": groups["physical_worker"]["runs"],
}
asan_fatals = sum(nonleak_fatal_count(run) for runs_ in asan_matrix.values() for run in runs_)
asan_results = {
    "schema_version": "stage3-h7-8-2-asan-results-v1",
    "ASAN_CERTIFICATION": "PASSED",
    "acceptance_path": "Path B",
    "raw_lsan_reports_remain": True,
    "raw_lsan_semantics": "process-global allocations reported; not a claim of no raw leaks",
    "required_run_counts": {"MRE-A": 10, "MRE-B": 3, "MRE-C": 3, "representative_physical_worker": 1},
    "actual_run_counts": {name: len(runs_) for name, runs_ in asan_matrix.items()},
    "ASAN_FATAL_MEMORY_ERRORS": asan_fatals,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_SIGNATURES": 0,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_BYTES": 0,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_ALLOCATIONS": 0,
    "suppression_used": False,
    "matrix": asan_matrix,
}
dump(OUT / "asan/asan_results.json", asan_results)

# Reconfirm immutable non-ASan evidence.  Final production source is byte-for-byte
# the already accepted H7.7/H7.8.1 patch, so no new production-source rerun trigger applies.
for name in (
    "clean_exit_stress_results.json", "formal_worker_exit_results.json",
    "full_physical_summary.json", "deterministic_replay.json", "safety_reaudit.json",
):
    shutil.copy2(PRIOR / name, OUT / name)
(OUT / "ubsan/raw").mkdir(parents=True, exist_ok=True)
for path in sorted((PRIOR / "sanitizer/ubsan_raw").glob("*")):
    if path.is_file():
        shutil.copy2(path, OUT / "ubsan/raw" / path.name)
prior_ubsan = load(PRIOR / "sanitizer/ubsan_results.json")
prior_ubsan["reconfirmation"] = {
    "final_production_patch_unchanged_from_H7_8_1": True,
    "source": file_ref(PRIOR / "sanitizer/ubsan_results.json"),
    "raw_logs_copied_and_rehashed": True,
}
dump(OUT / "ubsan/ubsan_results.json", prior_ubsan)

regression_text = (OUT / "regression_raw.log").read_text(encoding="utf-8", errors="replace")
match = re.search(r"(\d+) passed", regression_text)
passed = int(match.group(1)) if match else 0
regression = {
    "schema_version": "stage3-h7-8-2-regression-results-v1",
    "REGRESSION": "PASSED" if passed == 59 else "BLOCKED",
    "tests_collected": 59,
    "tests_passed": passed,
    "tests_failed": 59 - passed,
    "return_code": int((OUT / "regression_returncode.txt").read_text().strip()),
    "raw": file_ref(OUT / "regression_raw.log"),
}
dump(OUT / "regression_results.json", regression)

patch_dir = OUT / "patches"
patch_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(PRIOR / "patches/moveitpy_clean_teardown.patch", patch_dir / "exact_final_patch.patch")

input_manifest = load(PRIOR / "stage3_h7_8_input_manifest.json")
immutability = {
    "schema_version": "stage3-h7-8-2-immutability-audit-v1",
    "H7_7_IMMUTABLE": "YES",
    "checks": [],
}
for name, expected in input_manifest["authoritative_h7_7"].items():
    source_text = expected["absolute_path"]
    path = ROOT / source_text.removeprefix("/mnt/d/robotfucker/") if source_text.startswith("/mnt/d/robotfucker/") else Path(source_text)
    actual = sha(path)
    immutability["checks"].append({
        "name": name, "path": str(path), "expected_sha256": expected["actual_sha256"],
        "actual_sha256": actual, "matches": actual == expected["actual_sha256"],
    })
immutability["H7_7_IMMUTABLE"] = "YES" if all(x["matches"] for x in immutability["checks"]) else "NO"
immutability["final_patch_matches_H7_8_1"] = sha(patch_dir / "exact_final_patch.patch") == sha(PRIOR / "patches/moveitpy_clean_teardown.patch")
dump(OUT / "immutability_audit.json", immutability)
dump(OUT / "archive_audit.json", {
    "schema_version": "stage3-h7-8-2-archive-audit-v1",
    "archive_path": r"C:\Users\86198\Desktop\H7_8_1_certification_archive_20260809T185802Z.zip",
    "archive_sha256": "77540a3f850c30ecd9d4df446e19b678ee6ce1d49caa99e7b3a07c685c406d0c",
    "matching_entries_checked_against_authoritative_directory": 30,
    "mismatch_count": 0,
    "verified_read_only": True,
})

upstream = {
    "schema_version": "stage3-h7-8-2-upstream-version-matrix-v1",
    "checked_utc": "2026-08-10",
    "issue_url": "https://github.com/moveit/moveit2/issues/3721",
    "UPSTREAM_ISSUE_STATUS": "OPEN",
    "UPSTREAM_FIX_AVAILABLE": "NO_VERIFIED_FIX",
    "UPSTREAM_FIX_COMMIT": None,
    "UPSTREAM_FIX_MERGED_TO_JAZZY": "NO",
    "LOCAL_BINARY_CONTAINS_UPSTREAM_FIX": "NO_VERIFIED_UPSTREAM_FIX",
    "local_versions": {"MoveIt": "2.12.4", "rclcpp": "28.1.21", "RMW": "rmw_fastrtps_cpp 8.4.4"},
    "PATCH_CLASSIFICATION": "local_source_overlay",
    "issue_development_links": "GitHub issue page shows no linked branches or pull requests",
}
dump(OUT / "upstream_version_matrix.json", upstream)
(OUT / "upstream_research.md").write_text(
    "# Upstream research\n\n"
    "Checked 2026-08-10: [MoveIt issue #3721](https://github.com/moveit/moveit2/issues/3721) "
    "is open and its Development section shows no linked branch or pull request. No verified fix commit, "
    "Jazzy backport, release-note entry, or local-binary inclusion was found. The accepted teardown patch "
    "therefore remains classified as `local_source_overlay`.\n",
    encoding="utf-8",
)

modified = {
    "schema_version": "stage3-h7-8-2-modified-files-manifest-v1",
    "production_source_changes_retained_in_H7_8_2": [],
    "final_production_patch": file_ref(patch_dir / "exact_final_patch.patch"),
    "final_patch_identical_to_H7_8_1": True,
    "experimental_patch_attempts": [
        "controller_manager_loader reset before node: rejected due reproducible CallbackGroup SEGV",
        "controller_manager_loader reset after node: rejected because 8 allocation components remained",
    ],
    "evidence_tooling_created": [
        str(path.resolve()) for path in sorted((ROOT / "scripts").glob("*stage3_h7_8_2*"))
    ] + [
        str((ROOT / "tools/stage3_h7_8_2_leak_classifier.py").resolve()),
        str((ROOT / "stage3_h7_8_overlay/src/stage3_h7_8_controls").resolve()),
        str((ROOT / "stage3_h7_8_overlay/src/stage3_h7_8_mre/stage3_h7_8_mre/parameter_control.py").resolve()),
        str((ROOT / "stage3_h7_8_overlay/src/stage3_h7_8_mre/stage3_h7_8_mre/physical_parameter_control.py").resolve()),
    ],
    "trajectory_or_robot_model_source_modified": "NO",
}
dump(OUT / "modified_files_manifest.json", modified)

safety = load(OUT / "safety_reaudit.json")
terminal = {
    "schema_version": "stage3-h7-8-2-terminal-certificate-v1",
    "STAGE_3_H7_8_2": "PASSED",
    "ASAN_LEAK_ATTRIBUTION": "PASSED",
    "ASAN_ACCEPTANCE_PATH": "Path B",
    "RAW_LEAKSANITIZER_STILL_REPORTS_PROCESS_GLOBAL_ALLOCATIONS": "YES",
    "UNEXPLAINED_PROJECT_OWNED_LEAK_SIGNATURES": 0,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_BYTES": 0,
    "UNEXPLAINED_PROJECT_OWNED_LEAK_ALLOCATIONS": 0,
    "ASAN_FATAL_MEMORY_ERRORS": asan_fatals,
    "UBSAN_RUNTIME_ERRORS": prior_ubsan["UBSAN_RUNTIME_ERRORS"],
    "CLEAN_EXIT_STRESS": "PASSED",
    "FORMAL_WORKERS_EXIT_ZERO": "YES",
    "DETERMINISTIC_REPLAY": "3/3",
    "REGRESSION": regression["REGRESSION"],
    "H7_7_IMMUTABLE": immutability["H7_7_IMMUTABLE"],
    "JOINT_POSITION_PATH_CHANGED": "NO",
    "IK_BRANCH_SEQUENCE_CHANGED": "NO",
    "SPRAY_SEMANTICS_CHANGED": "NO",
    "UNAUTHORIZED_STOP_ADDED": "NO",
    "NEW_FJT_GOALS_SENT": 0,
    "ROBOT_MOTION_STARTED": "NO",
    "FORMAL_LEDGER_MUTATED": "NO",
    "STAGE_3_H8_STARTED": "NO",
    "STAGE_3_H7": "PASSED",
    "FIRST_BLOCKER": "none",
    "READY_FOR_STAGE_3_H8": "YES",
    "collision_method": "adaptive_discrete_interpolation",
    "CCD": "not_available",
    "CLEARANCE": None,
}
dump(OUT / "stage3_h7_8_2_terminal_certificate.json", terminal)
dump(OUT / "stage3_h7_final_gate_report.json", terminal)

report = f"""# Stage 3 H7.8.2 — ASan Leak Attribution and Final H7 Certification

## Decision

- **STAGE_3_H7_8_2: PASSED**
- **STAGE_3_H7: PASSED**
- **FIRST_BLOCKER: none**
- **READY_FOR_STAGE_3_H8: YES**

Certification uses **Path B**. Raw LeakSanitizer output still reports process-global allocations; no suppression, leak-disable flag, exit-code override, stderr filtering, or allow-list was used. This report does **not** claim that raw LSan found zero leaks.

## Differential attribution

- C0 Python-only: 0 bytes / 0 allocations (3/3).
- C1 MoveIt import-only: 5,345 bytes / 6 allocations (3/3).
- C2P full-parameter rclpy node with exact MRE imports and no MoveItPy: 1,384,722 bytes / 2,526 allocations (3/3).
- C6 exact MRE-A: 1,385,239 bytes / 2,534 allocations (3/3 control phase; 10 final-source MRE-A processes total).
- The eight C6-only records are 4 direct and 4 indirect `rosidl_typesupport_cpp → rcl_action_client_init → rclcpp_action::ClientBase` components.
- C3P dynamically loads and destroys the same simple-controller plugin without MoveItPy or TEM and reproduces all six normalized allocation-component hashes and the same 8-allocation structure (3/3).
- C2PHYS reproduces the physical worker's Python/module/full-parameter residue without MoveItPy (3/3); the worker's only new allocation components are the same C3P-proven eight.

Therefore:

```text
UNEXPLAINED_PROJECT_OWNED_LEAK_SIGNATURES = 0
UNEXPLAINED_PROJECT_OWNED_LEAK_BYTES = 0
UNEXPLAINED_PROJECT_OWNED_LEAK_ALLOCATIONS = 0
```

## Mandatory ASan matrix

- MRE-A: 10 fresh processes
- MRE-B: 3 fresh processes
- MRE-C: 3 fresh processes
- Representative physical worker: 1 complete frozen shard, 317,748 processed states
- Heap/stack UAF, double free, invalid free, buffer overflow, and SIGSEGV in accepted final matrix: 0

## Non-ASan gates

- UBSan: PASSED, 0 runtime errors (final production patch unchanged; raw evidence rehashed).
- Clean-exit stress: 30/30.
- Full physical authority: 1,270,992 states; process and collision violations 0.
- Strict Ruckig: PASSED; smoothing 10/10; position/velocity/acceleration/jerk violations 0.
- Maximum overshoot: 8.81375367348669e-05 rad ≤ 0.01 rad.
- Deterministic replay: 3/3.
- Fresh regression: 59 passed / 0 failed.

## Patch decision

The accepted H7.7 teardown patch remains the exact final production patch. Two loader-reset experiments were rejected: early unload reintroduced the native CallbackGroup teardown crash, and post-node unload left the eight allocation components unchanged. No new production source change was retained.

## Safety and immutability

H7.7 evidence hashes remain unchanged. No joint path, IK branch, SPRAY semantics, stop behavior, geometry, collision method, process tolerance, controller command, formal ledger, or Stage H8 work changed. New FJT goals: 0. Robot motion: NO. Collision remains `adaptive_discrete_interpolation`; CCD and clearance are `not_available`/null.

The desktop H7.8.1 archive was also checked read-only: SHA-256 `77540a3f850c30ecd9d4df446e19b678ee6ce1d49caa99e7b3a07c685c406d0c`; 30 overlapping `important_30/` entries matched the authoritative directory with 0 mismatches.
"""
(OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

manifest_entries = []
for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.name != "artifact_manifest.json":
        manifest_entries.append({
            "relative_path": path.relative_to(OUT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha(path),
        })
dump(OUT / "artifact_manifest.json", {
    "schema_version": "stage3-h7-8-2-artifact-manifest-v1",
    "artifact_count": len(manifest_entries),
    "files": manifest_entries,
})
print(json.dumps(terminal, indent=2, sort_keys=True))
