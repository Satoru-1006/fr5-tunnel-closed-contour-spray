#!/usr/bin/env python3
"""Read-only, fail-closed FR5 C5B1 baseline identity and regression gate.

This script does not generate, optimize, retime, or overwrite any trajectory.
It checks exact Git-blob identities of the October 8 C5B0 Hotfix inputs.
Only --run-regression executes the historically reported four-file pytest suite.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


HOTFIX_BASE_COMMIT = "ad842009562351c8b4a587fd9cd42dbfb3d03cb0"
C1_SEMANTIC_SHA256 = "be697bcb976cb69d349ee6c549364ba39041d9f0bda1b57d7f215129f8d5cbb3"
C4_CANONICAL_SHA256 = "f2a73cfe736d8379d9b5149bbd69fc98cfba09004f519c1975e2084e80eb0669"
PINS = {
    "outputs/p2b3_c2_c1_nominal_post_ruckig.csv": "987bc55f8d7c7b7543c64b24673c45cac5b10cf4",
    "outputs/p2b3_c4_authoritative_scene_manifest.json": "0a127c7a2042ac81825fe1a5adbeee99652ca0a1",
    "src/p2b3_c5b0_registration_spec.py": "9346789aa2c4e06c572d86c44a0e9e4d279238c6",
    "tests/test_p2b3_c5b0_registration_spec.py": "6fc475059b7b85162aee623cebdd4ca3d5ce82eb",
    "tests/test_p2b3_c5a_metrics.py": "b34a3005e69d67278ee864755315e7892e3be913",
    "tests/test_p2b3_c3_physical_uncertainty.py": "7f292c46152a0a27164ef33e05a9b76a1b34f962",
    "tests/test_p2b3_c4_scene_contract.py": "f09ee4c5a9aaa371553ba73612b8bf919e2b594c",
    "outputs/p2b3_c5b0_hotfix_result.json": "8f8411297a7dbcb119aec059603b5ae6924ad445",
}
TESTS = [
    "tests/test_p2b3_c5b0_registration_spec.py",
    "tests/test_p2b3_c5a_metrics.py",
    "tests/test_p2b3_c3_physical_uncertainty.py",
    "tests/test_p2b3_c4_scene_contract.py",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git_blob_sha1(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def git_worktree_blob_sha1(path: Path, root: Path) -> str:
    """Read the tracked blob through Git so Windows line-ending filters do not create false drift."""
    completed = subprocess.run(
        ["git", "-C", str(root), "hash-object", "--", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def check(root: Path) -> None:
    for relative, expected in PINS.items():
        path = root / relative
        require(path.is_file(), f"MISSING:{relative}")
        actual = git_worktree_blob_sha1(path, root)
        require(actual == expected, f"FROZEN_BLOB_MISMATCH:{relative}:{actual}")
    print(f"PINNED_FILES_PASS={len(PINS)}")

    nominal = root / "outputs/p2b3_c2_c1_nominal_post_ruckig.csv"
    with nominal.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "C1_MISSING_HEADER")
        require(reader.fieldnames[0] == "t", "C1_INVALID_TIME_FIELD")
        expected_joints = [f"j{i}_q" for i in range(1, 7)]
        require(all(name in reader.fieldnames for name in expected_joints), "C1_MISSING_JOINT")
        rows = list(reader)
    require(len(rows) == 181, f"C1_WAYPOINT_COUNT={len(rows)}")
    previous = float("-inf")
    for row in rows:
        t = float(row["t"])
        require(math.isfinite(t) and t > previous, "C1_TIMESTAMP_NOT_STRICTLY_INCREASING")
        require(all(math.isfinite(float(row[j])) for j in expected_joints), "C1_INVALID_JOINT")
        previous = t
    print("C1_TRAJECTORY_PASS=181_WAYPOINTS_STRICT_TIMESTAMPS")

    scene = json.loads((root / "outputs/p2b3_c4_authoritative_scene_manifest.json").read_text(encoding="utf-8"))
    objects = scene["collision_objects"]["ordered"]
    require(scene.get("project") == "FAIRINO_FR5", "WRONG_ROBOT_PROJECT")
    require(scene.get("frame_id") == "base_link", "C4_FRAME_MISMATCH")
    require(scene["collision_objects"]["count"] == 181 and len(objects) == 181, "C4_OBJECT_COUNT")
    ids = [item["id"] for item in objects]
    require(len(set(ids)) == 181, "C4_DUPLICATE_OBJECT")
    require(all(item["frame_id"] == "base_link" for item in objects), "C4_OBJECT_FRAME_MISMATCH")
    for item in objects:
        for primitive in item["primitives"]:
            dims = primitive["dimensions"]
            require(primitive["type"] == "BOX" and all(math.isfinite(v) and v > 0 for v in dims), "C4_INVALID_BOX")
    require(scene["source_c1"]["nominal_trajectory_sha256"] == C1_SEMANTIC_SHA256, "C1_SEMANTIC_AUTHORITY_MISMATCH")
    require(scene["expected_c1_baseline"]["waypoint_count"] == 181, "C4_C1_COUNT_MISMATCH")
    print("C4_SCENE_PASS=181_UNIQUE_BASE_LINK_BOX_OBJECTS")

    source = (root / "src/p2b3_c5b0_registration_spec.py").read_text(encoding="utf-8")
    assignments = {}
    for stmt in ast.parse(source).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assignments[stmt.targets[0].id] = stmt.value
    def constant(name: str) -> str | None:
        node = assignments.get(name)
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
    require(constant("BASE_LEFT") == "BASE_LEFT", "BASE_LEFT_CONSTANT_CHANGED")
    require(constant("BASE_FRAME") == "base_link", "BASE_FRAME_CHANGED")
    primary = assignments.get("PRIMARY_PARAMETERIZATION")
    require(isinstance(primary, ast.Name) and primary.id == "BASE_LEFT", "PRIMARY_REGISTRATION_CHANGED")
    require("return induced @ entity" in source, "BASE_LEFT_COMPOSITION_CHANGED")
    print("REGISTRATION_PASS=BASE_LEFT_IN_BASE_LINK")

    record = json.loads((root / "outputs/p2b3_c5b0_hotfix_result.json").read_text(encoding="utf-8"))
    require(record["scope"]["robot"] == "FAIRINO_FR5", "HOTFIX_SCOPE_WRONG")
    require(record["scope"]["c1_mutated"] is False and record["scope"]["c4_mutated"] is False, "HOTFIX_MUTATION_RECORDED")
    require(record["regression"]["python_result"] == "80 passed", "HOTFIX_80_TEST_RECEIPT_MISSING")
    require(record["status"]["historical_convergence"] == "INSUFFICIENT_EVIDENCE", "HOTFIX_CONVERGENCE_LIMIT_LOST")
    print("HISTORICAL_TEST_RECEIPT=80_PASSED_ON_2026_10_08")
    print("HISTORICAL_CONVERGENCE=INSUFFICIENT_EVIDENCE")
    print("BASELINE_FREEZE_GATE=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-regression", action="store_true", help="Rerun the Hotfix four-file pytest suite; requires its local dependencies.")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    try:
        check(root)
    except (ValueError, KeyError, OSError, SyntaxError, UnicodeError) as error:
        print(f"BASELINE_FREEZE_GATE=FAIL:{error}", file=sys.stderr)
        return 2
    if not args.run_regression:
        print("CURRENT_SESSION_80_TEST_REPLAY=NOT_RUN")
        return 0
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(completed.stdout, end="")
    if completed.returncode != 0 or not re.search(r"\b80 passed\b", completed.stdout):
        print("CURRENT_SESSION_80_TEST_REPLAY=FAIL_OR_NOT_80", file=sys.stderr)
        return 3
    print("CURRENT_SESSION_80_TEST_REPLAY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
