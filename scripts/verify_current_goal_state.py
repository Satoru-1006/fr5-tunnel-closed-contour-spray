from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str], allow_failure: bool = False) -> dict[str, Any]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(result.returncode)
    return {
        "command": " ".join(args),
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-20:],
        "stderr_tail": result.stderr.splitlines()[-20:],
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_summary(commands: list[dict[str, Any]]) -> dict[str, Any]:
    readiness = load_json(ROOT / "outputs/production_readiness_check.json")
    goal_audit = load_json(ROOT / "outputs/goal_resolution_audit.json")
    handoff = load_json(ROOT / "outputs/validation_handoff_manifest.json")
    commit_plan = load_json(ROOT / "outputs/commit_readiness_plan.json")
    pytest_cmd = next((cmd for cmd in commands if "pytest" in cmd["command"]), {})
    return {
        "schema_version": 1,
        "purpose": "Local verification summary for the active 150 mm placeholder TCP collision goal.",
        "overall_goal_status": goal_audit.get("overall_goal_status", "missing"),
        "production_readiness_status": readiness.get("overall_status", "missing"),
        "strict_handoff_status": handoff.get("overall_status", "missing"),
        "strict_audit_status": handoff.get("strict_audit_status", "missing"),
        "pytest_status": "pass" if pytest_cmd.get("returncode") == 0 else "fail",
        "handoff_unclassified_count": len(handoff.get("changed_files", {}).get("unclassified", [])),
        "commit_plan_unclassified_count": commit_plan.get("unclassified", {}).get("file_count", "missing"),
        "blocking_gates": [gate.get("gate", "") for gate in readiness.get("blocking_gates", [])],
        "missing_evidence": readiness.get("missing_evidence", []),
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh and verify the current active-goal evidence chain.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/current_goal_state_verification.json")
    parser.add_argument(
        "--require-production-ready",
        action="store_true",
        help="Return non-zero unless production readiness is pass and the active goal is complete.",
    )
    args = parser.parse_args()

    commands = [
        run_command([sys.executable, "tools/build_validation_action_plan.py"]),
        run_command([sys.executable, "tools/check_production_readiness.py"], allow_failure=True),
        run_command([sys.executable, "tools/build_validation_handoff_manifest.py"]),
        run_command([sys.executable, "tools/build_commit_readiness_plan.py"]),
        run_command([sys.executable, "tools/build_validation_handoff_manifest.py"]),
        run_command([sys.executable, "tools/build_goal_resolution_audit.py"]),
        run_command([sys.executable, "-m", "pytest"]),
    ]
    summary = build_summary(commands)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote current goal state verification to {args.out}")
    print(f"production_readiness_status={summary['production_readiness_status']}")
    print(f"overall_goal_status={summary['overall_goal_status']}")
    print(f"pytest_status={summary['pytest_status']}")
    if args.require_production_ready and (
        summary["production_readiness_status"] != "pass" or summary["overall_goal_status"] != "complete"
    ):
        print("required_production_ready=false")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
