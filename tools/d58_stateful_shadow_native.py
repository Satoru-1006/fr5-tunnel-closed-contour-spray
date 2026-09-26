"""Run a selected D57 optimizer shadow through the D52 stateful native repair.

This is a D58-only shadow harness.  It keeps the optimizer q path fixed while
rebuilding a consistent q/dq/ddq/t state before MoveIt2 native Ruckig.  It
does not touch D56/D57 protected artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

from moveit.planning import MoveItPy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from d52_stateful_ruckig_candidate import process_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--velocity-scaling", type=float, default=0.01)
    parser.add_argument("--acceleration-scaling", type=float, default=0.01)
    parser.add_argument("--time-scale-multiplier", type=float, default=1.0)
    args, _ = parser.parse_known_args()
    if args.case_id not in {"adversarial_0100", "adversarial_0101"}:
        raise ValueError("D58 harness only accepts the two selected adversarial optimizer shadows")
    source = args.source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    alias_dir = args.output_dir.resolve() / "input_alias"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / f"{args.case_id}.csv"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(source)
    moveit = MoveItPy(node_name=f"d58_stateful_{args.case_id}")
    try:
        record = process_case(
            moveit,
            alias,
            args.output_dir.resolve() / "trajectories" / f"{args.case_id}.csv",
            args.velocity_scaling,
            args.acceleration_scaling,
            repair_jerk_boundaries=False,
            repair_consistent_time=True,
            time_scale_multiplier=args.time_scale_multiplier,
        )
        record["d58_original_source_trajectory"] = str(source)
        record["input_alias"] = str(alias)
        summary = {
            "schema_version": "d58-stateful-shadow-native-v1",
            "status": "PASS" if record.get("status") == "PASS" else "BLOCKED",
            "case_count": 1,
            "native_post_ruckig_case_count": int(record.get("native_post_ruckig", False)),
            "velocity_scaling": float(args.velocity_scaling),
            "acceleration_scaling": float(args.acceleration_scaling),
            "repair_consistent_time": True,
            "time_scale_multiplier": float(args.time_scale_multiplier),
            "method": "D57 fixed q path -> consistent q/dq/ddq/t shadow repair -> MoveIt2 RobotTrajectory.apply_ruckig_smoothing",
            "cases": [record],
        }
        (args.output_dir.resolve() / "execution_form_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        exit_code = 0 if summary["status"] == "PASS" else 2
        print(json.dumps({"status": summary["status"], "case_id": args.case_id, "native_post_ruckig": bool(record.get("native_post_ruckig"))}, sort_keys=True))
        sys.stdout.flush()
        sys.stderr.flush()
        # MoveItPy/Jazzy can double-destroy MoveItCpp during interpreter
        # teardown after a completed native trajectory operation.  All output
        # is closed and the result is written before this one-shot exit.
        os._exit(exit_code)
    finally:
        moveit = None
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
