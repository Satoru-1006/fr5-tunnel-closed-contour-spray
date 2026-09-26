"""Audit a completed Stage 5B shadow FJT replay without promoting its result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


JOINTS = [f"j{i}" for i in range(1, 7)]
CONTINUITY_GATE_DEG = 20.0


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def max_abs(values):
    return max((abs(float(value)) for value in values), default=0.0)


def wrapped_delta_rad(a: float, b: float) -> float:
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def csv_metrics(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    q = [[float(row[f"j{i}_q"]) for i in range(1, 7)] for row in rows]
    t = [float(row["t"]) for row in rows]
    raw_steps = []
    wrapped_steps = []
    for left, right in zip(q, q[1:]):
        raw_steps.append([math.degrees(right[i] - left[i]) for i in range(6)])
        wrapped_steps.append([math.degrees(wrapped_delta_rad(left[i], right[i])) for i in range(6)])
    max_raw = max((max_abs(step) for step in raw_steps), default=0.0)
    max_wrapped = max((max_abs(step) for step in wrapped_steps), default=0.0)
    over_gate = [index for index, step in enumerate(wrapped_steps) if max_abs(step) > CONTINUITY_GATE_DEG]
    interval = {
        "left_waypoint": 66,
        "right_waypoint": 67,
        "raw_delta_deg": raw_steps[66],
        "wrapped_delta_deg": wrapped_steps[66],
        "raw_max_abs_deg": max_abs(raw_steps[66]),
        "wrapped_max_abs_deg": max_abs(wrapped_steps[66]),
    }
    return {
        "source_path": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "timestamps_monotonic": all(b >= a for a, b in zip(t, t[1:])),
        "duration_s": t[-1] if t else None,
        "continuity_gate_max_step_deg": CONTINUITY_GATE_DEG,
        "max_raw_step_deg": max_raw,
        "max_wrap_aware_step_deg": max_wrapped,
        "intervals_over_wrap_aware_gate": len(over_gate),
        "first_over_gate_interval_left_waypoint": over_gate[0] if over_gate else None,
        "wp66_to_wp67": interval,
        "source_q_finite": all(math.isfinite(value) for row in q for value in row),
        "source_endpoint_q": {"start": q[0], "end": q[-1]},
    }


def execution_metrics(runtime: Path, action: dict, goal: dict) -> dict:
    start = float(action["formal_start_monotonic_s"])
    end = float(action["formal_end_monotonic_s"])
    controller = [row for row in read_jsonl(runtime / "stage5a_controller_state_raw.jsonl") if start <= float(row["capture_monotonic_s"]) <= end]
    joints = [row for row in read_jsonl(runtime / "stage5a_joint_states_raw.jsonl") if start <= float(row["capture_monotonic_s"]) <= end]
    errors = []
    for row in controller:
        actual = row["feedback"]["positions"]
        reference = row["reference"]["positions"]
        errors.append([float(actual[i]) - float(reference[i]) for i in range(6)])
    actual_joint = [row["position"] for row in joints]
    joint_steps = []
    for left, right in zip(actual_joint, actual_joint[1:]):
        joint_steps.append([math.degrees(wrapped_delta_rad(float(left[i]), float(right[i]))) for i in range(6)])
    final_actual = controller[-1]["feedback"]["positions"] if controller else None
    final_goal = goal["points"][-1]["positions"]
    final_endpoint_error = [float(final_actual[i]) - float(final_goal[i]) for i in range(6)] if final_actual else None
    rms = [math.sqrt(sum(row[i] * row[i] for row in errors) / len(errors)) for i in range(6)] if errors else [None] * 6
    max_err = [max((abs(row[i]) for row in errors), default=0.0) for i in range(6)]
    return {
        "formal_controller_rows": len(controller),
        "formal_joint_state_rows": len(joints),
        "controller_error_max_abs_rad_by_joint": max_err,
        "controller_error_max_abs_deg": math.degrees(max(max_err, default=0.0)),
        "controller_error_rms_rad_by_joint": rms,
        "final_actual_q": final_actual,
        "final_goal_q": final_goal,
        "final_endpoint_error_rad": final_endpoint_error,
        "final_endpoint_error_max_abs_rad": max_abs(final_endpoint_error or []),
        "measured_joint_states_finite": all(math.isfinite(float(value)) for row in actual_joint for value in row),
        "measured_max_wrap_aware_sample_step_deg": max((max_abs(step) for step in joint_steps), default=0.0),
        "action_result": {
            "goal_sent": bool(action.get("goal_sent")),
            "goal_accepted": bool(action.get("goal_accepted")),
            "execution_completed": bool(action.get("execution_completed")),
            "result_code": int(action.get("result_code", -1)),
            "result_error_string": action.get("result_error_string"),
            "one_follow_joint_trajectory_goal": bool(action.get("one_follow_joint_trajectory_goal")),
            "second_action_goals": int(action.get("second_action_goals", -1)),
            "cancellations": int(action.get("cancellations", -1)),
            "preemptions": int(action.get("preemptions", -1)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    trajectory = args.trajectory.resolve()
    action = json.loads((runtime / "stage5a_action_execution.json").read_text(encoding="utf-8"))
    goal = json.loads((runtime / "stage5a_follow_joint_trajectory_goal.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "stage5b-shadow-execution-audit-v1",
        "status": "SOFTWARE_SIMULATION_EXECUTED_WITH_BASELINE_CONTINUITY_FAILURE",
        "scope": "x_minus_050_shadow_only",
        "world_authority": "STAGE5B_NOMINAL_WORLD_AUTHORITY.json; x=-0.50 m remains unverified shadow",
        "trajectory": csv_metrics(trajectory),
        "execution": execution_metrics(runtime, action, goal),
        "claim_fence": {
            "continuous_collision_detection": "not_available",
            "clearance": "not_available",
            "physical_installation_alignment": "UNVERIFIED",
            "hardware_safety": "UNVERIFIED",
            "collision_label": "adaptive_discrete_interpolation",
            "action_success_is_not_continuity_pass": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()), "max_wrap_aware_step_deg": report["trajectory"]["max_wrap_aware_step_deg"], "action_result": report["execution"]["action_result"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
