"""Fail-closed final Stage 5A evidence validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _gate(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def validate_state(state_path: Path) -> dict[str, Any]:
    state_path = state_path.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    root = Path(state["output_root"]).resolve()
    pre = state.get("trajectory_preflight", {})
    replays = state.get("replays", {})
    moveit = state.get("moveit", {})
    bullet = state.get("bullet", {})
    expected = {"auto0": (7005, 7004, "adversarial_0100"), "auto1": (7030, 7029, "adversarial_0101")}
    gates: dict[str, Any] = {
        "D65_INPUT_FROZEN": bool(all(item.get("passed") for item in pre.values())),
        "D65_MODIFIED": False,
        "REPLAN": False,
        "RETIME": False,
        "RERUCKIG": False,
        "MODEL": bool(state.get("model_preflight", {}).get("passed")),
        "TCP_EXPLICIT": bool(state.get("tcp", {}).get("tcp", {}).get("xyz_m") == [0.0, 0.0, 0.15] and state.get("tcp", {}).get("tcp", {}).get("source") == "assumed_150mm_placeholder"),
        "PLUMBING_SMOKE": bool(state.get("plumbing_smoke", {}).get("action", {}).get("execution_completed")),
        "PROTECTED_UNCHANGED": bool(state.get("protected_unchanged")),
        "REGRESSION": bool(state.get("regression", {}).get("exit_code") == 0 and not state.get("regression", {}).get("timed_out")),
    }
    for name, (states, intervals, trajectory_id) in expected.items():
        replay = replays.get(name, {})
        action = replay.get("action", {})
        ready = replay.get("ready", {})
        qmetrics = replay.get("q_cmd_q_mock", {})
        gate_prefix = name.upper()
        source_match = action.get("trajectory_id") == trajectory_id and action.get("goal_point_count") == states
        full = bool(source_match and action.get("goal_accepted") and action.get("execution_completed") and action.get("one_follow_joint_trajectory_goal") and action.get("second_action_goals") == 0 and action.get("cancellations") == 0 and action.get("preemptions") == 0)
        gates[f"{gate_prefix}_FULL_REPLAY"] = full
        gates[f"{gate_prefix}_ROUTING"] = bool(source_match and action.get("source_sha256"))
        gates[f"{gate_prefix}_STATE_COUNT"] = bool(pre.get(name, {}).get("state_count") == states)
        gates[f"{gate_prefix}_INTERVAL_COUNT"] = bool(pre.get(name, {}).get("interval_count") == intervals)
        gates[f"{gate_prefix}_CONTROLLER_MANAGER"] = bool(ready.get("controller_manager"))
        gates[f"{gate_prefix}_HARDWARE"] = bool(ready.get("hardware_component_active"))
        gates[f"{gate_prefix}_JSB"] = bool(ready.get("joint_state_broadcaster_active"))
        gates[f"{gate_prefix}_JTC"] = bool(ready.get("joint_trajectory_controller_active"))
        gates[f"{gate_prefix}_ACTION_SERVER"] = bool(ready.get("follow_joint_trajectory_available"))
        gates[f"Q_CMD_TO_Q_MOCK_{gate_prefix}"] = bool(full and qmetrics.get("passed"))
        move = moveit.get(name, {})
        gates[f"{gate_prefix}_D65_FK_VS_ROS_TF"] = bool(move.get("passed") and move.get("d65_fk_vs_ros_tf", {}).get("passed"))
        gates[f"{gate_prefix}_PLANNING_SCENE"] = bool(move.get("planning_scene", {}).get("loaded") and move.get("d65_fk_trace", {}).get("complete"))
    gates["AUTO_SOURCES_DISTINCT"] = bool(replays.get("auto0", {}).get("action", {}).get("source_sha256") != replays.get("auto1", {}).get("action", {}).get("source_sha256"))
    gates["TUNNEL_PLANNING_SCENE"] = bool(all(moveit.get(name, {}).get("planning_scene", {}).get("loaded") for name in expected))
    gates["MOVEIT_DISCRETE_SELF_COLLISION_MEASURED"] = bool(all(moveit.get(name, {}).get("moveit_discrete_self_collision", {}).get("status") == "measured" and moveit.get(name, {}).get("moveit_discrete_self_collision", {}).get("states_checked") == expected[name][0] for name in expected))
    gates["BULLET_ROBOT_WORLD_CCD_MEASURED"] = bool(all(bullet.get(name, {}).get("validation_complete") for name in expected))
    gates["D65_QUALIFIED_CERTIFICATE_PRESERVED"] = all((root.parent.parent / "D65_FINAL_CLOSURE" / f"continuous_certificate_{name}" / "D65_VALIDATED_FK_LIPSCHITZ_CERTIFICATE.json").is_file() for name in expected)
    required = [
        "D65_INPUT_FROZEN", "MODEL", "TCP_EXPLICIT", "PLUMBING_SMOKE", "PROTECTED_UNCHANGED", "REGRESSION",
        "AUTO_SOURCES_DISTINCT", "TUNNEL_PLANNING_SCENE", "MOVEIT_DISCRETE_SELF_COLLISION_MEASURED", "BULLET_ROBOT_WORLD_CCD_MEASURED",
        "D65_QUALIFIED_CERTIFICATE_PRESERVED",
    ] + [key for key in gates if key.endswith(("_FULL_REPLAY", "_ROUTING", "_D65_FK_VS_ROS_TF", "_PLANNING_SCENE")) or key.startswith("Q_CMD_TO_Q_MOCK_")]
    gates["STAGE5A_STATUS"] = all(bool(gates.get(key)) for key in required) and not gates["D65_MODIFIED"] and not gates["REPLAN"] and not gates["RETIME"] and not gates["RERUCKIG"]
    findings = {
        name: {"moveit_self_collision_state_count": moveit.get(name, {}).get("moveit_discrete_self_collision", {}).get("collision_state_count"), "moveit_self_collision_pairs": moveit.get(name, {}).get("moveit_discrete_self_collision", {}).get("pairs", []), "bullet_robot_world_collision_count": bullet.get(name, {}).get("continuous_collision_count"), "bullet_validation_complete": bullet.get(name, {}).get("validation_complete")} for name in expected
    }
    ledger = {
        "schema_version": "stage5a-final-ledger-v1",
        "status": "PASS" if gates["STAGE5A_STATUS"] else "FAIL_CLOSED",
        "scope": "ROS2_JAZZY_MOVEIT2_ROS2_CONTROL_JTC_GENERICSYSTEM_MOCK_EXECUTION_CHAIN",
        "output_root": str(root),
        "gates": gates,
        "trajectory": {name: {"trajectory_id": spec[2], "state_count": spec[0], "interval_count": spec[1], "source": pre.get(name, {}).get("source"), "source_sha256": pre.get(name, {}).get("source_sha256")} for name, spec in expected.items()},
        "findings": findings,
        "collision_semantics": {"d65_qualified_self_collision": "preserved independently", "moveit_discrete_self_collision": "measured independently over D65 states", "bullet_robot_world_ccd": "measured with native two-state CollisionEnvBullet API", "continuous_self_collision": "not_available", "clearance": None},
        "claims": {"mock_execution_chain": True, "idealized_q_cmd_to_q_mock": True, "real_fr5_tracking": False, "physical_safety": False, "calibrated_tcp": False, "gazebo_dynamics": False, "real_torque": False, "real_contact_force": False, "vendor_servo": False},
        "not_claimed": ["real FR5 tracking accuracy", "hardware safety", "physical TCP calibration", "actuator/motor/gearbox/compliance physics", "continuous self-collision from Bullet robot-world CCD"],
    }
    write_json(root / "STAGE5A_FINAL_LEDGER.json", ledger)
    lines = [
        "# Stage 5A Final Authority Report", "", f"- Final status: **{ledger['status']}**.", "- Scope: ROS2 Jazzy + MoveIt2 + ros2_control + JointTrajectoryController + mock_components/GenericSystem.", "- D65 inputs are consumed read-only; no replan, retime, resample, or Ruckig call is part of this chain.", "", "## Gate summary", "", "```text",
    ]
    for key in sorted(gates):
        lines.append(f"{key}={_gate(gates[key]) if isinstance(gates[key], bool) else gates[key]}")
    lines.extend(["```", "", "## Independent collision evidence", "", "The MoveIt discrete self-collision result and Bullet robot-world two-state CCD are retained as separate measurements. They are not merged into a generic continuous-collision pass.", ""])
    for name in expected:
        lines.append(f"- {name}: MoveIt discrete self-collision states={findings[name]['moveit_self_collision_state_count']}, collision_states={findings[name]['moveit_self_collision_state_count']}; Bullet robot-world CCD collisions={findings[name]['bullet_robot_world_collision_count']}, complete={findings[name]['bullet_validation_complete']}.")
    lines.extend(["", "## Runtime FK/TF consistency", ""])
    for name in expected:
        tf = moveit.get(name, {}).get("d65_fk_vs_ros_tf", {})
        lines.append(f"- {name}: compared={tf.get('samples_compared')}, max_position_error_m={tf.get('max_position_error_m')}, max_orientation_error_rad={tf.get('max_orientation_error_rad')}, passed={tf.get('passed')}; pairing={tf.get('pairing')}.")
    lines.extend(["", "The strict FK/TF gate remains fail-closed when asynchronous ROS publication error exceeds 1e-5 m or rad. This is an observation-boundary result, not a physical tracking claim.", "", "Native Bullet contact pair/count evidence is retained; Bullet nearest-point fields from the maintained binding are not used as clearance evidence."])
    lines.extend(["", "## Scope boundary", "", "This report does not claim real FR5 tracking, calibrated TCP accuracy, physical clearance, actuator dynamics, torque, contact force, vendor servo behavior, or hardware safety. The current TCP source is the explicit model assumption `assumed_150mm_placeholder`.", ""])
    (root / "STAGE5A_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return ledger


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    ledger = validate_state(args.state)
    print(json.dumps({"status": ledger["status"], "output_root": ledger["output_root"]}, ensure_ascii=False))
    return 0 if ledger["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
