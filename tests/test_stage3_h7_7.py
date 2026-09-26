from __future__ import annotations

import ast
from pathlib import Path

from src.stage3_h7_7_gate import evaluate_h7_7_gate, passing_fixture


ROOT = Path(__file__).resolve().parents[1]


def rejected(**changes):
    candidate = passing_fixture()
    candidate.update(changes)
    result = evaluate_h7_7_gate(candidate)
    assert result["passed"] is False
    return result


def test_positive_control_passes() -> None:
    assert evaluate_h7_7_gate(passing_fixture())["passed"] is True


def test_wrapper_true_but_smoothing_incomplete_must_fail() -> None:
    assert rejected(wrapper_boolean=True, smoothing_complete=False)["first_blocker"] == "smoothing_incomplete"


def test_positions_changed_must_fail() -> None:
    assert rejected(positions_changed=True)["first_blocker"] == "joint_position_path_changed"


def test_unauthorized_stop_must_fail() -> None:
    assert rejected(unauthorized_stop_added=True)["first_blocker"] == "unauthorized_stop_added"


def test_strict_ruckig_invalid_input_must_fail() -> None:
    assert rejected(strict_current_target_valid=False, native_input_errors=1)["first_blocker"] == "strict_ruckig_invalid_input"


def test_process_tolerance_violation_must_fail() -> None:
    assert rejected(process_violations=1)["first_blocker"] == "process_tolerance_violation"


def test_joint_limit_violation_must_fail() -> None:
    assert rejected(jerk_limit_violations=1)["first_blocker"] == "jerk_limit_violation"


def test_nondeterministic_replay_must_fail() -> None:
    assert rejected(deterministic_replay="2/3")["first_blocker"] == "nondeterministic_replay"


def test_workers_have_no_goal_or_ledger_surface() -> None:
    for path in (
        ROOT / "ros2_moveit_bridge/stage3_h7_7_native.py",
        ROOT / "ros2_moveit_bridge/stage3_h7_7_physical_native.py",
    ):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not any("action" in name.lower() or "ledger" in name.lower() for name in imported)
        assert "FollowJointTrajectory" not in source
        assert "send_goal" not in source
        assert "os._exit" not in source
