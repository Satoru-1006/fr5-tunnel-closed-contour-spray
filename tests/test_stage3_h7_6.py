from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage3_h7_6", ROOT / "scripts/stage3_h7_6.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_authoritative_h7_5_excludes_interrupted_directory() -> None:
    assert MODULE.H75.name.endswith("20260809T221000Z")
    assert MODULE.H75_INTERRUPTED.name.endswith("20260809T220000Z")
    assert MODULE.H75 != MODULE.H75_INTERRUPTED


def test_worker_has_no_execution_or_ledger_interfaces() -> None:
    source = (ROOT / "ros2_moveit_bridge/stage3_h7_6_native.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any("action" in name.lower() or "controller" in name.lower() or "ledger" in name.lower() for name in imported)
    assert "send_goal" not in source
    assert "FollowJointTrajectory" not in source
    assert "os._exit" not in source


def test_tier_a_windows_cover_all_three_strict_invalid_states() -> None:
    source = (ROOT / "ros2_moveit_bridge/stage3_h7_6_native.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "REMEDIATION_WINDOWS")
    windows = ast.literal_eval(assignment.value)
    assert windows["1001"]["start_state_index"] < 450 < windows["1001"]["end_state_index"]
    assert windows["1001"]["start_state_index"] < 784 < windows["1001"]["end_state_index"]
    assert windows["1003"]["start_state_index"] < 457 < windows["1003"]["end_state_index"]


def test_jerk_boundary_formula_reproduces_h7_5_failure() -> None:
    inp = {
        "current_velocity": [0.0] * 6,
        "current_acceleration": [0.0] * 6,
        "target_velocity": [-0.47229000000008287] + [0.0] * 5,
        "target_acceleration": [0.10499999995333954] + [0.0] * 5,
        "max_velocity": [0.4725] * 6,
        "min_velocity": None,
        "max_acceleration": [0.105] * 6,
        "min_acceleration": None,
        "max_jerk": [8.0] * 6,
    }
    result = MODULE.jerk_boundary_feasibility(inp, "target", 0)
    assert result["pass"] is False
    assert result["velocity_headroom_rad_s"] > 0.0
    assert result["acceleration_feasibility_margin_rad_s2"] < 0.0


def test_collision_claim_remains_discrete() -> None:
    assert MODULE.COLLISION_METHOD == "adaptive_discrete_interpolation"
