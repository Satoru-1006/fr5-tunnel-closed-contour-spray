"""Unit and negative tests for the Stage 5A frozen-input contract."""

from pathlib import Path
import csv
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage5a_trajectory_adapter import FIELDS, build_goal, load_frozen_trajectory  # noqa: E402
from tools.stage5_mock_validator import validate_state  # noqa: E402


def test_stage5_config_is_generic_system_and_tcp_is_explicit():
    xacro = (ROOT / "ros2_moveit_bridge/config/stage5_mock.urdf.xacro").read_text(encoding="utf-8")
    tcp = (ROOT / "ros2_moveit_bridge/config/stage5_mock_tcp.yaml").read_text(encoding="utf-8")
    assert "mock_components/GenericSystem" in xacro
    assert "FairinoHardwareInterface" not in xacro
    assert 'default="0 0 0.150"' in xacro
    assert "assumed_150mm_placeholder" in tcp


def test_adapter_rejects_header_or_shape_mutation(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t"])
        writer.writerow([0.0])
    with pytest.raises(ValueError, match="trajectory_header_mismatch"):
        load_frozen_trajectory(path, trajectory_id="x", expected_states=2, expected_intervals=1)


def test_goal_preserves_ros_representation_and_jerk_is_sidecar(tmp_path):
    path = ROOT / "outputs/D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE/timing_scale025_auto0/trajectories/adversarial_0100.csv"
    trajectory = load_frozen_trajectory(path, trajectory_id="adversarial_0100", expected_states=7005, expected_intervals=7004)
    goal = build_goal(trajectory)
    assert goal["joint_names"] == ["j1", "j2", "j3", "j4", "j5", "j6"]
    assert len(goal["points"]) == 7005
    assert goal["semantic_contract"]["replan"] is False
    assert goal["semantic_contract"]["jerk_is_sidecar_only"] is True
    assert "jerk" not in goal["points"][0]


def test_final_validator_fails_closed_on_missing_evidence(tmp_path):
    state = {"output_root": str(tmp_path), "trajectory_preflight": {}, "replays": {}, "moveit": {}, "bullet": {}, "protected_unchanged": False, "regression": {"exit_code": 2}}
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    ledger = validate_state(state_path)
    assert ledger["status"] == "FAIL_CLOSED"
    assert ledger["gates"]["STAGE5A_STATUS"] is False
