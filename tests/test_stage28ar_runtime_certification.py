"""Unit guards for Stage 2.8A-R; no ROS or robot process is started here."""

from pathlib import Path

from scripts.stage28ar_runtime_certification import (
    assess_live_state,
    classify_distro_compatibility,
    classify_hardware_component,
    classify_runtime,
    command_is_read_only,
    verify_sha_manifest,
)


def test_zero_motion_command_guard_rejects_writes_and_allows_discovery() -> None:
    assert command_is_read_only("ros2 action list -t")
    assert command_is_read_only("ros2 topic echo /joint_states --once")
    assert not command_is_read_only("ros2 action send_goal /arm/follow_joint_trajectory ...")
    assert not command_is_read_only("ros2 topic pub /joint_trajectory ...")
    assert not command_is_read_only("ros2 param set /controller_manager update_rate 100")


def test_mock_hardware_is_never_real_hardware_evidence() -> None:
    assert classify_hardware_component("mock_components/GenericSystem") == "rejected_mock_or_simulation"
    assert classify_hardware_component("Stage27SGenericSystem") == "rejected_mock_or_simulation"
    assert classify_hardware_component("fairino_hardware/FairinoHardwareInterface").startswith("candidate_fairino")


def test_frozen_hash_verification_detects_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("stable\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  payload.txt\n", encoding="utf-8")
    assert verify_sha_manifest(tmp_path)["passed"]
    payload.write_text("mutated\n", encoding="utf-8")
    result = verify_sha_manifest(tmp_path)
    assert not result["passed"]
    assert result["mismatches"]


def test_runtime_classification_requires_all_live_gates() -> None:
    assert classify_runtime(hardware_component_present=False, live_state=False, jtc_runtime=False, moveit_runtime=False) == "blocked_real_ros2_control_runtime"
    assert classify_runtime(hardware_component_present=True, live_state=True, jtc_runtime=True, moveit_runtime=True).startswith("candidate_runtime_complete")


def test_jazzy_humble_mismatch_is_a_blocker() -> None:
    assert classify_distro_compatibility("jazzy", "requires Humble") == "blocked_ros_distribution_driver_compatibility"
    assert classify_distro_compatibility("jazzy", None) == "compatible_with_existing_jazzy_stack"


def test_live_state_freshness_does_not_require_motion() -> None:
    result = assess_live_state([
        {"timestamp": 10.0, "positions": [0.0] * 6},
        {"timestamp": 10.01, "positions": [0.0] * 6},
    ])
    assert result["finite"]
    assert result["timestamps_monotonic"]
    assert result["freshness"]
    assert result["non_constant_or_sensor_refresh_proof"]


def test_semantics_transfer_guard_rejects_missing_runtime() -> None:
    assert classify_runtime(hardware_component_present=True, live_state=True, jtc_runtime=False, moveit_runtime=False) == "blocked_real_ros2_control_runtime"
