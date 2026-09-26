from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "cpp/stage26/stage26_continuous_probe.cpp"
RUNNER = ROOT / "scripts/run_stage26.py"
LAUNCH = ROOT / "tools/stage26_continuous_launch.py"


def test_stage26_native_probe_uses_two_state_moveit_api() -> None:
    text = PROBE.read_text(encoding="utf-8")
    assert "checkRobotCollision(request, result, state0, state1, acm)" in text
    assert "native_bullet_robot_world_ccd" in text
    assert "stage26_continuous_robot_world_intervals.jsonl" in text


def test_stage26_keeps_fcl_and_self_collision_capability_gaps_explicit() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "MoveIt_FCL_backend_does_not_implement_continuous_collision" in text
    assert "adaptive_discrete_self_collision_certification" in text
    assert "repository_stage26_contract_found\": False" in text


def test_stage26_launch_targets_isolated_package() -> None:
    text = LAUNCH.read_text(encoding="utf-8")
    assert 'package="stage26_continuous_collision"' in text
    assert 'executable="stage26_continuous_probe"' in text
