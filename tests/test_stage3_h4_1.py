from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_h4_1_bridge_calls_get_position_ik_without_retry_loop() -> None:
    source = (ROOT / "cpp/stage3_h4_1/src/stage3_h4_1_ik_bridge.cpp").read_text(encoding="utf-8")
    assert "solver->getPositionIK" in source
    assert "set_from_ik" not in source
    assert "while (!timedOut" not in source
    assert "attempt > 1" not in source
    assert "internal_attempt_count" in source


def test_h4_1_contract_keeps_frozen_seed_cardinality_and_call_count() -> None:
    source = (ROOT / "scripts/stage3_h4_1.py").read_text(encoding="utf-8")
    assert '"seed_count": 13' in source
    assert "192 * 13" in source
    assert "repeat_per_pair=3" in source


def test_h4_1_forbids_execution_paths_in_worker() -> None:
    source = (ROOT / "scripts/stage3_h4_1.py").read_text(encoding="utf-8")
    assert "send_goal_async" not in source
    assert "MoveItPy" in source
    assert "import ruckig" not in source
