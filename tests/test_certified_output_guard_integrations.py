from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_h10_orchestrator_checks_guard_before_output_creation() -> None:
    source = (ROOT / "scripts/stage3_h10_multi_trajectory_dataset.py").read_text(encoding="utf-8")
    guard = source.index('assert_output_path_writable(output, operation="stage3_h10_output_or_resume")')
    mkdir = source.index("output.mkdir(parents=True, exist_ok=True)", guard)
    assert guard < mkdir


def test_h10_native_writer_guards_json_and_jsonl() -> None:
    source = (ROOT / "ros2_moveit_bridge/stage3_h10_native.py").read_text(encoding="utf-8")
    assert 'assert_output_path_writable(path, operation="overwrite_json")' in source
    assert 'assert_output_path_writable(path, operation="truncate_jsonl")' in source
    assert 'except CertifiedOutputImmutableError as exc:' in source


def test_stage24g_root_and_leaf_writers_are_guarded() -> None:
    source = (ROOT / "scripts/run_stage24g_complete_edge_component_bridge.py").read_text(encoding="utf-8")
    assert 'assert_output_path_writable(out, operation="stage24g_output_root")' in source
    assert 'assert_output_path_writable(path, operation="truncate_jsonl")' in source
    assert 'assert_output_path_writable(path, operation="truncate_csv")' in source


def test_ruckig_interposer_guards_before_truncate() -> None:
    source = (ROOT / "tools/stage25r_ruckig_interposer.cpp").read_text(encoding="utf-8")
    guard = source.index("certified_output_guard::assert_output_path_writable(directory_);")
    first_truncate = source.index('calls_.open(directory_ / "stage25r_ruckig_calls.jsonl"')
    assert guard < first_truncate
    assert 'certified_output_guard.hpp' in source
    assert 'std::ios::trunc' in source
