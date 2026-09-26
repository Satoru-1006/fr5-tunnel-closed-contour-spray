from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.stage28sr2_h4_1_identity_migration import main as migration_cli_main
from src.stage28sr2_formal_ledger import FormalGoalLedger
from src.stage28sr2_ledger_identity_migration import (
    AlreadyMigratedError,
    MigrationError,
    migrate_runner_identity,
    read_ledger_snapshot,
    structural_diff,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / ".stage28sr2" / "formal_r2_global_one_shot_ledger.json"
RUNNER = ROOT / "scripts" / "stage28sr2_production_runner.py"
OLD_RUNNER_SHA = "7845953eb09e48cea53907828e04976e4d4a8356099e89c0a839f948abf462e0"
PROVENANCE = {
    "old_runner_sha256": OLD_RUNNER_SHA,
    "current_runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    "historical_runner_identity": [{"path": "test://H3", "evidence": "historical_runner_identity"}],
    "h4_current_runner_cross_check": [{"path": "test://H4", "evidence": "h4_current_runner_cross_check"}],
    "identity_transition_reason": "test provenance",
}


def _copy_ledger(tmp_path: Path, mutate=None) -> tuple[Path, str]:
    path = tmp_path / "ledger.json"
    # Build a fresh zero-consumption ledger in pytest's temporary directory.
    # The canonical ledger is a durable one-shot record and may already be
    # consumed/migrated; tests must never inherit or mutate that state.
    FormalGoalLedger(path, formal_runner_sha256=OLD_RUNNER_SHA)
    state = json.loads(path.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(state)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw = path.read_bytes()
    return path, hashlib.sha256(raw).hexdigest()


def _migrate(tmp_path: Path, *, mutate=None, expected_sha: str | None = None):
    path, actual_sha = _copy_ledger(tmp_path, mutate)
    return migrate_runner_identity(
        ledger_path=path,
        output_dir=tmp_path / "evidence",
        expected_old_runner_sha256=OLD_RUNNER_SHA,
        expected_ledger_sha256=expected_sha or actual_sha,
        runner_path=RUNNER,
        repo_root=ROOT,
        provenance=PROVENANCE,
        migration_tool_path=ROOT / "scripts" / "stage28sr2_h4_1_identity_migration.py",
    )


def test_identity_only_cas_migration_changes_one_field_and_writes_forensics(tmp_path: Path) -> None:
    certificate = _migrate(tmp_path)
    raw, snapshot = read_ledger_snapshot(tmp_path / "ledger.json")
    current_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()

    assert certificate["migration_status"] == "passed"
    assert certificate["old_formal_runner_sha256"] == OLD_RUNNER_SHA
    assert certificate["new_formal_runner_sha256"] == current_sha
    assert certificate["actual_changed_fields"] == ["formal_runner_sha256"]
    assert certificate["unexpected_changed_fields"] == []
    assert certificate["zero_consumption_state_unchanged"] is True
    assert snapshot["formal_runner_sha256"] == current_sha
    assert snapshot["consumed"] == 0
    assert (tmp_path / "evidence" / "pre_migration_ledger.raw").read_bytes() != raw
    for name in (
        "pre_migration_ledger_snapshot.json",
        "pre_migration_ledger.raw",
        "pre_migration_ledger.sha256",
        "post_migration_ledger_snapshot.json",
        "post_migration_ledger.raw",
        "post_migration_ledger.sha256",
        "ledger_pre_post_diff.json",
        "migration_certificate.json",
    ):
        assert (tmp_path / "evidence" / name).is_file(), name


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("N3_consumed", lambda state: state.update(consumed=1, goal_budget_consumed=True, consumed_utc="test")),
        ("N4_send_attempted", lambda state: state.update(send_attempted=True)),
        ("N5_send_count", lambda state: state.update(send_goal_async_call_count=1)),
        ("N7_immutable_identity", lambda state: state.update(frozen_goal_sha256="f" * 64)),
        ("N8_non_allowlisted_field", lambda state: state.update(goal_accepted=True)),
    ],
)
def test_migration_negative_states_fail_closed_without_write(tmp_path: Path, name, mutate) -> None:
    path, before_sha = _copy_ledger(tmp_path, mutate)
    with pytest.raises(MigrationError):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / name,
            expected_old_runner_sha256=OLD_RUNNER_SHA,
            expected_ledger_sha256=before_sha,
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_sha


def test_n1_unexpected_old_runner_sha_is_rejected(tmp_path: Path) -> None:
    path, before_sha = _copy_ledger(tmp_path)
    with pytest.raises(MigrationError, match="historical runner SHA|formal_runner_sha256"):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / "n1",
            expected_old_runner_sha256="a" * 64,
            expected_ledger_sha256=before_sha,
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_sha


def test_n2_stale_ledger_sha_is_rejected(tmp_path: Path) -> None:
    path, before_sha = _copy_ledger(tmp_path)
    stale_sha = "0" * 64
    with pytest.raises(MigrationError, match="preflight expected ledger SHA"):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / "n2",
            expected_old_runner_sha256=OLD_RUNNER_SHA,
            expected_ledger_sha256=stale_sha,
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_sha


def test_n2_file_changed_after_preflight_is_rejected(tmp_path: Path) -> None:
    path, preflight_sha = _copy_ledger(tmp_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["goal_accepted"] = True
    path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    changed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(MigrationError, match="preflight expected ledger SHA"):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / "n2_changed",
            expected_old_runner_sha256=OLD_RUNNER_SHA,
            expected_ledger_sha256=preflight_sha,
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == changed_sha


def test_n6_corrupt_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_bytes(b"{not-json\n")
    before = path.read_bytes()
    with pytest.raises(MigrationError, match="unreadable|JSON"):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / "n6",
            expected_old_runner_sha256=OLD_RUNNER_SHA,
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert path.read_bytes() == before


def test_n9_second_migration_performs_no_additional_mutation(tmp_path: Path) -> None:
    first = _migrate(tmp_path)
    path = tmp_path / "ledger.json"
    after_first = path.read_bytes()
    with pytest.raises(AlreadyMigratedError):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / "second",
            expected_old_runner_sha256=OLD_RUNNER_SHA,
            expected_ledger_sha256=first["post_ledger_sha256"],
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert path.read_bytes() == after_first


def test_n10_production_runner_mismatch_remains_fail_closed_and_has_no_migration_hook() -> None:
    source = (ROOT / "scripts" / "stage28sr2_production_runner.py").read_text(encoding="utf-8")
    assert "migrate_runner_identity" not in source
    assert "canonical_ledger_validation_failed" in source


def test_production_ledger_constructor_rejects_historical_identity(tmp_path: Path) -> None:
    path, _ = _copy_ledger(tmp_path)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        FormalGoalLedger(path)


def test_structural_diff_rejects_any_non_allowlisted_change() -> None:
    before = {"formal_runner_sha256": "a", "consumed": 0}
    after = copy.deepcopy(before)
    after["formal_runner_sha256"] = "b"
    after["consumed"] = 1
    diff = structural_diff(before, after)
    assert diff["actual_changed_fields"] == ["consumed", "formal_runner_sha256"]
    assert diff["unexpected_changed_fields"] == ["consumed"]


def test_two_migration_processes_have_exactly_one_winner(tmp_path: Path) -> None:
    path, expected_sha = _copy_ledger(tmp_path)
    child = """
from pathlib import Path
from src.stage28sr2_ledger_identity_migration import migrate_runner_identity, MigrationError
try:
    migrate_runner_identity(ledger_path=Path(__import__('sys').argv[1]), output_dir=Path(__import__('sys').argv[2]), expected_old_runner_sha256=__import__('sys').argv[3], expected_ledger_sha256=__import__('sys').argv[4], runner_path=Path(__import__('sys').argv[5]), repo_root=Path(__import__('sys').argv[6]), provenance={'test': True})
except Exception as error:
    print(type(error).__name__ + ':' + str(error))
    raise SystemExit(2)
print('winner')
"""
    commands = []
    for index in range(2):
        commands.append(
            subprocess.Popen(
                [sys.executable, "-c", child, str(path), str(tmp_path / f"proc{index}"), OLD_RUNNER_SHA, expected_sha, str(RUNNER), str(ROOT)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    results = [process.communicate(timeout=30) for process in commands]
    assert sorted(process.returncode for process in commands) == [0, 2]
    assert sum("winner" in stdout for stdout, _ in results) == 1
    _, snapshot = read_ledger_snapshot(path)
    assert snapshot["formal_runner_sha256"] == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert snapshot["consumed"] == 0


def test_migration_after_simulated_consume_is_rejected(tmp_path: Path) -> None:
    path, expected_sha = _copy_ledger(tmp_path)
    ledger = FormalGoalLedger(path, formal_runner_sha256=OLD_RUNNER_SHA)
    ledger.consume_before_send()
    with pytest.raises(MigrationError):
        migrate_runner_identity(
            ledger_path=path,
            output_dir=tmp_path / "consume_race",
            expected_old_runner_sha256=OLD_RUNNER_SHA,
            expected_ledger_sha256=expected_sha,
            runner_path=RUNNER,
            repo_root=ROOT,
            provenance=PROVENANCE,
        )
    assert json.loads(path.read_text(encoding="utf-8"))["consumed"] == 1


def test_cli_requires_explicit_old_identity_and_does_not_auto_migrate() -> None:
    with pytest.raises(SystemExit):
        migration_cli_main(["--ledger", str(CANONICAL), "--output", str(ROOT / "tmp" / "h4_1_cli_missing_old")])
