"""Fail-closed, administrative migration of the canonical runner identity.

This module intentionally contains no production-send or runtime code.  It
exists for the one-purpose transition from a historically proven runner hash
to the independently computed current production-runner hash.  The canonical
ledger is changed only when the expected raw-file SHA, historical runner SHA,
immutable identity, zero-consumption state, and allowlisted structural diff
all match while holding the same inter-process lock used by the ledger.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage28sr2_formal_ledger import (
    CANONICAL_LEDGER_PATH,
    LEDGER_SCHEMA_VERSION,
    default_ledger_identity,
    exclusive_ledger_lock,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RUNNER_PATH = ROOT / "scripts" / "stage28sr2_production_runner.py"
ALLOWLISTED_CHANGED_FIELDS = ["formal_runner_sha256"]
ZERO_STATE_FIELDS = [
    "maximum",
    "consumed",
    "send_attempted",
    "send_binding_committed",
    "send_goal_async_call_count",
    "transport_send_invocation",
    "goal_budget_consumed",
]


class MigrationError(RuntimeError):
    """A migration precondition or durable-write verification failed."""


class AlreadyMigratedError(MigrationError):
    """The ledger already contains the requested current runner identity."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise MigrationError(f"cannot hash file: {path}: {error}") from error


def _utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _read_raw(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise MigrationError(f"cannot read canonical ledger: {path}: {error}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MigrationError(f"canonical ledger JSON is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise MigrationError("canonical ledger root must be a JSON object")
    return raw, value


def _state_value(state: Mapping[str, Any], key: str) -> Any:
    if key == "transport_send_invocation":
        return state.get("transport_send_invocation_observed")
    return state.get(key)


def _budget_value(state: Mapping[str, Any], key: str) -> Any:
    budget = state.get("formal_goal_budget")
    if not isinstance(budget, Mapping):
        return None
    if key == "transport_send_invocation":
        return budget.get("transport_send_invocation_observed")
    return budget.get(key)


def _require_bool(value: Any, field: str) -> None:
    if not isinstance(value, bool):
        raise MigrationError(f"ledger field is not boolean: {field}")


def validate_ledger_shape(
    state: Mapping[str, Any],
    *,
    expected_runner_sha256: str,
    expected_immutable_identity: Mapping[str, Any] | None = None,
) -> None:
    """Validate the complete v2 ledger without mutating it."""

    if state.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise MigrationError("formal goal ledger schema mismatch")
    if state.get("formal_runner_sha256") != expected_runner_sha256:
        raise MigrationError("formal goal ledger runner identity mismatch")
    immutable = dict(expected_immutable_identity or default_ledger_identity())
    for key in ("certification_identity", "frozen_goal_sha256"):
        expected = immutable.get(key)
        if expected is not None and state.get(key) != expected:
            raise MigrationError(f"formal goal ledger immutable identity mismatch: {key}")

    budget = state.get("formal_goal_budget")
    if not isinstance(budget, Mapping):
        raise MigrationError("formal goal ledger budget is missing")
    if state.get("maximum") != 1 or budget.get("maximum") != 1:
        raise MigrationError("formal goal ledger maximum must remain exactly one")
    if state.get("consumed") != budget.get("consumed") or budget.get("consumed") not in (0, 1):
        raise MigrationError("formal goal ledger consumed state is corrupt")
    if state.get("goal_budget_consumed") is not (state.get("consumed") == 1):
        raise MigrationError("formal goal budget consumption flag disagrees with consumed")
    if budget.get("goal_budget_consumed") is not (budget.get("consumed") == 1):
        raise MigrationError("nested formal goal budget consumption flag disagrees with consumed")

    for key in ("send_attempted", "send_binding_committed", "transport_send_invocation"):
        top = _state_value(state, key)
        nested = _budget_value(state, key)
        _require_bool(top, key)
        _require_bool(nested, f"formal_goal_budget.{key}")
        if top is not nested:
            raise MigrationError(f"formal goal ledger {key} mirrors disagree")
    call_count = budget.get("send_goal_async_call_count")
    if call_count not in (0, 1) or isinstance(call_count, bool) or state.get("send_goal_async_call_count") != call_count:
        raise MigrationError("formal goal ledger send call count is corrupt")
    if state.get("send_attempted") and state.get("consumed") != 1:
        raise MigrationError("formal goal ledger is not fail-closed")
    if state.get("send_binding_committed") and not state.get("send_attempted"):
        raise MigrationError("formal goal ledger records binding without send attempt")
    if state.get("transport_send_invocation_observed") and not state.get("send_binding_committed"):
        raise MigrationError("formal goal ledger records invocation without binding")
    if call_count == 1 and not state.get("transport_send_invocation_observed"):
        raise MigrationError("formal goal ledger records send without invocation observation")
    if state.get("consumed") == 1 and not state.get("consumed_utc"):
        raise MigrationError("consumed ledger is missing its durable timestamp")


def validate_zero_consumption_state(state: Mapping[str, Any]) -> None:
    """Require every one-shot/send field to be in its untouched state."""

    expected = {
        "maximum": 1,
        "consumed": 0,
        "send_attempted": False,
        "send_binding_committed": False,
        "send_goal_async_call_count": 0,
        "transport_send_invocation": False,
        "goal_budget_consumed": False,
    }
    for key, wanted in expected.items():
        if _state_value(state, key) != wanted:
            raise MigrationError(f"migration requires zero-consumption state: {key}")
        if key in {"send_attempted", "send_binding_committed", "transport_send_invocation", "goal_budget_consumed"} and isinstance(_state_value(state, key), bool) is False:
            raise MigrationError(f"migration requires boolean zero state: {key}")
    if state.get("goal_request_acknowledged") is not False or state.get("goal_accepted") is not False:
        raise MigrationError("migration requires no goal response or acceptance state")
    budget = state.get("formal_goal_budget")
    if not isinstance(budget, Mapping):
        raise MigrationError("formal goal ledger budget is missing")
    nested_expected = {
        "maximum": 1,
        "consumed": 0,
        "goal_budget_consumed": False,
        "send_attempted": False,
        "send_binding_committed": False,
        "send_goal_async_call_count": 0,
        "transport_send_invocation_observed": False,
    }
    for key, wanted in nested_expected.items():
        if budget.get(key) != wanted:
            raise MigrationError(f"migration requires zero-consumption nested state: formal_goal_budget.{key}")


def _snapshot_from_raw(path: Path, raw: bytes, state: Mapping[str, Any]) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as error:
        raise MigrationError(f"cannot stat canonical ledger: {path}: {error}") from error
    file_identity = {
        "st_dev": getattr(stat, "st_dev", None),
        "st_ino": getattr(stat, "st_ino", None),
    }
    return {
        "canonical_path": str(path.resolve()),
        "file_sha256": sha256_bytes(raw),
        "raw_byte_size": len(raw),
        "mtime": _utc_from_timestamp(stat.st_mtime),
        "mtime_ns": getattr(stat, "st_mtime_ns", None),
        "file_identity": file_identity,
        "parsed_json": copy.deepcopy(dict(state)),
        "formal_runner_sha256": state.get("formal_runner_sha256"),
        "maximum": state.get("maximum"),
        "consumed": state.get("consumed"),
        "send_attempted": state.get("send_attempted"),
        "send_binding_committed": state.get("send_binding_committed"),
        "send_goal_async_call_count": state.get("send_goal_async_call_count"),
        "transport_send_invocation": state.get("transport_send_invocation_observed"),
        "goal_budget_consumed": state.get("goal_budget_consumed"),
        "formal_goal_budget": copy.deepcopy(state.get("formal_goal_budget")),
    }


def read_ledger_snapshot(path: Path) -> tuple[bytes, dict[str, Any]]:
    path = path.resolve()
    raw, state = _read_raw(path)
    return raw, _snapshot_from_raw(path, raw, state)


def write_forensic_snapshot(output: Path, prefix: str, raw: bytes, snapshot: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{prefix}.raw").write_bytes(raw)
    (output / f"{prefix}.sha256").write_text(sha256_bytes(raw) + "\n", encoding="ascii")
    (output / f"{prefix}_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_files_for_provenance(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "outputs" / "stage28sr2_h3_formal_plumbing_20260807",
        repo_root / "outputs" / "stage28sr2_r2_production_preformal",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            if path.name.endswith(".log") or path.stat().st_size > 12 * 1024 * 1024:
                continue
            candidates.append(path)
    return sorted(set(candidates))


def discover_runner_provenance(repo_root: Path, old_runner_sha256: str, current_runner_sha256: str) -> dict[str, Any]:
    """Cross-check old identity against H3 and current identity against H4."""

    old_matches: list[dict[str, Any]] = []
    h4_matches: list[dict[str, Any]] = []
    for path in _json_files_for_provenance(repo_root.resolve()):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if old_runner_sha256 in text:
            old_matches.append({"path": str(path.resolve()), "sha256": sha256_file(path), "evidence": "historical_runner_identity"})
        if current_runner_sha256 in text and ("h4_" in str(path).lower() or "h4" in path.name.lower()):
            h4_matches.append({"path": str(path.resolve()), "sha256": sha256_file(path), "evidence": "h4_current_runner_cross_check"})
    if not old_matches:
        raise MigrationError("historical old runner SHA has no H3/H4 provenance evidence")
    if not h4_matches:
        raise MigrationError("current runner SHA has no H4 certificate cross-check")
    return {
        "old_runner_sha256": old_runner_sha256,
        "current_runner_sha256": current_runner_sha256,
        "historical_runner_identity": old_matches,
        "h4_current_runner_cross_check": h4_matches,
        "identity_transition_reason": "H4 hardening changed the production runner source; the H3/H4-pre identity is migrated only after H4 current-SHA cross-check.",
    }


def _flatten_diff(before: Any, after: Any, prefix: str = "") -> list[str]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        fields: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{prefix}.{key}" if prefix else str(key)
            fields.extend(_flatten_diff(before.get(key), after.get(key), child))
        return fields
    return [] if before == after else [prefix]


def structural_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    changed = _flatten_diff(before, after)
    changes = []
    for field in changed:
        old: Any = before
        new: Any = after
        for part in field.split("."):
            old = old.get(part) if isinstance(old, Mapping) else None
            new = new.get(part) if isinstance(new, Mapping) else None
        changes.append({"field": field, "before": old, "after": new})
    unexpected = [field for field in changed if field not in ALLOWLISTED_CHANGED_FIELDS]
    return {
        "allowed_changed_fields": list(ALLOWLISTED_CHANGED_FIELDS),
        "actual_changed_fields": changed,
        "unexpected_changed_fields": unexpected,
        "changes": changes,
    }


def _atomic_durable_replace_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Replace JSON atomically and report each durability step."""

    path = path.resolve()
    parent = path.parent
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".migration.tmp", dir=parent)
    temp_fsync = False
    replaced = False
    directory_status = "not_supported_windows" if os.name == "nt" else "not_attempted"
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_fsync = True
        temp_raw = Path(temporary_name).read_bytes()
        try:
            parsed = json.loads(temp_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MigrationError(f"temporary ledger JSON verification failed: {error}") from error
        if parsed != dict(value):
            raise MigrationError("temporary ledger JSON differs from intended candidate")
        os.replace(temporary_name, path)
        replaced = True
        if os.name != "nt":
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as error:
                raise MigrationError(f"parent directory durability sync failed: {error}") from error
            directory_status = "completed"
        return {
            "atomic_replace_used": replaced,
            "file_fsync_completed": temp_fsync,
            "directory_durability_status": directory_status,
        }
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _counter_fields(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "maximum": state.get("maximum"),
        "consumed": state.get("consumed"),
        "send_attempted": state.get("send_attempted"),
        "send_binding_committed": state.get("send_binding_committed"),
        "send_goal_async_call_count": state.get("send_goal_async_call_count"),
        "transport_send_invocation": state.get("transport_send_invocation_observed"),
        "goal_budget_consumed": state.get("goal_budget_consumed"),
    }


def migrate_runner_identity(
    *,
    ledger_path: Path = CANONICAL_LEDGER_PATH,
    output_dir: Path,
    expected_old_runner_sha256: str,
    expected_ledger_sha256: str | None = None,
    runner_path: Path = PRODUCTION_RUNNER_PATH,
    repo_root: Path = ROOT,
    provenance: Mapping[str, Any] | None = None,
    migration_tool_path: Path | None = None,
) -> dict[str, Any]:
    """Perform one CAS-guarded identity-only migration.

    The preflight forensic snapshot is written before the lock is reacquired
    for the CAS write.  Any stale-state or validation mismatch raises before
    ``os.replace`` and therefore leaves the canonical file byte-for-byte
    unchanged.
    """

    path = ledger_path.resolve()
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    current_runner_sha256 = sha256_file(runner_path.resolve())
    if len(expected_old_runner_sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in expected_old_runner_sha256):
        raise MigrationError("expected old runner SHA must be a 64-character hexadecimal digest")
    if expected_old_runner_sha256.lower() == current_runner_sha256.lower():
        raise MigrationError("expected historical runner SHA equals current runner SHA")

    pre_raw, pre_snapshot = read_ledger_snapshot(path)
    write_forensic_snapshot(output, "pre_migration_ledger", pre_raw, pre_snapshot)
    pre_state = pre_snapshot["parsed_json"]
    if expected_ledger_sha256 is None:
        expected_ledger_sha256 = pre_snapshot["file_sha256"]
    if pre_snapshot["file_sha256"] != expected_ledger_sha256:
        raise MigrationError("preflight expected ledger SHA does not match pre-migration snapshot")
    if pre_state.get("formal_runner_sha256") == current_runner_sha256:
        raise AlreadyMigratedError("canonical ledger already contains current runner identity")
    if pre_state.get("formal_runner_sha256") != expected_old_runner_sha256:
        raise MigrationError("canonical ledger formal_runner_sha256 is not the expected historical SHA")
    validate_ledger_shape(pre_state, expected_runner_sha256=expected_old_runner_sha256)
    validate_zero_consumption_state(pre_state)
    provenance_value = dict(provenance or discover_runner_provenance(repo_root, expected_old_runner_sha256, current_runner_sha256))

    migration_timestamp = datetime.now(timezone.utc).isoformat()
    tool_path = (migration_tool_path or Path(__file__)).resolve()
    tool_sha256 = sha256_file(tool_path)
    transaction: dict[str, Any] = {
        "atomic_replace_used": False,
        "file_fsync_completed": False,
        "directory_durability_status": "not_attempted",
    }
    post_raw: bytes | None = None
    post_state: dict[str, Any] | None = None
    diff: dict[str, Any] | None = None
    lock_status = {"acquired": False, "lock_path": str(path.with_suffix(path.suffix + ".lock")), "exclusive": True}
    try:
        with exclusive_ledger_lock(path):
            lock_status["acquired"] = True
            current_raw, current_state = _read_raw(path)
            current_ledger_sha256 = sha256_bytes(current_raw)
            if current_ledger_sha256 != expected_ledger_sha256:
                raise MigrationError("CAS failure: canonical ledger changed after preflight")
            if current_state.get("formal_runner_sha256") == current_runner_sha256:
                raise AlreadyMigratedError("canonical ledger was migrated by another process")
            if current_state.get("formal_runner_sha256") != expected_old_runner_sha256:
                raise MigrationError("CAS failure: current historical runner identity differs from expected old SHA")
            validate_ledger_shape(current_state, expected_runner_sha256=expected_old_runner_sha256)
            validate_zero_consumption_state(current_state)
            candidate = copy.deepcopy(current_state)
            candidate["formal_runner_sha256"] = current_runner_sha256
            diff = structural_diff(current_state, candidate)
            if diff["unexpected_changed_fields"]:
                raise MigrationError(f"unexpected ledger diff: {diff['unexpected_changed_fields']}")
            validate_ledger_shape(candidate, expected_runner_sha256=current_runner_sha256)
            validate_zero_consumption_state(candidate)
            transaction = _atomic_durable_replace_json(path, candidate)
            post_raw, post_state = _read_raw(path)
            if sha256_bytes(post_raw) == expected_ledger_sha256:
                raise MigrationError("post-migration ledger SHA did not change")
            validate_ledger_shape(post_state, expected_runner_sha256=current_runner_sha256)
            validate_zero_consumption_state(post_state)
            if structural_diff(current_state, post_state)["unexpected_changed_fields"]:
                raise MigrationError("post-migration ledger has an unexpected structural diff")
    except (AlreadyMigratedError, MigrationError):
        raise
    except Exception as error:
        raise MigrationError(f"ledger migration failed closed: {type(error).__name__}: {error}") from error

    assert post_raw is not None and post_state is not None and diff is not None
    post_snapshot = _snapshot_from_raw(path, post_raw, post_state)
    write_forensic_snapshot(output, "post_migration_ledger", post_raw, post_snapshot)
    post_diff = dict(diff)
    post_diff.update({"pre_ledger_sha256": expected_ledger_sha256, "post_ledger_sha256": post_snapshot["file_sha256"]})
    (output / "ledger_pre_post_diff.json").write_text(json.dumps(post_diff, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    before = _counter_fields(pre_state)
    after = _counter_fields(post_state)
    certificate = {
        "migration_status": "passed",
        "canonical_path": str(path),
        "pre_ledger_sha256": expected_ledger_sha256,
        "post_ledger_sha256": post_snapshot["file_sha256"],
        "old_formal_runner_sha256": expected_old_runner_sha256,
        "new_formal_runner_sha256": current_runner_sha256,
        "independently_computed_current_runner_sha256": current_runner_sha256,
        "old_runner_provenance": provenance_value,
        "allowed_changed_fields": post_diff["allowed_changed_fields"],
        "actual_changed_fields": post_diff["actual_changed_fields"],
        "unexpected_changed_fields": post_diff["unexpected_changed_fields"],
        "maximum_before": before["maximum"],
        "maximum_after": after["maximum"],
        "consumed_before": before["consumed"],
        "consumed_after": after["consumed"],
        "send_attempted_before": before["send_attempted"],
        "send_attempted_after": after["send_attempted"],
        "send_binding_committed_before": before["send_binding_committed"],
        "send_binding_committed_after": after["send_binding_committed"],
        "send_goal_async_call_count_before": before["send_goal_async_call_count"],
        "send_goal_async_call_count_after": after["send_goal_async_call_count"],
        "transport_send_invocation_before": before["transport_send_invocation"],
        "transport_send_invocation_after": after["transport_send_invocation"],
        "goal_budget_consumed_before": before["goal_budget_consumed"],
        "goal_budget_consumed_after": after["goal_budget_consumed"],
        "migration_pid": os.getpid(),
        "timestamp": migration_timestamp,
        "migration_tool_sha256": tool_sha256,
        **transaction,
        "lock_status": lock_status,
        "CAS_status": "passed",
        "zero_consumption_state_unchanged": before == after,
    }
    (output / "migration_certificate.json").write_text(json.dumps(certificate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate
