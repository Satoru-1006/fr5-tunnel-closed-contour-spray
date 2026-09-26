"""Global, irreversible one-shot ledger for Stage 2.8S-R2 Formal execution.

The canonical ledger deliberately lives outside every per-run ``outputs``
directory.  Every read/compare/write transition is protected by an OS file
lock and durably replaced, so process restarts and concurrent runners share
one fail-closed budget.

There is intentionally no reset API.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_LEDGER_PATH = (ROOT / ".stage28sr2" / "formal_r2_global_one_shot_ledger.json").resolve()
LEDGER_SCHEMA_VERSION = "stage28sr2-formal-global-ledger-v2"
DEFAULT_CERTIFICATION_IDENTITY = "Stage-2.8S-R2-Formal-R2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_ledger_identity() -> dict[str, str]:
    frozen = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z/stage27r_clean_follow_joint_trajectory_goal.json"
    runner = ROOT / "scripts/stage28sr2_production_runner.py"
    return {
        "certification_identity": DEFAULT_CERTIFICATION_IDENTITY,
        "frozen_goal_sha256": _sha256(frozen) or "not_available",
        "formal_runner_sha256": _sha256(runner) or "not_available",
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON object."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass  # Windows has no directory fsync.
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an inter-process lock for one ledger compare-and-set operation."""

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# Administrative identity migration uses the exact same lock as budget
# validation/consumption.  Keeping this as a named public alias makes the
# shared inter-process boundary explicit without exposing a second lock file
# or a production migration entry point.
exclusive_ledger_lock = _exclusive_file_lock


class FormalGoalLedger:
    """Persistent global maximum-one ledger with atomic state transitions."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        certification_identity: str | None = None,
        frozen_goal_sha256: str | None = None,
        formal_runner_sha256: str | None = None,
    ) -> None:
        identity = default_ledger_identity()
        self.path = Path(path or CANONICAL_LEDGER_PATH).resolve()
        self.identity = {
            "certification_identity": certification_identity or identity["certification_identity"],
            "frozen_goal_sha256": frozen_goal_sha256 or identity["frozen_goal_sha256"],
            "formal_runner_sha256": formal_runner_sha256 or identity["formal_runner_sha256"],
        }
        with _exclusive_file_lock(self.path):
            if self.path.exists():
                self.state = self._read_unlocked()
            else:
                self.state = self._initial_state()
                atomic_write_json(self.path, self.state)
            self._validate(self.state)
            self._validate_identity(self.state)

    @property
    def is_canonical(self) -> bool:
        return self.path == CANONICAL_LEDGER_PATH

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            **self.identity,
            "maximum": 1,
            "consumed": 0,
            "goal_budget_consumed": False,
            "send_binding_committed": False,
            "transport_send_invocation_observed": False,
            "send_goal_async_call_count": 0,
            "send_attempted": False,
            "created_utc": _utc_now(),
            "consumed_utc": None,
            "send_binding_committed_utc": None,
            "transport_send_invocation_observed_utc": None,
            "goal_request_acknowledged": False,
            "goal_accepted": False,
            "formal_goal_budget": {
                "maximum": 1,
                "consumed": 0,
                "goal_budget_consumed": False,
                "send_attempted": False,
                "send_binding_committed": False,
                "transport_send_invocation_observed": False,
                "send_goal_async_call_count": 0,
            },
        }

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"formal goal ledger is unreadable: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("formal goal ledger root must be an object")
        return payload

    def _validate_identity(self, state: Mapping[str, Any]) -> None:
        for key, expected in self.identity.items():
            if state.get(key) != expected:
                raise RuntimeError(f"formal goal ledger identity mismatch: {key}")

    @staticmethod
    def _budget(state: Mapping[str, Any]) -> Mapping[str, Any]:
        budget = state.get("formal_goal_budget")
        if not isinstance(budget, Mapping):
            raise RuntimeError("formal goal ledger budget is missing")
        return budget

    def _validate(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise RuntimeError("formal goal ledger schema mismatch")
        budget = self._budget(state)
        if state.get("maximum") != 1 or budget.get("maximum") != 1:
            raise RuntimeError("formal goal ledger maximum must remain exactly one")
        consumed = budget.get("consumed")
        if consumed not in (0, 1) or state.get("consumed") != consumed:
            raise RuntimeError("formal goal ledger consumed state is corrupt")
        if budget.get("goal_budget_consumed") is not (consumed == 1):
            raise RuntimeError("formal goal budget consumption flag disagrees with consumed")
        for key in ("send_attempted", "send_binding_committed", "transport_send_invocation_observed"):
            if budget.get(key) not in (True, False) or state.get(key) is not budget.get(key):
                raise RuntimeError(f"formal goal ledger {key} is corrupt")
        if budget.get("send_attempted") and consumed != 1:
            raise RuntimeError("formal goal ledger is not fail-closed")
        if budget.get("send_binding_committed") and not budget.get("send_attempted"):
            raise RuntimeError("formal goal ledger records binding without send attempt")
        if budget.get("transport_send_invocation_observed") and not budget.get("send_binding_committed"):
            raise RuntimeError("formal goal ledger records invocation without binding")
        call_count = budget.get("send_goal_async_call_count")
        if call_count not in (0, 1) or isinstance(call_count, bool) or state.get("send_goal_async_call_count") != call_count:
            raise RuntimeError("formal goal ledger send call count is corrupt")
        if call_count == 1 and not budget.get("transport_send_invocation_observed"):
            raise RuntimeError("formal goal ledger records send without invocation observation")
        if consumed == 1 and not state.get("consumed_utc"):
            raise RuntimeError("consumed ledger is missing its durable timestamp")

    def _mutate(self, operation: str) -> dict[str, Any]:
        with _exclusive_file_lock(self.path):
            self.state = self._read_unlocked()
            self._validate(self.state)
            self._validate_identity(self.state)
            budget = self.state["formal_goal_budget"]
            now = _utc_now()
            if operation == "consume":
                if budget["consumed"] != 0 or budget["send_attempted"]:
                    raise RuntimeError("formal goal budget already consumed; retry is forbidden")
                budget["consumed"] = 1
                budget["goal_budget_consumed"] = True
                self.state["consumed"] = 1
                self.state["goal_budget_consumed"] = True
                self.state["consumed_utc"] = now
            elif operation == "bind":
                if budget["consumed"] != 1 or budget.get("goal_budget_consumed") is not True:
                    raise RuntimeError("send call cannot be bound before ledger consumption")
                if budget["send_binding_committed"] or budget["send_goal_async_call_count"] != 0:
                    raise RuntimeError("send call is already bound; retry is forbidden")
                budget["send_attempted"] = True
                budget["send_binding_committed"] = True
                self.state["send_attempted"] = True
                self.state["send_binding_committed"] = True
                self.state["send_binding_committed_utc"] = now
            elif operation == "invoke":
                if not budget["send_binding_committed"] or budget["send_goal_async_call_count"] != 0:
                    raise RuntimeError("transport invocation cannot be recorded before the single send binding")
                budget["transport_send_invocation_observed"] = True
                budget["send_goal_async_call_count"] = 1
                self.state["transport_send_invocation_observed"] = True
                self.state["send_goal_async_call_count"] = 1
                self.state["transport_send_invocation_observed_utc"] = now
            else:  # pragma: no cover - internal programming guard
                raise ValueError(operation)
            atomic_write_json(self.path, self.state)
            reread = self._read_unlocked()
            self._validate(reread)
            self._validate_identity(reread)
            self.state = reread
            return self.snapshot()

    def consume_before_send(self) -> dict[str, Any]:
        return self._mutate("consume")

    def bind_send_goal_async_call(self) -> dict[str, Any]:
        return self._mutate("bind")

    def record_transport_send_invocation(self) -> dict[str, Any]:
        return self._mutate("invoke")

    def record_goal_response(self, *, acknowledged: bool, accepted: bool | None) -> dict[str, Any]:
        with _exclusive_file_lock(self.path):
            self.state = self._read_unlocked()
            self._validate(self.state)
            self._validate_identity(self.state)
            budget = self.state["formal_goal_budget"]
            if not budget.get("transport_send_invocation_observed"):
                raise RuntimeError("goal response cannot be recorded before transport invocation")
            budget["goal_request_acknowledged"] = bool(acknowledged)
            budget["goal_accepted"] = bool(accepted) if accepted is not None else False
            self.state["goal_request_acknowledged"] = budget["goal_request_acknowledged"]
            self.state["goal_accepted"] = budget["goal_accepted"]
            self.state["goal_response_recorded_utc"] = _utc_now()
            atomic_write_json(self.path, self.state)
            self.state = self._read_unlocked()
            self._validate(self.state)
            return self.snapshot()

    def reread(self) -> dict[str, Any]:
        with _exclusive_file_lock(self.path):
            self.state = self._read_unlocked()
            self._validate(self.state)
            self._validate_identity(self.state)
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.state))
