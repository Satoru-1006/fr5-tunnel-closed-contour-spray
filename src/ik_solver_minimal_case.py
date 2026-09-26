"""Public Stage 1.9.2 minimal-case loader."""

from __future__ import annotations

import json
from pathlib import Path

from src.ik_solver_binary_snapshot import TARGET_CALL_ID


def load_minimal_case(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("target_call_id", data.get("first_differing_record_id")) != TARGET_CALL_ID:
        raise ValueError(f"expected {TARGET_CALL_ID}")
    return data
