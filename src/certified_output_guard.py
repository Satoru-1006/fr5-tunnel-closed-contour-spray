"""Fail-closed guard for frozen certification output trees.

Certified outputs opt in by adding a ``.certified_immutable`` marker at the
root.  Writers must call :func:`assert_output_path_writable` before creating,
opening, truncating, appending, replacing, renaming, or removing a path.
Readers are intentionally unaffected.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MARKER_NAME = ".certified_immutable"
MARKER_SCHEMA = "certified_output_immutable_v1"
WRITE_POLICY = "deny_existing_output_mutation"
REFUSAL_TOKEN = "WRITE_REFUSED_BY_IMMUTABLE_GUARD"


class CertifiedOutputImmutableError(PermissionError):
    """Raised before a writer can mutate a certified immutable output."""


def _absolute_without_requiring_existence(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().absolute()


def immutable_marker_for(path: os.PathLike[str] | str) -> Path | None:
    """Return the nearest ancestor marker protecting *path*, if any."""

    target = _absolute_without_requiring_existence(path)
    current = target if target.is_dir() else target.parent
    while True:
        marker = current / MARKER_NAME
        if marker.is_file():
            return marker
        if current.parent == current:
            return None
        current = current.parent


def assert_not_inside_certified_immutable_root(
    path: os.PathLike[str] | str, *, operation: str = "write"
) -> None:
    marker = immutable_marker_for(path)
    if marker is None:
        return
    raise CertifiedOutputImmutableError(
        f"{REFUSAL_TOKEN}: operation={operation}; target={_absolute_without_requiring_existence(path)}; "
        f"marker={marker}"
    )


def assert_output_path_writable(
    path: os.PathLike[str] | str, *, operation: str = "write"
) -> None:
    """Fail before a potentially mutating output operation."""

    assert_not_inside_certified_immutable_root(path, operation=operation)


def build_immutable_marker_payload(
    root: os.PathLike[str] | str,
    *,
    reason: str,
    certificate: os.PathLike[str] | str,
    semantic_sha256: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_root = _absolute_without_requiring_existence(root)
    payload: dict[str, Any] = {
        "schema": MARKER_SCHEMA,
        "root": str(resolved_root),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "certificate": str(_absolute_without_requiring_existence(certificate)),
        "semantic_sha256": semantic_sha256,
        "write_policy": WRITE_POLICY,
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def install_immutable_marker(root: os.PathLike[str] | str, payload: Mapping[str, Any]) -> Path:
    """Install a new marker without overwriting an existing one."""

    resolved_root = _absolute_without_requiring_existence(root)
    if not resolved_root.is_dir():
        raise FileNotFoundError(resolved_root)
    marker = resolved_root / MARKER_NAME
    if marker.exists():
        raise FileExistsError(marker)
    if payload.get("schema") != MARKER_SCHEMA:
        raise ValueError("invalid certified-output marker schema")
    if Path(str(payload.get("root", ""))).absolute() != resolved_root:
        raise ValueError("marker root does not match target root")
    # Exclusive creation is intentional: no overwrite/replacement semantics.
    with marker.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return marker


def load_immutable_marker(root: os.PathLike[str] | str) -> dict[str, Any]:
    marker = _absolute_without_requiring_existence(root) / MARKER_NAME
    return json.loads(marker.read_text(encoding="utf-8"))
