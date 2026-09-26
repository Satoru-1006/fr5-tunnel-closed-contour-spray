from __future__ import annotations

import argparse
import datetime as dt
import math
import shlex
from pathlib import Path

import yaml


def _vector(values: object, name: str) -> list[float]:
    if not isinstance(values, list | tuple) or len(values) != 3:
        raise ValueError(f"{name} must be a 3-element list.")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite numeric values.")
    return result


def _fmt(values: list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _required_text(tool_tcp: dict, key: str) -> str:
    value = str(tool_tcp.get(key, "")).strip()
    if not value:
        raise ValueError(f"tool_tcp.{key} is required for production measured TCP calibration.")
    return value


def load_calibration(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tool_tcp"), dict):
        raise ValueError("Calibration YAML must contain a tool_tcp mapping.")
    tool_tcp = data["tool_tcp"]
    source = str(tool_tcp.get("source", "")).strip()
    source_lower = source.lower()
    if not source or source_lower in {"measured_nozzle_tcp_required", "unknown"}:
        raise ValueError("tool_tcp.source must identify a measured calibration.")
    if "assumed" in source_lower or "placeholder" in source_lower or "required" in source_lower:
        raise ValueError(f"tool_tcp.source is not production measured: {source!r}.")
    xyz = _vector(tool_tcp.get("translation_xyz"), "tool_tcp.translation_xyz")
    rpy = _vector(tool_tcp.get("rotation_rpy"), "tool_tcp.rotation_rpy")
    if all(math.isclose(value, 0.0, abs_tol=1e-12) for value in xyz + rpy):
        raise ValueError("Measured tool TCP must not be exactly zero translation and zero rotation.")
    _required_text(tool_tcp, "measured_by")
    measured_date = _required_text(tool_tcp, "measured_date")
    try:
        dt.date.fromisoformat(measured_date)
    except ValueError as exc:
        raise ValueError("tool_tcp.measured_date must use YYYY-MM-DD format.") from exc
    _required_text(tool_tcp, "calibration_method")
    return {
        "TOOL_TCP_XYZ": _fmt(xyz),
        "TOOL_TCP_RPY": _fmt(rpy),
        "TOOL_TCP_SOURCE": source,
        "TOOL_TCP_MEASURED_BY": str(tool_tcp["measured_by"]).strip(),
        "TOOL_TCP_MEASURED_DATE": measured_date,
        "TOOL_TCP_CALIBRATION_METHOD": str(tool_tcp["calibration_method"]).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a measured tool TCP calibration YAML for strict validation.")
    parser.add_argument("calibration_yaml", type=Path)
    parser.add_argument("--format", choices=["shell", "env"], default="shell")
    args = parser.parse_args()

    values = load_calibration(args.calibration_yaml)
    if args.format == "env":
        for key, value in values.items():
            print(f"{key}={value}")
    else:
        for key, value in values.items():
            print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
