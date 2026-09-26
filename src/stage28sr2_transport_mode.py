"""Fail-closed Fast DDS transport selection for the ZERO-GOAL A/B."""

from __future__ import annotations

import os
from typing import Any, Mapping


DEFAULT_UDP_SHM = "default_udp_shm"
UDP4_ONLY = "udp4_only"
CONFLICT_KEYS = (
    "FASTDDS_DEFAULT_PROFILES_FILE",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "RMW_FASTRTPS_USE_QOS_FROM_XML",
)


def build_transport_environment(mode: str, source: Mapping[str, str] | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    if mode not in {DEFAULT_UDP_SHM, UDP4_ONLY}:
        raise ValueError(f"unsupported transport mode: {mode}")
    environment = dict(source or os.environ)
    conflicts = {key: environment[key] for key in CONFLICT_KEYS if environment.get(key)}
    existing_builtin = environment.get("FASTDDS_BUILTIN_TRANSPORTS")
    if existing_builtin and mode == DEFAULT_UDP_SHM:
        conflicts["FASTDDS_BUILTIN_TRANSPORTS"] = existing_builtin
    if existing_builtin and mode == UDP4_ONLY and existing_builtin.upper() != "UDPV4":
        conflicts["FASTDDS_BUILTIN_TRANSPORTS"] = existing_builtin
    if conflicts:
        return environment, {
            "passed": False,
            "requested_transport_mode": mode,
            "first_blocker": "fastdds_transport_configuration_conflict",
            "conflicts": conflicts,
        }
    if mode == UDP4_ONLY:
        environment["FASTDDS_BUILTIN_TRANSPORTS"] = "UDPv4"
        effective = "Fast DDS builtin UDPv4 transport only; SHM builtin disabled"
    else:
        environment.pop("FASTDDS_BUILTIN_TRANSPORTS", None)
        effective = "Fast DDS default builtin transports (UDPv4 + SHM)"
    return environment, {
        "passed": True,
        "requested_transport_mode": mode,
        "effective_transport_configuration": effective,
        "evidence": {
            "FASTDDS_BUILTIN_TRANSPORTS": environment.get("FASTDDS_BUILTIN_TRANSPORTS"),
            "conflicting_xml_profile_variables": {},
            "configuration_source": "Fast DDS 2.14.6 builtin transport environment contract",
        },
    }


def shell_prefix(mode: str) -> str:
    if mode == UDP4_ONLY:
        return "export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; "
    if mode == DEFAULT_UDP_SHM:
        return "unset FASTDDS_BUILTIN_TRANSPORTS; "
    raise ValueError(f"unsupported transport mode: {mode}")
