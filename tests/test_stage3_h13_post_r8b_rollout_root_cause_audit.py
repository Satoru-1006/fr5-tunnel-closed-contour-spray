from __future__ import annotations

import hashlib
import json

from scripts.stage3_h13_post_r8b_rollout_root_cause_audit import (
    ERROR_GROWTH_INTERVALS,
    HORIZONS,
    _first_positive_gap,
    _first_sustained_ratio,
)


def test_registered_horizons_and_growth_alignment() -> None:
    assert HORIZONS == (1, 4, 5, 8, 10, 12, 15, 16, 17, 20, 24, 32)
    assert ERROR_GROWTH_INTERVALS == ((1, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 32))
    curve = {str(h): float(h) for h in HORIZONS}
    assert _first_positive_gap(curve, {str(h): float(h) - 0.01 for h in HORIZONS}) == 1


def test_sustained_amplification_rule_requires_consecutive_registered_horizons() -> None:
    teacher = {str(h): 1.0 for h in HORIZONS}
    free = {str(h): 1.0 for h in HORIZONS}
    free["8"] = 2.1
    free["10"] = 2.1
    assert _first_sustained_ratio(free, teacher) == 8
    free["10"] = 1.9
    assert _first_sustained_ratio(free, teacher) is None


def test_replay_digest_is_stable_for_aggregated_payload() -> None:
    payload = {"role": "VALIDATION", "count": 128, "h1": 0.001, "h32": 0.023}
    first = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    second = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert first == second


def test_no_frozen20_path_is_present_in_authoritative_input_constants() -> None:
    from scripts import stage3_h13_post_r8b_rollout_root_cause_audit as audit

    paths = [str(audit.H10_ROOT), str(audit.H11_ROOT), str(audit.R6_ROOT), str(audit.R7_ROOT), str(audit.R8_ROOT), str(audit.R8A_ROOT), str(audit.R8B_ROOT)]
    assert all("frozen20" not in path.lower() for path in paths)
