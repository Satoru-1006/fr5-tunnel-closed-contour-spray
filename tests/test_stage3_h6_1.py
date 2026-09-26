from __future__ import annotations

from pathlib import Path

from scripts import stage3_h6_1_recertification as h61


def test_h6_1_contract_is_defined_and_hashed_before_use() -> None:
    contract = h61.load_json(h61.CONTRACT)
    info = h61.validate_contract(contract)
    assert info["valid"] is True
    assert info["contract_sha256"] == "9ad4ef2e78b59881c5fe612216474f77f7cfa0ea7d76706be9741b98f2040487"
    assert contract["tcp_link"] == "spray_tcp_link"
    assert contract["geometry"]["standoff_min_m"] == 0.25
    assert contract["geometry"]["standoff_max_m"] == 0.27


def test_h6_1_provenance_finds_no_prior_h6_authoritative_process_tolerance() -> None:
    audit = h61.build_provenance_audit(h61.load_json(h61.CONTRACT))
    assert audit["formal_spray_process_tolerance_found_before_h6_1"] is False
    assert audit["prior_h6_authoritative_tolerance_found"] is False
    assert audit["h5_formal_tcp_tolerance_applied_values"] == [False]


def test_h6_1_frozen_h6_path_has_628_on_and_1295_total_waypoints() -> None:
    rows = h61.frozen_full_waypoint_rows()
    assert len(rows) == 1295
    assert sum(row["spray_state"] == "SPRAY_ON" for row in rows) == 628
    assert sum(row["spray_state"] == "SPRAY_OFF" for row in rows) == 667


def test_shortest_orientation_error_is_sign_invariant() -> None:
    assert h61.quat_shortest_angle_deg([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]) == 0.0
