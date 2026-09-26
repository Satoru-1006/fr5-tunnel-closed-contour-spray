"""Offline guards for the Stage 2.8S-R2-H1 capture gate."""

from __future__ import annotations

from scripts.stage28sr2_h1_capture import controller_gap_gate


def test_controller_gap_is_accepted_only_with_publisher_evidence() -> None:
    assert controller_gap_gate({"long_callback_gap_over_100ms": {"count": 0}})
    assert controller_gap_gate(
        {
            "long_callback_gap_over_100ms": {"count": 3},
            "publisher_vs_recorder_gap_attribution": "publisher_did_not_publish_supported_by_synchronized_topics",
        }
    )


def test_unattributed_controller_gap_fails_closed() -> None:
    assert not controller_gap_gate(
        {
            "long_callback_gap_over_100ms": {"count": 1},
            "publisher_vs_recorder_gap_attribution": "not_available",
        }
    )

