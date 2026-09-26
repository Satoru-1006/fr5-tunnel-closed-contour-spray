import numpy as np

from src.stage3_h12_r7 import bounded_seeds, candidate_score, immutable_sample_hash, repair_eligible


def test_failed_sample_identification_is_exact_and_frozen():
    assert repair_eligible(spray_state="SPRAY_ON", segment_id=2, tcp_position_error_m=0.0010001)
    assert not repair_eligible(spray_state="SPRAY_ON", segment_id=2, tcp_position_error_m=0.001)
    assert not repair_eligible(spray_state="SPRAY_OFF", segment_id=2, tcp_position_error_m=0.1)
    assert not repair_eligible(spray_state="SPRAY_ON", segment_id=5, tcp_position_error_m=0.1)


def test_bounded_seed_order_is_deterministic_and_local():
    q = np.arange(24, dtype=float).reshape(4, 6)
    first = bounded_seeds(q, 2)
    second = bounded_seeds(q, 2)
    assert [name for name, _ in first] == ["current", "previous_neighbor_biased", "next_neighbor_biased"]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(first, second))
    assert len(first) == 3


def test_candidate_score_prefers_minimum_joint_correction_then_continuity():
    q = np.zeros(6)
    small = candidate_score(q, np.full(6, 0.01), q, q)
    large = candidate_score(q, np.full(6, 0.02), q, q)
    assert small < large


def test_immutable_sample_hash_ignores_only_declared_repairs():
    q = np.arange(18, dtype=float).reshape(3, 6)
    changed = q.copy(); changed[1] += 100.0
    assert immutable_sample_hash(q, [1]) == immutable_sample_hash(changed, [1])
    changed[0, 0] += 1.0
    assert immutable_sample_hash(q, [1]) != immutable_sample_hash(changed, [1])
