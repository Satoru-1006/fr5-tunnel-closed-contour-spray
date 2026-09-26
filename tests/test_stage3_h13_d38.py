import csv
from pathlib import Path

import numpy as np

from src.stage3_h11_dataset import POSITION_LOWER, POSITION_UPPER
from tools import stage3_h13_d36_system_evaluation as d36
from tools import stage3_h13_d38_rollout_stabilization as d38


ROOT = Path(__file__).resolve().parents[1]


def test_matched_teacher_forced_and_self_fed_rollouts_diverge_without_future_reads():
    case = d38.make_case(d38.load_channels())
    model = d38.load_model(d36.PANEL[-1][-1], trainable=False)
    teacher_forced = d38.trace_rollout(model, case, "TF")
    self_fed = d38.trace_rollout(model, case, "SF")

    assert not np.allclose(teacher_forced.full_positions[16:], self_fed.full_positions[16:])
    assert d38.mode_metrics(self_fed, case)["future_reference_joint_rows_read"] == 0


def test_authentic_bounds_funnel_detects_first_violation_and_shield_is_causal():
    case = d38.make_case(d38.load_channels())
    midpoint = np.tile((POSITION_LOWER + POSITION_UPPER) / 2.0, (181, 1))
    legal = d38.candidate_metrics(midpoint, case)
    assert legal["full_181_bounds_pass"] is True
    assert legal["bounds_valid_rows"] == 181

    invalid = midpoint.copy()
    invalid[35, 3] = POSITION_UPPER[3] + 0.1
    invalid_metrics = d38.candidate_metrics(invalid, case)
    assert invalid_metrics["full_181_bounds_pass"] is False
    assert invalid_metrics["first_bound_failure_index"] == 35

    no_future_case = d38.AuthoritativeCase(
        positions=np.full_like(case.positions, np.nan),
        times=case.times,
        features=case.features,
        channels=case.channels,
    )
    shielded = d38.feasibility_shield_candidate(midpoint, no_future_case)
    assert np.isfinite(shielded).all()
    assert d38.candidate_metrics(shielded, case)["full_181_bounds_pass"] is True


def test_feature_shift_inverts_model_normalization_before_raw_range_comparison():
    case = d38.make_case(d38.load_channels())
    trace = d38.Trace(
        mode="TF",
        full_positions=case.positions.copy(),
        residuals=np.zeros((165, 6)),
        input_last_rows=case.features[16:].astype(np.float64),
        hidden=np.zeros((165, 128)),
        hidden_norms=np.zeros(165),
        hidden_delta_norms=np.zeros(165),
    )
    rows = d38.feature_shift_rows("test", case, {"TF": trace})
    q1 = next(row for row in rows if row["feature"] == "planned_joint_position_1")
    assert np.isclose(q1["stage01_mean"], np.mean(case.positions[:, 0]))


def test_strict_certification_requires_every_gate_and_frozen_checkpoints_are_external():
    gates = {name: True for name in d38.CERTIFICATION_GATE_ORDER}
    assert d38.certification_passes(gates) is True
    gates["post_ruckig_collision"] = False
    assert d38.certification_passes(gates) is False
    assert d38.first_failed_certification_gate(gates) == "post_ruckig_collision"

    d38_output = ROOT / "outputs" / "stage3_h13_d38_rollout_stabilization"
    for update, _stage, _h1, _h32, checkpoint in d36.PANEL:
        assert Path(checkpoint).is_file()
        assert d38_output not in Path(checkpoint).parents


def test_corrected_native_certification_logs_are_on_state_only():
    csv_path = ROOT / "outputs" / "stage3_h13_d38_rollout_stabilization" / "D38_ROBOT_CERTIFICATION.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    assert rows
    assert all(row["native_invocation_mode"] == "ON_STATE_ONLY_SEGMENTED_EXECUTION_FALSE" for row in rows)
    assert all("spray-off reorientation transitions" not in row["stderr_tail"] for row in rows)
    assert all(row["FULL_ROBOT_CERTIFICATION"] == "FAIL" for row in rows)
