"""Focused D49 verifier and campaign-driver contract tests."""

from pathlib import Path

from tools.stage4d_continuous_self_collision import MANIFEST, read_manifest, aggregate


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCE = ROOT / "cpp" / "stage4d_continuous_self_collision" / "stage4d_continuous_self_collision.cpp"


def test_d49_manifest_is_the_frozen_d48_twelve_case_set() -> None:
    cases = read_manifest(MANIFEST)
    assert len(cases) == 12
    assert {case.family for case in cases} == {
        "REGRESSION", "NORMAL", "BOUNDARY", "COLLISION_SENSITIVE", "ADVERSARIAL", "PERTURBATION"
    }
    assert all("C3_TIMING_SHADOW/ultra_slow_005/native_postprocess/trajectories" in case.trajectory_csv for case in cases)


def test_partial_campaign_cannot_certify_continuous_collision(tmp_path: Path) -> None:
    result = aggregate(
        [{
            "case_id": "smoke",
            "family": "TEST",
            "return_code": 0,
            "output": str(tmp_path / "smoke"),
            "complete": False,
            "status": "PASS_PARTIAL_SMOKE",
            "summary": {
                "complete_trajectory": False,
                "swept_interval_count": 1,
                "swept_pair_calls": 4,
                "broadphase_rejected_pairs": 2,
                "continuous_collision_count": 0,
                "continuous_api_error_count": 0,
                "endpoint_contact_count": 0,
                "checked_link_pairs": [],
            },
        }],
        tmp_path / "campaign",
        max_workers=1,
        max_segments=1,
    )
    assert result["status"] == "INCOMPLETE_OR_CONTACT"
    assert result["continuous_self_collision_status"] == "NOT_CERTIFIED"


def test_native_verifier_uses_exact_swept_fcl_and_acm_contract() -> None:
    source = VERIFIER_SOURCE.read_text(encoding="utf-8")
    for token in (
        "fcl::continuousCollide",
        "fcl::CCDM_SCREW",
        "PlanningScene",
        "getAllowedCollisionMatrix",
        "complete_trajectory",
        'required_column(columns, "j" + std::to_string(joint + 1) + "_q")',
        "penetration_depth_m",
        "continuous_return",
    ):
        assert token in source
