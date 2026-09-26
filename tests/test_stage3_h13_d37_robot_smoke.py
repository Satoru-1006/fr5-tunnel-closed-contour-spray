import json
from pathlib import Path
from tools.stage3_h13_d37_robot_smoke import canonicalize

ROOT=Path(__file__).resolve().parents[1]
def test_canonical_waterfall_is_fail_closed(tmp_path):
    result=canonicalize(ROOT/"outputs/stage3_h13_d37_robot_smoke",tmp_path)
    assert result["checkpoint_fully_certified"]=="0/5"
    assert result["baseline_control"].startswith("1/1 PASS")
    assert result["post_processing_collision_recheck"]=="NOT_REACHED"
    assert all(r["first_failure_reason"]=="joint_bounds_violation" for r in result["results"])
    assert json.loads((tmp_path/"D37_SUMMARY.json").read_text())["clearance"] is None
