"""Static regression tests for the formal Stage 2.5 parameter contract."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage25_contract import (  # noqa: E402
    FORMAL_RUCKIG_PARAMETERS,
    FORMAL_TOTG_PARAMETERS,
    REPAIR_REASON,
)


def source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_formal_stage25_path_tolerance_is_002():
    assert FORMAL_TOTG_PARAMETERS == {
        "velocity_scaling_factor": 0.15,
        "acceleration_scaling_factor": 0.15,
        "path_tolerance": 0.002,
        "resample_dt": 0.01,
        "min_angle_change": 0.0005,
    }
    assert REPAIR_REASON["previous_path_tolerance_rad"] == 0.01
    assert REPAIR_REASON["selected_path_tolerance_rad"] == 0.002


def test_launch_runner_and_formal_driver_share_contract():
    driver = source("scripts/run_stage25.py")
    launch = source("tools/stage25_moveit_launch.py")
    runner = source("tools/stage25_moveit_runner.py")
    for text in (driver, launch, runner):
        assert "FORMAL_TOTG_PARAMETERS" in text
        assert "FORMAL_RUCKIG_PARAMETERS" in text
    assert "path_tolerance:=0.01" not in driver
    assert 'declare_parameter("path_tolerance", 0.01)' not in runner
    assert 'DeclareLaunchArgument("path_tolerance", default_value="0.01")' not in launch


def test_formal_gate_records_effective_parameters_and_repair():
    driver = source("scripts/run_stage25.py")
    assert '"formal_totg_parameters"' in driver
    assert '"formal_ruckig_parameters"' in driver
    assert '"repair_reason"' in driver
    assert '"samples_over_6mm"' in source("tools/stage25_moveit_runner.py")


def test_pose_threshold_is_not_relaxed():
    runner = source("tools/stage25_moveit_runner.py")
    assert "position_errors <= 6.0" in runner
    assert '"position_constraint_limit_mm": 6.0' in runner
    assert "np.all(position_errors <= 6.1" not in runner
    assert "6.066" not in runner


def test_stage24t_input_is_read_only_in_formal_driver():
    driver = source("scripts/run_stage25.py")
    assert "no_stage24t_rebuild_requested" in driver
    assert "no_ik_or_segmentation_mutation" in driver
    assert "STAGE24T / \"stage24t_input_manifest.json\"" in driver
