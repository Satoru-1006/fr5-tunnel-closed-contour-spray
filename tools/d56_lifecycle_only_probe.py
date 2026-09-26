"""D56 controlled-box lifecycle regression for the MoveItPy embedding.

This deliberately does not run a trajectory or repeat the runtime jerk probe.  It
only verifies that the locally built MoveItCpp/MoveItPy dependency graph can be
constructed and destroyed without the historical SIGSEGV.
"""

from __future__ import annotations

import gc
import json
import copy
from pathlib import Path

import moveit.planning as moveit_planning
from moveit_configs_utils import MoveItConfigsBuilder


ROOT = Path("/mnt/d/robotfucker")
OUT = ROOT / "outputs/D56_STAGE4B_SOFTWARE_CLOSURE/lifecycle_overlay_regression"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    root = str(ROOT)
    config = (
        MoveItConfigsBuilder("fairino5_v6_robot", package_name="fairino5_v6_moveit2_config")
        .robot_description(
            file_path=root
            + "/outputs/stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z/derived_robot_model.urdf"
        )
        .robot_description_semantic(file_path=root + "/ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf")
        .joint_limits(file_path=root + "/ros2_moveit_bridge/config/joint_limits_with_jerk.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    config_dict = config.to_dict()
    config_dict["planning_pipelines"] = {
        "pipeline_names": ["ompl"],
        "ompl": copy.deepcopy(config_dict.get("ompl", {})),
    }

    evidence: dict[str, object] = {
        "status": "BLOCKED",
        "planning_module": str(moveit_planning.__file__),
        "constructed": False,
        "destroyed": False,
        "moveit_libraries_before_destroy": [],
        "error": None,
    }
    moveit = None
    try:
        moveit = moveit_planning.MoveItPy(
            node_name="d56_lifecycle_only",
            config_dict=config_dict,
            provide_planning_service=False,
        )
        evidence["constructed"] = True
        evidence["moveit_libraries_before_destroy"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "moveit" in line and "r-xp" in line
            }
        )
        moveit = None
        gc.collect()
        evidence["destroyed"] = True
        evidence["status"] = "PASS"
    except Exception as exc:  # pragma: no cover - exercised by the native runtime
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (OUT / "D56_LIFECYCLE_ONLY.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
