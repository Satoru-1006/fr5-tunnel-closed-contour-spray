from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .robot_model import FR5Robot


@dataclass(frozen=True)
class FairinoFR5Assets:
    repo_root: Path
    urdf: Path
    moveit_config: Path
    joint_limits: Path
    srdf: Path
    kinematics: Path


def find_official_fr5_assets(project_root: Path) -> FairinoFR5Assets | None:
    repo = project_root / "external" / "frcobot_ros2"
    urdf = repo / "fairino_description" / "urdf" / "fairino5_v6.urdf"
    moveit = repo / "fairino5_v6_moveit2_config"
    joint_limits = moveit / "config" / "joint_limits.yaml"
    srdf = moveit / "config" / "fairino5_v6_robot.srdf"
    kinematics = moveit / "config" / "kinematics.yaml"
    required = [urdf, moveit, joint_limits, srdf, kinematics]
    if all(path.exists() for path in required):
        return FairinoFR5Assets(repo, urdf, moveit, joint_limits, srdf, kinematics)
    return None


def load_official_fr5_robot(project_root: Path) -> tuple[FR5Robot, FairinoFR5Assets]:
    assets = find_official_fr5_assets(project_root)
    if assets is None:
        raise FileNotFoundError(
            "Official FAIRINO FR5 assets not found. Expected external/frcobot_ros2 with "
            "fairino_description/urdf/fairino5_v6.urdf and fairino5_v6_moveit2_config."
        )
    return FR5Robot.from_official_urdf(assets.urdf, assets.joint_limits), assets
