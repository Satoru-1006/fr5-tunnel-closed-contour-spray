"""Run the H4.5 derived robot-model candidate through the frozen H4.2.1 worker.

This entry point is additive.  It changes only the URDF path used by the
standalone deterministic IK bridge; the caller supplies a matching MoveIt
robot_description through the derived launch file.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import stage3_h4_2 as h42  # noqa: E402
import stage3_h4_2_1 as h421  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--replay-index", type=int, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--h3-targets", type=Path, required=True)
    parser.add_argument("--h2-manifest", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--srdf", type=Path, required=True)
    # ros2 launch appends --ros-args/--params-file to executable nodes.  These
    # parameters configure MoveItPy through the launch action and are not
    # worker payload arguments, so accept and record them without interpreting
    # them here.
    args, _ros_args = parser.parse_known_args()

    # h42.worker_result receives the URDF argument for validation, but its
    # frozen bridge helper reads h41.URDF.  Redirect that helper explicitly so
    # this candidate cannot accidentally use the frozen file.
    h42.h41.URDF = args.urdf.resolve()
    h42.h41.SRDF = args.srdf.resolve()
    return h421.run_clean_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
