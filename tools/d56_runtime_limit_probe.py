"""D56 runtime MoveIt model/limit probe.

This is an evidence-only node. It uses the same MoveItConfigsBuilder inputs as
the D52 stateful Ruckig launch and writes only to the caller-provided D56
output directory.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

from moveit.planning import MoveItPy


GROUP = "fairino5_v6_group"
JOINTS = [f"j{i}" for i in range(1, 7)]


def _first(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError("empty variable bounds")
        return value[0]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args()
    output_value = str(args.output_dir)
    if not output_value:
        raise RuntimeError("output_dir is required")
    output = Path(output_value).resolve()
    output.mkdir(parents=True, exist_ok=True)
    moveit = None
    status = "BLOCKED"
    result: dict[str, object] = {
        "schema_version": "d56-runtime-limit-probe-v1",
        "scope": "Stage 0/1 ON-state open-arch; D56 shadow evidence",
        "collision_method": "adaptive_discrete_interpolation",
        "fallback_defaults_used": None,
        "runtime_bounds": {},
        "errors": [],
    }
    try:
        moveit = MoveItPy(node_name="d56_runtime_limit_probe")
        model = moveit.get_robot_model()
        group = model.get_joint_model_group(GROUP)
        active = list(group.active_joint_model_names)
        result["robot_model_name"] = str(model.name)
        result["planning_group"] = GROUP
        result["active_joint_names"] = active
        if active != JOINTS:
            result["errors"].append(f"joint_order_mismatch:{active}")
        bounds: dict[str, object] = {}
        for joint, raw in zip(active, group.active_joint_model_bounds):
            item = _first(raw)
            row = {
                "position_lower_rad": float(item.min_position),
                "position_upper_rad": float(item.max_position),
                "velocity_rad_s": float(item.max_velocity),
                "acceleration_rad_s2": float(item.max_acceleration),
                "jerk_rad_s3": float(item.max_jerk),
                "position_bounded": bool(item.position_bounded),
                "velocity_bounded": bool(item.velocity_bounded),
                "acceleration_bounded": bool(item.acceleration_bounded),
                "jerk_bounded": bool(item.jerk_bounded),
            }
            numeric = [value for key, value in row.items() if key.endswith(("_rad", "_rad_s", "_rad_s2", "_rad_s3"))]
            if not all(math.isfinite(float(value)) for value in numeric):
                result["errors"].append(f"nonfinite_bound:{joint}")
            bounds[joint] = row
        result["runtime_bounds"] = bounds
        result["ruckig_input_mapping"] = {
            "source": "MoveIt2 RuckigSmoothing::getRobotModelBounds",
            "velocity": "max_velocity_scaling_factor * VariableBounds.max_velocity",
            "acceleration": "max_acceleration_scaling_factor * VariableBounds.max_acceleration",
            "jerk": "VariableBounds.max_jerk without velocity/acceleration scaling",
            "active_jerk_limit_rad_s3": [bounds[j]["jerk_rad_s3"] for j in JOINTS],
        }
        result["fallback_defaults_used"] = False
        result["status"] = "PASS" if not result["errors"] else "BLOCKED"
        status = str(result["status"])
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}:{exc}")
        result["status"] = "BLOCKED"
    finally:
        # Persist the runtime evidence before touching MoveIt teardown.  The
        # teardown path is itself under investigation and must not erase the
        # useful limit/mapping evidence if it terminates the process.
        output.joinpath("D56_RUNTIME_LIMITS.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if moveit is not None:
            # MoveItPy owns the rclcpp context in this process.  There is no
            # Python-side rclpy Node/context to shut down afterward.
            del moveit
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
