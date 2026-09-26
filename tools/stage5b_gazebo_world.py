"""Generate an isolated Gazebo SDF world from a Stage5B environment JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment_json", type=Path)
    parser.add_argument("output_sdf", type=Path)
    args = parser.parse_args()
    environment = json.loads(args.environment_json.read_text(encoding="utf-8"))
    models = []
    for index, obj in enumerate(environment.get("objects", [])):
        center = obj["center_m"]
        dims = obj["dimensions_m"]
        qx, qy, qz, qw = obj["quaternion_xyzw"]
        models.append(
            f'''    <model name="{obj.get("id", f"stage5b_wall_{index:03d}")}">
      <static>true</static>
      <pose>{center[0]} {center[1]} {center[2]} 0 0 0</pose>
      <link name="wall_link">
        <collision name="collision">
          <pose>0 0 0 0 0 0</pose>
          <geometry><box><size>{dims[0]} {dims[1]} {dims[2]}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{dims[0]} {dims[1]} {dims[2]}</size></box></geometry>
          <material><ambient>0.32 0.32 0.36 1</ambient><diffuse>0.45 0.45 0.50 1</diffuse></material>
        </visual>
      </link>
    </model>'''
        )
        # SDF box poses carry Euler angles; convert the source quaternion in a
        # tiny local helper below rather than silently dropping orientation.
        import math
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        models[-1] = models[-1].replace(f"{center[0]} {center[1]} {center[2]} 0 0 0", f"{center[0]} {center[1]} {center[2]} {roll} {pitch} {yaw}")
    sdf = """<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="stage5b_shadow_world">
    <gravity>0 0 -9.81</gravity>
    <!-- Match the ros2_control update period so the shadow backend remains
         serviceable while still running a deterministic fixed-step simulation. -->
    <physics name="ode" type="ode"><max_step_size>0.01</max_step_size><real_time_factor>1</real_time_factor></physics>
    <scene><ambient>0.4 0.4 0.4 1</ambient><background>0.08 0.08 0.10 1</background></scene>
""" + "\n".join(models) + "\n  </world>\n</sdf>\n"
    args.output_sdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_sdf.write_text(sdf, encoding="utf-8")
    print(json.dumps({"status": "GENERATED_SHADOW_WORLD", "objects": len(models), "output": str(args.output_sdf)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
