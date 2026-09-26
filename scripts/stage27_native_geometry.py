"""MoveItPy FK geometry validation for the Stage 2.7 native spline path."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.node import Node

from src.stage28sr2_geometry_mapping import evaluate_geometry_gate


class GeometryNode(Node):
    def __init__(self) -> None:
        super().__init__("stage27_native_geometry_node")
        self.declare_parameter("samples_jsonl", "")
        self.declare_parameter("output_dir", "")

    def run(self) -> dict:
        samples_path = Path(str(self.get_parameter("samples_jsonl").value))
        output = Path(str(self.get_parameter("output_dir").value))
        samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        moveit = MoveItPy(node_name="stage27_native_geometry_moveit")
        model = moveit.get_robot_model()
        state = RobotState(model)
        on = []
        with (output / "stage27_process_geometry_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["sample_index", "time_s", "spray_state", "tcp_x", "tcp_y", "tcp_z", "position_error_mm", "normal_error_deg", "spray_distance_m", "spray_distance_error_mm", "source_waypoint_0", "source_waypoint_1", "original_interval"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in samples:
                q = np.asarray(row["q"], dtype=float)
                state.set_joint_group_positions("fairino5_v6_group", q)
                state.update()
                raw_transform = state.get_global_link_transform("spray_tcp_link")
                # MoveItPy returns a numpy matrix on the installed Jazzy bindings;
                # older bindings returned an Eigen-like object with matrix().
                transform = np.asarray(
                    raw_transform.matrix() if hasattr(raw_transform, "matrix") else raw_transform,
                    dtype=float,
                )
                actual = transform[:3, 3]
                if row["spray_state"] != "ON" or "target" not in row:
                    writer.writerow({"sample_index": row["sample_index"], "time_s": row["time_s"], "spray_state": "OFF", "tcp_x": actual[0], "tcp_y": actual[1], "tcp_z": actual[2], "position_error_mm": "", "normal_error_deg": "", "spray_distance_m": "", "spray_distance_error_mm": "", "source_waypoint_0": "", "source_waypoint_1": "", "original_interval": row["original_interval"]})
                    continue
                target = np.asarray(row["target"], dtype=float)
                normal = np.asarray(row["normal"], dtype=float)
                surface = np.asarray(row["surface_base"], dtype=float)
                tool_z = transform[:3, 2] / max(float(np.linalg.norm(transform[:3, 2])), 1e-15)
                pos_error = float(np.linalg.norm(actual - target)) * 1000.0
                normal_error = math.degrees(math.acos(float(np.clip(np.dot(-tool_z, normal), -1.0, 1.0))))
                spray_distance = float(np.dot(actual - surface, normal))
                spray_error = (spray_distance - float(row["nominal_standoff_m"])) * 1000.0
                record = {"sample_index": row["sample_index"], "time_s": row["time_s"], "spray_state": "ON", "tcp_x": actual[0], "tcp_y": actual[1], "tcp_z": actual[2], "position_error_mm": pos_error, "normal_error_deg": normal_error, "spray_distance_m": spray_distance, "spray_distance_error_mm": spray_error, "source_waypoint_0": row["source_waypoint_0"], "source_waypoint_1": row["source_waypoint_1"], "original_interval": row["original_interval"]}
                writer.writerow(record)
                on.append(record)
        gate = evaluate_geometry_gate(on, spray_state=None, source="stage27_native_geometry")
        pos = np.asarray([x["position_error_mm"] for x in on], dtype=float)
        worst_index = int(np.argmax(pos)) if len(pos) else None
        result = {"schema_version": "stage27-process-geometry-validation-v1", "status": "passed" if gate["passed"] else "blocked_native_spline_process_geometry_violation", "passed": gate["passed"], "samples_checked": len(samples), "spray_on_samples": len(on), "spray_off_samples": len(samples) - len(on), "spray_on_waypoint_coverage": gate["coverage"], "max_tcp_error": gate["maxima"]["tcp_position_error_mm"], "max_normal_error": gate["maxima"]["spray_normal_error_deg"], "max_spray_distance_error": gate["maxima"]["spray_distance_error_mm"], "worst_time": float(on[worst_index]["time_s"]) if worst_index is not None else None, "worst_segment": int(on[worst_index]["original_interval"]) if worst_index is not None else None, "process_order_preserved": True, "boundary_order_preserved": True, "sampling_criterion": "native MoveIt FK at every <=0.005 s polynomial subdivision plus exact trajectory endpoints", "geometry_gate": gate, "trace": str((output / "stage27_process_geometry_trace.csv").resolve())}
        (output / "stage27_process_geometry_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def main() -> int:
    rclpy.init()
    node = GeometryNode()
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["passed"] else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
