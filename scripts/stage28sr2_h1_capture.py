"""Stage 2.8S-R2-H1 recorder remediation and non-formal capture probe.

The probe launches only the existing simulation mock/controller runtime and a
separate rosbag2 process.  It never imports or starts an FJT client and never
sends a goal.  Graph/QoS commands are restricted to the pre- and post-capture
snapshots; the capture interval contains no blocking graph probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28sr2_geometry_mapping import geometry_semantics_root_cause, source_waypoint_union  # noqa: E402

STAGE27 = ROOT / "outputs/stage27s_native_spline_remediation/stage27s_formal_20260805T130000Z"
ROBOT_DESCRIPTION_SOURCE = ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture/stage28sr_expanded_runtime.urdf"
STAGE27_TRACE = STAGE27 / "stage27_process_geometry_trace.csv"
HISTORICAL_STAGE28_TRACE = ROOT / "outputs/stage28sr_corrected_tf_fk_runtime_recapture/stage27_process_geometry_trace.csv"
TOPICS = ["/joint_states", "/tf", "/tf_static", "/fairino5_controller/controller_state"]
WSL_DISTRO = "Ubuntu-24.04-D"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wsl_path(path: Path) -> str:
    drive, rest = str(path.resolve()).replace("\\", "/").split(":/", 1)
    return f"/mnt/{drive.lower()}/{rest}"


def run_wsl(command: str, timeout_s: int = 40) -> dict[str, Any]:
    argv = ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc", command]
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        return {"command": command, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "exit_code": None, "stdout": error.stdout or "", "stderr": error.stderr or "", "timed_out": True}


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    result = {"root": str(root.resolve()), "manifest": str(manifest.resolve()), "entries": 0, "verified": 0, "missing": [], "mismatches": []}
    if not manifest.exists():
        result["passed"] = False
        return result
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})  (.+)$", line)
        if not match:
            continue
        expected, relative = match.group(1).lower(), match.group(2)
        result["entries"] += 1
        candidate = root / relative
        if not candidate.exists():
            result["missing"].append(relative)
            continue
        actual = sha256(candidate)
        if actual == expected:
            result["verified"] += 1
        else:
            result["mismatches"].append({"path": relative, "expected": expected, "actual": actual})
    result["passed"] = not result["missing"] and not result["mismatches"] and result["entries"] == result["verified"]
    return result


def frozen_snapshot() -> dict[str, Any]:
    paths = [
        ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv",
        ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv",
        ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro",
        ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
        ROBOT_DESCRIPTION_SOURCE,
        STAGE27 / "stage27s_candidate_trajectory.csv",
        STAGE27 / "stage27r_clean_follow_joint_trajectory_goal.json",
    ]
    return {
        "files": {str(path.relative_to(ROOT)): {"exists": path.exists(), "sha256": sha256(path) if path.is_file() else None, "bytes": path.stat().st_size if path.is_file() else None} for path in paths},
        "stage27_manifest": verify_manifest(STAGE27),
    }


def parse_qos(text: str, topic: str) -> dict[str, Any]:
    publisher_match = re.search(r"Publisher count:\s*(\d+)", text, re.I)
    publisher_count = int(publisher_match.group(1)) if publisher_match else None
    offers: list[dict[str, Any]] = []
    blocks = re.findall(r"Endpoint type:\s*PUBLISHER(?P<body>.*?)(?=Endpoint type:|$)", text, flags=re.I | re.S)
    for body in blocks:
        reliability_match = re.search(r"Reliability:\s*([A-Z_]+)", body, re.I)
        durability_match = re.search(r"Durability:\s*([A-Z_]+)", body, re.I)
        history_match = re.search(r"History(?: \(Depth\))?:\s*([A-Z_]+)", body, re.I)
        depth_match = re.search(r"History \(Depth\):\s*[A-Z_]+\s*\((\d+)\)|Depth:\s*(\d+)", body, re.I)
        reliability = reliability_match.group(1).lower() if reliability_match else None
        durability = durability_match.group(1).lower() if durability_match else None
        history = history_match.group(1).lower() if history_match else None
        depth = None
        if depth_match:
            depth = int(next(value for value in depth_match.groups() if value is not None))
        offers.append({"reliability": reliability, "durability": durability, "history": history, "depth": depth})
    # Jazzy endpoint introspection can expose publisher history/depth as
    # UNKNOWN. Keep those offer fields as not_available; compatibility is
    # determined from the observed reliability/durability, while the recorder
    # queue policy is explicit and reported separately.
    valid_offers = [item for item in offers if item["reliability"] and item["durability"]]
    if valid_offers:
        reliability = "best_effort" if any(item["reliability"] == "best_effort" for item in valid_offers) else "reliable"
        durability = "volatile" if any(item["durability"] == "volatile" for item in valid_offers) else "transient_local"
        history = "keep_last"
        minimum_depth = 100 if topic == "/tf_static" else 1000
        override = {"reliability": reliability, "durability": durability, "history": history, "depth": minimum_depth}
    else:
        override = {"reliability": "not_available", "durability": "not_available", "history": "not_available", "depth": None}
    return {"publisher_count": publisher_count, "offers": offers, "valid_offers": valid_offers, "recorder_qos_override": override, "recorder_qos_basis": "publisher offer reliability/durability; recorder queue minimum policy for unknown publisher history/depth", "parse_passed": bool(publisher_count and valid_offers)}


def graph_snapshot(out: Path, name: str) -> dict[str, Any]:
    commands = {
        "node_list": "source /opt/ros/jazzy/setup.bash; ros2 node list --no-daemon",
        "robot_state_publisher_info": "source /opt/ros/jazzy/setup.bash; ros2 node info /robot_state_publisher --no-daemon",
        "controller_manager_info": "source /opt/ros/jazzy/setup.bash; ros2 node info /controller_manager --no-daemon",
    }
    for index, topic in enumerate(TOPICS):
        commands[f"topic_info_{index}"] = f"source /opt/ros/jazzy/setup.bash; ros2 topic info -v --spin-time 5 --no-daemon {topic}"
    records: dict[str, Any] = {"phase": name, "captured_utc": now_utc(), "commands": {}}
    lines = [f"Stage 2.8S-R2-H1 graph/QoS snapshot: {name}", f"captured_utc: {records['captured_utc']}", ""]
    for key, command in commands.items():
        result = run_wsl(command, 30)
        records["commands"][key] = result
        lines.extend([f"$ {command}", f"exit_code: {result['exit_code']}", result.get("stdout", "").rstrip(), "stderr:", result.get("stderr", "").rstrip(), "", "---", ""])
    (out / f"{name}_execution_graph.txt").write_text("\n".join(lines), encoding="utf-8")
    # The user-facing artifact names are intentionally stable and phase-specific.
    (out / f"{name}_graph.txt").write_text("\n".join(lines), encoding="utf-8")
    return records


def parse_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    topics: dict[str, Any] = {}
    for index, topic in enumerate(TOPICS):
        raw = snapshot["commands"].get(f"topic_info_{index}", {})
        topics[topic] = {"raw": raw.get("stdout", ""), **parse_qos(raw.get("stdout", ""), topic)}
    return {"phase": snapshot["phase"], "captured_utc": snapshot["captured_utc"], "topics": topics}


def write_runtime_files(out: Path) -> tuple[Path, Path]:
    if not ROBOT_DESCRIPTION_SOURCE.is_file():
        raise RuntimeError(f"robot_description source missing: {ROBOT_DESCRIPTION_SOURCE}")
    description = out / "stage28sr2_h1_robot_description.urdf"
    description.write_bytes(ROBOT_DESCRIPTION_SOURCE.read_bytes())
    controllers = out / "stage28sr2_h1_controllers.yaml"
    controllers.write_text(
        """# H1 static mock/controller runtime; no trajectory goal is sent.
controller_manager:
  ros__parameters:
    update_rate: 100
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    fairino5_controller:
      type: joint_trajectory_controller/JointTrajectoryController

fairino5_controller:
  ros__parameters:
    joints: [j1, j2, j3, j4, j5, j6]
    command_interfaces: [position]
    state_interfaces: [position]
""",
        encoding="utf-8",
    )
    launch = out / "stage28sr2_h1_runtime.launch.py"
    launch.write_text(
        f"""from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DESCRIPTION = Path('{wsl_path(description)}')
CONTROLLERS = Path('{wsl_path(controllers)}')

def generate_launch_description():
    robot_description = ParameterValue(DESCRIPTION.read_text(encoding='utf-8'), value_type=str)
    manager = Node(package='controller_manager', executable='ros2_control_node', parameters=[{{'robot_description': robot_description}}, str(CONTROLLERS)], output='screen')
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', parameters=[{{'robot_description': robot_description, 'publish_frequency': 20.0, 'ignore_timestamp': False}}], output='screen')
    world_tf = ExecuteProcess(cmd=['ros2', 'run', 'tf2_ros', 'static_transform_publisher', '0', '0', '0', '0', '0', '0', 'world', 'base_link'], output='screen')
    jsb = ExecuteProcess(cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster', '--controller-manager', '/controller_manager', '--controller-manager-timeout', '30'], output='screen')
    jtc = ExecuteProcess(cmd=['ros2', 'run', 'controller_manager', 'spawner', 'fairino5_controller', '--controller-manager', '/controller_manager', '--controller-manager-timeout', '30'], output='screen')
    return LaunchDescription([DeclareLaunchArgument('runtime_tag', default_value='stage28sr2_h1'), manager, rsp, world_tf, jsb, jtc])
""",
        encoding="utf-8",
    )
    return launch, controllers


def wait_runtime_ready(timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    last: dict[str, Any] = {}
    while time.monotonic() - started < timeout_s:
        controllers = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 control list_controllers", 20)
        publish_frequency = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 param get /robot_state_publisher publish_frequency", 20)
        ignore_timestamp = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 param get /robot_state_publisher ignore_timestamp", 20)
        text = controllers.get("stdout", "")
        last = {
            "controllers": controllers,
            "publish_frequency": publish_frequency,
            "ignore_timestamp": ignore_timestamp,
            "joint_state_broadcaster_active": bool(re.search(r"joint_state_broadcaster\s+joint_state_broadcaster/JointStateBroadcaster\s+active", text, re.I)),
            "controller_active": bool(re.search(r"fairino5_controller\s+joint_trajectory_controller/JointTrajectoryController\s+active", text, re.I)),
        }
        if last["joint_state_broadcaster_active"] and last["controller_active"] and publish_frequency.get("exit_code") == 0 and ignore_timestamp.get("exit_code") == 0:
            return {"runtime_verified": True, **last}
        time.sleep(2.0)
    return {"runtime_verified": False, **last}


def start_process(command: str, log_path: Path) -> tuple[subprocess.Popen, Any]:
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc", command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log


def start_bag(out: Path, override: Path, bag_native: str) -> tuple[subprocess.Popen, Any, Path]:
    # Keep rosbag2 as the direct child of a WSL shell whose PID is recorded.
    # stop_bag() sends SIGINT to that exact process at the requested duration,
    # allowing rosbag2 to finalize MCAP metadata without an extra recording
    # window introduced by GNU timeout.
    pid_file = out / "stage28sr2_rosbag.pid"
    command = (
        f"source /opt/ros/jazzy/setup.bash; "
        f"echo $$ > {wsl_path(pid_file)}; "
        f"exec ros2 bag record --storage mcap --disable-keyboard-controls --node-name stage28sr2_h1_rosbag_recorder "
        f"--qos-profile-overrides-path {wsl_path(override)} -o {bag_native} --topics " + " ".join(TOPICS)
    )
    proc, log = start_process(command, out / "stage28sr2_rosbag_record.log")
    return proc, log, pid_file


def copy_native_bag(bag_native: str, bag_output: Path) -> dict[str, Any]:
    # The target is a fresh, uniquely named output directory.  No existing
    # artifact is removed or overwritten.
    return run_wsl(f"cp -a {bag_native} {wsl_path(bag_output)}", 90)


def stop_bag(proc: subprocess.Popen | None, pid_file: Path | None) -> dict[str, Any]:
    signal_result = None
    wait_result = None
    if pid_file is not None:
        signal_result = run_wsl(
            f"if test -s {wsl_path(pid_file)}; then kill -INT \"$(cat {wsl_path(pid_file)})\"; fi",
            20,
        )
    if proc is not None:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            terminate_result = run_wsl(
                f"if test -s {wsl_path(pid_file)}; then kill -TERM \"$(cat {wsl_path(pid_file)})\"; fi"
                if pid_file is not None
                else "true",
                20,
            )
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.terminate()
                wait_result = "terminated_after_timeout"
            else:
                wait_result = proc.returncode
            return {"signal": signal_result, "terminate": terminate_result, "wait": wait_result}
        else:
            wait_result = proc.returncode
    return {"signal": signal_result, "wait": wait_result}


def stop_runtime(proc: subprocess.Popen | None, launch: Path | None) -> dict[str, Any]:
    result = None
    if launch is not None:
        result = run_wsl(f"pkill -INT -f '{wsl_path(launch)}' 2>/dev/null || true", 20)
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    return {"signal": result, "returncode": None if proc is None else proc.returncode}


def load_trace(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def waypoint_union(rows: list[dict[str, str]]) -> set[int]:
    return source_waypoint_union(rows)


def geometry_regression(out: Path) -> dict[str, Any]:
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage28sr2_h1_geometry_mapping.py"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    stage27_coverage = waypoint_union(load_trace(STAGE27_TRACE)) if STAGE27_TRACE.is_file() else set()
    historical_coverage = waypoint_union(load_trace(HISTORICAL_STAGE28_TRACE)) if HISTORICAL_STAGE28_TRACE.is_file() else set()
    result = {
        "pytest_exit_code": test.returncode,
        "pytest_stdout": test.stdout,
        "pytest_stderr": test.stderr,
        "stage27s": {"actual": len(stage27_coverage), "expected": 720, "missing": sorted(set(range(720)) - stage27_coverage)},
        "historical_stage28s_r": {"actual": len(historical_coverage), "expected": 720, "missing": sorted(set(range(720)) - historical_coverage)},
        "four_waypoint_policy": "raw controller-state/native geometry union only; no interpolation substitution; fail_closed_if_absent",
        **geometry_semantics_root_cause(),
    }
    result["passed"] = bool(test.returncode == 0 and result["stage27s"]["actual"] == 720 and not result["stage27s"]["missing"] and result["historical_stage28s_r"]["actual"] == 716 and result["historical_stage28s_r"]["missing"] == [91, 92, 93, 634])
    (out / "stage28sr2_geometry_regression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def topic_summary(qos: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for topic in TOPICS:
        item = qos.get("topics", {}).get(topic, {})
        offers = item.get("valid_offers", [])
        offer = offers[0] if offers else {}
        summary[topic] = {
            "reliability": offer.get("reliability", "not_available"),
            "durability": offer.get("durability", "not_available"),
            "history": offer.get("history", "not_available"),
            "depth": offer.get("depth"),
            "publisher_count": item.get("publisher_count"),
            "subscription_qos": item.get("recorder_qos_override", {}),
            "recorder_qos_basis": item.get("recorder_qos_basis"),
        }
    return summary


def controller_gap_gate(controller_coverage: dict[str, Any]) -> bool:
    """Accept a long controller gap only with independent publisher evidence.

    A synchronized gap in both controller-origin topics and in bag arrival is
    evidence that the publisher did not publish.  An isolated/unattributed gap
    remains a fail-closed H1 blocker because it could still be recorder loss.
    """
    count = controller_coverage.get("long_callback_gap_over_100ms", {}).get("count")
    if count in (None, 0):
        return count == 0
    attribution = str(controller_coverage.get("publisher_vs_recorder_gap_attribution", ""))
    return attribution.startswith("publisher_did_not_publish_supported")


def finalize(out: Path, state: dict[str, Any]) -> dict[str, Any]:
    after = frozen_snapshot()
    before = state["frozen_before"]
    frozen_mismatch_count = len(after["stage27_manifest"].get("missing", [])) + len(after["stage27_manifest"].get("mismatches", []))
    for key, before_item in before["files"].items():
        after_item = after["files"].get(key, {})
        if before_item.get("sha256") != after_item.get("sha256"):
            frozen_mismatch_count += 1
    state["frozen_after"] = after
    state["frozen_hash_mismatch_count"] = frozen_mismatch_count
    audit = state.get("audit") or {}
    geometry = state.get("geometry") or {"passed": False}
    qos = state.get("qos") or {"topics": {}}
    duration_s = audit.get("rosbag_metadata", {}).get("duration_s")
    js = audit.get("joint_states", {})
    cs = audit.get("controller_state", {})
    tf = audit.get("tf", {})
    controller_coverage = audit.get("controller_state_coverage", {})
    conditions = {
        "independent_recorder_process": bool(state.get("recorder_process_independent")),
        "blocking_graph_probe_during_capture": state.get("graph_probe_during_capture", 0) == 0,
        "probe_duration_at_least_60s": bool(duration_s is not None and duration_s >= 60.0),
        "probe_topic_counts_complete": all(audit.get("topic_message_counts", {}).get(topic, 0) > 0 for topic in TOPICS),
        "probe_tf_exact_correspondence": bool(tf.get("unmatched") == 0 and tf.get("ambiguous_duplicate") == 0 and tf.get("messages", 0) > 0),
        "probe_joint_state_integrity": bool(js.get("invalid_header_stamps") == 0 and js.get("out_of_order_header_stamps") == 0 and js.get("messages", 0) > 0),
        "probe_controller_state_integrity": bool(cs.get("invalid_header_stamps") == 0 and cs.get("out_of_order_header_stamps") == 0 and cs.get("messages", 0) > 0),
        "controller_gap_not_recorder_loss": controller_gap_gate(controller_coverage),
        "runtime_qos_snapshot_complete": bool(qos.get("all_topics_parse_passed")),
        "geometry_coverage_regression": bool(geometry.get("passed")),
        "frozen_artifacts_unchanged": frozen_mismatch_count == 0,
        "formal_fjt_goal_budget_used": state.get("formal_fjt_goals_sent", 0) == 0,
    }
    blocker = next((name for name, passed in conditions.items() if not passed), "none")
    passed = all(conditions.values())
    status = "passed" if passed else f"blocked_{blocker}"
    report = {
        "Stage_2_8S_R2_H1": {
            "recorder_architecture": {
                "type": "rosbag2 independent process",
                "independent_process": bool(state.get("recorder_process_independent")),
                "graph_probe_during_capture": state.get("graph_probe_during_capture", 0),
                "blocking_subprocess_calls_during_capture": 0,
                "formal_client_process_started": False,
            },
            "runtime_qos": topic_summary(qos),
            "probe": {
                "formal_FJT_goal": False,
                "counts_against_formal_goal_budget": False,
                "duration_s": duration_s,
                "joint_states": {
                    "messages": js.get("messages"),
                    "duplicate_stamps": js.get("duplicate_header_stamps"),
                    "out_of_order": js.get("out_of_order_header_stamps"),
                    "invalid_header_stamps": js.get("invalid_header_stamps"),
                    "max_gap_ms": js.get("capture_header_intervals_ms", {}).get("max"),
                    "capture_header_intervals": js.get("capture_header_intervals_ms"),
                },
                "controller_state": {
                    "messages": cs.get("messages"),
                    "duplicate_stamps": cs.get("duplicate_header_stamps"),
                    "out_of_order": cs.get("out_of_order_header_stamps"),
                    "invalid_header_stamps": cs.get("invalid_header_stamps"),
                    "max_gap_ms": cs.get("capture_header_intervals_ms", {}).get("max"),
                    "capture_header_intervals": cs.get("capture_header_intervals_ms"),
                    "capture_monotonic_gap": controller_coverage.get("capture_monotonic_gap"),
                    "long_callback_gap_over_100ms": controller_coverage.get("long_callback_gap_over_100ms"),
                    "publisher_vs_recorder_gap_attribution": controller_coverage.get("publisher_vs_recorder_gap_attribution"),
                    "gap_evidence": controller_coverage.get("evidence"),
                },
                "tf": {
                    "messages": tf.get("messages"),
                    "ros_messages": tf.get("ros_messages"),
                    "transform_records": tf.get("transform_records"),
                    "exact_joint_state_matches": tf.get("exact_joint_state_matches"),
                    "unmatched": tf.get("unmatched"),
                    "ambiguous": tf.get("ambiguous_duplicate"),
                    "unmatched_header_stamps_sample": tf.get("unmatched_header_stamps_sample", []),
                },
                "topic_message_counts": audit.get("topic_message_counts", {}),
                "rosbag_metadata": audit.get("rosbag_metadata", {}),
                "message_lost_event_counter": audit.get("message_lost_event_counter", {"status": "not_available"}),
                "deadline_event_counter": audit.get("deadline_event_counter", {"status": "not_available"}),
                "liveliness_event_counter": audit.get("liveliness_event_counter", {"status": "not_available"}),
            },
            "old_starvation_root_cause": {
                "positive_control_performed": False,
                "final_confidence": "strongly_supported_not_formally_proven",
                "working_classification": "recorder_executor_starvation_and_asymmetric_queue_eviction",
                "h0_forensic_reanalysis_repeated": False,
            },
            "geometry_mapping": {
                "source_waypoint_union_semantics": "union(source_waypoint_0, source_waypoint_1) from raw native/controller-state evidence",
                "stage27s_regression_coverage": f"{geometry.get('stage27s', {}).get('actual')}/720",
                "historical_stage28sr_coverage": f"{geometry.get('historical_stage28s_r', {}).get('actual')}/720",
                "historical_missing": geometry.get("historical_stage28s_r", {}).get("missing", []),
                "four_waypoints_91_92_93_634": "fail_closed_if_absent_from_raw_capture",
            },
            "frozen_hash_mismatch_count": frozen_mismatch_count,
            "frozen_artifacts_modified": frozen_mismatch_count != 0,
            "formal_FJT_goals_sent": state.get("formal_fjt_goals_sent", 0),
            "Stage_2_8S_R2_H1": status,
            "gate_conditions": conditions,
            "ready_for_formal_R2": passed,
            "Can_we_start_Stage_2_8S_R2_formal": passed,
            "Can_we_start_Stage_2_9": False,
            "Can_we_start_Stage_3": False,
            "artifacts": {"output_root": str(out.resolve()), "pre_execution_graph": str((out / "pre_execution_graph.txt").resolve()), "post_execution_graph": str((out / "post_execution_graph.txt").resolve()), "qos_snapshot": str((out / "stage28sr2_qos_runtime_snapshot.yaml").resolve()), "qos_override": str((out / "stage28sr2_rosbag_qos_override.yaml").resolve()), "bag_audit": str((out / "stage28sr2_bag_audit.json").resolve())},
        }
    }
    (out / "stage28sr2_h1_gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "stage28sr2_h1_final_report.yaml").write_text(yaml.safe_dump(report, allow_unicode=True, sort_keys=False), encoding="utf-8")
    md = [
        "# Stage 2.8S-R2-H1 — Recorder Remediation + Capture Integrity Certification",
        "",
        f"- H1: **{status}**",
        f"- Independent recorder: `{report['Stage_2_8S_R2_H1']['recorder_architecture']['independent_process']}` (`rosbag2`).",
        f"- Probe duration: `{duration_s}` s; TF exact unmatched/ambiguous: `{tf.get('unmatched')}/{tf.get('ambiguous_duplicate')}`.",
        f"- Geometry regression: Stage 2.7S `{geometry.get('stage27s', {}).get('actual')}/720`; historical Stage 2.8S-R `{geometry.get('historical_stage28s_r', {}).get('actual')}/720`, missing `{geometry.get('historical_stage28s_r', {}).get('missing', [])}`.",
        f"- Formal FJT goals sent: `{state.get('formal_fjt_goals_sent', 0)}`; next Stage 2.9/3: `false/false`.",
        f"- First blocker: `{blocker}`.",
        "",
        "This H1 probe used a static simulation mock/controller runtime only. No formal Stage 2.8S-R2 FJT goal, real FAIRINO connection, hardware plugin, trajectory change, waypoint change, geometry threshold change, URDF/SRDF change, or RSP workaround was used.",
        "",
    ]
    (out / "stage28sr2_h1_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.output or ROOT / "outputs/stage28sr2_h1_certification" / f"stage28sr2_h1_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {"formal_fjt_goals_sent": 0, "graph_probe_during_capture": 0, "frozen_before": frozen_snapshot(), "recorder_process_independent": False}
    runtime_proc = None
    runtime_log = None
    bag_proc = None
    bag_log = None
    bag_pid = None
    launch = None
    try:
        runtime_launch, _controllers = write_runtime_files(out)
        launch = runtime_launch
        initial_nodes = run_wsl("source /opt/ros/jazzy/setup.bash; ros2 node list --no-daemon", 30)
        state["initial_node_list"] = initial_nodes
        if "/controller_manager" in initial_nodes.get("stdout", "") or "/robot_state_publisher" in initial_nodes.get("stdout", ""):
            raise RuntimeError("pre-existing controller_manager or robot_state_publisher detected; refusing to attach to a non-fresh runtime")
        runtime_command = f"source /opt/ros/jazzy/setup.bash; source {wsl_path(ROOT / 'install/setup.bash')}; ros2 launch {wsl_path(launch)} runtime_tag:=stage28sr2_h1"
        runtime_proc, runtime_log = start_process(runtime_command, out / "stage28sr2_h1_pre_runtime.log")
        state["runtime_process_started"] = True
        readiness = wait_runtime_ready(120.0)
        state["runtime_ready_pre_capture"] = readiness
        if not readiness.get("runtime_verified"):
            raise RuntimeError("static mock/controller runtime did not become ready")
        pre_raw = graph_snapshot(out, "pre_execution")
        state["pre_graph"] = parse_snapshot(pre_raw)
        qos = state["pre_graph"]
        state["qos"] = qos
        (out / "stage28sr2_qos_runtime_snapshot.yaml").write_text(yaml.safe_dump(qos, allow_unicode=True, sort_keys=False), encoding="utf-8")
        all_parse = all(qos["topics"][topic].get("parse_passed") for topic in TOPICS)
        qos["all_topics_parse_passed"] = all_parse
        (out / "stage28sr2_qos_runtime_snapshot.yaml").write_text(yaml.safe_dump(qos, allow_unicode=True, sort_keys=False), encoding="utf-8")
        overrides = {topic: qos["topics"][topic]["recorder_qos_override"] for topic in TOPICS}
        (out / "stage28sr2_rosbag_qos_override.yaml").write_text(yaml.safe_dump(overrides, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if not all_parse:
            raise RuntimeError("actual runtime publisher QoS could not be parsed for every required topic")

        # Close the first runtime before opening the recorder.  The recorder
        # therefore discovers the required topics before their first capture
        # sample, eliminating startup-only TF stamps whose JointState sample
        # can otherwise race recorder subscription discovery.  The same
        # frozen mock/controller launch is then started once for the actual
        # non-formal probe.  No FJT client or goal is involved in either run.
        state["pre_capture_runtime_stop"] = stop_runtime(runtime_proc, launch)
        runtime_proc = None
        if runtime_log is not None:
            runtime_log.close()
            runtime_log = None

        bag = out / "stage28sr2_probe_bag"
        bag_native = f"/tmp/stage28sr2_h1_{out.name}_bag"
        state["bag_native_path"] = bag_native
        bag_proc, bag_log, bag_pid = start_bag(out, out / "stage28sr2_rosbag_qos_override.yaml", bag_native)
        state["recorder_process_independent"] = True

        runtime_proc, runtime_log = start_process(runtime_command, out / "stage28sr2_h1_runtime.log")
        state["capture_runtime_process_started"] = True
        capture_readiness = wait_runtime_ready(120.0)
        state["runtime_ready"] = capture_readiness
        if not capture_readiness.get("runtime_verified"):
            raise RuntimeError("capture mock/controller runtime did not become ready")
        if bag_proc.poll() is not None:
            raise RuntimeError(f"rosbag2 exited before capture with code {bag_proc.returncode}")

        # Give discovery a short non-blocking settling interval.  This is
        # outside the timed probe interval and performs no graph/CLI probe.
        time.sleep(2.0)
        state["capture_start_utc"] = now_utc()
        started = time.monotonic()
        while time.monotonic() - started < args.duration:
            if bag_proc.poll() is not None:
                raise RuntimeError(f"rosbag2 exited during capture with code {bag_proc.returncode}")
            time.sleep(min(1.0, max(0.05, args.duration - (time.monotonic() - started))))
        state["capture_end_utc"] = now_utc()
        state["bag_stop"] = stop_bag(bag_proc, bag_pid)
        bag_proc = None
        state["bag_copy"] = copy_native_bag(bag_native, bag)
        if not (bag / "metadata.yaml").is_file():
            raise RuntimeError("finalized native rosbag2 output did not contain metadata.yaml after copy")
        # The only graph snapshot after this point is post-capture.
        post_raw = graph_snapshot(out, "post_execution")
        state["post_graph"] = parse_snapshot(post_raw)
        state["graph_probe_during_capture"] = 0
        bag_info = run_wsl(f"source /opt/ros/jazzy/setup.bash; ros2 bag info --storage mcap {wsl_path(bag)}", 60)
        (out / "stage28sr2_rosbag_info.txt").write_text(bag_info.get("stdout", "") + "\nSTDERR:\n" + bag_info.get("stderr", ""), encoding="utf-8")
        state["rosbag_info"] = bag_info
        audit_path = out / "stage28sr2_bag_audit.json"
        audit_command = f"source /opt/ros/jazzy/setup.bash; python3 {wsl_path(ROOT / 'scripts/stage28sr2_h1_bag_audit.py')} --bag {wsl_path(bag)} --output {wsl_path(audit_path)}"
        audit_run = run_wsl(audit_command, 180)
        (out / "stage28sr2_bag_audit_runtime.log").write_text(json.dumps(audit_run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if audit_path.is_file():
            state["audit"] = json.loads(audit_path.read_text(encoding="utf-8"))
            metadata = state["audit"].setdefault("rosbag_metadata", {})
            start_ns = metadata.get("starting_time_ns")
            end_ns = metadata.get("ending_time_ns")
            if start_ns is not None:
                metadata["record_start_time"] = datetime.fromtimestamp(start_ns / 1_000_000_000.0, timezone.utc).isoformat()
            if end_ns is not None:
                metadata["record_end_time"] = datetime.fromtimestamp(end_ns / 1_000_000_000.0, timezone.utc).isoformat()
            metadata["capture_wall_start_time"] = state.get("capture_start_utc")
            metadata["capture_wall_end_time"] = state.get("capture_end_utc")
            metadata["publisher_count"] = {topic: qos["topics"][topic].get("publisher_count") for topic in TOPICS}
            metadata["subscription_qos"] = {topic: qos["topics"][topic].get("recorder_qos_override") for topic in TOPICS}
            audit_path.write_text(json.dumps(state["audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            raise RuntimeError("offline rosbag2 audit did not produce stage28sr2_bag_audit.json")
        state["geometry"] = geometry_regression(out)
    except Exception as error:
        state["exception"] = repr(error)
        (out / "stage28sr2_h1_runtime_error.txt").write_text(repr(error) + "\n", encoding="utf-8")
    finally:
        if bag_proc is not None:
            state["bag_stop"] = stop_bag(bag_proc, bag_pid)
        if runtime_proc is not None:
            state["runtime_stop"] = stop_runtime(runtime_proc, launch)
        if runtime_log is not None:
            runtime_log.close()
        if bag_log is not None:
            bag_log.close()
    report = finalize(out, state)
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    return 0 if report["Stage_2_8S_R2_H1"]["Stage_2_8S_R2_H1"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
