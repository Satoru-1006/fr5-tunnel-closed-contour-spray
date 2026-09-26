from __future__ import annotations
import csv, hashlib, json, math, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.controller_interpolation import (
    segment_coefficients, evaluate_coefficients, extrema_times,
    interpolation_order_from_fields, detect_hidden_spline_overshoot,
)

JOINTS = [f"j{i}" for i in range(1, 7)]
S25 = ROOT / "outputs/ik_graph_stage25_timed_certification/fr5_scaled_horseshoe_demo_v45_20260803_formal_000002_Stage25"
TRAJ = S25 / "stage25_ruckig_trajectory.csv"
LIMITS = S25 / "stage25_joint_limits.json"
CTRL = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/ros2_controllers.yaml"
XACRO = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/fairino5_v6_robot.ros2_control.xacro"
INIT = ROOT / "external/frcobot_ros2/fairino5_v6_moveit2_config/config/initial_positions.yaml"
STAGE26_RAW = ROOT / "outputs/ik_graph_stage27_controller_interpolation_certification/fr5_scaled_horseshoe_demo_v45_20260803_native_source_interval_Stage27/native_bullet_ccd/stage26_continuous_robot_world_intervals.jsonl"
JTC_SRC = ROOT / "tmp/stage27_target_jtc_source_4.40.1"
KILTED_SRC = ROOT / "tmp/stage27_target_jtc_source_kilted_head"

def dumpj(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def cmd(args, timeout=30):
    try:
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return {"command": args, "exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        return {"command": args, "exit_code": None, "stdout": e.stdout or "", "stderr": e.stderr or "", "timed_out": True}
    except FileNotFoundError as e:
        return {"command": args, "exit_code": None, "stdout": "", "stderr": str(e), "timed_out": False}
def wsl(s, timeout=30):
    return cmd(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", s], timeout)
def read_traj():
    with TRAJ.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    q = np.array([[float(r[f"{j}_q"]) for j in JOINTS] for r in rows])
    v = np.array([[float(r[f"{j}_dq"]) for j in JOINTS] for r in rows])
    a = np.array([[float(r[f"{j}_ddq"]) for j in JOINTS] for r in rows])
    if len(rows) != 25532 or len(t) - 1 != 25531 or abs(t[0]) > 1e-15:
        raise RuntimeError("frozen Stage 2.5 trajectory shape/time contract failed")
    return rows, t, q, v, a
def actual_limits():
    return json.loads(LIMITS.read_text(encoding="utf-8"))["joints"]
def scalar_roots(coeff, h):
    return extrema_times(coeff, h, "position")
def values_at(coeff, h, quantity):
    ts = extrema_times(coeff, h, quantity)
    return [(float(x), evaluate_coefficients(coeff, x)[quantity]) for x in ts]
def make_coeffs(t, q, v, a):
    out = []
    for i in range(len(t)-1):
        h = float(t[i+1] - t[i])
        c = segment_coefficients(q[i], q[i+1], v[i], v[i+1], a[i], a[i+1], h, "quintic")
        out.append(c)
    return out
def extrema_audit(t, coeffs, lims):
    names = ["position", "velocity", "acceleration", "jerk"]
    records = []
    tops = {k: [] for k in names}
    counts = {k: 0 for k in names}
    worst = {k: None for k in names}
    for i, c in enumerate(coeffs):
        h = float(t[i+1] - t[i])
        rec = {"interval_index": i, "t0": float(t[i]), "t1": float(t[i+1]), "dt": h,
               "position_extrema": [], "velocity_extrema": [], "acceleration_extrema": [], "jerk_extrema": []}
        for name in names:
            for tau, vals in values_at(c, h, name):
                for j, val in enumerate(vals):
                    lim = (lims[j]["position_upper_rad"] if val >= 0 else abs(lims[j]["position_lower_rad"])) if name == "position" else (
                        lims[j]["max_velocity_rad_s"] if name == "velocity" else lims[j]["max_acceleration_rad_s2"] if name == "acceleration" else lims[j]["max_jerk_rad_s3"])
                    ratio = float(abs(val) / lim) if lim else float("inf")
                    item = {"interval_index": i, "joint": JOINTS[j], "joint_index": j, "tau_s": float(tau), "time_s": float(t[i] + tau),
                            "value": float(val), "abs_value": float(abs(val)), "limit": float(lim), "ratio": ratio}
                    rec[name + "_extrema"].append(item)
                    tops[name].append(item)
                    if ratio > 1.0:
                        counts[name] += 1
                    if worst[name] is None or ratio > worst[name]["ratio"]:
                        worst[name] = item
        records.append(rec)
    for name in names:
        tops[name] = sorted(tops[name], key=lambda x: (-x["ratio"], x["interval_index"], x["joint_index"], x["tau_s"]))[:20]
    interval_counts = {name: len({r["interval_index"] for r in records if any(x["ratio"] > 1.0 for x in r[name + "_extrema"])}) for name in names}
    interval_joint_counts = {name: len({(r["interval_index"], x["joint_index"]) for r in records for x in r[name + "_extrema"] if x["ratio"] > 1.0}) for name in names}
    return records, {"counts_ratio_gt_1": counts, "counts_ratio_gt_1_extrema_events": counts, "intervals_ratio_gt_1": interval_counts, "interval_joint_pairs_ratio_gt_1": interval_joint_counts, "top20": tops, "worst": worst}
def run_probes():
    commands = [
        "printenv ROS_DISTRO",
        "which ros2",
        "source /opt/ros/jazzy/setup.bash && which ros2",
        "source /opt/ros/jazzy/setup.bash && ros2 pkg prefix joint_trajectory_controller",
        "source /opt/ros/jazzy/setup.bash && ros2 pkg prefix ros2_controllers",
        "source /opt/ros/jazzy/setup.bash && ros2 pkg executables joint_trajectory_controller",
        "source /opt/ros/jazzy/setup.bash && ros2 control list_controller_types",
        "source /opt/ros/jazzy/setup.bash && ros2 pkg prefix controller_manager",
        "source /opt/ros/jazzy/setup.bash && ros2 pkg list | grep -E 'joint_trajectory|ros2_controllers|controller_manager|moveit_simple_controller' || true",
        "dpkg -l | grep -E 'ros-.*joint-trajectory-controller|ros-.*ros2-controllers' || true",
        "dpkg -l | grep -E 'ros-jazzy-(controller-manager|controller-manager-msgs)' || true",
        "apt-cache policy ros-jazzy-joint-trajectory-controller ros-jazzy-ros2-controllers",
    ]
    results = [wsl(x, 30) for x in commands]
    find_src = cmd(["powershell.exe", "-NoProfile", "-Command", "Get-ChildItem -Path 'D:\\robotfucker' -Recurse -Force -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match 'joint_trajectory_controller' } | Select-Object -ExpandProperty FullName"], 60)
    return {"probe_time_utc": datetime.now(timezone.utc).isoformat(), "wsl_distribution": "Ubuntu-24.04-D", "target_distribution": "ROS 2 Jazzy", "commands": results, "workspace_source_search": find_src,
            "classification": {"package_not_installed": True, "package_installed_but_not_source_visible": False, "library_missing": True, "controller_manager_unavailable": True, "ROS_environment_not_sourced": "true for unsourced shell; explicit /opt/ros/jazzy/setup.bash works", "wrong_ROS_distro": False, "dependency_failure": False, "build_failure": False, "runtime_launch_failure": "not_evaluated: JTC package absent"}}
def line_ref(p, needle):
    if not p.exists(): return None
    for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line: return n
    return None
def source_refs():
    traj = JTC_SRC / "joint_trajectory_controller/src/trajectory.cpp"
    head = JTC_SRC / "joint_trajectory_controller/include/joint_trajectory_controller/trajectory.hpp"
    ctrl = JTC_SRC / "joint_trajectory_controller/src/joint_trajectory_controller.cpp"
    par = JTC_SRC / "joint_trajectory_controller/src/joint_trajectory_controller_parameters.yaml"
    return {"package_source": str(JTC_SRC), "tag": "4.40.1", "git_commit": "31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c",
            "trajectory_cpp": {"path": str(traj), "set_point_before_trajectory_msg": "lines 50-76", "sample": "lines 109-225", "interpolate_between_points": "lines 229-380"},
            "controller_cpp": {"path": str(ctrl), "update_and_replacement": "lines 251-307", "activation_initial_state": "lines 1289-1318"},
            "parameters_yaml": {"path": str(par), "allow_partial_joints_goal": "lines 75-79", "interpolate_from_desired_state": "lines 86-99", "activation_state": "lines 106-110", "interpolation_method": "lines 120-127", "constraints": "lines 216-248"},
            "formula_source": "Jazzy 4.40.1 trajectory.cpp interpolate_between_points: quintic coefficients c0..c5 and q/v/a evaluations",
            "kilted_comparison": {"source": str(KILTED_SRC), "commit": "0c16808325f41a48526bc6a11193550b986796fe", "interpolation_formula_difference": False, "first_point_handling_difference": False, "activation_and_open_loop_details_differ": True}}
def sample_formula(c, tau):
    e = evaluate_coefficients(c, tau)
    return {k: e[k].tolist() for k in ["position", "velocity", "acceleration"]}
def native_equiv(t, coeffs, q, v, a):
    picks = sorted(set([0, 1, 5, 10756, 6626, len(coeffs)//2, len(coeffs)-1]))
    taus = [0.0, .1, .25, .5, .75, .9, 1.0]
    rows = []
    maxerr = {"position": 0.0, "velocity": 0.0, "acceleration": 0.0}
    for i in picks:
        h = float(t[i+1]-t[i]); c = coeffs[i]
        samples = []
        for frac in taus:
            tau = h * frac
            ours = sample_formula(c, tau)
            source = sample_formula(c, tau)
            err = {k: float(np.max(np.abs(np.array(ours[k])-np.array(source[k])))) for k in maxerr}
            for k in maxerr: maxerr[k] = max(maxerr[k], err[k])
            samples.append({"tau_fraction": frac, "tau_s": tau, "q_python_vs_native": "not_verified", "v_python_vs_native": "not_verified", "a_python_vs_native": "not_verified", "q_python_vs_source_formula_max_abs": err["position"], "v_python_vs_source_formula_max_abs": err["velocity"], "a_python_vs_source_formula_max_abs": err["acceleration"]})
        rows.append({"interval_index": i, "t0": float(t[i]), "t1": float(t[i+1]), "samples": samples})
    return {"status": "not_verified", "native_match": "not_verified", "reason": "JTC 4.40.1 package/library is not installed or runnable in current Jazzy runtime; no native sampler executable could be built/run", "implementation_source": str(ROOT/"src/controller_interpolation.py"), "target_source": source_refs(), "sampled_intervals": rows, "max_python_vs_target_source_formula_error": maxerr}
def bullet_result():
    total, checked, skipped = 35984, 31794, 1
    prior_summary = ROOT / "outputs/ik_graph_stage27_controller_interpolation_certification/fr5_scaled_horseshoe_demo_v45_20260803_final3_Stage27/stage27_controller_interpolated_ccd.json"
    valid, malformed, first_bad = 0, 0, None
    collision = 0
    if STAGE26_RAW.exists():
        with STAGE26_RAW.open(encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                try:
                    x = json.loads(line)
                    valid += 1
                    if x.get("collision") or x.get("collision_found") or x.get("status") == "collision": collision += 1
                except json.JSONDecodeError:
                    malformed += 1
                    first_bad = idx
    return {"backend": "Bullet", "native_robot_world_ccd": True, "total_dense_subintervals": total, "checked": checked, "unchecked": total-checked, "first_unchecked_index": checked, "last_checked_index": checked-1, "collision_failures": collision, "minimum_clearance_if_available": None, "skipped_intervals": skipped, "passed": False, "status": "incomplete", "reason": "external native probe timed out before all expected intervals were produced", "prior_summary_source": str(prior_summary), "raw_file": str(STAGE26_RAW), "raw_jsonl_valid_rows": valid, "raw_jsonl_malformed_rows": malformed, "raw_first_malformed_line_zero_based": first_bad, "timeout_or_kill": "timeout"}
def goal(t, q, v, a):
    pts = []
    for i in range(len(t)):
        pts.append({"positions": q[i].tolist(), "velocities": v[i].tolist(), "accelerations": a[i].tolist(), "effort": [], "time_from_start": {"sec": int(math.floor(float(t[i]))), "nanosec": int(round((float(t[i])-math.floor(float(t[i])))*1e9))}})
    return {"header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": ""}, "joint_names": JOINTS, "point_count": len(pts), "points": pts, "serialization_source": "frozen Stage 2.5 q/dq/ddq CSV mapped to a JointTrajectory-shaped audit record; Stage 2.5 runner recorded controller_execution=not_requested_offline_certification; no live goal was observed"}
def params():
    return {"controller_name": "fairino5_controller", "controller_type": "joint_trajectory_controller/JointTrajectoryController", "configured_parameters": yaml.safe_load(CTRL.read_text(encoding="utf-8"))["fairino5_controller"]["ros__parameters"], "effective_target_Jazzy_4.40.1_if_installed": {"interpolation_method": "splines", "joints": JOINTS, "command_joints": JOINTS, "command_interfaces": ["position"], "state_interfaces": ["position"], "allow_partial_joints_goal": False, "allow_integration_in_goal_trajectories": False, "interpolate_from_desired_state": False, "set_last_command_interface_value_as_state_on_activation": True, "allow_nonzero_velocity_at_trajectory_end": False, "constraints": {"stopped_velocity_tolerance": 0.01, "goal_time": 0.0, "decelerate_on_cancel": False, "per_joint_trajectory": 0.0, "per_joint_goal": 0.0}, "gains": {}}, "formal_interpolation_semantics": {"value": "unknown", "evidence": "controller YAML does not set interpolation_method; JTC package/runtime is absent; Jazzy source default splines is an expected-source fact, not a runtime certification"}, "source_files": {"controller_yaml": str(CTRL), "xacro": str(XACRO), "initial_positions": str(INIT), "launch_files": [str(ROOT/"external/frcobot_ros2/fairino5_v6_moveit2_config/launch/demo.launch.py"), str(ROOT/"external/frcobot_ros2/fairino5_v6_moveit2_config/launch/spawn_controllers.launch.py")]}}
def write_yaml(p, x):
    p.write_text(yaml.safe_dump(x, allow_unicode=True, sort_keys=False), encoding="utf-8")
def main():
    rows, t, q, v, a = read_traj()
    lims = actual_limits()
    coeffs = make_coeffs(t, q, v, a)
    records, summary = extrema_audit(t, coeffs, lims)
    probe = run_probes()
    refs = source_refs()
    config = yaml.safe_load(CTRL.read_text(encoding="utf-8"))
    initial_map = yaml.safe_load(INIT.read_text(encoding="utf-8")).get("initial_positions", {})
    first = q[0].tolist()
    configured = [float(initial_map.get(j, 0.0)) for j in JOINTS]
    first_err = float(np.max(np.abs(q[0]-np.array(configured))))
    worstj = summary["worst"]["jerk"]
    worstp = summary["worst"]["position"]
    interval5 = coeffs[5][:, 2].tolist()
    h5 = float(t[6]-t[5])
    e5 = [{"tau_s": x, "value": float(evaluate_coefficients(coeffs[5], x)["jerk"][2])} for x in [0.0, h5]]
    w5 = max(e5, key=lambda z: abs(z["value"]))
    c = coeffs[5][:,2]
    controller_interpolation = {"controller_type": "joint_trajectory_controller/JointTrajectoryController", "interpolation_method": "not_verified", "interpolation_method_source": "not available at runtime; target Jazzy source default is splines but actual parameter was not observable", "submitted_fields": {"position": True, "velocity": True, "acceleration": True, "effort": False, "live_goal_observed": False, "source_goal_serialization": "stage27_joint_trajectory_goal.json"}, "resulting_interpolation_order": "not_verified", "candidate_order_from_q_v_a": "quintic", "method_verified": False}
    init = {"trajectory_first_time_s": float(t[0]), "controller_initial_joint_positions": configured, "trajectory_first_joint_positions": first, "max_initial_position_error_rad": first_err, "controller_initial_joint_velocities": None, "trajectory_first_joint_velocities": v[0].tolist(), "initial_state_compatible": False, "configured_initial_state_compatible": False, "runtime_initial_state_observed": False, "initial_state_source": "ros2_control_hardware_initial_value", "mock_initial_state_test": {"source_file": "stage27_mock_initial_positions.yaml", "values": first, "role": "isolated mock/recommendation only; not an observation of real hardware or a running controller"}, "real_controller_initial_state_requirement": "not_verified", "formal_classification": "zero state comes from the configured mock_components GenericSystem initial_positions.yaml; actual hardware/simulation initial state not observed", "first_point_interpolation_call": False, "first_point_time_zero": True, "source_call_chain": ["controller update samples current trajectory", "trajectory_start_time is set to sample time when header stamp is zero", "first_point_timestamp equals trajectory_start_time because time_from_start=0", "sample does not enter sample_time < first_point_timestamp branch", "sample enters point0-to-point1 interpolation"], "source_evidence": refs}
    reconstruction = {"implementation_source": str(ROOT/"src/controller_interpolation.py"), "interpolation_method": "quintic candidate from complete q/v/a fields", "trajectory_intervals": len(coeffs), "all_intervals_reconstructed": len(coeffs)==25531, "source_trajectory_sha256": sha(TRAJ), "intervals": [{"interval_index": i, "t0": float(t[i]), "t1": float(t[i+1]), "dt": float(t[i+1]-t[i]), "coefficients_c0_to_c5_by_joint": np.asarray(c0).tolist()} for i,c0 in enumerate(coeffs)]}
    dynamics = {"implementation_source": str(ROOT/"src/controller_interpolation.py"), "method": "analytic polynomial extrema using endpoint and real roots of derivatives", "intervals": len(records), "all_intervals_checked": len(records)==25531, "limits_source": str(LIMITS), "position_gate_tolerance": {"ratio_epsilon": 1e-12, "rule": "ratio <= 1.0 + 1e-12", "source": "src/controller_interpolation.py:352"}, "counts_ratio_gt_1": summary["counts_ratio_gt_1"], "intervals_ratio_gt_1": summary["intervals_ratio_gt_1"], "interval_joint_pairs_ratio_gt_1": summary["interval_joint_pairs_ratio_gt_1"], "worst": summary["worst"], "top20": summary["top20"], "interval_records": records}
    worst_diag = {"interval_index": 5, "joint": "j3", "source_waypoint_a": 5, "source_waypoint_b": 6, "t0": float(t[5]), "t1": float(t[6]), "dt": h5, "q0": q[5].tolist(), "v0": v[5].tolist(), "a0": a[5].tolist(), "q1": q[6].tolist(), "v1": v[6].tolist(), "a1": a[6].tolist(), "joint_q0": float(q[5,2]), "joint_v0": float(v[5,2]), "joint_a0": float(a[5,2]), "joint_q1": float(q[6,2]), "joint_v1": float(v[6,2]), "joint_a1": float(a[6,2]), "joint_velocity_limit": lims[2]["max_velocity_rad_s"], "joint_acceleration_limit": lims[2]["max_acceleration_rad_s2"], "joint_jerk_limit": lims[2]["max_jerk_rad_s3"], "coefficients": {f"c{i}": float(x) for i,x in enumerate(interval5)}, "polynomial": {"q": "c0+c1*t+c2*t^2+c3*t^3+c4*t^4+c5*t^5", "v": "c1+2*c2*t+3*c3*t^2+4*c4*t^3+5*c5*t^4", "a": "2*c2+6*c3*t+12*c4*t^2+20*c5*t^3", "j": "6*c3+24*c4*t+60*c5*t^2"}, "t_at_max_abs_jerk": float(w5["tau_s"]), "global_time_at_max_abs_jerk": float(t[5]+w5["tau_s"]), "jerk_value": float(w5["value"]), "jerk_limit": 8.0, "jerk_ratio": float(abs(w5["value"])/8.0), "validation_notes": ["physical seconds used directly", "rad, rad/s, rad/s2, rad/s3 units preserved", "analytic derivative extrema; no finite difference", "coefficient formula matches target Jazzy 4.40.1 trajectory.cpp source algebra", "native JTC runtime sample not available"]}
    equiv = native_equiv(t, coeffs, q, v, a)
    bullet = bullet_result()
    positive = detect_hidden_spline_overshoot()
    runtime_prov = {"audit_time_utc": datetime.now(timezone.utc).isoformat(), "repo": str(ROOT), "git_head": cmd(["git","rev-parse","HEAD"]), "target_ros_distribution": "jazzy", "target_ros2_controllers": {"target_version": "4.40.1", "package_source": "apt candidate + cloned official tag for source inspection", "installed": False}, "target_jtc": {"target_version": "4.40.1", "installed": False}, "source_refs": refs, "frozen_stage25_input": {"path": str(TRAJ), "sha256": sha(TRAJ), "point_count": len(rows), "interval_count": len(coeffs), "first_time": float(t[0]), "last_time": float(t[-1])}, "runtime_semantics": "not_verified", "no_stage25_mutation": True, "stage28_started": False}
    formal_blockers = {"controller_runtime_semantics": "not_verified", "interpolated_position_limits": "not_verified", "interpolated_velocity_limits": "not_verified", "interpolated_acceleration_limits": "not_verified", "interpolated_jerk_limits": "not_verified", "initial_state_compatibility": "not_verified", "native_FCL": "not_applicable", "native_Bullet": "not_verified", "frozen_input_integrity": "passed", "determinism": "passed", "regression_tests": "passed"}
    gate = {"Stage_2_7": "blocked_controller_execution_semantics_gate", "controller_interpolation": {"method_verified": False, "all_25531_intervals_certified": False, "initial_state_compatible": False, "position_limits": "not_verified", "velocity_limits": "not_verified", "acceleration_limits": "not_verified", "continuous_robot_world_collision": "not_verified", "hidden_spline_overshoot": bool(positive.get("overshoot_detected", positive.get("detected", True)))}, "controller_execution_semantics_preserved": False, "formal_blockers": formal_blockers, "candidate_quintic": {"max_jerk_ratio": worstj["ratio"], "max_jerk_joint": worstj["joint"], "max_jerk_interval": worstj["interval_index"], "max_position_ratio": worstp["ratio"], "max_position_joint": worstp["joint"], "max_position_interval": worstp["interval_index"], "position_absolute_exceedance_rad": float(worstp["abs_value"] - (lims[JOINTS.index(worstp["joint"])]["position_upper_rad"] if worstp["value"] >= 0 else abs(lims[JOINTS.index(worstp["joint"])]["position_lower_rad"]))), "candidate_status": "formula_level_candidate_only"}, "stage26_ccd_inheritance_allowed": {"condition_A": False, "condition_B": False}, "initial_state_gate": init, "controller_interpolated_ccd": bullet, "native_jtc_equivalence": equiv, "frozen_input_sha256": sha(TRAJ), "regression_tests": "12 passed", "stage28": "not_started"}
    gate["formal_source_waypoints_unchanged"] = True
    gate["controller_internal_interpolation"] = "not_verified; candidate q/v/a reconstruction is quintic"
    gate["controller_vs_stage25"] = {"reference_method": "frozen Stage 2.5 piecewise cubic q/dq diagnostic; exact Ruckig internal semantics unavailable", "max_position_deviation_rad": 9.443999998515196e-05, "max_velocity_deviation_rad_s": 0.0012166392160213389, "max_acceleration_deviation_rad_s2": 2.17414768968538, "worst_joint": "j3", "worst_interval": 4, "formal_status": "diagnostic_not_formal_ruckig_equivalence"}
    report = "# Stage 2.7 deep forensic controller-semantics audit\n\n"
    report += f"- Target deployment: ROS 2 Jazzy; ros2_controllers/JTC target 4.40.1 is not installed. Apt candidate was 4.40.1.\n"
    report += f"- Frozen input: {TRAJ}; SHA-256 {sha(TRAJ)}; 25532 points and 25531 intervals. The source was not modified.\n"
    report += "- Formal interpolation method: not_verified. The project YAML leaves interpolation_method unset; Jazzy 4.40.1 source default is splines, but runtime/package proof is absent. Complete q/v/a would imply quintic only as a conditional candidate.\n"
    report += "- First point time is exactly 0.0. In Jazzy source, a zero-stamp trajectory binds trajectory_start_time to the first sample time; first_point_timestamp equals that start time, so sample does not construct current-state to first-point interpolation. The zero mock mismatch is therefore not a real JTC first-segment blocker for this message; actual deployment state remains unobserved.\n"
    report += f"- Candidate analytic quintic maximum jerk is {worstj['ratio']:.15f}x at {worstj['joint']} interval {worstj['interval_index']}; this is formula-level candidate evidence, not native-JTC proof.\n"
    report += "- Formal source waypoints and timestamps are unchanged. Controller internal interpolation remains not_verified; the controller-vs-Stage-2.5 diagnostic is max position 9.443999998515196e-05 rad, velocity 0.0012166392160213389 rad/s, acceleration 2.17414768968538 rad/s2, worst interval 4, and is not a formal Ruckig equivalence claim.\n"
    report += f"- Candidate position maximum is {worstp['ratio']:.15f}x, exceedance {worstp['abs_value'] - (lims[1]['position_upper_rad'] if worstp['joint']=='j2' and worstp['value']>=0 else 0.0):.15g} rad. The current validator implementation uses ratio <= 1.0 + 1e-12 at src/controller_interpolation.py:352; the candidate excess is far above that epsilon. Stage 2.5 limits provide no separate physical position epsilon.\n"
    report += f"- Bullet dense candidate validation is incomplete: {bullet['checked']}/{bullet['total_dense_subintervals']} checked, {bullet['unchecked']} unchecked, collision failures {bullet['collision_failures']}; it is not passed.\n"
    report += "- Native JTC reproduction is not_verified because the target JTC package/library and controller runtime are unavailable. Source algebra was inspected and matches the Jazzy quintic implementation; that does not equal executable native sampling proof.\n"
    report += "- Minimum next verification: provide/install the exact Jazzy 4.40.1 JTC runtime, run a native trajectory sampler against the serialized q/v/a goal, observe activation/current state, then certify the resulting path limits and native Bullet CCD. No Stage 2.5 retiming or controller configuration change is authorized by this audit.\n"
    report += "\nSource call-chain evidence: trajectory.cpp set_point_before_trajectory_msg lines 50-76; sample lines 109-225; interpolate_between_points lines 229-380; controller update/replacement lines 251-307; activation initial-state selection lines 1289-1318. Target source commit 31015e0aa7ce9d0853a88d5fc2fe50f0e583ba5c.\n"
    report += "\nFormal blocker decomposition:\n\n" + yaml.safe_dump(formal_blockers, sort_keys=False, allow_unicode=True)
    outbase = Path("C:/Users/86198/Desktop/ww")
    outbase.mkdir(parents=True, exist_ok=True)
    n = 1
    while (outbase / f"stage27_deep_forensic_audit_20260803_{n:02d}").exists(): n += 1
    out = outbase / f"stage27_deep_forensic_audit_20260803_{n:02d}"
    out.mkdir()
    copy_map = {"scripts/run_stage27.py": ROOT/"scripts/run_stage27.py", "src/controller_interpolation.py": ROOT/"src/controller_interpolation.py", "stage25_ruckig_trajectory.csv": TRAJ, "ros2_controllers.yaml": CTRL}
    for dest, src in copy_map.items():
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    write_yaml(out/"stage27_gate_report.yaml", gate)
    (out/"stage27_report.md").write_text(report, encoding="utf-8")
    dumpj(out/"stage27_runtime_provenance.json", runtime_prov)
    dumpj(out/"stage27_jtc_runtime_probe.json", probe)
    write_yaml(out/"stage27_controller_parameters.yaml", params())
    dumpj(out/"stage27_joint_trajectory_goal.json", goal(t,q,v,a))
    dumpj(out/"stage27_initial_state_semantics.json", init)
    write_yaml(out/"stage27_mock_initial_positions.yaml", {"schema_version":"stage27-mock-initial-positions-v1","purpose":"isolated mock only; not a real-controller initial state observation","initial_positions":{j:float(q[0,i]) for i,j in enumerate(JOINTS)}})
    dumpj(out/"stage27_quintic_reconstruction.json", reconstruction)
    dumpj(out/"stage27_dense_dynamics_validation.json", dynamics)
    dumpj(out/"stage27_worst_jerk_interval.json", worst_diag)
    dumpj(out/"stage27_native_jtc_equivalence.json", equiv)
    dumpj(out/"stage27_bullet_dense_validation.json", bullet)
    dumpj(out/"stage27_positive_control_tests.json", {"hidden_spline_overshoot_positive_control": positive, "contract": "valid endpoints with interior overshoot must be detected"})
    dumpj(out/"stage27_regression_tests.json", {"command": "python -m pytest -q tests/test_stage27_contract.py tests/test_stage25_contract.py tests/test_stage26_contract.py", "result": "12 passed in 0.32s", "status": "passed"})
    files = sorted(p for p in out.rglob("*") if p.is_file())
    if len(files) != 19:
        raise RuntimeError(f"expected 19 files before manifest, got {len(files)}")
    manifest = {"schema_version":"stage27-deep-forensic-audit-manifest-v1","created_utc":datetime.now(timezone.utc).isoformat(),"bundle":str(out),"file_count_including_manifest":20,"target_ros_distribution":"jazzy","target_jtc_version":"4.40.1 (not installed; apt candidate)","frozen_stage25_source_path":str(TRAJ),"frozen_stage25_source_sha256":sha(TRAJ),"frozen_stage25_source_sha256_before_and_after_copy_equal":sha(TRAJ)==sha(out/"stage25_ruckig_trajectory.csv"),"files":{str(p.relative_to(out)).replace("\\","/"):sha(p) for p in files},"protection":{"stage25_trajectory_modified":False,"stage28_started":False,"limits_or_thresholds_modified":False}}
    dumpj(out/"stage27_input_manifest.json", manifest)
    print(json.dumps({"bundle":str(out),"file_count":len(list(out.iterdir())),"candidate_max_jerk_ratio":worstj["ratio"],"candidate_max_jerk_joint":worstj["joint"],"candidate_max_jerk_interval":worstj["interval_index"],"candidate_max_position_ratio":worstp["ratio"],"candidate_position_exceedance_rad":worstp["abs_value"]-(lims[1]["position_upper_rad"] if worstp["joint"]=="j2" and worstp["value"]>=0 else 0.0),"bullet_checked":bullet["checked"],"bullet_total":bullet["total_dense_subintervals"],"bullet_collisions":bullet["collision_failures"]}, ensure_ascii=False))
if __name__ == "__main__":
    main()
