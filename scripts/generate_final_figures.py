from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def metric_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_float(value: str | float | int, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def status_color(status: str) -> str:
    status = status.lower()
    if status == "pass":
        return "#2e7d32"
    if status in {"waived", "pass or waived"}:
        return "#8a6d1d"
    if status in {"fail", "missing"}:
        return "#b3261e"
    return "#5f6368"


def plot_final_status_dashboard(outputs: Path, out_dir: Path) -> Path:
    final_quality = metric_map(outputs / "final_quality_report.csv")
    readiness = load_json(outputs / "production_readiness_check.json")
    current = load_json(outputs / "current_goal_state_verification.json")

    cards = [
        ("总目标", current.get("overall_goal_status", "missing"), "complete"),
        ("生产就绪", current.get("production_readiness_status", readiness.get("overall_status", "missing")), "pass"),
        ("严格审计", current.get("strict_audit_status", final_quality.get("strict_audit_status", "missing")), "pass"),
        ("碰撞检查", final_quality.get("collision_status", "missing"), "pass"),
        ("TCP 速度", final_quality.get("tcp_speed_production_status", "missing"), "pass"),
        ("IK 连续性", final_quality.get("joint_continuity_status", "missing"), "pass"),
        ("TCP 口径", final_quality.get("tool_tcp_acceptance_status", "missing"), "waived"),
        ("自动测试", current.get("pytest_status", "missing"), "pass"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("最终状态总览", fontsize=18, fontweight="bold")
    for ax, (title, actual, expected) in zip(axes.ravel(), cards):
        actual_str = str(actual)
        ok = actual_str == expected or (title == "总目标" and actual_str == "complete")
        color = status_color("pass" if ok else actual_str)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f7f9fb")
        ax.text(0.05, 0.78, title, transform=ax.transAxes, fontsize=13, color="#303134")
        ax.text(0.05, 0.42, actual_str, transform=ax.transAxes, fontsize=24, fontweight="bold", color=color)
        ax.text(0.05, 0.15, f"期望: {expected}", transform=ax.transAxes, fontsize=10, color="#5f6368")

    metrics = (
        f"碰撞数={final_quality.get('collision_count', 'missing')} | "
        f"工作段最大步长={final_quality.get('process_max_joint_step_deg', 'missing')} deg | "
        f"停喷换支最大插值步长={final_quality.get('reorientation_transition_max_interpolated_step_deg', 'missing')} deg | "
        f"TCP source={final_quality.get('tool_tcp_source', 'missing')}"
    )
    fig.text(0.5, 0.02, metrics, ha="center", fontsize=10, color="#303134")
    out = out_dir / "final_status_dashboard.png"
    save(fig, out)
    return out


def plot_closed_contour_path(outputs: Path, out_dir: Path) -> Path:
    images = [
        outputs / "closed_contour_section.png",
        outputs / "path_3d.png",
    ]
    existing = [path for path in images if path.exists()]
    if not existing:
        raise FileNotFoundError("No closed contour/path image exists in outputs.")

    fig, axes = plt.subplots(1, len(existing), figsize=(6.5 * len(existing), 5.2))
    if len(existing) == 1:
        axes = [axes]
    titles = {
        "closed_contour_section.png": "闭合轮廓截面",
        "path_3d.png": "三维 TCP/壁面路径",
    }
    for ax, path in zip(axes, existing):
        ax.imshow(plt.imread(path))
        ax.set_title(titles.get(path.name, path.stem), fontsize=13)
        ax.axis("off")
    out = out_dir / "closed_contour_path_overview.png"
    save(fig, out)
    return out


def copy_existing_figure(source: Path, out_dir: Path, target_name: str) -> Path | None:
    if not source.exists():
        return None
    out = out_dir / target_name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, out)
    return out


def plot_tcp_speed_profile(outputs: Path, out_dir: Path) -> Path:
    df = pd.read_csv(outputs / "moveit_fk_tcp_trace.csv")
    final_quality = metric_map(outputs / "final_quality_report.csv")
    mean_speed = as_float(final_quality.get("fk_tcp_speed_mean_m_s", "nan"))
    min_speed = as_float(final_quality.get("production_min_tcp_speed_m_s", "0.003"))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(df["t"], df["tcp_speed_m_s"], color="#1565c0", linewidth=1.8, label="TCP 速度")
    ax.axhline(min_speed, color="#c62828", linestyle="--", linewidth=1.4, label=f"最低要求 {min_speed:g} m/s")
    if np.isfinite(mean_speed):
        ax.axhline(mean_speed, color="#2e7d32", linestyle=":", linewidth=1.8, label=f"平均 {mean_speed:.6f} m/s")
    ax.set_title("TCP 速度曲线", fontsize=15, fontweight="bold")
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("速度 (m/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    out = out_dir / "tcp_speed_profile.png"
    save(fig, out)
    return out


def plot_joint_dynamics(outputs: Path, out_dir: Path) -> Path:
    df = pd.read_csv(outputs / "moveit_joint_dynamics_report.csv")
    joints = df["joint"].astype(str).tolist()
    x = np.arange(len(joints))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - width, df["velocity_ratio"], width, label="速度比", color="#1565c0")
    ax.bar(x, df["acceleration_ratio"], width, label="加速度比", color="#7b1fa2")
    ax.bar(x + width, df["jerk_ratio"], width, label="Jerk 比", color="#ef6c00")
    ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.2, label="限制=1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(joints)
    ax.set_ylim(0, max(1.05, float(df[["velocity_ratio", "acceleration_ratio", "jerk_ratio"]].max().max()) * 1.25))
    ax.set_ylabel("占限制比例")
    ax.set_title("关节速度 / 加速度 / Jerk 限制占比", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    out = out_dir / "joint_dynamics_ratios.png"
    save(fig, out)
    return out


def plot_strict_moveit_ruckig_velocity_acceleration(outputs: Path, out_dir: Path) -> Path:
    trajectory = pd.read_csv(outputs / "moveit_smoothed_joint_trajectory.csv")
    dynamics = pd.read_csv(outputs / "moveit_joint_dynamics_report.csv")
    t = trajectory["t"].astype(float)
    joints = dynamics["joint"].astype(str).tolist()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True)
    for joint in joints:
        dq = trajectory[f"{joint}_dq"].astype(float)
        ddq = trajectory[f"{joint}_ddq"].astype(float)
        v_limit = float(dynamics.loc[dynamics["joint"] == joint, "velocity_limit"].iloc[0])
        a_limit = float(dynamics.loc[dynamics["joint"] == joint, "acceleration_limit"].iloc[0])
        axes[0].plot(t, dq / v_limit, linewidth=1.1, label=joint)
        axes[1].plot(t, ddq / a_limit, linewidth=1.1, label=joint)

    axes[0].axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.0)
    axes[0].axhline(-1.0, color="#b3261e", linestyle="--", linewidth=1.0)
    axes[1].axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.0)
    axes[1].axhline(-1.0, color="#b3261e", linestyle="--", linewidth=1.0)
    axes[0].set_title("Strict MoveIt/Ruckig Velocity Ratio Time Series", fontsize=14, fontweight="bold")
    axes[1].set_title("Strict MoveIt/Ruckig Acceleration Ratio Time Series", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("dq / limit")
    axes[1].set_ylabel("ddq / limit")
    axes[1].set_xlabel("time / s")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    out = out_dir / "strict_moveit_ruckig_velocity_acceleration.png"
    save(fig, out)
    return out


def plot_strict_moveit_ruckig_jerk(outputs: Path, out_dir: Path) -> Path:
    trajectory = pd.read_csv(outputs / "moveit_smoothed_joint_trajectory.csv")
    dynamics = pd.read_csv(outputs / "moveit_joint_dynamics_report.csv")
    t = trajectory["t"].astype(float)
    joints = dynamics["joint"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for joint in joints:
        jerk = trajectory[f"{joint}_jerk"].astype(float)
        jerk_limit = float(dynamics.loc[dynamics["joint"] == joint, "jerk_limit"].iloc[0])
        ax.plot(t, jerk / jerk_limit, linewidth=1.1, label=joint)

    ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.0, label="limit")
    ax.axhline(-1.0, color="#b3261e", linestyle="--", linewidth=1.0)
    max_ratio = float(dynamics["jerk_ratio"].max())
    p95_ratio = float((dynamics["p95_jerk"] / dynamics["jerk_limit"]).max())
    ax.set_title("Strict MoveIt/Ruckig Jerk Ratio Time Series", fontsize=14, fontweight="bold")
    ax.set_xlabel("time / s")
    ax.set_ylabel("jerk / limit")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.text(
        0.01,
        0.96,
        f"max jerk ratio = {max_ratio:.4f}; max p95 jerk ratio = {p95_ratio:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        color="#303134",
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "alpha": 0.88},
    )
    out = out_dir / "strict_moveit_ruckig_jerk_timeseries.png"
    save(fig, out)
    return out


def plot_segmented_reorientation_plan(outputs: Path, out_dir: Path) -> Path:
    plan = pd.read_csv(outputs / "segmented_process_plan.csv")
    final_quality = metric_map(outputs / "final_quality_report.csv")
    limit = as_float(final_quality.get("production_joint_step_limit_deg", "20.0"), 20.0)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[1.1, 2.0], sharex=True)
    colors = {"process": "#2e7d32", "smooth_reorientation_stop": "#ef6c00"}
    labels_seen: set[str] = set()
    for _, row in plan.iterrows():
        typ = str(row["type"])
        start = float(row["start_index"])
        end = float(row["end_index"])
        label = "工作段(喷涂)" if typ == "process" else "停喷换支"
        ax0.barh(
            0,
            max(end - start, 1.0),
            left=start,
            height=0.55,
            color=colors.get(typ, "#5f6368"),
            label=label if label not in labels_seen else None,
        )
        labels_seen.add(label)
    ax0.set_yticks([])
    ax0.set_title("IK 大跳变与分段停喷换支计划", fontsize=15, fontweight="bold")
    ax0.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.35))

    process = plan[plan["type"] == "process"]
    transition = plan[plan["type"] != "process"]
    ax1.bar(
        process["segment_id"].astype(int),
        process["max_joint_step_deg"].astype(float),
        color="#2e7d32",
        label="工作段最大关节步长",
    )
    ax1.scatter(
        transition["segment_id"].astype(int),
        transition["max_joint_step_deg"].astype(float),
        color="#ef6c00",
        s=55,
        label="原始大跳变(已停喷换支)",
        zorder=3,
    )
    ax1.axhline(limit, color="#b3261e", linestyle="--", label=f"工作段限制 {limit:g} deg")
    ax1.set_xlabel("分段 ID")
    ax1.set_ylabel("最大相邻关节步长 (deg)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right")

    note = (
        f"工作段最大 {final_quality.get('process_max_joint_step_deg', 'missing')} deg；"
        f"停喷换支最大插值步长 {final_quality.get('reorientation_transition_max_interpolated_step_deg', 'missing')} deg"
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=10, color="#303134")
    out = out_dir / "segmented_reorientation_plan.png"
    save(fig, out)
    return out


def plot_process_joint_step_distribution(outputs: Path, out_dir: Path) -> Path:
    steps = pd.read_csv(outputs / "moveit_joint_step_report.csv")
    plan = pd.read_csv(outputs / "segmented_process_plan.csv")
    final_quality = metric_map(outputs / "final_quality_report.csv")
    limit = as_float(final_quality.get("production_joint_step_limit_deg", "20.0"), 20.0)
    transition_pairs = {
        (int(row.start_index), int(row.end_index))
        for row in plan.itertuples()
        if str(row.type) != "process"
    }
    process_steps = steps[
        ~steps.apply(lambda row: (int(row["from_index"]), int(row["to_index"])) in transition_pairs, axis=1)
    ]["step_deg"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.hist(process_steps, bins=35, color="#1565c0", alpha=0.82)
    ax.axvline(limit, color="#b3261e", linestyle="--", linewidth=1.5, label=f"限制 {limit:g} deg")
    ax.axvline(process_steps.max(), color="#2e7d32", linestyle=":", linewidth=1.8, label=f"最大 {process_steps.max():.3f} deg")
    ax.set_title("工作段关节步长分布", fontsize=15, fontweight="bold")
    ax.set_xlabel("相邻采样点关节步长 (deg)")
    ax.set_ylabel("数量")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    out = out_dir / "process_joint_step_distribution.png"
    save(fig, out)
    return out


def plot_collision_summary(outputs: Path, out_dir: Path) -> Path:
    collision = metric_map(outputs / "moveit_collision_report.csv")
    rows = [
        ("环境物体数", collision.get("collision_environment_object_count", "missing")),
        ("检查状态数", collision.get("collision_checked_state_count", "missing")),
        ("碰撞次数", collision.get("collision_count", "missing")),
        ("首个碰撞索引", collision.get("first_collision_index", "missing")),
        ("底部闭合碰撞", collision.get("include_bottom_closure_collision", "missing")),
        ("状态", collision.get("status", "missing")),
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.0, 1.35]})
    checked = as_float(collision.get("collision_checked_state_count", "0"), 0.0)
    count = as_float(collision.get("collision_count", "0"), 0.0)
    ax0.bar(["检查状态", "碰撞"], [checked, count], color=["#1565c0", "#2e7d32" if count == 0 else "#b3261e"])
    ax0.set_title("碰撞检查数量", fontsize=14, fontweight="bold")
    ax0.set_ylabel("数量")
    ax0.grid(True, axis="y", alpha=0.25)

    ax1.axis("off")
    table = ax1.table(
        cellText=rows,
        colLabels=["指标", "值"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.45)
    ax1.set_title("碰撞检查结果表", fontsize=14, fontweight="bold")

    out = out_dir / "collision_check_summary.png"
    save(fig, out)
    return out


def write_index(generated: list[Path], out_dir: Path) -> Path:
    index = out_dir / "README_final_figures.md"
    lines = [
        "# 最终数据图清单",
        "",
        "本目录由 `python scripts/generate_final_figures.py` 自动生成。",
        "",
    ]
    for path in generated:
        lines.append(f"- `{path.name}`")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def generate(outputs: Path, out_dir: Path) -> list[Path]:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    generated.append(plot_final_status_dashboard(outputs, out_dir))
    generated.append(plot_closed_contour_path(outputs, out_dir))
    generated.append(plot_tcp_speed_profile(outputs, out_dir))
    generated.append(plot_joint_dynamics(outputs, out_dir))
    generated.append(plot_strict_moveit_ruckig_velocity_acceleration(outputs, out_dir))
    generated.append(plot_strict_moveit_ruckig_jerk(outputs, out_dir))
    generated.append(plot_segmented_reorientation_plan(outputs, out_dir))
    generated.append(plot_process_joint_step_distribution(outputs, out_dir))
    generated.append(plot_collision_summary(outputs, out_dir))

    for source, name in [
        (outputs / "coverage_heatmap.png", "coverage_heatmap.png"),
        (outputs / "dynamics_main.png", "offline_dynamics_main.png"),
        (outputs / "dynamics_jerk.png", "offline_dynamics_jerk.png"),
        (outputs / "final_moveit_dynamics_main.png", "moveit_dynamics_main.png"),
        (outputs / "final_moveit_dynamics_jerk.png", "moveit_dynamics_jerk.png"),
        (outputs / "animation.gif", "animation.gif"),
        (outputs / "animation_moveit.gif", "animation_moveit.gif"),
    ]:
        copied = copy_existing_figure(source, out_dir, name)
        if copied is not None:
            generated.append(copied)

    generated.append(write_index(generated, out_dir))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final delivery figures from existing validation outputs.")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "final_figures")
    args = parser.parse_args()

    generated = generate(args.outputs_dir, args.out_dir)
    print(f"Wrote {len(generated)} final figure/artifact files to {args.out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
