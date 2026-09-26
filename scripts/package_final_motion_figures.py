from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_open_arch_chapter_doc import make_figures


OUTPUTS = ROOT / "outputs"
STRICT = OUTPUTS / "open_arch_moveit_strict"
SOURCE_FIGURES = STRICT / "chapter_figures"
DEST = Path(r"C:\Users\86198\Desktop\hh")


def metric(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def coverage_scope(out: Path) -> None:
    wall = pd.read_csv(OUTPUTS / "surface_path.csv").iloc[:240][["x", "y", "z"]].to_numpy()
    tcp = pd.read_csv(OUTPUTS / "tcp_path.csv").iloc[30:211][["x", "y", "z"]].to_numpy()
    process_wall = wall[30:211]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(wall[:, 0], wall[:, 2], color="#9aa0a6", lw=2.0, label="full standard horseshoe wall")
    ax.plot(process_wall[:, 0], process_wall[:, 2], color="#2e7d32", lw=4.0, label="final motion work scope")
    ax.plot(tcp[:, 0], tcp[:, 2], color="#c62828", lw=2.2, label="TCP path")
    ax.plot(wall[:31, 0], wall[:31, 2], color="#9aa0a6", lw=4.0, alpha=0.6)
    ax.plot(wall[210:, 0], wall[210:, 2], color="#9aa0a6", lw=4.0, alpha=0.6, label="bottom closure excluded")
    ax.scatter(tcp[0, 0], tcp[0, 2], color="#1565c0", s=45, marker="o", label="start")
    ax.scatter(tcp[-1, 0], tcp[-1, 2], color="#1565c0", s=45, marker="s", label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / m"); ax.set_ylabel("Z / m")
    ax.set_title("Final motion scope on the standard horseshoe section")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=.76); fig.savefig(out, dpi=240, bbox_inches="tight"); plt.close(fig)


def tcp_speed(out: Path) -> None:
    trace = pd.read_csv(STRICT / "moveit_fk_tcp_trace.csv")
    speed = trace["tcp_speed_m_s"].to_numpy() * 1000.0
    t = trace["t"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.plot(t, speed, color="#1565c0", lw=1.7, label="FK TCP speed")
    ax.axhline(3.1, color="#2e7d32", ls="--", lw=1.3, label="commanded speed 3.1 mm/s")
    ax.axhline(3.0, color="#c62828", ls=":", lw=1.3, label="production minimum 3.0 mm/s")
    ax.fill_between(t, np.percentile(speed, 5), np.percentile(speed, 95), color="#1565c0", alpha=0.10, label="P05-P95 band")
    ax.set_xlabel("Time / s"); ax.set_ylabel("TCP speed / mm s$^{-1}$")
    ax.set_title("WSL MoveIt2/Ruckig TCP constant-speed verification")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(left=.10, right=.76, bottom=.15, top=.88)
    fig.savefig(out, dpi=240, bbox_inches="tight"); plt.close(fig)


def joint_steps(out: Path) -> None:
    steps = pd.read_csv(STRICT / "moveit_joint_step_report.csv")
    maximum = steps.groupby("from_index")["step_deg"].max()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].plot(maximum.index, maximum.values, color="#1565c0", lw=1.4)
    axes[0].axhline(20.0, color="#c62828", ls="--", lw=1.2, label="20 deg limit")
    axes[0].set_xlabel("Transition index"); axes[0].set_ylabel("Maximum joint step / deg")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)
    axes[1].hist(maximum.values, bins=16, color="#2e7d32", edgecolor="white")
    axes[1].axvline(float(maximum.max()), color="#c62828", ls="--", lw=1.2, label=f"max = {maximum.max():.2f} deg")
    axes[1].set_xlabel("Maximum joint step / deg"); axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25); axes[1].legend(fontsize=8)
    fig.suptitle("MoveIt2 adjacent-joint continuity distribution")
    fig.tight_layout(); fig.savefig(out, dpi=240); plt.close(fig)


def python_moveit_consistency(out: Path) -> None:
    offline = pd.read_csv(OUTPUTS / "animation_open_arch_lshape_ik.csv")
    moveit = pd.read_csv(STRICT / "moveit_smoothed_joint_trajectory.csv")
    q_off = offline[[f"j{i}_q" for i in range(1, 7)]].to_numpy()
    q_moveit = moveit[[f"j{i}_q" for i in range(1, 7)]].to_numpy()
    difference = np.rad2deg((q_moveit - q_off + np.pi) % (2.0 * np.pi) - np.pi)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True)
    for joint in range(6):
        axes[0].plot(np.rad2deg(q_off[:, joint]), lw=1.1, label=f"J{joint + 1}")
        axes[1].plot(difference[:, joint], lw=1.1, label=f"J{joint + 1}")
    axes[0].set_ylabel("Python IK q / deg"); axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("MoveIt - Python / deg"); axes[1].set_xlabel("Open-arch waypoint index"); axes[1].grid(alpha=0.25)
    axes[0].legend(ncol=6, fontsize=7, loc="upper center")
    axes[1].text(0.02, 0.88, f"maximum absolute difference = {np.max(np.abs(difference)):.3e} deg", transform=axes[1].transAxes)
    fig.suptitle("Python offline IK and WSL MoveIt2 waypoint consistency")
    fig.tight_layout(); fig.savefig(out, dpi=240); plt.close(fig)


def status_dashboard(out: Path) -> None:
    quality = metric(STRICT / "final_quality_report.csv")
    dynamics = pd.read_csv(STRICT / "moveit_joint_dynamics_report.csv")
    collision = metric(STRICT / "moveit_collision_report.csv")
    labels = ["normal error", "path deviation", "joint step", "velocity", "acceleration", "jerk"]
    values = [
        float(quality["fk_normal_error_max_deg"]) / 10.0,
        float(quality["fk_path_deviation_max_mm"]) / 5.0,
        float(quality["max_joint_step_deg"]) / 20.0,
        float(dynamics["velocity_ratio"].max()),
        float(dynamics["acceleration_ratio"].max()),
        float(dynamics["jerk_ratio"].max()),
    ]
    colors = ["#2e7d32" if value <= 1.0 else "#c62828" for value in values]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.6, 1.0]})
    y = np.arange(len(labels))
    axes[0].barh(y, values, color=colors)
    axes[0].axvline(1.0, color="#c62828", ls="--", lw=1.2, label="acceptance limit")
    axes[0].set_yticks(y, labels); axes[0].invert_yaxis(); axes[0].set_xlabel("ratio to acceptance limit")
    axes[0].set_xlim(0, max(1.05, max(values) * 1.15)); axes[0].grid(axis="x", alpha=0.25); axes[0].legend(fontsize=8)
    axes[1].axis("off")
    summary = [
        f"Overall status: {quality['status'].upper()}",
        f"Collision count: {collision['collision_count']}",
        f"TCP speed mean: {float(quality['fk_tcp_speed_mean_m_s']) * 1000:.4f} mm/s",
        f"Maximum joint step: {float(quality['max_joint_step_deg']):.3f} deg",
        f"Path deviation max: {float(quality['fk_path_deviation_max_mm']):.4f} mm",
        f"TCP source: {quality['tool_tcp_source']}",
        "Trajectory: 181-point open arch",
        "Bottom closure: excluded",
    ]
    axes[1].text(0.03, 0.95, "\n".join(summary), va="top", fontsize=10.5, linespacing=1.55, bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f5f7fa", "edgecolor": "#c8cdd2"})
    fig.suptitle("Final Python + WSL MoveIt2 motion-validation dashboard")
    fig.tight_layout(); fig.savefig(out, dpi=240); plt.close(fig)


def write_readme(path: Path, figures: list[tuple[str, str]]) -> None:
    lines = [
        "# FR5标准马蹄形最终运动仿真结果图包",
        "",
        "本图包仅包含机械臂运动仿真，不包含流体、粒子、沉积或膜厚结果。",
        "最终方案为左侧竖段—上部拱形—右侧竖段的181点开口拱形雨刮式轨迹，底部闭合段已删除。",
        "",
        "## 图表清单",
        "",
    ]
    for filename, description in figures:
        lines.append(f"- `{filename}`：{description}")
    lines.extend(
        [
            "",
            "## 数据来源",
            "",
            "- Python：`animation_open_arch_lshape_ik.csv`、`surface_path.csv`、`tcp_path.csv`。",
            "- WSL/MoveIt2：`moveit_smoothed_joint_trajectory.csv`、`moveit_fk_tcp_trace.csv`、动力学、碰撞与连续性报告。",
            "- TCP约定：150 mm虚拟TCP，来源仍标记为`assumed_150mm_placeholder`。",
            "- 最终动画：15秒四视角正交投影版本。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    style()
    DEST.mkdir(parents=True, exist_ok=True)
    python_dir = DEST / "01_Python离线计算"
    moveit_dir = DEST / "02_WSL_MoveIt2验证"
    summary_dir = DEST / "03_综合结果"
    data_dir = DEST / "04_源数据"
    for directory in (python_dir, moveit_dir, summary_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    make_figures()

    figure_map = [
        (SOURCE_FIGURES / "fig1_open_arch_geometry.png", python_dir / "01_开口拱形几何与TCP法向.png", "最终开口拱形壁面、TCP和朝墙法向"),
        (SOURCE_FIGURES / "fig6_wall_tcp_3d.png", python_dir / "02_壁面与TCP三维路径.png", "Python几何路径的三维关系"),
        (None, python_dir / "03_标准马蹄形运动范围.png", "标准马蹄形上最终工作范围及排除的底边"),
        (SOURCE_FIGURES / "fig7_python_offline_dynamics.png", python_dir / "04_Python离线关节位置速度加速度.png", "Python连续IK的q、dq和ddq"),
        (SOURCE_FIGURES / "fig8_python_offline_jerk.png", python_dir / "05_Python离线关节Jerk.png", "Python离线轨迹六关节jerk"),
        (SOURCE_FIGURES / "fig2_fk_quality.png", moveit_dir / "06_MoveIt2_FK轨迹质量.png", "法向误差、路径偏差与TCP速度"),
        (SOURCE_FIGURES / "fig3_dynamics_ratios.png", moveit_dir / "07_MoveIt2动力学限值比.png", "速度、加速度和jerk相对限值"),
        (SOURCE_FIGURES / "fig4_continuity_collision.png", moveit_dir / "08_MoveIt2连续性与碰撞.png", "相邻关节步长及碰撞采样"),
        (SOURCE_FIGURES / "fig9_moveit_dynamics.png", moveit_dir / "09_MoveIt2_Ruckig关节位置速度加速度.png", "Ruckig定时后的q、dq和ddq"),
        (SOURCE_FIGURES / "fig10_moveit_jerk.png", moveit_dir / "10_MoveIt2_Ruckig关节Jerk.png", "Ruckig轨迹六关节jerk"),
        (None, moveit_dir / "11_MoveIt2_TCP匀速验证.png", "FK TCP速度及3.0 mm/s生产下限"),
        (None, moveit_dir / "12_MoveIt2关节步长分布.png", "相邻轨迹点最大关节变化分布"),
        (None, summary_dir / "13_Python与MoveIt2关节轨迹一致性.png", "Python IK与MoveIt2 waypoint逐点一致性"),
        (SOURCE_FIGURES / "fig5_motion_states.png", summary_dir / "14_机械臂典型运动状态.png", "起点、中点和终点真实六轴构型"),
        (None, summary_dir / "15_最终验收状态总览.png", "几何、动力学、连续性和碰撞综合状态"),
        (OUTPUTS / "_animation_standard_horseshoe_wiper_layout_review.png", summary_dir / "16_最终四视角动画版式.png", "最终动画代表帧与四视角布局"),
    ]

    coverage_scope(python_dir / "03_标准马蹄形运动范围.png")
    tcp_speed(moveit_dir / "11_MoveIt2_TCP匀速验证.png")
    joint_steps(moveit_dir / "12_MoveIt2关节步长分布.png")
    python_moveit_consistency(summary_dir / "13_Python与MoveIt2关节轨迹一致性.png")
    status_dashboard(summary_dir / "15_最终验收状态总览.png")
    for source, destination, _ in figure_map:
        if source is not None:
            copy(source, destination)

    animation_out = summary_dir / "17_最终15秒四视角运动动画.gif"
    copy(OUTPUTS / "animation_standard_horseshoe_wiper_multiview.gif", animation_out)
    validation_out = summary_dir / "最终动画验证数据.json"
    copy(OUTPUTS / "animation_standard_horseshoe_wiper_multiview_validation.json", validation_out)

    data_sources = [
        OUTPUTS / "animation_open_arch_lshape_ik.csv",
        OUTPUTS / "surface_path.csv",
        OUTPUTS / "tcp_path.csv",
        STRICT / "moveit_smoothed_joint_trajectory.csv",
        STRICT / "moveit_fk_tcp_trace.csv",
        STRICT / "moveit_joint_dynamics_report.csv",
        STRICT / "moveit_joint_step_report.csv",
        STRICT / "moveit_collision_report.csv",
        STRICT / "final_quality_report.csv",
        STRICT / "final_acceptance_summary.json",
        STRICT / "moveit_runtime.log",
    ]
    for source in data_sources:
        copy(source, data_dir / source.name)

    descriptions = [(str(destination.relative_to(DEST)), description) for _, destination, description in figure_map]
    descriptions.append((str(animation_out.relative_to(DEST)), "最终15秒四视角运动动画"))
    write_readme(DEST / "README_图表与数据说明.md", descriptions)

    manifest = {
        "package_root": str(DEST),
        "scope": "motion simulation only; no fluid, particle or deposition figures",
        "final_path": "181-point open arch: left wall -> upper arch -> right wall",
        "python_figure_count": 5,
        "wsl_moveit_figure_count": 7,
        "summary_png_count": 4,
        "animation_count": 1,
        "total_visual_count": 17,
        "source_data_file_count": len(data_sources),
        "figures": [str(destination.relative_to(DEST)) for _, destination, _ in figure_map] + [str(animation_out.relative_to(DEST))],
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
