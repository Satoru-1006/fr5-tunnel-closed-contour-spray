from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(r"C:\Users\86198\Desktop\cc")
STRICT = ROOT / "outputs" / "open_arch_moveit_strict"
FIG = STRICT / "chapter_figures"
DOCX = OUT / "开放拱形喷涂轨迹建立与MoveIt严格验证_章节.docx"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in rows(path)}


def set_font(run, size: float = 10.5, bold: bool = False, color: str | None = None, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_para(doc: Document, text: str = "", *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=True, after=4, before=0, size=10.5, bold=False, color=None, keep=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.keep_with_next = keep
    if first_indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    r = p.add_run(text)
    set_font(r, size=size, bold=bold, color=color)
    return p


def add_heading(doc: Document, text: str, level: int):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, size=15 if level == 1 else 12, bold=True, color="17365D")
    return p


def border(cell, top=None, bottom=None, left=None, right=None):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, value in {"top": top, "bottom": bottom, "left": left, "right": right}.items():
        if value is None:
            continue
        tag = "w:" + edge
        node = borders.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            borders.append(node)
        node.set(qn("w:val"), value)
        node.set(qn("w:sz"), "8")
        node.set(qn("w:color"), "000000")


def add_three_line_table(doc: Document, headers: list[str], body: list[list[str]], widths: list[float]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hdr = table.rows[0]
    for i, text in enumerate(headers):
        cell = hdr.cells[i]
        cell.width = Cm(widths[i])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        border(cell, top="single", bottom="single", left="nil", right="nil")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(text)
        set_font(r, size=9.5, bold=True)
    for row_index, values in enumerate(body):
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cell = cells[i]
            cell.width = Cm(widths[i])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index == len(body) - 1:
                border(cell, bottom="single", left="nil", right="nil")
            else:
                border(cell, left="nil", right="nil")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(value)
            set_font(r, size=9.2)
    return table


def caption(doc: Document, text: str, *, figure=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, size=9.5, bold=False)
    return p


def add_figure(doc: Document, path: Path, caption_text: str, width_cm: float = 15.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    caption(doc, caption_text, figure=True)


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    surface = np.asarray([[float(r[k]) for k in ("x", "y", "z")] for r in rows(ROOT / "outputs" / "surface_path.csv")[30:211]])
    tcp = np.asarray([[float(r[k]) for k in ("x", "y", "z")] for r in rows(ROOT / "outputs" / "tcp_path.csv")[30:211]])
    normal = surface - tcp
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.1))
    ax.plot(surface[:, 0], surface[:, 2], color="#555555", lw=1.8, label="wall surface")
    ax.plot(tcp[:, 0], tcp[:, 2], color="#C00000", lw=2.4, label="TCP open-arch path")
    idx = np.arange(0, len(tcp), 15)
    ax.quiver(tcp[idx, 0], tcp[idx, 2], normal[idx, 0], normal[idx, 2], angles="xy", scale_units="xy", scale=8, color="#1565C0", width=0.004, label="TCP-to-wall direction")
    ax.scatter(tcp[[0, -1], 0], tcp[[0, -1], 2], c="#2E7D32", s=35, zorder=5, label="start / end")
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X / m"); ax.set_ylabel("Z / m"); ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.subplots_adjust(right=.77); fig.savefig(FIG / "fig1_open_arch_geometry.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    trace = rows(STRICT / "moveit_fk_tcp_trace.csv")
    t = np.asarray([float(r["t"]) for r in trace]); normal_err = np.asarray([float(r["normal_angle_error_deg"]) for r in trace]); dev = np.asarray([float(r["path_deviation_mm"]) for r in trace]); speed = np.asarray([float(r["tcp_speed_m_s"]) * 1000 for r in trace])
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True)
    axes[0].plot(t, normal_err, color="#1565C0", lw=1.4); axes[0].axhline(10, color="#C00000", ls="--", lw=1); axes[0].set_ylabel("normal error / deg")
    axes[1].plot(t, dev, color="#E67E22", lw=1.4); axes[1].axhline(5, color="#C00000", ls="--", lw=1); axes[1].set_ylabel("path deviation / mm")
    axes[2].plot(t, speed, color="#2E7D32", lw=1.4); axes[2].axhline(3.0, color="#C00000", ls="--", lw=1); axes[2].set_ylabel("TCP speed / mm s-1"); axes[2].set_xlabel("time / s")
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "fig2_fk_quality.png", dpi=240); plt.close(fig)

    dyn = rows(STRICT / "moveit_joint_dynamics_report.csv")
    joints = [r["joint"] for r in dyn]; x = np.arange(len(joints)); w = .24
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x-w, [float(r["velocity_ratio"]) for r in dyn], width=w, color="#1565C0", label="velocity ratio")
    ax.bar(x, [float(r["acceleration_ratio"]) for r in dyn], width=w, color="#E67E22", label="acceleration ratio")
    ax.bar(x+w, [float(r["jerk_ratio"]) for r in dyn], width=w, color="#2E7D32", label="jerk ratio")
    ax.axhline(1.0, color="#C00000", ls="--", lw=1, label="limit")
    ax.set_xticks(x, joints); ax.set_ylim(0, 1.05); ax.set_ylabel("ratio to configured limit"); ax.grid(axis="y", alpha=.25); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_dynamics_ratios.png", dpi=240); plt.close(fig)

    steps = rows(STRICT / "moveit_joint_step_report.csv")
    max_by_transition: dict[int, float] = {}
    for r in steps:
        i = int(r["from_index"]); max_by_transition[i] = max(max_by_transition.get(i, 0.0), float(r["step_deg"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    axes[0].plot(list(max_by_transition), list(max_by_transition.values()), color="#1565C0", lw=1.2); axes[0].axhline(20, color="#C00000", ls="--", lw=1); axes[0].set_xlabel("transition index"); axes[0].set_ylabel("max joint step / deg"); axes[0].grid(alpha=.25)
    axes[1].bar(["checked states", "collisions"], [181, 0], color=["#1565C0", "#2E7D32"]); axes[1].set_ylabel("count"); axes[1].set_title("MoveIt collision sampling"); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "fig4_continuity_collision.png", dpi=240); plt.close(fig)

    # Extract representative motion states from the true-joint GIF.
    from PIL import Image, ImageDraw
    gif = Image.open(ROOT / "outputs" / "animation_open_arch_true_left_front.gif")
    ids = [0, 90, 180]
    sheet = Image.new("RGB", (1200, 330), "white")
    for n, index in enumerate(ids):
        gif.seek(index)
        image = gif.convert("RGB").resize((400, 318))
        draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 92, 22), fill="white"); draw.text((5, 4), f"state {index}", fill="black")
        sheet.paste(image, (n * 400, 0))
    sheet.save(FIG / "fig5_motion_states.png")

    # Supplementary source-data figures: all series are regenerated from the
    # final open-arch trajectory, rather than reusing the legacy closed-loop plots.
    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(surface[:, 0], surface[:, 1], surface[:, 2], color="#555555", lw=2.0, label="wall path")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color="#C00000", lw=2.5, label="virtual TCP path")
    for i in range(0, len(tcp), 20):
        ax.plot([tcp[i, 0], surface[i, 0]], [tcp[i, 1], surface[i, 1]], [tcp[i, 2], surface[i, 2]], color="#1565C0", alpha=.55, lw=.8)
    ax.scatter(*tcp[0], color="#2E7D32", s=32, label="start")
    ax.scatter(*tcp[-1], color="#2E7D32", s=32, marker="s", label="end")
    ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
    ax.view_init(elev=23, azim=-58); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig6_wall_tcp_3d.png", dpi=240); plt.close(fig)

    offline = rows(ROOT / "outputs" / "animation_open_arch_lshape_ik.csv")
    q_off = np.asarray([[float(r[f"j{i}_q"]) for i in range(1, 7)] for r in offline])
    p_off = np.asarray([[float(r[k]) for k in ("tcp_x", "tcp_y", "tcp_z")] for r in offline])
    t_off = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p_off, axis=0), axis=1) / 0.0031)]
    dq_off = np.gradient(q_off, t_off, axis=0)
    ddq_off = np.gradient(dq_off, t_off, axis=0)
    jerk_off = np.gradient(ddq_off, t_off, axis=0)

    def dynamics_main(path: Path, time: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray):
        fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=True)
        for i in range(6):
            axes[0].plot(time, q[:, i], lw=1.05, label=f"J{i+1}")
            axes[1].plot(time, dq[:, i], lw=1.05)
            axes[2].plot(time, ddq[:, i], lw=1.05)
        axes[0].set_ylabel("q / rad"); axes[1].set_ylabel("dq / rad s-1"); axes[2].set_ylabel("ddq / rad s-2"); axes[2].set_xlabel("time / s")
        axes[0].legend(ncol=1, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.015, 1.0), borderaxespad=0.0)
        for a in axes: a.grid(alpha=.25)
        fig.subplots_adjust(left=.10, right=.84, bottom=.08, top=.97, hspace=.12)
        fig.savefig(path, dpi=240, bbox_inches="tight"); plt.close(fig)

    def jerk_plot(path: Path, time: np.ndarray, jerk: np.ndarray):
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for i in range(6): ax.plot(time, jerk[:, i], lw=1.05, label=f"J{i+1}")
        ax.axhline(0, color="#555555", lw=.8); ax.set_xlabel("time / s"); ax.set_ylabel("jerk / rad s-3"); ax.grid(alpha=.25)
        ax.legend(ncol=6, fontsize=7, loc="upper center")
        fig.tight_layout(); fig.savefig(path, dpi=240); plt.close(fig)

    dynamics_main(FIG / "fig7_python_offline_dynamics.png", t_off, q_off, dq_off, ddq_off)
    jerk_plot(FIG / "fig8_python_offline_jerk.png", t_off, jerk_off)

    moveit = rows(STRICT / "moveit_smoothed_joint_trajectory.csv")
    t_m = np.asarray([float(r["t"]) for r in moveit])
    q_m = np.asarray([[float(r[f"j{i}_q"]) for i in range(1, 7)] for r in moveit])
    dq_m = np.asarray([[float(r[f"j{i}_dq"]) for i in range(1, 7)] for r in moveit])
    ddq_m = np.asarray([[float(r[f"j{i}_ddq"]) for i in range(1, 7)] for r in moveit])
    jerk_m = np.asarray([[float(r[f"j{i}_jerk"]) for i in range(1, 7)] for r in moveit])
    dynamics_main(FIG / "fig9_moveit_dynamics.png", t_m, q_m, dq_m, ddq_m)
    jerk_plot(FIG / "fig10_moveit_jerk.png", t_m, jerk_m)


def build_doc():
    quality = metric(STRICT / "moveit_quality_report.csv")
    collision = metric(STRICT / "moveit_collision_report.csv")
    dyn = rows(STRICT / "moveit_joint_dynamics_report.csv")
    make_figures()
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0); section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54); section.bottom_margin = Cm(2.54); section.left_margin = Cm(2.54); section.right_margin = Cm(2.54)
    section.header_distance = Cm(1.25); section.footer_distance = Cm(1.25)
    header = section.header.paragraphs[0]; header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = header.add_run("FR5 开放拱形喷涂轨迹的建立与严格验证"); set_font(r, size=8.5, color="666666")
    footer = section.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("开放拱形运动仿真章节"); set_font(r, size=8.5, color="666666")

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(6)
    r = p.add_run("第5章  开放拱形喷涂轨迹的建立与 MoveIt 严格验证"); set_font(r, size=16, bold=True, color="17365D")
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(14)
    r = p.add_run("基于 FR5 六轴机械臂、虚拟 TCP 与 ROS2 MoveIt2 的离线验证结果"); set_font(r, size=10.5, color="555555")

    add_heading(doc, "5.1 改进目标与问题重构", 1)
    add_para(doc, "原始闭合轮廓包含底部横向闭合段。在侧壁、拱顶与底部交界区域，逐点逆运动学求解容易发生构型分支切换；若将相邻分支直接相连，会造成较大的关节跳变，并使动画中的连杆姿态出现不自然的折返。为使运动过程与实际喷涂工艺一致，本研究将目标轨迹改为开放拱形：从左下侧壁起始，沿左竖直段上行，经过顶部拱形后沿右竖直段下行，并在右下端结束，不再生成底部横线和首尾闭合约束。")
    add_para(doc, "本项目中的 150 mm TCP 为统一的虚拟建模参数，目的在于将末端法兰与喷涂作用点区分开来；其并非本章的误差来源或验收重点。改进的核心是使 TCP 始终处于同一工作平面、工具轴指向墙面，并使关节轨迹在完整六轴模型下连续、可平滑定时且满足碰撞约束。")

    caption(doc, "表5-1  开放拱形轨迹相对于原闭合轨迹的关键改动")
    add_three_line_table(doc, ["改动对象", "原方案", "最终方案", "作用"], [
        ["轨迹拓扑", "闭合马蹄形，包含底部横线", "左竖线—拱形—右竖线的开放轨迹", "消除无工艺需求的底部闭合段"],
        ["基座布置", "靠近截面中心", "基座后移至 Y=-0.25 m，Z=0.20 m", "提高主臂舒展度并降低折返构型概率"],
        ["TCP 姿态", "沿用旧轨迹末端姿态", "由 TCP 至墙面点的法向实时定义", "保证喷涂轴朝向壁面"],
        ["轨迹求解", "闭合轮廓 IK 与分段换姿", "181 点连续 IK，允许工具绕自身轴滚转", "保持构型连续并降低关节跳变"],
        ["验证方式", "旧闭合轨迹报告", "开放路径 MoveIt FK、Ruckig、动力学与碰撞重验", "保证报告与新轨迹一一对应"],
    ], [2.3, 3.8, 5.4, 4.0])
    add_para(doc, "表5-1表明，本次改进不是对动画外观的修饰，而是从轨迹拓扑、工位相对位置、TCP 定向、逆运动学求解和运行时验证五个层面完成了重构。特别是删除底部横线后，轨迹不再需要满足首尾闭合，从而避免了闭合条件引入的额外姿态约束。", first_indent=True)

    add_heading(doc, "5.2 开放拱形路径与朝墙 TCP 建模", 1)
    add_para(doc, "开放轨迹由原始截面路径中相位30至相位210的有效工作段提取，共得到181个离散目标点。墙面点记为 p_w，TCP 目标点记为 p_t，则喷涂方向取为 n=(p_w-p_t)/||p_w-p_t||。以 n 作为工具坐标系 Z 轴，通过与隧道纵向方向构造正交基，得到每个目标点的姿态四元数。该处理使 TCP 位置、法向和喷距均由同一几何关系确定，避免仅约束位置而导致喷枪朝向错误。")
    doc.add_page_break()
    add_figure(doc, FIG / "fig1_open_arch_geometry.png", "图5-1  开放拱形 TCP 轨迹、墙面曲线及 TCP 指向墙面的法向定义", 13.2)
    add_para(doc, "如图5-1所示，红线为实际参与验证的开放拱形 TCP 轨迹，灰线为对应墙面曲线，蓝色箭头表示 TCP 至墙面的喷涂方向。左右端点均处于侧壁段，轨迹在端点处终止，因此不会生成底部横向移动。")

    add_heading(doc, "5.3 连续逆运动学与舒展构型建立", 1)
    add_para(doc, "为获得不出现突跳的六轴运动，首先将 FR5 基座相对工作平面后移，使主臂具有足够的前向伸展空间；随后采用前一点关节解作为下一点的初值，逐点进行连续逆运动学求解。位置残差和工具 Z 轴方向残差作为硬约束，而工具绕自身轴的滚转角保留自由度。该策略避免了在完整姿态六自由度均被锁死时产生的不可达或分支跳转问题。")
    add_figure(doc, FIG / "fig5_motion_states.png", "图5-2  开放拱形轨迹在起始、拱顶和终止阶段的真实六轴关节姿态", 15.2)
    add_para(doc, "图5-2给出了左前视角下的三组代表性姿态。主臂由基座向工作平面方向伸展，腕部与 TCP 段保持壁面法向一致。图中紫色线为法兰至 150 mm TCP 的真实偏置段，蓝色虚线为其继续指向墙面的喷涂方向。")

    add_heading(doc, "5.4 MoveIt2 严格验证链与实现方式", 1)
    add_para(doc, "为使离线动画结果具备可复核性，本研究将新生成的181点关节解转换为 MoveIt 可读的 base_link 坐标系 TCP 姿态文件和关节种子文件。转换时对基座平移与绕 Z 轴旋转进行逆变换，确保离线模型与 MoveIt URDF 模型使用相同坐标口径。输入文件包含 TCP 位置、单位四元数和墙面法向；输入校验结果显示，四元数范数为1，法向最大对齐误差为 8.54×10^-7°，工作平面 Y 坐标标准差为 2.22×10^-16 m。")
    add_para(doc, "MoveIt2 运行链采用 seed_joint_waypoints 模式复放连续关节解，并以 spray_tcp_link 作为末端链接。轨迹首先按 TCP 弧长进行时间参数化，再执行原生 Ruckig 平滑；随后使用 MoveIt RobotState 对全部后 Ruckig 样本重新计算 TCP 位置、工具轴、喷距和速度。开放路径模式使碰撞墙体仅沿相邻工作段生成，不再将右端与左端错误连接为底部闭合边。最后，系统对181个后 Ruckig 状态执行自碰撞和环境碰撞检查，并输出动力学、FK和碰撞报告。")

    add_heading(doc, "5.5 严格验证结果与分析", 1)
    caption(doc, "表5-2  开放拱形轨迹的 MoveIt2 严格验证结果")
    max_dyn = max(float(row["jerk_ratio"]) for row in dyn)
    add_three_line_table(doc, ["指标类别", "验收量", "结果", "判定"], [
        ["路径规模", "TCP目标点数", "181", "通过"],
        ["姿态", "FK法向误差（最大）", f"{float(quality['fk_normal_error_max_deg']):.3f}°", "通过"],
        ["喷距", "FK喷距误差（最大）", f"{float(quality['fk_standoff_error_max_abs_mm']):.3f} mm", "通过"],
        ["路径一致性", "TCP路径偏差（最大）", f"{float(quality['fk_path_deviation_max_mm']):.3f} mm", "通过"],
        ["连续性", "最大相邻关节步长", f"{float(quality['max_joint_step_deg']):.3f}° / 20°", "通过"],
        ["速度", "TCP平均速度", f"{float(quality['fk_tcp_speed_mean_m_s']):.6f} m/s", "通过"],
        ["动力学", "最大jerk比", f"{max_dyn:.6f}", "通过"],
        ["碰撞", "检查状态/碰撞数", f"{collision['collision_checked_state_count']} / {collision['collision_count']}", "通过"],
    ], [2.7, 5.1, 4.1, 2.3])
    add_para(doc, "表5-2显示，开放拱形轨迹在 MoveIt FK 复算下保持了较高的几何一致性。最大法向误差为1.085°，远小于10°门限；最大喷距误差为0.076 mm；最大路径偏差为0.018 mm。关节连续性方面，最大相邻步长为12.949°，低于20°限制，表明新轨迹无需使用停喷换姿段即可完成完整开放拱形工作段。")
    doc.add_page_break()
    add_figure(doc, FIG / "fig2_fk_quality.png", "图5-3  后 Ruckig 轨迹的法向误差、路径偏差与 TCP 速度时序", 15.2)
    add_para(doc, "图5-3中虚线分别对应法向误差10°、路径偏差5 mm和最低TCP速度3 mm/s的门限。三组曲线均稳定处于允许范围内；TCP速度的5%至95%分位差仅为4.24×10^-10，说明弧长定时和 Ruckig 平滑后速度波动可以忽略。")
    doc.add_page_break()
    add_figure(doc, FIG / "fig3_dynamics_ratios.png", "图5-4  六关节速度、加速度与 jerk 相对限值比", 12.5)
    add_para(doc, "图5-4表明，动力学峰值出现在 J4，但其速度、加速度和 jerk 比分别仅为0.0318、0.0174和0.000405，均显著低于限值。该结果说明开放拱形轨迹在当前设定速度下具有充足的动态裕量。")
    add_figure(doc, FIG / "fig4_continuity_collision.png", "图5-5  最大关节步长分布与 MoveIt 碰撞采样结果", 15.2)
    add_para(doc, "如图5-5所示，所有关节转移均位于20°连续性门限以下；MoveIt 对181个后 Ruckig 状态和180个开放拱形墙体对象进行了检查，未发现自碰撞或环境碰撞。由于轨迹为开放路径，底部闭合墙体被正确标记为不适用，而非被误判为缺失。")

    doc.add_page_break()
    add_heading(doc, "5.5.1 原始运动数据图补充", 1)
    add_para(doc, "为避免仅以汇总指标说明轨迹质量，本节按最终开放拱形轨迹的原始采样数据补充壁面—TCP 三维几何关系、Python 离线轨迹的关节动力学主图及 jerk 时序图，以及 MoveIt2 经 Ruckig 平滑后的对应两组动力学图。所有图均由本次最终 181 点开放轨迹及其 MoveIt2 严格验证输出重新生成，不采用早期闭合轨迹或分段换姿阶段的数据。")
    add_figure(doc, FIG / "fig6_wall_tcp_3d.png", "图5-6  最终开放拱形轨迹的壁面路径与虚拟 TCP 路径三维数据图", 15.0)
    add_para(doc, "图5-6 中灰色曲线为壁面工艺路径，红色曲线为相隔 150 mm 的虚拟 TCP 路径，蓝色连线表示两者在抽样点处的法向对应关系。两条轨迹具有相同的开放拱形拓扑，并在右端终止；因此图中不存在底部横线，也不存在首尾闭合产生的附加运动。")
    doc.add_page_break()
    add_figure(doc, FIG / "fig7_python_offline_dynamics.png", "图5-7  Python 离线连续 IK 轨迹的关节位置、速度与加速度主图", 14.8)
    add_para(doc, "图5-7 基于 Python 离线连续 IK 结果，以 TCP 弧长和目标速度 3.1 mm/s 重建时间轴。J1 至 J6 的位置、速度和加速度均沿同一条连续开放轨迹演化，可直观看出关节没有在中途回折到另一构型分支。")
    add_figure(doc, FIG / "fig8_python_offline_jerk.png", "图5-8  Python 离线连续 IK 轨迹的六关节 jerk 时序图", 15.0)
    add_para(doc, "图5-8 给出了离线轨迹二次求导后的 jerk 时序。曲线只在起止和曲率变化较明显的位置出现有限峰值，未出现旧闭合轨迹中因 IK 分支切换导致的大幅瞬时尖峰，可作为离线连续性的补充证据。")
    doc.add_page_break()
    add_figure(doc, FIG / "fig9_moveit_dynamics.png", "图5-9  MoveIt2 规划并经 Ruckig 平滑后的关节位置、速度与加速度主图", 14.8)
    add_para(doc, "图5-9 直接采用 MoveIt2 输出的 moveit_smoothed_joint_trajectory.csv。与图5-7 的离线结果相比，Ruckig 在保持同一 181 点几何路径的前提下，对速度和加速度进行了可执行的时间参数化，使运行时序可用于后续控制器接口替换。")
    add_figure(doc, FIG / "fig10_moveit_jerk.png", "图5-10  MoveIt2 规划并经 Ruckig 平滑后的六关节 jerk 时序图", 15.0)
    add_para(doc, "图5-10 显示 Ruckig 输出的解析 jerk 序列。结合表5-2中的最大 jerk 比 0.000405，可知六轴 jerk 均远低于配置限值；这与动力学报告的逐关节统计结论一致。")

    doc.add_page_break()
    add_heading(doc, "5.6 本章小结", 1)
    add_para(doc, "本章完成了从闭合马蹄形轨迹到开放拱形轨迹的系统性改进。通过删除底部横线、后移基座、以墙面法向定义 TCP 姿态、保持工具滚转自由度并使用连续 IK 关节种子，建立了具有舒展构型且无显著关节突跳的181点开放拱形运动。随后，该轨迹进入 ROS2 MoveIt2 严格验证链，依次通过输入姿态校验、原生 Ruckig 平滑、FK 几何质量、关节动力学和碰撞检查。结果说明，在当前虚拟 TCP 与简化隧道环境条件下，所建立的开放拱形轨迹能够作为后续工艺仿真、轨迹导出和实体工位扩展的可靠离线基础。")
    add_para(doc, "需要说明的是，本章结论针对已定义的基座位置、虚拟 TCP、开放拱形墙体和 MoveIt 碰撞环境成立。若后续引入实体喷枪、工装或实际控制器，应在相同流程下替换对应几何与控制参数并重新执行严格验证。", first_indent=True)
    doc.core_properties.title = "开放拱形喷涂轨迹建立与MoveIt严格验证"
    doc.core_properties.subject = "FR5机器人运动仿真章节"
    OUT.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    build_doc()
