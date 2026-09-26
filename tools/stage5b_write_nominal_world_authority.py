"""Persist the Stage 5B nominal-world authority and its claim fences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "STAGE5B_SHADOW"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    pose = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_tcp_poses_base_link.csv"
    seed = ROOT / "outputs" / "internal_wiper_moveit_inputs" / "open_arch_seed_joints.csv"
    config = ROOT / "config" / "ik_graph.yaml"
    generator = ROOT / "scripts" / "stage5a_mock_execution.py"
    candidate_env = ROOT / "outputs" / "STAGE5AC_FINAL_CLOSURE" / "STAGE5AC_CANDIDATE_ENVIRONMENT.json"
    h45 = ROOT / "outputs" / "stage3_h4_5_robot_fixture_geometry_codesign_20260809T000000Z" / "stage3_h4_5_selected_candidate.json"
    artifact = {
        "schema_version": "stage5b-nominal-world-authority-v1",
        "stage": "5B",
        "status": "AUTHORITATIVE_FOR_SOFTWARE_SIMULATION_ONLY",
        "claim_fence": {
            "physical_installation_alignment": "UNVERIFIED",
            "hardware_clearance": "UNVERIFIED",
            "exact_articulated_self_ccd": "not_available",
            "clearance": "not_available",
            "collision_method": "adaptive_discrete_interpolation",
        },
        "authoritative_inputs": {
            "pose_csv": str(pose),
            "pose_csv_sha256": sha256(pose),
            "seed_csv": str(seed),
            "seed_csv_sha256": sha256(seed),
            "ik_graph_config": str(config),
            "ik_graph_config_sha256": sha256(config),
        },
        "nominal_simulation_world": {
            "frame_id": "base_link",
            "robot_base_frame": "base_link",
            "tunnel_frame": "base_link",
            "base_to_tunnel_transform": {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0], "identity": True},
            "source_geometry": "181-point ON-state open-arch TCP pose/normal sequence",
            "generator": "scripts/stage5a_mock_execution.py::make_environment",
            "generator_sha256": sha256(generator),
            "stand_off_m": 0.260,
            "wall_thickness_m": 0.025,
            "transverse_extent_m": 1.10,
            "segment_stride": 1,
            "open_path": True,
            "floor_included": False,
            "candidate_environment_snapshot": str(candidate_env),
        },
        "x_minus_050_shadow": {
            "status": "SHADOW_ONLY_UNVERIFIED_NOT_PROMOTED",
            "environment_snapshot": str(candidate_env),
            "translation_m": [-0.50, 0.0, 0.0],
            "physical_pose_authority": "unverified_shadow_only",
            "reason": "derived Stage5AC layout probe; no authoritative CAD, calibration, URDF/SDF installation transform, or hardware measurement supports this translation",
        },
        "historical_context_not_authority": {
            "h45_selected_candidate": str(h45),
            "h45_status": "bounded experimental fixture placement; not the Stage5B nominal transform",
            "stage5ar_bullet_mesh_compensation": "native probe bookkeeping; not physical installation calibration",
        },
        "decision": "Use the identity base_link software scene as the nominal simulation authority. Keep x=-0.50 m available only as a separately labelled shadow; neither establishes physical safety or promotes a trajectory.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STAGE5B_NOMINAL_WORLD_AUTHORITY.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = f"""# Stage 5B nominal world authority

## Decision

The nominal world for software simulation is the deterministic ON-state open-arch scene generated from the authoritative 181-point input pair in `outputs/internal_wiper_moveit_inputs/`. The robot and tunnel are represented in `base_link`; the software planning-scene transform is identity, with the formal 0.260 m process stand-off, 0.025 m wall thickness, 1.10 m transverse extent, stride 1, open path, and no floor closure.

This is an authority for reproducible MoveIt2/Gazebo software simulation only. It is not a calibrated physical installation model.

## `x=-0.50 m` decision

`x=-0.50 m` remains `SHADOW_ONLY_UNVERIFIED_NOT_PROMOTED`. It is a Stage5AC derived layout probe that produced a collision-free node screen in one shadow, but it has no authoritative CAD, calibration, URDF/SDF installation transform, or hardware measurement behind it. It cannot be used to erase the nominal-world collision finding or to claim clearance.

## Consequences

- Nominal-world collisions remain genuine software-model findings and must stay visible.
- The exact articulated self-CCD and physical clearance domains remain unavailable/unverified.
- Any candidate trajectory must pass full FK, dynamics, post-Ruckig, collision, and execution checks in the selected software world before it can be called a software-only pass.
- The JSON beside this report records the input identities and claim fences.
"""
    (OUT / "STAGE5B_NOMINAL_WORLD_AUTHORITY.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": artifact["status"], "output": str(OUT / "STAGE5B_NOMINAL_WORLD_AUTHORITY.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
