#!/usr/bin/env python3
"""Add the exact native execution provenance to an already-built bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def native_command(output_dir: str, variant: str, export: bool, global_limit: int, pair_limit: int) -> str:
    return (
        "source /opt/ros/jazzy/setup.bash; "
        "source /home/robot/fr5_ros2_ws/install/setup.bash; "
        "source /home/robot/robotfucker_stage23a1/native_install/"
        "stage21_native_diagnostics/share/stage21_native_diagnostics/local_setup.bash; "
        "ros2 launch /mnt/c/Users/86198/Desktop/robotfucker/tools/"
        "stage23a_native_contact_launch.py "
        "candidate_csv:=/home/robot/robotfucker_stage23a1/raw/candidate_provenance.csv "
        "pose_csv:=/home/robot/robotfucker_stage23a1/raw/tcp_poses_base_link.csv "
        f"output_dir:={output_dir} export_all_contacts:={'true' if export else 'false'} "
        "distance_request:=false "
        f"max_contacts:={global_limit} max_contacts_per_pair:={pair_limit} "
        f"variant_filter:={variant}"
    )


def main() -> None:
    root = Path("/home/robot/robotfucker_stage23a1")
    bundle = root / "merged/final_bundle_v2"
    frozen = json.loads((bundle / "frozen_input_manifest.json").read_text(encoding="utf-8"))
    fcl_repro = json.loads((bundle / "fcl_reproducibility.json").read_text(encoding="utf-8"))
    bullet_repro = json.loads((bundle / "bullet_reproducibility.json").read_text(encoding="utf-8"))
    protected = json.loads((bundle / "protected_outputs_hash.json").read_text(encoding="utf-8"))["files"]
    complete = json.loads((bundle / "run_manifests/stage23a1_complete_bundle.json").read_text(encoding="utf-8"))
    fcl_canonical = complete["backend_results"]["fcl"]["canonical_contact_sha256"]
    bullet_canonical = complete["backend_results"]["bullet"]["canonical_contact_sha256"]
    for run in fcl_repro["runs"]:
        run["canonical_contact_sha256"] = fcl_canonical
    for run in bullet_repro["runs"]:
        run["canonical_contact_sha256"] = bullet_canonical
    atomic_json(bundle / "fcl_reproducibility.json", fcl_repro)
    atomic_json(bundle / "bullet_reproducibility.json", bullet_repro)
    env = {
        "ROS_VERSION": "2",
        "ROS_DISTRO": "jazzy",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "ROS_LOCALHOST_ONLY": "0",
        "ROS_DOMAIN_ID": "0",
        "native_prefix": "/home/robot/robotfucker_stage23a1/native_install/stage21_native_diagnostics",
        "moveit_workspace_prefix": "/home/robot/fr5_ros2_ws/install",
        "ros_prefix": "/opt/ros/jazzy",
        "filesystem": "WSL2 Ubuntu-24.04-D ext4 (/home/robot/robotfucker_stage23a1)",
        "thread_policy": "one native probe process per run; no shared mutable Bullet manager",
    }
    env_path = bundle / "run_manifests/native_environment.json"
    atomic_json(env_path, env)
    runs = []
    for run in fcl_repro["runs"]:
        runs.append({
            **run, "backend": "FCL", "variant": "case_C_official_baseline",
            "collision_request": {"contacts": True, "max_contacts": 524288, "max_contacts_per_pair": 65536, "is_done": "nullptr", "early_termination": False},
            "command": native_command(run["raw_contact_file"].rsplit("/native_collision_contacts.csv", 1)[0], "case_C_official_baseline", True, 524288, 65536),
            "environment": env,
        })
    for run in bullet_repro["runs"]:
        runs.append({
            **run, "backend": "Bullet", "variant": "alternative_bullet_diagnostic",
            "collision_request": {"contacts": True, "max_contacts": 524288, "max_contacts_per_pair": 65536, "is_done": "nullptr", "early_termination": False},
            "command": native_command(run["raw_contact_file"].rsplit("/native_collision_contacts.csv", 1)[0], "alternative_bullet_diagnostic", True, 524288, 65536),
            "environment": env,
        })
    diagnostic_runs = [
        ("FCL", "fcl_official_262144_32768_diag", "case_C_official_baseline", 262144, 32768),
        ("FCL", "fcl_official_524288_65536_diag", "case_C_official_baseline", 524288, 65536),
        ("Bullet", "alternative_bullet_262144_32768_diag", "alternative_bullet_diagnostic", 262144, 32768),
        ("Bullet", "alternative_bullet_full_524288_65536", "alternative_bullet_diagnostic", 524288, 65536),
    ]
    for backend, directory, variant, global_limit, pair_limit in diagnostic_runs:
        runs.append({
            "run_id": f"{backend.lower()}_{directory}", "backend": backend, "variant": variant,
            "output_dir": str(root / "raw" / directory),
            "collision_request": {"contacts": True, "max_contacts": global_limit, "max_contacts_per_pair": pair_limit, "is_done": "nullptr", "early_termination": False},
            "command": native_command(str(root / "raw" / directory), variant, False, global_limit, pair_limit),
            "environment": env,
        })
    native = {
        "schema_version": "2.3A.1",
        "purpose": "native MoveIt2 FCL/Bullet contact evidence provenance",
        "input_hashes": {key: value for key, value in frozen.items() if key.endswith("sha256") or key.endswith("_hash")},
        "candidate_count": 1439, "waypoint_count": 720,
        "collision_request": {"contacts": True, "max_contacts": 524288, "max_contacts_per_pair": 65536, "is_done": "nullptr", "early_termination": False},
        "limit_ladder": [{"global": 64, "per_pair": 8}, {"global": 256, "per_pair": 32}, {"global": 1024, "per_pair": 128}, {"global": 4096, "per_pair": 512}, {"global": 16384, "per_pair": 2048}, {"global": 262144, "per_pair": 32768}, {"global": 524288, "per_pair": 65536}],
        "environment": env,
        "environment_manifest": "run_manifests/native_environment.json",
        "native_probe_source": "/mnt/c/Users/86198/Desktop/robotfucker/cpp/stage21/stage21_native_contact_probe.cpp",
        "native_launch_source": "/mnt/c/Users/86198/Desktop/robotfucker/tools/stage23a_native_contact_launch.py",
        "runs": runs,
        "output_hashes": {
            "fcl_final_zst": protected["collision_contacts_all_fcl.csv.zst"],
            "bullet_final_zst": protected["collision_contacts_all_bullet.csv.zst"],
            "fcl_native_raw_runs": [run["raw_contact_sha256"] for run in fcl_repro["runs"]],
            "bullet_native_raw_runs": [run["raw_contact_sha256"] for run in bullet_repro["runs"]],
        },
    }
    native_path = bundle / "run_manifests/native_execution_manifest.json"
    atomic_json(native_path, native)
    complete["native_execution_manifest"] = "run_manifests/native_execution_manifest.json"
    complete["native_environment_manifest"] = "run_manifests/native_environment.json"
    atomic_json(bundle / "run_manifests/stage23a1_complete_bundle.json", complete)
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "protected_outputs_hash.json":
            files[str(path.relative_to(bundle))] = sha256(path)
    atomic_json(bundle / "protected_outputs_hash.json", {"schema_version": "2.3A.1", "files": files})
    print(json.dumps({"native_manifest": str(native_path), "environment": str(env_path), "protected_file_count": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
