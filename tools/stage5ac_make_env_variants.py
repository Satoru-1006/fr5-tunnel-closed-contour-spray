"""Materialize isolated rigid environment-layout variants for Stage 5A-C.

This is a shadow-only geometry/layout tool.  It changes neither the source
environment nor any D65/Stage5AR artifact.  Every object receives the same
translation and optional yaw about the base-link z axis; the original object
dimensions, source segment ids, and geometry metadata are retained.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def variant(environment: dict[str, Any], name: str, dx: float, dy: float, dz: float, yaw: float) -> dict[str, Any]:
    out = copy.deepcopy(environment)
    out["stage5ac_variant"] = {
        "name": name,
        "translation_m": [dx, dy, dz],
        "yaw_rad": yaw,
        "source_environment_unchanged": True,
    }
    base_xyz = environment.get("base_to_tunnel_transform", {}).get("xyz_m", [0.0, 0.0, 0.0])
    base_rpy = environment.get("base_to_tunnel_transform", {}).get("rpy_rad", [0.0, 0.0, 0.0])
    out["base_to_tunnel_transform"] = {
        "identity": bool(abs(dx) < 1e-15 and abs(dy) < 1e-15 and abs(dz) < 1e-15 and abs(yaw) < 1e-15),
        "xyz_m": [float(base_xyz[0]) + dx, float(base_xyz[1]) + dy, float(base_xyz[2]) + dz],
        "rpy_rad": [float(base_rpy[0]), float(base_rpy[1]), float(base_rpy[2]) + yaw],
    }
    for obj in out["objects"]:
        x, y = rotate_xy(float(obj["center_m"][0]), float(obj["center_m"][1]), yaw)
        obj["center_m"] = [x + dx, y + dy, float(obj["center_m"][2]) + dz]
        # Existing tunnel segment boxes have no roll/pitch.  Compose yaw with
        # the existing quaternion using the z-axis rotation in xyzw form.
        qx, qy, qz, qw = [float(v) for v in obj["quaternion_xyzw"]]
        hy = 0.5 * yaw
        rz = (0.0, 0.0, math.sin(hy), math.cos(hy))
        # Normalize and use a general quaternion product for correctness.
        obj["quaternion_xyzw"] = [
            rz[3] * qx - rz[2] * qy,
            rz[3] * qy + rz[2] * qx,
            rz[3] * qz + rz[2] * qw,
            rz[3] * qw - rz[2] * qz,
        ]
        norm = math.sqrt(sum(float(v) * float(v) for v in obj["quaternion_xyzw"]))
        obj["quaternion_xyzw"] = [float(v) / norm for v in obj["quaternion_xyzw"]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source = json.loads(args.environment.read_text(encoding="utf-8"))
    specs = [
        ("nominal", 0.0, 0.0, 0.0, 0.0),
        ("x_minus_050", -0.50, 0.0, 0.0, 0.0),
        ("x_minus_045", -0.45, 0.0, 0.0, 0.0),
        ("x_minus_040", -0.40, 0.0, 0.0, 0.0),
        ("x_minus_035", -0.35, 0.0, 0.0, 0.0),
        ("x_minus_030", -0.30, 0.0, 0.0, 0.0),
        ("x_plus_050", 0.50, 0.0, 0.0, 0.0),
        ("z_minus_025", 0.0, 0.0, -0.25, 0.0),
        ("z_plus_025", 0.0, 0.0, 0.25, 0.0),
        ("y_minus_025", 0.0, -0.25, 0.0, 0.0),
        ("y_plus_025", 0.0, 0.25, 0.0, 0.0),
        ("xy_minus_025", -0.25, 0.0, 0.0, 0.0),
        ("xy_plus_025", 0.25, 0.0, 0.0, 0.0),
        ("yaw_minus_020", 0.0, 0.0, 0.0, math.radians(-20.0)),
        ("yaw_plus_020", 0.0, 0.0, 0.0, math.radians(20.0)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, dx, dy, dz, yaw in specs:
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(variant(source, name, dx, dy, dz, yaw), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append({"name": name, "path": str(path.resolve()), "translation_m": [dx, dy, dz], "yaw_rad": yaw})
    (args.output_dir / "STAGE5AC_ENV_VARIANTS.json").write_text(json.dumps({"source": str(args.environment.resolve()), "variants": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"variant_count": len(manifest), "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
