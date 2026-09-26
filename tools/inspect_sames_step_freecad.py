"""Inspect downloaded official Sames STEP files through FreeCAD's importer.

This is a geometry inventory only.  It does not assign a gun origin, mounting
interface, nozzle axis, or TCP; those assignments require the actual installed
configuration and measured adapter data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: FreeCADCmd inspect_sames_step_freecad.py INPUT.step OUTPUT.json")
    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    import FreeCAD
    import Part

    document = FreeCAD.newDocument("sames_step_inventory")
    Part.insert(str(input_path), document.Name)
    document.recompute()
    rows = []
    for obj in document.Objects:
        shape = getattr(obj, "Shape", None)
        if shape is None or shape.isNull():
            continue
        box = shape.BoundBox
        rows.append(
            {
                "object_name": str(obj.Name),
                "label": str(obj.Label),
                "shape_type": str(shape.ShapeType),
                "solid_count": int(len(shape.Solids)),
                "shell_count": int(len(shape.Shells)),
                "face_count": int(len(shape.Faces)),
                "edge_count": int(len(shape.Edges)),
                "vertex_count": int(len(shape.Vertexes)),
                "bbox_min_mm": [float(box.XMin), float(box.YMin), float(box.ZMin)],
                "bbox_max_mm": [float(box.XMax), float(box.YMax), float(box.ZMax)],
                "extent_mm": [float(box.XLength), float(box.YLength), float(box.ZLength)],
                "volume_mm3": float(shape.Volume),
                "center_of_mass_mm": [float(shape.CenterOfMass.x), float(shape.CenterOfMass.y), float(shape.CenterOfMass.z)],
            }
        )
    payload = {
        "status": "PASS" if rows else "FAIL",
        "step": str(input_path),
        "freecad_version": str(FreeCAD.Version()),
        "object_count": len(rows),
        "objects": rows,
        "interpretation": "official geometry inventory only; no TCP or adapter assignment",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "object_count": len(rows), "output": str(output_path)}))
    FreeCAD.closeDocument(document.Name)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
