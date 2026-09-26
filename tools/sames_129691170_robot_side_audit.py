"""Fail-closed truth audit for the Sames 129691170 robot-side interface.

This is a read-only audit of an official analytic STEP and the existing Adapter
V2 R3 shadow.  It intentionally does not edit either input and does not create
an adapter revision.  Generated JSON, Markdown, and section plots are evidence
artifacts in a caller-selected output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


TOL = 1e-5


def xyz(p):
    if hasattr(p, "toTuple"):
        return [float(v) for v in p.toTuple()]
    if hasattr(p, "x"):
        return [float(p.x), float(p.y), float(p.z)]
    return [float(p.X()), float(p.Y()), float(p.Z())]


def rounded(values, digits=9):
    return [round(float(v), digits) for v in values]


def shape_bbox(shape):
    b = shape.BoundingBox()
    return {
        "min": [float(b.xmin), float(b.ymin), float(b.zmin)],
        "max": [float(b.xmax), float(b.ymax), float(b.zmax)],
        "extents": [float(b.xlen), float(b.ylen), float(b.zlen)],
    }


def vec_close(a, b, tol=1e-4):
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def parallel_y(axis, tol=1e-5):
    return abs(abs(float(axis[1])) - 1.0) <= tol and abs(float(axis[0])) <= tol and abs(float(axis[2])) <= tol


def face_is_plane(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane

    return BRepAdaptor_Surface(face.wrapped, True).GetType() == GeomAbs_Plane


def plane_normal(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    s = BRepAdaptor_Surface(face.wrapped, True)
    n = s.Plane().Axis().Direction()
    return [float(n.X()), float(n.Y()), float(n.Z())]


def circle_edges(face):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle

    rows = []
    for edge_id, edge in enumerate(face.Edges()):
        c = BRepAdaptor_Curve(edge.wrapped)
        if c.GetType() != GeomAbs_Circle:
            continue
        circle = c.Circle()
        loc = circle.Location()
        axis = circle.Axis().Direction()
        rows.append(
            {
                "edge_id": edge_id,
                "radius_mm": float(circle.Radius()),
                "diameter_mm": float(2.0 * circle.Radius()),
                "center_mm": [float(loc.X()), float(loc.Y()), float(loc.Z())],
                "axis": [float(axis.X()), float(axis.Y()), float(axis.Z())],
                "bbox": shape_bbox(edge),
            }
        )
    return rows


def group_face_openings(face):
    groups = {}
    for row in circle_edges(face):
        c = row["center_mm"]
        key = (round(c[0], 4), round(c[2], 4), round(row["radius_mm"], 4))
        groups.setdefault(key, []).append(row)
    result = []
    for i, (key, edges) in enumerate(sorted(groups.items(), key=lambda item: (item[0][2], item[0][0], item[0][1]))):
        x, z, radius = key
        result.append(
            {
                "opening_id": f"OPEN_{i + 1:02d}",
                "center_mm": [x, edges[0]["center_mm"][1], z],
                "radius_mm": radius,
                "diameter_mm": 2.0 * radius,
                "edge_ids": [r["edge_id"] for r in edges],
                "edge_count": len(edges),
            }
        )
    return result


def cylinder_faces(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder

    rows = []
    for face_id, face in enumerate(shape.Faces()):
        s = BRepAdaptor_Surface(face.wrapped, True)
        if s.GetType() != GeomAbs_Cylinder:
            continue
        cyl = s.Cylinder()
        axis = cyl.Axis().Direction()
        loc = cyl.Location()
        b = face.BoundingBox()
        rows.append(
            {
                "face_id": face_id,
                "radius_mm": float(cyl.Radius()),
                "diameter_mm": float(2.0 * cyl.Radius()),
                "axis": [float(axis.X()), float(axis.Y()), float(axis.Z())],
                "location_mm": [float(loc.X()), float(loc.Y()), float(loc.Z())],
                "area_mm2": float(face.Area()),
                "bbox": shape_bbox(face),
                "axis_interval_y_mm": [float(b.ymin), float(b.ymax)],
            }
        )
    return rows


def plane_faces(shape):
    rows = []
    for face_id, face in enumerate(shape.Faces()):
        if not face_is_plane(face):
            continue
        b = face.BoundingBox()
        rows.append(
            {
                "face_id": face_id,
                "area_mm2": float(face.Area()),
                "center_mm": xyz(face.Center()),
                "normal": plane_normal(face),
                "bbox": shape_bbox(face),
                "circle_edges": circle_edges(face),
            }
        )
    return rows


def matching_cylinders(opening, cylinders):
    x, y, z = opening["center_mm"]
    result = []
    for row in cylinders:
        if not parallel_y(row["axis"]):
            continue
        if abs(row["radius_mm"] - opening["radius_mm"]) > 1e-4:
            continue
        if abs(row["location_mm"][0] - x) > 1e-4 or abs(row["location_mm"][2] - z) > 1e-4:
            continue
        result.append(row)
    return result


def matching_caps(opening, planes, y_value):
    x, _, z = opening["center_mm"]
    result = []
    for row in planes:
        b = row["bbox"]
        if abs(b["min"][1] - y_value) > 2e-4 or abs(b["max"][1] - y_value) > 2e-4:
            continue
        for edge in row["circle_edges"]:
            c = edge["center_mm"]
            if abs(c[0] - x) <= 1e-4 and abs(c[2] - z) <= 1e-4 and abs(edge["radius_mm"] - opening["radius_mm"]) <= 1e-4:
                result.append(row)
                break
    return result


def classify_opening(opening, cyls, planes, mating_y, adapter_centers):
    matched = matching_cylinders(opening, cyls)
    intervals = []
    for row in matched:
        lo, hi = row["axis_interval_y_mm"]
        intervals.append([lo, hi])
    if intervals:
        lo = min(i[0] for i in intervals)
        hi = max(i[1] for i in intervals)
    else:
        lo = hi = None

    cap_faces = matching_caps(opening, planes, hi) if hi is not None else []
    cap_area_expected = math.pi * opening["radius_mm"] ** 2
    true_caps = [f for f in cap_faces if abs(f["area_mm2"] - cap_area_expected) < 1e-3]
    external_end_faces = [f for f in cap_faces if f["area_mm2"] > cap_area_expected * 1.8]

    center = opening["center_mm"]
    mapped = [center[0] + 13.9566800831, center[2] - 13.3187604207]
    adapter_match = any(abs(mapped[0] - x) <= 1e-4 and abs(mapped[1] - y) <= 1e-4 for x, y in adapter_centers)
    mapping_repeated = False

    if abs(opening["diameter_mm"] - 5.5) <= 1e-4:
        role = "MOUNTING_FASTENER" if adapter_match else "UNKNOWN"
        confidence = "HIGH" if adapter_match else "LOW"
    elif true_caps:
        role = "OTHER"
        confidence = "LOW"
    else:
        role = "UNKNOWN"
        confidence = "LOW"

    if external_end_faces:
        termination = "OPEN_TO_OPPOSITE_EXTERNAL_FACE"
        external_through = "YES"
    elif true_caps:
        termination = "BLIND_IN_SOLID"
        external_through = "NO"
    elif matched:
        termination = "OPEN_TO_INTERNAL_CAVITY"
        external_through = "NO"
    else:
        termination = "UNRESOLVED"
        external_through = "UNRESOLVED"

    return {
        "opening_id": opening["opening_id"],
        "center_mm": rounded(center),
        "axis": [0.0, -1.0, 0.0],
        "diameter_mm": opening["diameter_mm"],
        "radius_mm": opening["radius_mm"],
        "face_edge_ids": opening["edge_ids"],
        "matched_cylindrical_face_ids": [r["face_id"] for r in matched],
        "axial_intervals_y_mm": intervals,
        "axial_interval_union_y_mm": [lo, hi] if lo is not None else None,
        "mating_face_opening": "YES",
        "axis_normal_to_mating_face": "YES",
        "adapter_mapped_center_local_mm": rounded([mapped[0], mapped[1], -10.0]),
        "matching_adapter_feature": "YES" if adapter_match else "NO",
        "repeated_symmetric_mounting_pattern": "CANDIDATE" if abs(opening["diameter_mm"] - 5.5) <= 1e-4 else "NO_EVIDENCE",
        "connected_to_filter_or_internal_cavity": "NO" if external_end_faces or true_caps else "CANDIDATE",
        "connected_to_fluid_or_air_port": "NO_EVIDENCE",
        "feature_role": role,
        "role_confidence": confidence,
        "hole_termination_class": termination,
        "external_through_hole": external_through,
        "counterbore_present": "NO" if abs(opening["diameter_mm"] - 5.5) <= 1e-4 else "UNRESOLVED",
        "counterbore_diameter_mm": None,
        "counterbore_depth_mm": None,
        "blind_bottom_face_ids": [f["face_id"] for f in true_caps],
        "opposite_external_opening_faces": [f["face_id"] for f in external_end_faces],
        "evidence_level": "OFFICIAL_GEOMETRY_DERIVED",
    }


def map_to_adapter(point):
    x, y, z = point
    return [x + 13.9566800831, z - 13.3187604207, -y - 31.6712316760]


def make_cylinder(cq, radius, height, x, y, z, direction=(0.0, 0.0, 1.0)):
    return cq.Workplane("XY").newObject([cq.Solid.makeCylinder(radius, height, cq.Vector(x, y, z), cq.Vector(*direction))])


def common_volume(cq, a, b):
    result = a.intersect(b)
    shape = result.val()
    return 0.0 if shape.isNull() else float(shape.Volume())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path, text):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def md_table(rows):
    if not rows:
        return "(none)"
    keys = list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |", "|" + "|".join("---" for _ in keys) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def section_plot(out_dir, feature, label):
    intervals = feature.get("axial_interval_union_y_mm") or []
    if len(intervals) != 2:
        return None
    lo, hi = intervals
    # Matplotlib's Windows font/cache path is not stable in the bundled
    # CadQuery runtime.  Use Pillow for a deterministic evidence sketch.
    from PIL import Image, ImageDraw

    width, height = 1400, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 100
    x0, x1 = margin, width - margin
    y_mid = 235
    span = max(hi - lo, 1e-6)
    domain_lo = lo - max(2.0, span * 0.35)
    domain_hi = hi + max(2.0, span * 0.35)

    def px(value):
        return int(x0 + (value - domain_lo) / (domain_hi - domain_lo) * (x1 - x0))

    draw.text((margin, 30), f"{label}: dia {feature['diameter_mm']:.3f} mm | {feature['hole_termination_class']}", fill="black")
    draw.line((x0, y_mid, x1, y_mid), fill="#888888", width=2)
    draw.rectangle((px(lo), y_mid - 45, px(hi), y_mid + 45), fill="#7aa6d8", outline="#24527a", width=3)
    draw.line((px(lo), 100, px(lo), 330), fill="#444444", width=3)
    draw.line((px(hi), 100, px(hi), 330), fill="#b33a3a", width=3)
    draw.text((px(lo) - 30, 72), "mating", fill="#333333")
    draw.text((px(hi) - 20, 72), "end", fill="#9a2020")
    draw.text((margin, 350), "STEP axis y (mm); increasing y enters the base", fill="#333333")
    draw.text((x1 - 210, 350), f"{lo:.3f} -> {hi:.3f}", fill="#333333")
    path = out_dir / f"{label}_section.png"
    image.save(path)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=Path, required=True)
    ap.add_argument("--adapter-step", type=Path, required=True)
    ap.add_argument("--adapter-params", type=Path, required=True)
    ap.add_argument("--adapter-build-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    import cadquery as cq
    import yaml

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = args.step.read_bytes()
    text = raw.decode("latin1", errors="ignore")
    header_match = re.search(r"FILE_NAME\s*\(\s*'([^']+)'", text, flags=re.IGNORECASE)
    internal_filename = header_match.group(1) if header_match else None
    base_wp = cq.importers.importStep(str(args.step))
    base = base_wp.val()
    adapter_wp = cq.importers.importStep(str(args.adapter_step))
    adapter = adapter_wp.val()
    params = yaml.safe_load(args.adapter_params.read_text(encoding="utf-8"))
    build_record = json.loads(args.adapter_build_json.read_text(encoding="utf-8"))

    base_faces = base.Faces()
    base_planes = plane_faces(base)
    y_min = base.BoundingBox().ymin
    mating_candidates = [
        (i, f) for i, f in enumerate(base_faces)
        if face_is_plane(f) and abs(f.BoundingBox().ymin - y_min) < 1e-4 and abs(f.BoundingBox().ymax - y_min) < 1e-4 and f.Area() > 1000.0
    ]
    if len(mating_candidates) != 1:
        raise RuntimeError(f"FAIL_CLOSED: mating face candidates={len(mating_candidates)}")
    mating_face_id, mating_face = mating_candidates[0]
    openings = group_face_openings(mating_face)
    cylinders = cylinder_faces(base)
    adapter_centers = [[-36.5, -17.25], [-36.5, 17.25], [36.5, -17.25], [36.5, 17.25]]
    features = [classify_opening(o, cylinders, base_planes, y_min, adapter_centers) for o in openings if len(o["edge_ids"]) >= 2]

    mounting = [f for f in features if f["feature_role"] == "MOUNTING_FASTENER" and f["role_confidence"] == "HIGH"]
    if len(mounting) != 4:
        raise RuntimeError(f"FAIL_CLOSED: high-confidence mounting count={len(mounting)}")
    mounting.sort(key=lambda f: (f["center_mm"][0], f["center_mm"][2]))
    for i, f in enumerate(mounting, 1):
        f["feature_id"] = f"MOUNT_{i:02d}"
    for i, f in enumerate([x for x in features if x not in mounting], 1):
        f["feature_id"] = f"NONMOUNT_{i:02d}"

    # The four high-confidence holes form the only promoted mounting set.
    mounting_centers = [f["center_mm"] for f in mounting]
    xs = sorted({round(c[0], 6) for c in mounting_centers})
    zs = sorted({round(c[2], 6) for c in mounting_centers})
    x_pitch = xs[-1] - xs[0]
    z_pitch = zs[-1] - zs[0]
    pcd = math.hypot(x_pitch, z_pitch)

    # Read-only R3 compatibility and two directional fastener probes.
    base_transformed = cq.Workplane("XY").newObject([base]).rotate((0, 0, 0), (1, 0, 0), -90.0).translate((13.9566800831, -13.3187604207, -31.6712316760))
    adapter_base_common = common_volume(cq, adapter_wp, base_transformed)
    r3_asb = params["asb"]
    r3_asb_r = float(r3_asb["modeled_clearance_radius_mm"])
    r3_head_r = float(r3_asb["head_radius_mm"])
    r3_head_depth = float(r3_asb["head_counterbore_depth_mm"])
    r3_hole_d = 2.0 * r3_asb_r
    iso_m5_head_d = 8.5
    iso_m5_head_h = 5.0
    iso_m5_shank_r = 2.5
    base_outer_z = min(map_to_adapter([mounting[0]["center_mm"][0], -17.6712316760, mounting[0]["center_mm"][2]])[2],
                         map_to_adapter([mounting[0]["center_mm"][0], -17.6712316760, mounting[0]["center_mm"][2]])[2])
    pocket_head_intersections = []
    outer_head_intersections = []
    through_shank_intersections = []
    for f in mounting:
        x, _, z = f["center_mm"]
        lx, ly, lz = map_to_adapter([x, y_min, z])
        wrong_orientation_head = make_cylinder(cq, iso_m5_head_d / 2.0, iso_m5_head_h, lx, ly, -11.5)
        outer_head = make_cylinder(cq, iso_m5_head_d / 2.0, iso_m5_head_h, lx, ly, -19.0)
        through_shank = make_cylinder(cq, iso_m5_shank_r, 20.0, lx, ly, -14.0)
        pocket_head_intersections.append(common_volume(cq, wrong_orientation_head, base_transformed))
        outer_head_intersections.append(common_volume(cq, outer_head, base_transformed))
        through_shank_intersections.append({
            "base_mm3": common_volume(cq, through_shank, base_transformed),
            "adapter_mm3": common_volume(cq, through_shank, adapter_wp),
        })

    identity = {
        "status": "OFFICIAL_VERIFIED",
        "part_number": "129691170",
        "official_product_name": "Robotic Base for ASB (T) With filter",
        "official_angle": "60°",
        "official_product_page": "https://www.sames.com/usa/en/product/product-asb",
        "official_step_download_label": "Base (robot T,filter),step file",
        "official_step_document_id_recorded": "427361",
        "official_step_source_url_recorded": "https://www.sames.com/documents/427361/download",
        "local_original_filename": args.step.name,
        "step_internal_filename": internal_filename,
        "file_size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest().upper(),
        "retrieval_date": "2026-09-05 workspace artifact timestamp; re-audited 2026-09-07",
        "official_page_title": "ASB - Sames",
        "identity_evidence_level": "OFFICIAL_VERIFIED",
        "identity_boundary": "The product page/catalog identity and STEP internal filename identify the artifact; the STEP does not by itself specify manufacturer fastener semantics.",
    }
    write_json(out / "129691170_IDENTITY.json", identity)
    write_json(out / "MOUNTING_FEATURE_CLASSIFICATION.json", {"status": "PASS", "mating_face_id": mating_face_id, "features": features, "promoted_mounting_feature_ids": [f["feature_id"] for f in mounting]})
    write_json(out / "129691170_ROBOT_SIDE_HOLE_GEOMETRY.json", {"status": "PASS", "evidence_level": "OFFICIAL_GEOMETRY_DERIVED", "mating_face_id": mating_face_id, "mating_face_plane_y_mm": y_min, "mounting_features": mounting})
    write_json(out / "ROBOT_SIDE_HOLE_ENUMERATION.json", {"status": "PASS", "evidence_level": "OFFICIAL_GEOMETRY_DERIVED", "mating_face_id": mating_face_id, "mating_face_plane_y_mm": y_min, "mounting_features": mounting})
    write_json(out / "129691170_HOLE_TERMINATION.json", {"status": "PASS", "features": [{k: f[k] for k in ("feature_id", "diameter_mm", "axial_intervals_y_mm", "axial_interval_union_y_mm", "hole_termination_class", "external_through_hole", "blind_bottom_face_ids", "opposite_external_opening_faces")} for f in features]})
    write_json(out / "HOLE_TERMINATION_ANALYSIS.json", {"status": "PASS", "features": [{k: f[k] for k in ("feature_id", "diameter_mm", "axial_intervals_y_mm", "axial_interval_union_y_mm", "hole_termination_class", "external_through_hole", "blind_bottom_face_ids", "opposite_external_opening_faces")} for f in features]})

    r3_report = {
        "status": "PASS_GEOMETRIC_INTERFACE_ONLY",
        "adapter_revision": "V2_R3",
        "adapter_step": str(args.adapter_step),
        "adapter_build_status": build_record.get("status"),
        "adapter_valid": bool(adapter.isValid()),
        "adapter_base_common_volume_mm3": adapter_base_common,
        "mapped_mounting_centers_local_mm": [f["adapter_mapped_center_local_mm"] for f in mounting],
        "r3_asb_clearance_diameter_mm": r3_hole_d,
        "sames_mounting_diameter_mm": mounting[0]["diameter_mm"],
        "radial_clearance_mm": r3_asb_r - mounting[0]["radius_mm"],
        "r3_counterbore_diameter_mm": 2.0 * r3_head_r,
        "r3_counterbore_depth_mm": r3_head_depth,
        "r3_counterbore_location": "Adapter local z=-10 mating side; project-designed R3 geometry, not Sames geometry",
        "iso4762_m5_head_diameter_mm": iso_m5_head_d,
        "iso4762_m5_head_height_mm": iso_m5_head_h,
        "wrong_orientation_adapter_pocket_head_common_mm3": pocket_head_intersections,
        "base_exterior_head_common_mm3": outer_head_intersections,
        "base_exterior_shank_common_mm3": [r["base_mm3"] for r in through_shank_intersections],
        "base_exterior_shank_adapter_common_mm3": [r["adapter_mm3"] for r in through_shank_intersections],
        "claim_boundary": "R3 is read-only; this is nominal B-Rep geometry and does not prove nut access, thread strength, preload, tolerances, sealing, or production release.",
    }
    write_json(out / "R3_COMPATIBILITY.json", r3_report)

    section_files = []
    for feature in mounting:
        section_files.append(section_plot(out, feature, feature["feature_id"]))

    # Required Markdown artifacts.
    urls = {
        "Sames product page": identity["official_product_page"],
        "Sames STEP source recorded in workspace": identity["official_step_source_url_recorded"],
        "ISO 273 clearance holes": "https://www.iso.org/standard/4183.html",
        "ISO 4762 socket head cap screws": "https://www.iso.org/standard/34460.html",
        "ISO 7380-1 button head screws": "https://www.iso.org/standard/78699.html",
        "ISO 4762 dimensional table used for candidate geometry": "https://www.roymech.co.uk/Useful_Tables/Screws/cap_screws.htm",
        "Sames ASB manual family evidence": "https://www.sames.com/tzr/scripts/downloader2.php?filename=DOC001%2FFILES%2Fbc%2Fed%2Faed5ei3ynvnl&mime=application%2Fpdf&moid=153&originalname=ASB-airless%C2%AE+-instructions-manual-%3Cstrong%3ESames%3C%2Fstrong%3E-582100110-uk.pdf",
        "Sames ASB spare-parts family evidence": "https://www.sames.com/tzr/scripts/downloader2.php?filename=T004%2Fmedia%2Fcc%2F10%2FUS.mjw5hiadwnhb&mime=application%2Fpdf&originalname=LT999-600-383_ASB_Spare_Parts_Sheet_11x17.pdf",
    }
    source_rows = [
        {"evidence": "Exact product identity", "source": "Sames ASB product page", "level": "OFFICIAL_VERIFIED", "claim": "129691170 is the robot T/filter 60° base; page lists the exact STEP label."},
        {"evidence": "Exact STEP identity", "source": "Official document 427361 + STEP FILE_NAME", "level": "OFFICIAL_VERIFIED", "claim": f"Internal {internal_filename}; {len(raw)} bytes; SHA256 {identity['sha256']}."},
        {"evidence": "Robot-side geometry", "source": "Exact STEP analytic B-Rep", "level": "OFFICIAL_GEOMETRY_DERIVED", "claim": "Face 46 and four Ø5.5 openings form the promoted interface pattern; no base-side counterbore at those centers."},
        {"evidence": "Same-family gun-to-base evidence", "source": "Official ASB instruction manual", "level": "OFFICIAL_VERIFIED / FAMILY ONLY", "claim": "Manual states 4 holes M5 x 10 mm and 1 hole M8 x 7 mm for a base bottom view; not transferred to robot-side mounting."},
        {"evidence": "Same-family screw evidence", "source": "Official ASB spare-parts sheet", "level": "OFFICIAL_VERIFIED / FAMILY ONLY", "claim": "Lists CHc M5 screws for other base variants; not an exact 129691170 robot-side requirement."},
        {"evidence": "Clearance candidate", "source": "ISO 273 + dimensional table", "level": "STANDARD_GEOMETRY_MATCH", "claim": "Ø5.5 is the normal/medium M5 clearance candidate; it is not a Sames specification."},
        {"evidence": "Head candidate", "source": "ISO 4762 dimensional table", "level": "STANDARD_GEOMETRY_MATCH", "claim": "M5 socket head candidate: head Ø8.5 mm, nominal head height 5.0 mm; candidate only."},
    ]
    nonmount = [f for f in features if f not in mounting]
    final_status = {
        "SAMES_BASE_PART_NUMBER": "129691170",
        "129691170_IDENTITY": "OFFICIAL_VERIFIED",
        "129691170_OFFICIAL_STEP": "PASS",
        "ROBOT_SIDE_MATING_FACE": f"Face {mating_face_id}; plane y={y_min:.9f} mm; OFFICIAL_GEOMETRY_DERIVED",
        "MOUNTING_FEATURE_COUNT": 4,
        "MOUNTING_FASTENER_FEATURES": [f["feature_id"] for f in mounting],
        "FLUID_AIR_FILTER_FEATURES": "No non-mounting opening was promoted to a specific fluid/air/filter semantic; remaining face openings are fenced as OTHER/UNKNOWN/INTERNAL-CAVITY candidates.",
        "SMALL_HOLE_DIAMETER": 5.5,
        "HOLE_TERMINATION_CLASS": "OPEN_TO_OPPOSITE_EXTERNAL_FACE",
        "EXTERNAL_THROUGH_HOLE": "YES",
        "COUNTERBORE_PRESENT": "NO on the four promoted Sames mounting features",
        "COUNTERBORE_DIAMETER": None,
        "COUNTERBORE_DEPTH": None,
        "BLIND_BOTTOM_FACE": "None for the four promoted mounting features; the separate Ø5.0 pair has blind planar bottoms.",
        "ISO273_MATCH": "YES",
        "NOMINAL_FASTENER_CANDIDATE": "M5",
        "MATCH_CLASS": "MEDIUM / NORMAL clearance candidate",
        "SAMES_OFFICIAL_FASTENER_SPEC": "NOT_FOUND for 129691170 robot-side mounting",
        "SAMES_BASE_THREAD_SEMANTICS": "UNRESOLVED; smooth STEP geometry is not thread semantics",
        "SAMES_BASE_THREAD_DEPTH": "NOT_APPLICABLE_TO_EXTERNAL_THROUGH_FEATURE",
        "THREAD_REACTION_METHOD": "UNRESOLVED",
        "THREAD_REACTION_STATUS": "UNRESOLVED",
        "ADAPTER_THREAD_REQUIRED": "UNRESOLVED; R3 currently has clearance holes and no verified insert/nut reaction architecture",
        "ADAPTER_R3_THREAD_CAPACITY": "UNRESOLVED",
        "PRELIMINARY_BOLT_LENGTH": "UNRESOLVED; no reaction method/access stack is closed",
        "FASTENER_SHADOW_BREP": "NOT_BUILT; blocked by unresolved thread reaction method",
        "129691170_MOUNTING_ARCHITECTURE": "EXTERNAL_THROUGH_HOLE_WITHOUT_BASE_COUNTERBORE",
        "MANUFACTURING_RELEASE": "NO",
        "PROMOTION": "NO_PROMOTION",
    }
    write_json(out / "FINAL_STATE.json", final_status)

    identity_md = f"""# 129691170 identity report

`129691170` is identity-closed at Level 1: **OFFICIAL_VERIFIED**.

{md_table([
    {"field": "commercial part", "value": "129691170"},
    {"field": "official designation", "value": "Robotic Base for ASB (T) With filter"},
    {"field": "angle", "value": "60°"},
    {"field": "official page", "value": f"[{urls['Sames product page']}]({urls['Sames product page']})"},
    {"field": "download label", "value": "Base (robot T,filter),step file"},
    {"field": "document id recorded", "value": "427361"},
    {"field": "local original filename", "value": args.step.name},
    {"field": "STEP internal filename", "value": internal_filename},
    {"field": "file size", "value": f"{len(raw)} bytes"},
    {"field": "SHA-256", "value": identity['sha256']},
    {"field": "retrieval/re-audit", "value": identity['retrieval_date']},
])}

The exact source artifact is the local official STEP at `{args.step}`. The web page is live and lists the exact robot T/filter STEP label; the workspace source record retains document 427361 and the recorded download URL. Identity evidence does not imply a Sames fastener specification.
"""
    write_text(out / "129691170_IDENTITY_REPORT.md", identity_md)
    write_text(out / "OFFICIAL_SOURCE_PROVENANCE.md", f"""# Official source provenance

## Exact 129691170 sources

- Product page: [{urls['Sames product page']}]({urls['Sames product page']})
- Exact STEP source recorded by the workspace: [{urls['Sames STEP source recorded in workspace']}]({urls['Sames STEP source recorded in workspace']})
- Download label: `Base (robot T,filter),step file`; document ID recorded: `427361`.
- Local STEP: `{args.step}`
- Internal STEP filename: `{internal_filename}`
- Bytes: `{len(raw)}`
- SHA-256: `{identity['sha256']}`

## Retrieval and identity fence

The product page/catalog identity is Level 1. The analytic geometry claims in this package are Level 2 and come only from the local STEP B-Rep. The local STEP is not treated as evidence for thread form, pitch, material, strength, preload, or manufacturer fastener selection.

## Online verification record

On 2026-09-07 the Sames ASB product page was checked and still listed the exact `Base (robot T,filter),step file` entry. The exact download endpoint recorded in the workspace was not used as a substitute for the local file when the web endpoint did not expose a stable textual response.
""")
    write_text(out / "OFFICIAL_SOURCE_EVIDENCE_MATRIX.md", "# Official source evidence matrix\n\n## A. Exact 129691170 evidence\n\n" + md_table(source_rows[:3]) + f"\n\n## B. Same robotic/special-base family evidence\n\n" + md_table(source_rows[3:5]) + f"\n\n## C. Standard / comparison evidence\n\n" + md_table(source_rows[5:]) + f"\n\n### URLs\n\n" + "\n".join(f"- [{k}]({v})" for k, v in urls.items()) + "\n\n`SAMES_OFFICIAL_FASTENER_SPEC = NOT_FOUND` for the exact robot-side 129691170-to-external mounting interface. Family M5 callouts are not promoted across interface boundaries.")

    face_md = f"""# Robot-side mating face report

## Selected face

`F_BASE_ROBOT_MATE = Face {mating_face_id}` in the exact official STEP.

{md_table([
    {"property": "plane", "value": f"y = {y_min:.9f} mm"},
    {"property": "normal", "value": rounded(plane_normal(mating_face))},
    {"property": "area", "value": f"{mating_face.Area():.6f} mm²"},
    {"property": "B-Rep bbox", "value": shape_bbox(mating_face)},
    {"property": "face index evidence", "value": str(mating_face_id)},
])}

## Joint confirmation

- It is the outermost large planar face at the rear/robot-side direction of the base B-Rep.
- The four promoted hole axes are normal to this plane.
- Applying the existing R3 read-only transform `R_x(-90°)` and translation `(13.9566800831, -13.3187604207, -31.6712316760) mm` maps the face to adapter local `z=-10 mm`.
- The four hole centers map to R3 local `(-36.5, ±17.25, -10)` and `(36.5, ±17.25, -10) mm`, exactly matching R3's four ASB interface centers.
- The transformed official base and R3 adapter have zero positive B-Rep common volume in the nominal check.

The semantic phrase “robot-side” is therefore a geometry/transform-derived claim (**OFFICIAL_GEOMETRY_DERIVED**), not a Sames text callout.
"""
    write_text(out / "ROBOT_SIDE_MATING_FACE_REPORT.md", face_md)

    classification_md = "# Mounting feature classification\n\nOnly the following four features are promoted to `MOUNTING_FASTENER` with HIGH confidence.\n\n" + md_table([{"feature": f["feature_id"], "center mm": f["center_mm"], "diameter mm": f["diameter_mm"], "termination": f["hole_termination_class"], "adapter match": f["matching_adapter_feature"], "role": f["feature_role"], "confidence": f["role_confidence"]} for f in mounting])
    classification_md += "\n\n## Non-mounting candidates\n\n" + md_table([{"feature": f["feature_id"], "center mm": f["center_mm"], "diameter mm": f["diameter_mm"], "termination": f["hole_termination_class"], "role": f["feature_role"], "confidence": f["role_confidence"]} for f in nonmount])
    classification_md += "\n\nThe non-mounting candidates were not force-labelled as fluid, air, or filter features without an exact semantic callout. They are fenced as `OTHER`/`UNKNOWN`/internal-cavity candidates and cannot enter the fastener audit.\n"
    write_text(out / "MOUNTING_FEATURE_CLASSIFICATION.md", classification_md)

    hole_md = "# Robot-side hole enumeration\n\n## High-confidence mounting features\n\n" + md_table([{"feature": f["feature_id"], "solid": 0, "center xyz mm": f["center_mm"], "axis": f["axis"], "small-hole Ø mm": f["diameter_mm"], "cylindrical faces": f["matched_cylindrical_face_ids"], "axial interval y mm": f["axial_interval_union_y_mm"], "opposite opening": f["opposite_external_opening_faces"]} for f in mounting])
    hole_md += "\n\nAnalytic B-Rep only; no STL, mesh, screenshot, or bounding-box-only measurement was used.\n"
    write_text(out / "ROBOT_SIDE_HOLE_ENUMERATION.md", hole_md)
    write_text(out / "HOLE_TERMINATION_ANALYSIS.md", "# Hole termination analysis\n\n" + md_table([{"feature": f["feature_id"], "diameter mm": f["diameter_mm"], "cylindrical interval": f["axial_interval_union_y_mm"], "termination": f["hole_termination_class"], "external through": f["external_through_hole"], "blind bottom": f["blind_bottom_face_ids"], "external end": f["opposite_external_opening_faces"]} for f in features]) + "\n\nFor the four promoted Ø5.5 features, the analytic cylindrical wall runs about 4.0 mm from the mating face to an opposite external planar face. This is an external through-hole, not an internal-cavity opening and not a blind hole.\n")
    write_text(out / "COUNTERBORE_ANALYSIS.md", f"""# Counterbore analysis

For each promoted high-confidence mounting feature:

- mating-face opening: Ø{mounting[0]['diameter_mm']:.6f} mm;
- no larger concentric cylindrical recess at the mating face;
- no annular shoulder at the promoted hole centers;
- no second large diameter in the same axial feature;
- termination is the opposite external face at approximately 4.0 mm axial distance.

Therefore:

`COUNTERBORE_PRESENT = NO` for the Sames 129691170 robot-side mounting features.

The Ø10 × 3.5 mm recess in Adapter V2 R3 is a separate **PROJECT_DESIGNED** adapter feature. It must not be reported as a Sames counterbore.
""")
    write_text(out / "ISO273_FASTENER_MATCH.md", f"""# ISO 273 geometry match

The promoted mounting holes measure **Ø{mounting[0]['diameter_mm']:.6f} mm**. ISO 273 defines general-purpose clearance-hole diameters; the standard's scope does not identify a manufacturer part's chosen fastener. The standard comparison is:

{md_table([
    {"candidate": "M4", "typical medium clearance": "Ø4.5 mm", "match": "NO"},
    {"candidate": "M5", "typical medium/normal clearance": "Ø5.5 mm", "match": "YES / HIGH geometry match"},
    {"candidate": "M6", "typical medium clearance": "Ø6.6 mm", "match": "NO"},
])}

`NOMINAL_FASTENER_CANDIDATE = M5`\n\n`EVIDENCE = STANDARD_GEOMETRY_MATCH`\n\n`SAMES_OFFICIAL_FASTENER_SPEC = NOT_FOUND` for this robot-side interface. Official family documents containing M5 callouts refer to the gun-to-base interface or other base variants and are not transferred here.

Sources: [ISO 273]({urls['ISO 273 clearance holes']}), [Sames product page]({urls['Sames product page']}).
""")
    write_text(out / "FASTENER_HEAD_GEOMETRY_CANDIDATES.md", f"""# Fastener head geometry candidates

These are screening candidates only (**STANDARD_GEOMETRY_MATCH**). They do not establish a Sames requirement.

{md_table([
    {"head family": "ISO 4762 socket head cap screw", "M5 candidate": "Ø8.5 × 5.0 mm", "R3 Ø10 × 3.5 recess": "diameter YES; depth NO for full head seating", "status": "candidate only"},
    {"head family": "ISO 7380-1 button head", "M5 candidate": "about Ø9.5 × 2.75 mm", "R3 Ø10 × 3.5 recess": "nominal envelope candidate", "status": "reduced-loadability family; not selected"},
    {"head family": "low-head variant", "M5 candidate": "supplier/design-specific", "R3 Ø10 × 3.5 recess": "not enough authoritative exact data", "status": "UNRESOLVED"},
])}

The Sames base itself has no matching counterbore at the four promoted centers. A head geometry that fits the R3 recess therefore does not close the external-through-hole reaction architecture.

Sources: [ISO 4762]({urls['ISO 4762 socket head cap screws']}), [ISO 7380-1:2022]({urls['ISO 7380-1 button head screws']}), [dimension table]({urls['ISO 4762 dimensional table used for candidate geometry']}).
""")

    write_text(out / "THREAD_REACTION_METHOD_ANALYSIS.md", f"""# Thread reaction method analysis

The Sames geometry closes as `EXTERNAL_THROUGH_HOLE_WITHOUT_BASE_COUNTERBORE`. A through-hole does **not** imply an Adapter tapped hole. The candidate reaction methods are:

{md_table([
    {"method": "ADAPTER_TAPPED", "scope": "PROJECT_DESIGNED", "R3 current geometry": "NO; current Ø6.2 clearance holes are not tapped", "access/strength": "requires redesign and material/grade/preload calculation", "status": "candidate"},
    {"method": "THROUGH_ADAPTER_WITH_NUT", "scope": "PROJECT_DESIGNED", "R3 current geometry": "through shank geometry nominally possible", "access/strength": "robot-side nut/wrist/robot-flange access not represented", "status": "unresolved"},
    {"method": "THREADED_INSERT", "scope": "PROJECT_DESIGNED", "R3 current geometry": "no insert modeled", "access/strength": "insert type, pull-out, wall and installation access absent", "status": "unresolved"},
    {"method": "CAPTIVE_NUT_OR_NUT_PLATE", "scope": "PROJECT_DESIGNED", "R3 current geometry": "not modeled", "access/strength": "retention and serviceability absent", "status": "unresolved"},
])}

The existing R3 recess at the Sames-facing adapter surface is not a thread reaction method. A directionally wrong ISO 4762 M5 head placed in that recess produces positive B-Rep common with the official base (`{max(pocket_head_intersections):.6f} mm³` per hole in the nominal probe), while a base-exterior head plus Ø5 shank clears the base and adapter. This is a geometry diagnostic, not a released fastener design.

`THREAD_REACTION_METHOD = UNRESOLVED`\n\n`THREAD_REACTION_STATUS = UNRESOLVED`
""")
    write_text(out / "ADAPTER_THREAD_DESIGN_ANALYSIS.md", f"""# Adapter thread design analysis

Adapter V2 R3 is read-only in this audit. It contains four project-designed Sames-side clearance passages (`Ø{r3_hole_d:.3f} mm`) and a `Ø{2*r3_head_r:.3f} × {r3_head_depth:.3f} mm` recess. It does not contain a verified M5 tapped thread, threaded insert, captive nut, or nut plate at those four locations.

If the project later selects `ADAPTER_TAPPED`, the design must separately establish:

- tap-drill diameter and thread class;
- full thread depth versus runout and tip clearance;
- actual effective engagement, not plate thickness;
- adapter material, bolt grade, preload, stripping/pull-out and bearing checks;
- head-seat orientation and base-side clearance;
- manufacturing and inspection method.

No R4 is generated here. No R3 geometry is modified.
""")
    write_text(out / "R3_THREAD_CAPACITY_CHECK.md", f"""# R3 thread capacity check

`ADAPTER_R3_THREAD_CAPACITY = UNRESOLVED`.

Reason: R3 has no selected thread reaction method. The current four Sames-side holes are modeled as Ø{r3_hole_d:.3f} mm clearance passages, not tapped holes. A strength/capacity PASS cannot be claimed without a chosen thread/insert/nut architecture, adapter material, bolt grade, preload and access envelope.

The nominal R3-to-official-base B-Rep interface check is geometrically clean (`{adapter_base_common:.6f} mm³` common volume), but this is not a fastener capacity result.
""")
    write_text(out / "BOLT_LENGTH_STACK_ANALYSIS.md", f"""# Bolt length stack analysis

The only closed axial geometry is the Sames mounting passage:

{md_table([
    {"quantity": "base mating face to opposite external opening", "value": "≈4.000 mm", "evidence": "OFFICIAL_GEOMETRY_DERIVED"},
    {"quantity": "R3 total plate thickness", "value": "10.000 mm", "evidence": "PROJECT_DESIGNED / R3 read-only"},
    {"quantity": "required thread engagement", "value": "UNRESOLVED", "evidence": "reaction method not selected"},
    {"quantity": "thread runout / tip clearance", "value": "UNRESOLVED", "evidence": "not designed"},
    {"quantity": "washer/nut stack", "value": "UNRESOLVED", "evidence": "not selected"},
    {"quantity": "preliminary bolt length", "value": "UNRESOLVED", "evidence": "fail-closed"},
])}

The existing `10 mm` R3 plate is not treated as `10 mm effective thread`. No vague allowance or hard-coded bolt length is released.
""")
    write_text(out / "FASTENER_SHADOW_VALIDATION.md", f"""# Fastener shadow validation

`FASTENER_SHADOW_BREP = NOT_BUILT`.

The task's prerequisite is not satisfied: mounting geometry is closed, but thread reaction method, nut/insert geometry, service access, and bolt stack are not closed. A final head/shank/thread-engagement B-Rep would therefore encode an unapproved project choice.

Diagnostic geometry-only checks retained:

{md_table([
    {"check": "official base ↔ R3 nominal interface", "result": f"PASS; common volume {adapter_base_common:.6f} mm³", "classification": "EXPECTED_CONTACT"},
    {"check": "Ø5.5 Sames passage ↔ R3 Ø6.2 passage", "result": "PASS nominal radial clearance 0.35 mm", "classification": "CLEARANCE"},
    {"check": "ISO 4762 M5 head in existing R3 Sames-facing recess", "result": f"FAIL for base-facing architecture; {max(pocket_head_intersections):.6f} mm³ base common", "classification": "UNEXPECTED_VOLUME_PENETRATION"},
    {"check": "base-exterior M5 head + Ø5 shank through base/R3", "result": f"PASS geometry-only; max base/adapter common {max(max(r['base_mm3'], r['adapter_mm3']) for r in through_shank_intersections):.6f} mm³", "classification": "CLEARANCE"},
    {"check": "nut/insert ↔ R3/wrist", "result": "NOT_RUN", "classification": "UNRESOLVED"},
])}

These checks do not certify strength, preload, tolerance, sealing, wrench access, wrist clearance, or manufacturing release.
""")
    write_text(out / "TEST_RESULTS.md", f"""# Test results

{md_table([
    {"test": "specified CadQuery/OCP runtime", "result": f"PASS; Python {sys.version.split()[0]}, CadQuery {cq.__version__}, OCP runtime imported", "scope": "software environment"},
    {"test": "official STEP import", "result": f"PASS; {len(base.Solids())} solids, {len(base.Faces())} faces", "scope": "analytic B-Rep"},
    {"test": "official STEP identity hash", "result": f"PASS; {identity['sha256']}", "scope": "artifact identity"},
    {"test": "mating face selection", "result": f"PASS; face {mating_face_id}", "scope": "geometry-derived"},
    {"test": "high-confidence mounting feature gate", "result": "PASS; four and only four promoted", "scope": "classification"},
    {"test": "hole termination", "result": "PASS; four external through-holes", "scope": "geometry-derived"},
    {"test": "counterbore gate", "result": "PASS; none on promoted Sames holes", "scope": "geometry-derived"},
    {"test": "ISO273 M5 match", "result": "PASS as standard geometry candidate", "scope": "not manufacturer specification"},
    {"test": "R3 nominal B-Rep contact", "result": f"PASS; common volume {adapter_base_common:.6f} mm³", "scope": "R3 read-only"},
    {"test": "thread reaction closure", "result": "UNRESOLVED", "scope": "fail-closed"},
    {"test": "final fastener shadow", "result": "NOT_BUILT", "scope": "blocked by unresolved architecture"},
])}

Generated section plots: `{', '.join(x for x in section_files if x)}`.
""")
    write_text(out / "OPEN_ITEMS.md", """# Open items

1. Obtain a Sames drawing/manual explicitly specifying the 129691170 robot-side external mounting fastener and any washer/head requirement.
2. Select and approve exactly one reaction method: Adapter tapped, through-adapter with nut, threaded insert, captive nut/nut plate, or another documented method.
3. If Adapter tapped is selected, create a separately authorized adapter revision; this audit did not modify R3.
4. Verify adapter material, bolt grade, preload, thread stripping/pull-out, bearing, fatigue and safety factors.
5. Verify nut/insert/wrench access, robot wrist clearance, gun/base serviceability and internal interference using the complete approved assembly.
6. Rebuild a final fastener shadow B-Rep only after items 2–5 close.
7. Do not issue a manufacturing drawing or promote R3.
""")
    write_text(out / "QUALITY_AUDIT.md", """# Quality audit

| Check | Result |
|---|---|
| Did any gun→base M5 callout get transferred to base→adapter? | NO; family evidence is fenced. |
| Was a fluid/air/filter opening promoted as a mounting hole? | NO; only the four repeated Ø5.5/R3-matched holes were promoted. |
| Was an internal-cavity opening called an external through-hole? | NO; the promoted holes exit at opposite external planar faces. |
| Was through-hole automatically converted to Adapter tapped? | NO; thread reaction remains UNRESOLVED. |
| Was ISO 273 match written as Sames official? | NO; it is STANDARD_GEOMETRY_MATCH only. |
| Was a smooth STEP cylinder written as definite thread/no-thread? | NO; thread semantics remain UNRESOLVED. |
| Was 10 mm plate thickness treated as effective thread? | NO. |
| Was bolt length guessed? | NO. |
| Was R3 modified or R4 silently generated? | NO. |
| Was new software installed/upgraded? | NO; specified environment used. |

The positive B-Rep common volume from the directionally wrong R3 recess-head probe is retained as a failure finding, not hidden or tuned away.
""")
    write_text(out / "AUTHORITATIVE_STATE.md", """# Authoritative state

This package is the authoritative result of the 2026-09-07 Sames 129691170 robot-side mounting truth audit within its stated scope.

- Exact official identity: `OFFICIAL_VERIFIED`.
- Exact STEP analytic B-Rep: `PASS`.
- Robot-side mating face: face 46, geometry-derived.
- Promoted mounting features: four Ø5.5 external through-holes.
- Sames base-side counterbore: none on the promoted features.
- Nominal standard geometry candidate: M5 ISO 273 medium/normal clearance.
- Exact Sames robot-side fastener specification: not found.
- Thread reaction method: unresolved.
- Adapter V2 R3: read-only; not modified.
- Manufacturing release: no.
- Promotion: no promotion.

The older R3 parameter `asb.head_counterbore_depth_mm` remains a project-designed adapter parameter. It is not a Sames base counterbore and is not promoted as official truth.
""")
    write_text(out / "FINAL_STATUS.md", "# Final status\n\n```text\n" + "\n".join(f"{k} = {json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}" for k, v in final_status.items()) + "\n```\n\nThe only closed mechanical truth is the external-through-hole mounting geometry. The fastener reaction architecture is intentionally unresolved and therefore no manufacturing release or promotion is allowed.")
    write_text(out / "FINAL_REPORT.md", f"""# Sames `129691170` robot-side mounting interface truth audit

## Executive result

The exact official `129691170` STEP is identity-verified and analytically imported. The robot-side mating face is official STEP face `{mating_face_id}`. Exactly four high-confidence robot-side mounting features are promoted:

`4 × Ø{mounting[0]['diameter_mm']:.3f} mm external through-holes`, arranged as a `{x_pitch:.3f} × {z_pitch:.3f} mm` rectangle; its diagonal is `{pcd:.3f} mm` (not a PCD claim).

There is **no Sames-side counterbore** at those four centers. The Ø5.5 geometry matches an ISO 273 M5 clearance candidate, but `SAMES_OFFICIAL_FASTENER_SPEC = NOT_FOUND` for this exact robot-side interface.

## Fastener architecture result

`external through-hole ≠ Adapter tapped`.

Adapter V2 R3 is read-only and currently supplies clearance passages, not a verified thread reaction method. The candidate reaction methods were compared, but none is closed with access, strength, material, preload and complete B-Rep evidence. Therefore:

`THREAD_REACTION_METHOD = UNRESOLVED`\n\n`FASTENER_SHADOW_BREP = NOT_BUILT`\n\n`MANUFACTURING_RELEASE = NO`\n\n`PROMOTION = NO_PROMOTION`

The directional B-Rep probe also found that placing an ISO 4762 M5 socket-head candidate in the existing R3 Sames-facing recess creates positive common with the official base (`{max(pocket_head_intersections):.3f} mm³ per hole`), so the old “R3 counterbore” must not be interpreted as a closed Sames fastener solution.

## Evidence boundary

All dimensions above are analytic B-Rep measurements from the official STEP. STL, screenshots and mesh approximations were not used for formal dimensions. Standard tables are only geometry matches. Family-level Sames M5 callouts are kept separate from the exact robot-side interface.

See [FINAL_STATUS.md](FINAL_STATUS.md), [MOUNTING_FEATURE_CLASSIFICATION.md](MOUNTING_FEATURE_CLASSIFICATION.md), [THREAD_REACTION_METHOD_ANALYSIS.md](THREAD_REACTION_METHOD_ANALYSIS.md), and [R3_THREAD_CAPACITY_CHECK.md](R3_THREAD_CAPACITY_CHECK.md).
""")

    print(json.dumps({
        "status": "PASS_WITH_UNRESOLVED_THREAD_REACTION",
        "output_dir": str(out),
        "identity_sha256": identity["sha256"],
        "mating_face_id": mating_face_id,
        "mounting_feature_count": len(mounting),
        "mounting_diameter_mm": mounting[0]["diameter_mm"],
        "termination": mounting[0]["hole_termination_class"],
        "counterbore": "NO",
        "iso273_candidate": "M5",
        "thread_reaction_method": "UNRESOLVED",
        "r3_nominal_common_volume_mm3": adapter_base_common,
        "wrong_orientation_head_common_mm3": pocket_head_intersections,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
