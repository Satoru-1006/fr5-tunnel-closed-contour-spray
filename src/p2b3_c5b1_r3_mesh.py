"""C5B1-R3: instrument original nearest-polyline normal and intersect C1 wall strip.

Finite open section: 181 frozen target wall points (TCP + 0.260 * normal),
180 consecutive *straight* segments extruded by C4 Y_THICKNESS=1.10 m.
This is the actual C1 reference *polygonal* surface representation, not
a physical tunnel scan, smoothed CAD surface, certified coating model or CCD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.p2b3_c5b1_r2_attribution import (
    C1_FK_HASH, C1_TARGET_HASH, C5B1_SCAN_HASH, STANDOFF_M,
    dot, length, minus, plus, read_csv, rotation, samples, scale, sha, unit,
)

Y_WIDTH_M = 1.10
EPS = 1e-10


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def transform(v, axis, sign, amount, *, direction=False):
    if axis[0] == "r":
        return rotation(v, axis, sign*amount)
    if direction:
        return v
    change = [0., 0., 0.]
    change["xyz".index(axis[1])] = sign*amount
    return plus(v, change)


def inverse(v, axis, sign, amount, *, direction=False):
    if axis[0] == "r":
        return rotation(v, axis, -sign*amount)
    if direction:
        return v
    change = [0., 0., 0.]
    change["xyz".index(axis[1])] = -sign*amount
    return plus(v, change)


def wall(rows):
    if len(rows) != 181:
        raise ValueError("FROZEN_181_WALL_POINTS_REQUIRED")
    out = [plus(row["t"], scale(row["n"], STANDOFF_M)) for row in rows]
    if max(p[1] for p in out) - min(p[1] for p in out) > 1e-8:
        raise ValueError("C1_WALL_SECTION_NOT_SINGLE_Y_STATION")
    if any(length(minus(a, b)) < 1e-9 for a, b in zip(out, out[1:])):
        raise ValueError("DEGENERATE_C1_WALL_SEGMENT")
    return out


def closest_segment(point, section, normals):
    """Exactly reproduce C5A's open-polyline nearest-point and normal formula.

    Strict < comparison retains first candidate in a squared-distance tie.
    """
    best = None
    for k, (a, b) in enumerate(zip(section, section[1:])):
        edge = minus(b, a)
        alpha = max(0., min(1., dot(minus(point, a), edge) /
                              max(dot(edge, edge), 1e-15)))
        q = plus(a, scale(edge, alpha))
        d2 = dot(minus(point, q), minus(point, q))
        if best is None or d2 < best["squared_distance_m2"]:
            normal = unit(plus(scale(normals[k], 1-alpha),
                               scale(normals[k+1], alpha)))
            best = {"segment": k, "alpha": alpha,
                    "squared_distance_m2": d2, "normal": normal}
    return best


def projected_normal_diagnostic(rows, axis, sign, amount):
    if axis not in ("tx","ty","tz","rx","ry","rz") or sign not in (-1, 1):
        raise ValueError("INVALID_RAY")
    points = [transform(x, axis, sign, amount) for x in wall(rows)]
    normals = [unit(transform(r["n"], axis, sign, amount, direction=True))
               for r in rows]
    best = None
    changed = 0
    original_wall = wall(rows)
    original_normals = [r["n"] for r in rows]
    for i, r in enumerate(rows):
        near = closest_segment(r["p"], points, normals)
        baseline = closest_segment(r["p"], original_wall, original_normals)
        changed += int(near["segment"] != baseline["segment"])
        same = unit(transform(r["n"], axis, sign, amount, direction=True))
        angle = math.degrees(math.acos(max(-1., min(1., dot(r["a"], near["normal"])))))
        corresponding = math.degrees(math.acos(max(-1., min(1., dot(r["a"], same)))))
        record = {"waypoint": i, "projected_segment": near["segment"],
                  "projected_alpha": near["alpha"],
                  "baseline_segment": baseline["segment"],
                  "projected_angle_deg": angle,
                  "same_material_angle_deg": corresponding}
        if best is None or angle > best["projected_angle_deg"]:
            best = record
    return {"critical": best, "changed_nearest_segment_waypoints": changed,
            "max_angle_mechanism": "NEAREST_POLYLINE_REASSOCIATION_AND_MAX_WAYPOINT_SWITCH",
            "projected_reference": "C5A_REPRODUCED_OPEN_POLYLINE"}


def intersect_open_wall(rows, axis, sign, amount, waypoint):
    """Exact forward-ray/finite *polygonal* open extrusion intersection.

    Analytic ray/quad intersection for each 3D ruled facet. Since all nominal
    contour points share Y, solve in nominal x-z cross-section, checking Y range.
    The inverse BASE_LEFT transform keeps ray length unchanged. Ray first hit
    is nearest positive t; coincident segment/edge hits are deduplicated.
    """
    if axis not in ("tx","ty","tz","rx","ry","rz") or sign not in (-1, 1):
        raise ValueError("INVALID_RAY")
    section = wall(rows)
    r = rows[waypoint]
    origin = inverse(r["p"], axis, sign, amount)
    ray = unit(inverse(r["a"], axis, sign, amount, direction=True))
    ymid = section[0][1]
    intersections = []
    degenerate_parallel = False
    for k, (u, v) in enumerate(zip(section, section[1:])):
        edge = minus(v, u)
        # In-plane normal to the ruled surface, cross(edge, y_axis).
        normal = cross(edge, (0.,1.,0.))
        den = dot(normal, ray)
        numerator = dot(normal, minus(u, origin))
        if abs(den) < EPS:
            if abs(numerator) < EPS:
                degenerate_parallel = True
            continue
        t = numerator / den
        if not math.isfinite(t) or t <= EPS:
            continue
        hit = plus(origin, scale(ray, t))
        xy = (hit[0]-u[0])*edge[0] + (hit[2]-u[2])*edge[2]
        s = xy / (edge[0]*edge[0]+edge[2]*edge[2])
        if -1e-9 <= s <= 1+1e-9 and abs(hit[1]-ymid) <= Y_WIDTH_M/2+1e-9:
            s = max(0., min(1., s))
            # Interpolate material normals for an oriented proxy (NOT actual smooth-surface normal).
            n = unit(plus(scale(rows[k]["n"], 1-s), scale(rows[k+1]["n"], s)))
            n = unit(transform(n, axis, sign, amount, direction=True))
            intersections.append({
                "distance_m": t, "segment": k, "segment_fraction": s,
                "point_base_xyz_m": list(transform(hit, axis, sign, amount)),
                "normal_proxy_base_xyz": list(n)})
    intersections.sort(key=lambda x:(x["distance_m"],x["segment"]))
    distinct = []
    for candidate in intersections:
        if not distinct or candidate["distance_m"]-distinct[-1]["distance_m"] > 1e-8:
            distinct.append(candidate)
    if degenerate_parallel:
        return {"status": "INDETERMINATE_COPLANAR_RAY", "first_hit": None,
                "distinct_hit_count": len(distinct)}
    if not distinct:
        return {"status": "NO_FORWARD_SURFACE_HIT", "first_hit": None,
                "distinct_hit_count": 0}
    first = distinct[0]
    boundary = any(abs(first["distance_m"]-other["distance_m"]) <= 1e-8
                   and first["segment"] != other["segment"] for other in intersections)
    ref = transform(plus(r["t"], scale(r["n"], STANDOFF_M)), axis, sign, amount)
    return {
        "status": "HIT_MULTIPLE" if len(distinct)>1 else
                  ("HIT_FACET_EDGE_AMBIGUOUS_NORMAL" if boundary else "HIT"),
        "first_hit": first, "distinct_hit_count": len(distinct),
        "first_hit_axial_standoff_delta_m": first["distance_m"]-STANDOFF_M,
        "first_hit_landing_offset_from_material_reference_m":
            length(minus(first["point_base_xyz_m"], ref)),
        "ray_normal_angle_deg": math.degrees(math.acos(max(-1.,min(1.,dot(r["a"],first["normal_proxy_base_xyz"]))))),
        "material_point_reference": "frozen_C1_correspondence_not_actual_target",
        "facet_edge_ambiguous_normal": boundary}


def evaluate_direction(rows, axis, sign, magnitude):
    per_wp = [intersect_open_wall(rows, axis, sign, magnitude, i)
              for i in range(len(rows))]
    invalid = [i for i, r in enumerate(per_wp)
               if r["status"] not in ("HIT", "HIT_MULTIPLE", "HIT_FACET_EDGE_AMBIGUOUS_NORMAL")]
    multi = [i for i,r in enumerate(per_wp) if r["status"]=="HIT_MULTIPLE"]
    edge = [i for i,r in enumerate(per_wp) if r["status"]=="HIT_FACET_EDGE_AMBIGUOUS_NORMAL"]
    extrema = {}
    for name in ("first_hit_axial_standoff_delta_m",
                 "first_hit_landing_offset_from_material_reference_m",
                 "ray_normal_angle_deg"):
        valid = [(i, abs(r[name]) if name.endswith("delta_m") else r[name])
                 for i,r in enumerate(per_wp) if r.get(name) is not None]
        extrema[name] = ({"value": max(valid,key=lambda v:v[1])[1],
                           "waypoint": max(valid,key=lambda v:v[1])[0]}
                         if valid else None)
    return {
        "surface_model": "C1_OPEN_181_POINT_PIECEWISE_PLANAR_RULED_STRIP",
        "intersection_method": "ANALYTIC_RAY_FINITE_QUAD_NOT_TANGENT_PLANE",
        "intersection_status": "COMPLETE_FINITE_SURFACE_INTERSECTIONS" if not invalid else "INDETERMINATE",
        "ray_hit_waypoints": len(rows)-len(invalid),
        "invalid_waypoints": invalid, "multiple_hit_waypoints": multi,
        "ambiguous_edge_waypoints": edge,
        "extrema": extrema,
        "first_hit_critical_example": per_wp[extrema["first_hit_axial_standoff_delta_m"]["waypoint"]]
            if extrema["first_hit_axial_standoff_delta_m"] else None,
        "physical_tunnel_scan": "NOT_AVAILABLE",
        "smooth_surface_intersection": "NOT_ASSESSED",
        "coating_quality": "NOT_ASSESSED",
        "continuous_collision": "NOT_ASSESSED"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c1-fk",required=True,type=Path)
    p.add_argument("--c1-target",required=True,type=Path)
    p.add_argument("--c5b1-scan",required=True,type=Path)
    p.add_argument("--output-json",required=True,type=Path)
    p.add_argument("--allow-hash-mismatch",action="store_true")
    a=p.parse_args()
    if a.output_json.exists(): p.error("OUTPUT_ALREADY_EXISTS")
    if not a.allow_hash_mismatch and (
        sha(a.c1_fk)!=C1_FK_HASH or sha(a.c1_target)!=C1_TARGET_HASH or
        sha(a.c5b1_scan)!=C5B1_SCAN_HASH):
        p.error("AUTHORITATIVE_INPUT_SHA256_MISMATCH")
    rows=samples(read_csv(a.c1_fk), read_csv(a.c1_target))
    scan=json.loads(a.c5b1_scan.read_text(encoding="utf-8-sig"))
    archived=scan["stages"]["all"]["rays"]
    if len(archived)!=12 or scan.get("measurement_type")!="NATIVE_FR5_C5A_FCL_SAMPLED":
        p.error("WRONG_C5B1_NATIVE_SCAN_CONTRACT")
    diagnostics=[]
    for sign in (-1,1):
        orig=next(x for x in archived if x["axis"]=="tz" and x["sign"]==sign)
        bracket=orig["predicates"]["normal_diagnostic"]["first_observed_entry"]
        if not bracket: p.error("MISSING_TZ_BOUNDARY_BRACKET")
        magnitudes=[0., bracket["a"],bracket["b"],.020]
        diagnostics.append({"ray":"tz-" if sign<0 else "tz+",
            "archived_normal_bracket_m":[bracket["a"],bracket["b"]],
            "probes":[{"magnitude_m":v,
                       "nearest_polyline":projected_normal_diagnostic(rows,"tz",sign,v),
                       "mesh_intersection":evaluate_direction(rows,"tz",sign,v)}
                      for v in magnitudes]})
    result={"schema":"P2B3_C5B1_R3_FINITE_SURFACE_RAYCAST_V1",
        "project":"FAIRINO_FR5","status":"REFERENCE_MESH_ANALYSIS_NOT_PHYSICAL_CERTIFICATION",
        "source_hashes":{"C1_FK":sha(a.c1_fk),"C1_TARGET":sha(a.c1_target),
                         "C5B1_NATIVE_SCAN":sha(a.c5b1_scan)},
        "source_identity":"UNVERIFIED_EXPLORATORY" if a.allow_hash_mismatch else "FROZEN_HASH_VERIFIED",
        "reference_surface":"C1 181-point open wall polyline extruded 1.10m in Y; finite, polygonal, not smooth or measured",
        "registration":"BASE_LEFT", "frozen_joint_trajectory":True,
        "process_quality":"NOT_ASSESSED","physical_clearance":"NOT_AVAILABLE",
        "diagnostics":diagnostics}
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(result,indent=2,allow_nan=False,sort_keys=True)+"\n",
                             encoding="utf-8")
    print("C5B1_R3_JSON="+str(a.output_json))


if __name__=="__main__":
    main()
