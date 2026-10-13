"""FR5 R3-R2 finite ruled-surface raycast: explicit hit and normal certainty.

Reference: C1 open 181-point wall polyline, extruded 1.10 m along Y.
C1 q(t), C4 scene, C5B1 native output never mutated; pure offline geometry.
This is NOT measured tunnel geometry, coating-quality validation or CCD.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.p2b3_c5b1_r2_attribution import (
    C1_FK_HASH, C1_TARGET_HASH, C5B1_SCAN_HASH, STANDOFF_M,
    dot, length, minus, plus, read_csv, rotation, samples, scale, sha, unit,
)

WIDTH_M = 1.10
# Pure numerical classification tolerances, not FR5 or coating tolerances.
PLANE_TOL_M = 1e-10
EDGE_PARAMETER_TOL = 1e-9
DISTINCT_HIT_TOL_M = 1e-9
PARALLEL_COS_TOL = 1e-12
NORMAL_EQUIVALENT_DEG = 1e-6
_AXES = ("tx", "ty", "tz", "rx", "ry", "rz")


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1., min(1., dot(unit(a), unit(b))))))


def transform(v, axis, sign, amplitude, *, direction=False, inverse=False):
    signed = sign*amplitude * (-1 if inverse else 1)
    if axis.startswith("r"):
        return rotation(v, axis, signed)
    if direction:
        return v
    q = [0., 0., 0.]
    q["xyz".index(axis[1])] = signed
    return plus(v, q)


def wall_points(rows):
    if len(rows) != 181:
        raise ValueError("FROZEN_181_WAYPOINTS_REQUIRED")
    section = [plus(row["t"], scale(row["n"], STANDOFF_M)) for row in rows]
    if max(p[1] for p in section)-min(p[1] for p in section) > PLANE_TOL_M:
        raise ValueError("C1_WALL_NOT_SINGLE_STATION")
    for p, q in zip(section, section[1:]):
        if length(minus(p, q)) < PLANE_TOL_M:
            raise ValueError("DEGENERATE_OPEN_WALL_SEGMENT")
    return section


def _clip_linear(lo, hi, base, delta, left, right):
    """Clip nonnegative ray t by the affine inequality left <= base+t*delta <= right."""
    if abs(delta) <= PARALLEL_COS_TOL:
        return (lo, hi) if left-EDGE_PARAMETER_TOL <= base <= right+EDGE_PARAMETER_TOL else None
    t0, t1 = (left-base)/delta, (right-base)/delta
    a, b = min(t0, t1), max(t0, t1)
    lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if hi >= lo-PLANE_TOL_M else None


def _coplanar_interval(origin, direction, start, edge, ymid):
    """Ray--finite-quad overlap interval (not infinite supporting-plane overlap)."""
    edge2 = dot(edge, edge)
    if edge2 < 1e-22:
        raise ValueError("DEGENERATE_WALL_FACET")
    along = dot(minus(origin, start), edge)/edge2
    along_rate = dot(direction, edge)/edge2
    z = _clip_linear(0., float("inf"), along, along_rate, 0., 1.)
    if z is None:
        return None
    y0, dy = origin[1]-ymid, direction[1]
    z = _clip_linear(z[0], z[1], y0, dy, -WIDTH_M/2, WIDTH_M/2)
    if z is None or not math.isfinite(z[0]):
        return None
    return (max(0., z[0]), z[1])


def _facet_intersections(section, material_normals, origin, direction):
    """Return all bounded ray hits plus finite coplanar-overlap intervals."""
    hits, overlaps = [], []
    ymid = section[0][1]
    for seg, (start, end) in enumerate(zip(section, section[1:])):
        edge = minus(end, start)
        facet_normal = unit(cross(edge, (0., 1., 0.)))
        den = dot(facet_normal, direction)
        numerator = dot(facet_normal, minus(start, origin))
        if abs(den) <= PARALLEL_COS_TOL:
            if abs(numerator) <= PLANE_TOL_M:
                overlap = _coplanar_interval(origin, direction, start, edge, ymid)
                if overlap is not None:
                    overlaps.append({"segment": seg, "t_interval_m": list(overlap)})
            continue
        t = numerator/den
        if not math.isfinite(t) or t <= PLANE_TOL_M:
            continue
        p = plus(origin, scale(direction, t))
        s = dot(minus(p, start), edge)/max(dot(edge, edge), 1e-30)
        if not (-EDGE_PARAMETER_TOL <= s <= 1.+EDGE_PARAMETER_TOL):
            continue
        if abs(p[1]-ymid) > WIDTH_M/2+PLANE_TOL_M:
            continue
        s = max(0., min(1., s))
        ref_n = unit(plus(scale(material_normals[seg], 1-s), scale(material_normals[seg+1], s)))
        # Align geometric facet normal with frozen wall material normal.
        geo_n = facet_normal if dot(facet_normal, ref_n) >= 0 else scale(facet_normal, -1)
        hits.append({"segment":seg,"fraction":s,"distance_m":t,"point_local_xyz_m":list(p),
                     "geometric_normal_local_xyz":list(geo_n),
                     "interpolated_material_normal_local_xyz":list(ref_n),
                     "at_cross_section_edge":s <= EDGE_PARAMETER_TOL or s>=1.-EDGE_PARAMETER_TOL,
                     "at_longitudinal_edge":abs(abs(p[1]-ymid)-WIDTH_M/2) <= PLANE_TOL_M})
    hits.sort(key=lambda h:(h["distance_m"],h["segment"]))
    return hits, overlaps


def _group_adjacent_hits(hits):
    """Merge shared-edge duplicates ONLY for adjacent facets, never merely by t distance."""
    groups: list[list[dict[str,Any]]] = []
    for h in hits:
        matches=[]
        for gi, group in enumerate(groups):
            if any(abs(h["segment"]-g["segment"])==1 and
                   h["at_cross_section_edge"] and g["at_cross_section_edge"] and
                   abs(h["distance_m"]-g["distance_m"])<=DISTINCT_HIT_TOL_M
                   for g in group):
                matches.append(gi)
        if not matches:
            groups.append([h])
        else:
            base=matches[0]
            groups[base].append(h)
            for gi in reversed(matches[1:]):
                groups[base].extend(groups.pop(gi))
    groups.sort(key=lambda a:min(x["distance_m"] for x in a))
    return groups


def intersect_waypoint(rows, section, waypoint, axis, sign, amplitude):
    if axis not in _AXES or sign not in (-1,1) or not math.isfinite(amplitude) or amplitude<0:
        raise ValueError("INVALID_SIGNED_BASE_LEFT_RAY")
    p0=rows[waypoint]["p"]
    origin=transform(p0,axis,sign,amplitude,inverse=True)
    direction=unit(transform(rows[waypoint]["a"],axis,sign,amplitude,inverse=True,direction=True))
    hits, coplanar=_facet_intersections(section,[r["n"] for r in rows],origin,direction)
    groups=_group_adjacent_hits(hits)
    known_first = min((h["distance_m"] for h in hits),default=float("inf"))
    blocking=[o for o in coplanar if o["t_interval_m"][0] <= known_first+DISTINCT_HIT_TOL_M]
    if blocking:
        return {"position_status":"INDETERMINATE_COPLANAR_FINITE_OVERLAP",
                "normal_status":"NOT_ASSESSED", "first_hit":None,
                "known_point_hit_count":len(hits),"distinct_point_hit_groups":len(groups),
                "finite_coplanar_intervals":coplanar,
                "known_first_point_hit_distance_m":None if not hits else known_first}
    if not groups:
        return {"position_status":"NO_FORWARD_FINITE_SURFACE_HIT","normal_status":"NOT_ASSESSED",
                "first_hit":None,"known_point_hit_count":0,"distinct_point_hit_groups":0,
                "finite_coplanar_intervals":coplanar}
    first=groups[0]
    t=min(c["distance_m"] for c in first)
    # Nonadjacent near-equal-distance groups must remain distinct and flag first-hit ambiguity.
    coincident_nonadjacent=any(abs(min(v["distance_m"] for v in gg)-t)<=DISTINCT_HIT_TOL_M for gg in groups[1:])
    normals_geo=[]
    normals_material=[]
    def world_dir(x):return unit(transform(x,axis,sign,amplitude,direction=True))
    for c in first:
        normals_geo.append(world_dir(c["geometric_normal_local_xyz"]))
        normals_material.append(world_dir(c["interpolated_material_normal_local_xyz"]))
    geo_angles=[angle_deg(rows[waypoint]["a"],x) for x in normals_geo]
    material_angles=[angle_deg(rows[waypoint]["a"],x) for x in normals_material]
    normal_difference=max((angle_deg(normals_geo[i],normals_geo[j])
                           for i in range(len(first)) for j in range(i+1,len(first))),default=0.)
    normal_status=("AMBIGUOUS_SHARED_EDGE_NORMAL" if normal_difference>NORMAL_EQUIVALENT_DEG
                   or coincident_nonadjacent else "UNIQUE_FACET_NORMAL")
    position_status=("AMBIGUOUS_COINCIDENT_NONADJACENT_FACETS" if coincident_nonadjacent
                     else "FIRST_HIT_POSITION_DETERMINATE")
    representative=first[0]
    ref=transform(plus(rows[waypoint]["t"],scale(rows[waypoint]["n"],STANDOFF_M)),axis,sign,amplitude)
    hit_world=transform(representative["point_local_xyz_m"],axis,sign,amplitude)
    first_record={"distance_m":t,"point_base_xyz_m":list(hit_world),
                  "participating_segment_ids":sorted(c["segment"] for c in first),
                  "member_hit_count":len(first),
                  "normal_geometric_angle_interval_deg":[min(geo_angles),max(geo_angles)],
                  "normal_interpolated_angle_interval_deg":[min(material_angles),max(material_angles)],
                  "normal_geometric_candidate_vectors_base":normals_geo,
                  "normal_interpolated_candidate_vectors_base":normals_material,
                  "facet_normal_separation_deg":normal_difference,
                  "landing_offset_material_reference_m":length(minus(hit_world,ref)),
                  "axial_standoff_delta_m":t-STANDOFF_M}
    return {"position_status":position_status,"normal_status":normal_status,
            "first_hit":first_record,"known_point_hit_count":len(hits),
            "distinct_point_hit_groups":len(groups),
            "additional_occluded_groups":len(groups)-1,
            "other_group_distances_m":[min(h["distance_m"] for h in g) for g in groups[1:]],
            "finite_coplanar_intervals":coplanar}


def evaluate_direction(rows, axis, sign, amplitude):
    section=wall_points(rows)
    rec=[intersect_waypoint(rows,section,i,axis,sign,amplitude) for i in range(181)]
    pos_bad=[i for i,r in enumerate(rec) if r["position_status"]!="FIRST_HIT_POSITION_DETERMINATE"]
    normal_bad=[i for i,r in enumerate(rec) if r["normal_status"]!="UNIQUE_FACET_NORMAL"]
    no_hit=[i for i,r in enumerate(rec) if r["position_status"]=="NO_FORWARD_FINITE_SURFACE_HIT"]
    coplanar=[i for i,r in enumerate(rec) if r["position_status"]=="INDETERMINATE_COPLANAR_FINITE_OVERLAP"]
    ambiguous=[i for i,r in enumerate(rec) if r["normal_status"]=="AMBIGUOUS_SHARED_EDGE_NORMAL"]
    occluded=[i for i,r in enumerate(rec) if r.get("additional_occluded_groups",0)]
    extrema={}
    for key in ("axial_standoff_delta_m","landing_offset_material_reference_m"):
        measures=[(i,abs(r["first_hit"][key])) for i,r in enumerate(rec) if r.get("first_hit")]
        extrema[key]={"abs_value_m":max(measures,key=lambda v:v[1])[1],"waypoint":max(measures,key=lambda v:v[1])[0]} if measures else None
    for field in ("normal_geometric_angle_interval_deg","normal_interpolated_angle_interval_deg"):
        measures=[(i,max(r["first_hit"][field])) for i,r in enumerate(rec) if r.get("first_hit")]
        extrema[field]={"upper_deg":max(measures,key=lambda v:v[1])[1],"waypoint":max(measures,key=lambda v:v[1])[0]} if measures else None
    return {"waypoint_count":181,"point_hit_waypoints":181-len(no_hit)-len(coplanar),
            "position_certainty":"COMPLETE" if not pos_bad else "PARTIAL_OR_INDETERMINATE",
            "normal_certainty":"COMPLETE" if not normal_bad else "PARTIAL_OR_AMBIGUOUS",
            "position_indeterminate_waypoints":pos_bad,"normal_nonunique_or_unavailable_waypoints":normal_bad,
            "no_forward_hit_waypoints":no_hit,"coplanar_uncertain_waypoints":coplanar,
            "shared_edge_multiple_normal_waypoints":ambiguous,"multiple_distinct_hits_waypoints":occluded,
            "extrema_on_confirmed_hits_only":extrema,
            "first_critical_hit":next((r["first_hit"] for i,r in enumerate(rec)
                                      if extrema["axial_standoff_delta_m"] and i==extrema["axial_standoff_delta_m"]["waypoint"]),None),
            "scope":"FINITE_C1_DESIGN_POLYGONAL_STRIP_NOT_MEASURED_TUNNEL",
            "physical_coating_quality":"NOT_ASSESSED","continuous_collision":"NOT_ASSESSED"}


def evaluate_authoritative(scan, rows):
    rays=scan["stages"]["all"]["rays"]
    if len(rays)!=12 or scan.get("measurement_type")!="NATIVE_FR5_C5A_FCL_SAMPLED":
        raise ValueError("C5B1_NATIVE_12_RAY_INPUT_REQUIRED")
    if scan.get("frozen_inputs",{}).get("q_waypoint_count")!=181:
        raise ValueError("C5B1_FROZEN_WAYPOINT_INCONSISTENCY")
    data=[]
    for sign in (-1,1):
        original=next((x for x in rays if x.get("axis")=="tz" and x.get("sign")==sign),None)
        if original is None:
            raise ValueError("MISSING_TZ_ARCHIVED_RAY")
        bracket=original["predicates"]["normal_diagnostic"].get("first_observed_entry")
        if not bracket or not (0<bracket["a"]<bracket["b"]<=0.020):
            raise ValueError("INVALID_ARCHIVED_TZ_NORMAL_BRACKET")
        probes=[0.,bracket["a"],bracket["b"],.02]
        data.append({"ray":"tz-" if sign<0 else "tz+",
                     "archive_normal_10deg_bracket_m":[bracket["a"],bracket["b"]],
                     "probes":[{"amplitude_m":m,"signed_base_tz_m":sign*m,
                                "measurement":evaluate_direction(rows,"tz",sign,m)} for m in probes]})
    return {"schema":"FR5_C5B1_R3_R2_FINITE_RAYCAST_V1","project":"FAIRINO_FR5",
            "status":"OFFLINE_FINITE_REFERENCE_SURFACE_REPLAY_NOT_PHYSICAL_CERTIFICATE",
            "semantic_change":"SPLIT_POSITION_AND_NORMAL_CERTAINTY; FINITE_COPLANAR_INTERVALS; ADJACENT_DUPLICATE_ONLY",
            "numerical_tolerances":{
                "plane_m":PLANE_TOL_M,"parallel_cos":PARALLEL_COS_TOL,
                "hit_group_m":DISTINCT_HIT_TOL_M,"edge_parameter":EDGE_PARAMETER_TOL,
                "equivalent_normal_deg":NORMAL_EQUIVALENT_DEG},
            "source":"C1_181_POINT_OPEN_RULLED_STRIP_Y_1.10M",
            "coating_quality":"NOT_ASSESSED", "measured_surface":"NOT_AVAILABLE",
            "ccd":"NOT_ASSESSED","rays":data}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c1-fk",type=Path,required=True)
    p.add_argument("--c1-target",type=Path,required=True)
    p.add_argument("--c5b1-scan",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    args=p.parse_args(argv)
    if args.output_json.exists():
        p.error("OUTPUT_EXISTS_NO_OVERWRITE")
    identities={"C1_FK":sha(args.c1_fk),"C1_TARGET":sha(args.c1_target),"C5B1_NATIVE_RESULT":sha(args.c5b1_scan)}
    if identities!={"C1_FK":C1_FK_HASH,"C1_TARGET":C1_TARGET_HASH,"C5B1_NATIVE_RESULT":C5B1_SCAN_HASH}:
        p.error("AUTHORITATIVE_INPUT_SHA256_MISMATCH")
    scan=json.loads(args.c5b1_scan.read_text(encoding="utf-8-sig"))
    rows=samples(read_csv(args.c1_fk),read_csv(args.c1_target))
    result=evaluate_authoritative(scan,rows)
    result["input_sha256"]=identities
    result["input_identity_status"]="FROZEN_HASH_VERIFIED"
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("R3R2_RESULT="+str(args.output_json))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
