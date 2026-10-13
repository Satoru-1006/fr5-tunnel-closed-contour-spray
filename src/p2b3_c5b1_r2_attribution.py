"""FR5 C5B1-R2 offline geometric attribution; not a coating or CCD certificate.

Consumes frozen C1 FK, C1 target/material normals and authoritative C5B1 scan JSON.
Does not invoke IK, Ruckig, FCL or mutate C1/C4. All ray intersections use
corresponding-material-point LOCAL TANGENT PLANES, not a curved-surface mesh.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

STANDOFF_M = 0.260
THRESHOLDS_M = (0.003, 0.006, 0.010)
AXES = ("tx", "ty", "tz", "rx", "ry", "rz")
C1_TARGET_HASH = "8a7d4a22c9e701bf840da255e1619e81b4a29bac4183528ad76fc918e16fbde7"
C1_FK_HASH = "ed8cffa87dae845df0dcc3cf50f88225e3f72b1fa425acce6def08e7ee60acee"

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))

def minus(a, b):
    return tuple(x-y for x,y in zip(a,b))

def plus(a, b):
    return tuple(x+y for x,y in zip(a,b))

def scale(a, v):
    return tuple(x*v for x in a)

def length(a):
    return math.sqrt(dot(a,a))

def unit(a):
    size=length(a)
    if not math.isfinite(size) or size<=1e-12:
        raise ValueError("invalid_direction_or_normal")
    return scale(a,1/size)

def rotation(v, axis, angle):
    i="xyz".index(axis[1]); c,s=math.cos(angle),math.sin(angle)
    if i==0: return (v[0],c*v[1]-s*v[2],s*v[1]+c*v[2])
    if i==1: return (c*v[0]+s*v[2],v[1],-s*v[0]+c*v[2])
    return (c*v[0]-s*v[1],s*v[0]+c*v[1],v[2])

def samples(fk_rows, target_rows):
    if len(fk_rows)!=181 or len(target_rows)!=181:
        raise ValueError("frozen_181_waypoint_correspondence_required")
    out=[]
    for fk,target in zip(fk_rows,target_rows):
        coords=lambda d,keys: tuple(float(d[k]) for k in keys)
        p=coords(fk,("actual_tcp_x","actual_tcp_y","actual_tcp_z"))
        a=unit(coords(fk,("tool_z_x","tool_z_y","tool_z_z")))
        t=coords(target,("x","y","z"))
        n=unit(coords(target,("nx","ny","nz")))
        if not all(math.isfinite(v) for group in (p,a,t,n) for v in group):
            raise ValueError("nonfinite_frozen_C1_data")
        out.append({"p":p,"a":a,"t":t,"n":n})
    return out

def metrics(rows, axis, sign, magnitude):
    if axis not in AXES or sign not in (-1,1) or not math.isfinite(magnitude) or magnitude<0:
        raise ValueError("invalid_ray")
    d=[0.0]*3
    if axis[0]=="t": d["xyz".index(axis[1])]=sign*magnitude
    d=tuple(d)
    names=("max_position_error_m",
           "max_abs_plane_axial_standoff_delta_m",
           "max_plane_lateral_offset_m",
           "max_material_normal_angle_deg")
    best={name:(-1.0,None) for name in names}
    invalid=[]; hit_dist=[]
    for i,row in enumerate(rows):
        p,a,t,n=row["p"],row["a"],row["t"],row["n"]
        if axis[0]=="r":
            t=rotation(t,axis,sign*magnitude)
            n=rotation(n,axis,sign*magnitude)
        t=plus(t,d)
        pos=length(minus(p,t))
        angle=math.degrees(math.acos(max(-1.0,min(1.0,dot(a,n)))))
        # C5A reference wall = registered TCP target + 0.260*material normal.
        wall=plus(t,scale(n,STANDOFF_M))
        den=dot(n,a)
        hit=None; lateral=None
        if den>1e-9:
            hit=dot(n,minus(wall,p))/den
            if not math.isfinite(hit) or hit<0:hit=None
        if hit is None:
            invalid.append(i)
        else:
            lateral=length(minus(plus(p,scale(a,hit)),wall))
            hit_dist.append(hit)
        observed={
            "max_position_error_m":pos,
            "max_abs_plane_axial_standoff_delta_m":abs(hit-STANDOFF_M) if hit is not None else None,
            "max_plane_lateral_offset_m":lateral,
            "max_material_normal_angle_deg":angle,
        }
        for name,value in observed.items():
            if value is not None and value>best[name][0]: best[name]=(value,i)
    result={name:({"value":v,"waypoint":i} if i is not None else None)
            for name,(v,i) in best.items()}
    result.update({
        "ray_hit_proxy_status":"VALID_ALL_MATERIAL_PLANES" if not invalid else "INDETERMINATE_RAY_INTERSECTION",
        "invalid_material_waypoints":invalid,
        "plane_ray_distance_min_m":min(hit_dist) if hit_dist else None,
        "plane_ray_distance_max_m":max(hit_dist) if hit_dist else None,
        "quality_failure":"NOT_ASSESSED",
        "full_curved_surface_ray_intersection":"NOT_ASSESSED",
        "proxy_geometry":"CORRESPONDING_MATERIAL_POINT_LOCAL_TANGENT_PLANE_NOT_MESH",
    })
    return result

def threshold_root(rows,axis,sign,gate):
    upper=0.020 if axis[0]=="t" else math.radians(2)
    def error(m):return metrics(rows,axis,sign,m)["max_position_error_m"]["value"]
    if error(0)>gate:
        return {"status":"BASELINE_EXCEEDS_HYPOTHETICAL_GATE","gate_m":gate}
    if error(upper)<=gate:
        return {"status":"NO_CROSSING_OBSERVED_AT_DOMAIN_ENDPOINT","gate_m":gate,"domain_upper":upper}
    lo,hi=0.0,upper
    for _ in range(36):
        mid=(lo+hi)/2
        if error(mid)>gate:hi=mid
        else:lo=mid
    return {"status":"ANALYTIC_ENGINEERING_THRESHOLD_CROSSING","gate_m":gate,
            "bracket":[lo,hi],"unit":"m" if axis[0]=="t" else "rad",
            "not_a_physical_failure":True}

def analyze(scan:dict[str,Any],rows):
    rays=scan["stages"]["all"]["rays"]
    if (len(rays)!=12 or scan.get("frozen_inputs",{}).get("q_waypoint_count")!=181
        or scan.get("measurement_type")!="NATIVE_FR5_C5A_FCL_SAMPLED"):
        raise ValueError("authoritative_native_c5b1_12_ray_input_required")
    output=[]
    for r in rays:
        axis,sign=r["axis"],r["sign"]
        if axis not in AXES or sign not in (-1,1):raise ValueError("invalid_archival_ray")
        roots={f"{int(g*1000)}mm":threshold_root(rows,axis,sign,g) for g in THRESHOLDS_M}
        positions=(0.003,0.006,0.010) if axis[0]=="t" else tuple(math.radians(x) for x in (0.25,0.5,1.0))
        probes=[{"magnitude":m,"unit":"m" if axis[0]=="t" else "rad",
                 **metrics(rows,axis,sign,m)} for m in positions]
        archive=r["predicates"]["position_diagnostic"]["first_observed_entry"]
        archived_mid=(archive["a"]+archive["b"])/2 if archive else None
        analytic=roots["6mm"].get("bracket")
        discrepancy=(abs(sum(analytic)/2-archived_mid)
                     if analytic and archived_mid is not None else None)
        output.append({
            "direction":axis+("+" if sign>0 else "-"),
            "threshold_attribution":roots,
            "original_position_bracket":archive,
            "original_vs_analytic_6mm_midpoint_discrepancy_native_units":discrepancy,
            "local_tangent_plane_probes":probes,
            "archived_collision_status":{
                key:r["predicates"][key]["status"]
                for key in ("robot_world_collision","self_collision")}
        })
    if len({r["direction"] for r in output})!=12:raise ValueError("duplicate_ray")
    return {
        "schema":"P2B3_C5B1_R2_ATTRIBUTION_V1","project":"FAIRINO_FR5",
        "status":"OFFLINE_GEOMETRIC_ATTRIBUTION_COMPLETE_NOT_PHYSICAL_VALIDATION",
        "classification":{
            "legacy_position":"ENGINEERING_THRESHOLD_CROSSING",
            "legacy_normal":"ENGINEERING_PROJECTION_DIAGNOSTIC",
            "sampled_collision":"C5B1_ARCHIVAL_SAMPLED_MODEL_ONLY",
            "process_quality":"NOT_ASSESSED",
            "continuous_collision":"NOT_ASSESSED"
        },
        "frozen_C1_waypoints":len(rows),"reference_standoff_m":STANDOFF_M,
        "rotation_origin":"base_link origin","registration":"BASE_LEFT",
        "physical_failure_margin":"NOT_AVAILABLE",
        "methodology":"analytic frozen C1 position metric plus local material tangent plane nozzle ray; no native reevaluation",
        "rays":output
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-json",type=Path,required=True)
    ap.add_argument("--c1-fk",type=Path,required=True)
    ap.add_argument("--c1-target",type=Path,required=True)
    ap.add_argument("--output-json",type=Path,required=True)
    ap.add_argument("--allow-hash-mismatch",action="store_true",help="exploratory ONLY; never authoritative")
    args=ap.parse_args()
    if args.output_json.exists():ap.error("output_json_already_exists")
    if not args.allow_hash_mismatch:
        if sha(args.c1_fk)!=C1_FK_HASH or sha(args.c1_target)!=C1_TARGET_HASH:
            ap.error("frozen_C1_input_sha256_mismatch")
    scan=json.loads(args.scan_json.read_text(encoding="utf-8-sig"))
    rows=samples(read_csv(args.c1_fk),read_csv(args.c1_target))
    result=analyze(scan,rows)
    result["input_sha256"]={
        "C1_FK":sha(args.c1_fk),"C1_TARGET":sha(args.c1_target),
        "C5B1_NATIVE_RESULT":sha(args.scan_json)}
    result["input_identity_status"]=(
        "UNVERIFIED_EXPLORATORY" if args.allow_hash_mismatch else "FROZEN_HASH_VERIFIED")
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(
        json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print("R2_OUTPUT="+str(args.output_json))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
