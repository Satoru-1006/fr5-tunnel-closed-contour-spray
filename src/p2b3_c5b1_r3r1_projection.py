"""FR5 C5B1 R3-R1: explain nearest-polyline normal diagnostic discontinuities.

Pure offline, standard-library tool. No IK, FK, C4 changes, FCL or physical claims.
Historical C5A open-polyline strict '<' minimization is retained at stride=1.
Refining straight segments is an *invariance* control, not a smoother surface.
Coarsening alters reference geometry and is a sensitivity experiment only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

C1_FK_SHA256 = "ed8cffa87dae845df0dcc3cf50f88225e3f72b1fa425acce6def08e7ee60acee"
C1_TARGET_SHA256 = "8a7d4a22c9e701bf840da255e1619e81b4a29bac4183528ad76fc918e16fbde7"
C5B1_SHA256 = "20dd1df5a2c4a1239f3a8079d70cb0932c80b9b57b70c94064d3614ea14ee31a"
STANDOFF_M = 0.260
NEAR_TIE_DISTANCE_M = 1e-6  # Diagnostic label ONLY, not physical or numerical error bound.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def scale(a, t):
    return tuple(x*t for x in a)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    d=norm(a)
    if not math.isfinite(d) or d < 1e-14:
        raise ValueError("invalid_direction_or_normal")
    return scale(a, 1/d)


def angle_deg(a,b):
    return math.degrees(math.acos(max(-1.,min(1.,dot(unit(a),unit(b))))))


def lerp(a,b,t):
    return add(scale(a,1.-t),scale(b,t))


def load_frozen(fk, target):
    fk_rows=read_csv(fk); target_rows=read_csv(target)
    if len(fk_rows)!=181 or len(target_rows)!=181:
        raise ValueError("FROZEN_WAYPOINT_COUNT_NOT_181")
    points=[]; normals=[]; positions=[]; axes=[]
    for f,r in zip(fk_rows,target_rows):
        x=tuple(float(r[k]) for k in ("x","y","z"))
        n=unit(tuple(float(r[k]) for k in ("nx","ny","nz")))
        points.append(add(x,scale(n,STANDOFF_M)))
        normals.append(n)
        positions.append(tuple(float(f[k]) for k in ("actual_tcp_x","actual_tcp_y","actual_tcp_z")))
        axes.append(unit(tuple(float(f[k]) for k in ("tool_z_x","tool_z_y","tool_z_z"))))
    if any(not math.isfinite(x) for line in (points+normals+positions+axes) for x in line):
        raise ValueError("NONFINITE_FROZEN_VALUE")
    return points,normals,positions,axes


def build_segments(points, normals, *, stride=1, subdivide=1):
    """Preserve the original *open* chain and strict segment order.

    For stride>1, skip original vertices (different approximation).
    For subdivide>1, split straight edges, preserving piecewise geometry.
    Segment 'source_id' identifies original/coarsened parent; comparisons of
    top two use distinct parent IDs so a split edge cannot tie with itself.
    """
    if stride<1 or subdivide<1 or len(points)!=len(normals) or len(points)<2:
        raise ValueError("INVALID_POLYLINE")
    ids=list(range(0,len(points)-1,stride)) + [len(points)-1]
    segments=[]
    for start,end in zip(ids,ids[1:]):
        a,b=points[start],points[end]
        na,nb=normals[start],normals[end]
        for j in range(subdivide):
            lo=j/subdivide; hi=(j+1)/subdivide
            segments.append((start,lo,hi,lerp(a,b,lo),lerp(a,b,hi),na,nb))
    return segments


def all_candidates(point, axis, segments, tz_m):
    """Return closest candidate per original parent edge (not per split)."""
    selected={}
    for source,lo,hi,a0,b0,na,nb in segments:
        a=(a0[0],a0[1],a0[2]+tz_m)
        b=(b0[0],b0[1],b0[2]+tz_m)
        e=sub(b,a)
        u=max(0.,min(1.,dot(sub(point,a),e)/max(dot(e,e),1e-15)))
        p=add(a,scale(e,u)); d2=dot(sub(point,p),sub(point,p))
        alpha=lo+(hi-lo)*u
        n=unit(lerp(na,nb,alpha))
        v={"segment":source,"alpha":alpha,"distance_m":math.sqrt(d2),
           "distance_sq_m2":d2,"angle_deg":angle_deg(axis,n),"normal":n}
        # Historical strict '<' tie-break for the same parent edge.
        if source not in selected or d2<selected[source]["distance_sq_m2"]:
            selected[source]=v
    # Distance and earliest source segment tie-break. This reproduces the
    # original C5A loop at stride=subdivide=1 for unique minima.
    return sorted(selected.values(),key=lambda v:(v["distance_sq_m2"],v["segment"]))


def critical_probe(point,axis,segments,tz_m,waypoint,*,near_tie_m=NEAR_TIE_DISTANCE_M):
    cand=all_candidates(point,axis,segments,tz_m)
    first,second=cand[:2]
    gap=second["distance_m"]-first["distance_m"]
    return {"waypoint":waypoint,"tz_m":tz_m,
        "winner":{k:v for k,v in first.items() if k!="normal"},
        "runner_up":{k:v for k,v in second.items() if k!="normal"},
        "runner_up_minus_winner_distance_m":gap,
        "runner_up_minus_winner_distance_sq_m2":second["distance_sq_m2"]-first["distance_sq_m2"],
        "competing_normals_angle_deg":angle_deg(first["normal"],second["normal"]),
        "competing_tool_angles_difference_deg":second["angle_deg"]-first["angle_deg"],
        "near_tie_diagnostic":gap<=near_tie_m,
        "near_tie_cutoff_m":near_tie_m,
        "cutoff_semantics":"INVESTIGATORY_ABSOLUTE_DISTANCE_GAP_NOT_PHYSICAL_TOLERANCE"}


def pair_squared_difference(point,axis,segments,tz_m,first_id,second_id):
    c=all_candidates(point,axis,segments,tz_m)
    lookup={v["segment"]:v for v in c}
    if first_id not in lookup or second_id not in lookup:
        raise ValueError("SEGMENT_ID_MISSING")
    return lookup[first_id]["distance_sq_m2"]-lookup[second_id]["distance_sq_m2"]


def crossing_bisection(point,axis,segments,lo,hi,first_id,second_id):
    """Isolate a nearest-distance pair equality, not a physical failure."""
    flo=pair_squared_difference(point,axis,segments,lo,first_id,second_id)
    fhi=pair_squared_difference(point,axis,segments,hi,first_id,second_id)
    if flo*fhi>=0.:
        return {"status":"NO_PAIR_SIGN_CROSSING_IN_INTERVAL","values_sq_m2":[flo,fhi]}
    for _ in range(54):
        mid=(lo+hi)/2
        fm=pair_squared_difference(point,axis,segments,mid,first_id,second_id)
        if fm==0.:lo=hi=mid;break
        if (fm>0.)==(flo>0.):lo=mid;flo=fm
        else:hi=mid;fhi=fm
        if hi-lo<1e-13:break
    root=(hi+lo)/2
    h=min(1e-8,(hi-lo)*5+1e-9)
    before=critical_probe(point,axis,segments,root-h,0)
    after=critical_probe(point,axis,segments,root+h,0)
    return {"status":"PAIR_EQUAL_DISTANCE_CROSSING","pair":[first_id,second_id],
       "signed_tz_root_m":root,"bracket_signed_tz_m":[lo,hi],
       "left_winner":before["winner"]["segment"],
       "right_winner":after["winner"]["segment"],
       "left_angle_deg":before["winner"]["angle_deg"],
       "right_angle_deg":after["winner"]["angle_deg"],
       "one_sided_angle_jump_deg":after["winner"]["angle_deg"]-before["winner"]["angle_deg"],
       "pair_switch_not_necessarily_global_crossing":True}


def worst_angle(positions,axes,segments,tz_m):
    best=None
    for i,(p,a) in enumerate(zip(positions,axes)):
        c=all_candidates(p,a,segments,tz_m)[0]
        if best is None or c["angle_deg"]>best["angle_deg"]:
            best={"waypoint":i,"segment":c["segment"],"angle_deg":c["angle_deg"]}
    return best


def analyze(points,normals,positions,axes,scan):
    rays=scan["stages"]["all"]["rays"]
    if len(rays)!=12 or len(positions)!=181 or scan.get("measurement_type")!="NATIVE_FR5_C5A_FCL_SAMPLED":
        raise ValueError("C5B1_CONTRACT_MISMATCH")
    baseline=build_segments(points,normals)
    records=[]
    for sign,wp,pair in ((-1,11,(14,15)),(1,16,(11,12))):
        r=next(x for x in rays if x["axis"]=="tz" and x["sign"]==sign)
        bracket=r["predicates"]["normal_diagnostic"]["first_observed_entry"]
        if not bracket:raise ValueError("MISSING_ARCHIVED_NORMAL_BRACKET")
        # all signed offsets from historical positive ray magnitudes
        mlo,mhi=bracket["a"],bracket["b"]
        probes=[]
        for magnitude in (mlo,(mlo+mhi)/2,mhi):
            signed=sign*magnitude
            x=critical_probe(positions[wp],axes[wp],baseline,signed,wp)
            x["global"] = worst_angle(positions,axes,baseline,signed)
            probes.append(x)
        root=crossing_bisection(positions[wp],axes[wp],baseline,
            min(sign*mlo,sign*mhi), max(sign*mlo,sign*mhi),pair[0],pair[1])
        density=[]
        for stride,subdiv in ((1,1),(1,2),(1,4),(1,8),(2,1),(3,1),(4,1)):
            edges=build_segments(points,normals,stride=stride,subdivide=subdiv)
            samples=[]
            for magnitude in (mlo,mhi):
                c=critical_probe(positions[wp],axes[wp],edges,sign*magnitude,wp)
                samples.append({"magnitude_m":magnitude,"winner":c["winner"]["segment"],
                    "angle_deg":c["winner"]["angle_deg"],
                    "gap_m":c["runner_up_minus_winner_distance_m"]})
            density.append({"stride":stride,"linear_subdivision":subdiv,
                            "preserves_reference_geometry":stride==1,"probes":samples})
        records.append({"ray":"tz+" if sign>0 else "tz-","critical_waypoint":wp,
            "historical_normal_bracket_m":[mlo,mhi],"probes":probes,
            "pair_distance_root":root,"discretization_sensitivity":density})
    return {"schema":"FR5_C5B1_R3_R1_NEAREST_PROJECTION_V1",
        "status":"OFFLINE_REASSOCIATION_EXPLANATION_NOT_PHYSICAL_FAILURE",
        "scope":"frozen C1 FK vs displaced C1 open reference polyline; no native recomputation",
        "closest_point_tie_policy":"strict <, first segment wins", 
        "fixed_material_angle":"unchanged under pure tz translation",
        "historical_threshold_deg":10.,
        "near_tie_cutoff_m":NEAR_TIE_DISTANCE_M,
        "near_tie_cutoff_is_not_physical_tolerance":True,
        "records":records,
        "process_quality":"NOT_ASSESSED","continuous_collision":"NOT_ASSESSED",
        "surface_scan":"NOT_AVAILABLE"}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c1-fk",type=Path,required=True)
    ap.add_argument("--c1-target",type=Path,required=True)
    ap.add_argument("--c5b1-scan",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--allow-hash-mismatch",action="store_true",help="Exploratory only")
    args=ap.parse_args()
    if args.output.exists():ap.error("OUTPUT_EXISTS_NO_OVERWRITE")
    hashes={"C1_FK":sha256(args.c1_fk),"C1_TARGET":sha256(args.c1_target),
            "C5B1":sha256(args.c5b1_scan)}
    if not args.allow_hash_mismatch and (hashes!={"C1_FK":C1_FK_SHA256,
                  "C1_TARGET":C1_TARGET_SHA256,"C5B1":C5B1_SHA256}):
        ap.error("FROZEN_INPUT_SHA256_MISMATCH")
    points,normals,positions,axes=load_frozen(args.c1_fk,args.c1_target)
    scan=json.loads(args.c5b1_scan.read_text(encoding="utf-8-sig"))
    out=analyze(points,normals,positions,axes,scan)
    out["provenance"]={"input_sha256":hashes,
       "verified":"NO_EXPLORATORY_HASH_OVERRIDE" if args.allow_hash_mismatch else "FROZEN_INPUTS_VERIFIED"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print("FR5_R3R1_RESULT="+str(args.output))


if __name__=="__main__":main()
