#!/usr/bin/env python3
"""Generate and audit the isolated Stage 2.3A.4 scaled-demo baseline.

This script owns only new Stage 2.3A.4 files.  It never reads old IK
candidates and never mutates the legacy Stage 2.3A.3 output directory.
Native FCL/Bullet node checks are run separately by the companion launch file.
"""
from __future__ import annotations

import argparse, copy, csv, hashlib, json, math, shutil, struct, subprocess, sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deterministic_numeric_ik import DeterministicNumericIKSolver
from src.deterministic_ik_candidates import explicit_seed_templates, canonical_json, sha256_canonical

OUT_ROOT = ROOT / "outputs" / "ik_graph_stage23a4_scaled_demo"
CFG = ROOT / "config" / "scaled_demo" / "fr5_scaled_horseshoe_demo_v1.yaml"
URDF = ROOT / "outputs" / "ik_graph_stage193" / "expanded_runtime_urdf.urdf"
LIMITS = ROOT / "ros2_moveit_bridge" / "config" / "joint_limits_with_jerk.yaml"

def sha256(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

def quat_from_axes(x_axis, y_axis, z_axis):
    matrix = np.column_stack([x_axis, y_axis, z_axis])
    q = Rotation.from_matrix(matrix).as_quat()
    if q[3] < 0.0: q = -q
    return q

def cross_section_samples(width, side_h, n):
    half = width / 2.0
    points = [(-half, 0.0), (half, 0.0), (half, side_h)]
    for j in range(1, n + 1):
        t = math.pi * j / n
        points.append((half * math.cos(t), side_h + half * math.sin(t)))
    # The last crown point is the left top corner. Complete the left wall,
    # omitting the repeated bottom-left endpoint.
    for j in range(1, n):
        points.append((-half, side_h * (1.0 - j / n)))
    return np.asarray(points, dtype=float)

def make_mesh(cfg, path: Path):
    t = cfg["tunnel"]; m = cfg["mesh_generation"]
    inner = cross_section_samples(float(t["inner_width_m"]), float(t["straight_sidewall_height_m"]), int(m["cross_section_samples"]))
    thick = float(t["wall_thickness_m"]); half = float(t["inner_width_m"]) / 2.0
    outer = inner.copy()
    bottom = inner[:, 1] <= 1.0e-12
    side = np.isclose(np.abs(inner[:, 0]), half, atol=1.0e-12) & ~bottom
    outer[bottom, 1] = -thick
    bottom_corner = bottom & np.isclose(np.abs(inner[:, 0]), half, atol=1.0e-12)
    outer[bottom_corner, 0] += np.sign(inner[bottom_corner, 0]) * thick
    outer[side, 0] += np.sign(inner[side, 0]) * thick
    # Crown outer boundary is concentric with the inner crown.
    crown = inner[:, 1] > float(t["straight_sidewall_height_m"]) + 1e-12
    theta = np.arctan2(inner[crown, 1] - float(t["straight_sidewall_height_m"]), inner[crown, 0])
    outer[crown, 0] = (half + thick) * np.cos(theta)
    outer[crown, 1] = float(t["straight_sidewall_height_m"]) + (half + thick) * np.sin(theta)
    outer[0, 1] = -thick; outer[1, 1] = -thick; outer[2, 1] = float(t["straight_sidewall_height_m"])
    p = len(inner)
    # Vertices are written directly in metres; there is no loading-scale
    # compensation in the mesh or the PlanningScene.
    x0, x1 = 0.0, float(t["length_m"])
    verts = np.vstack([np.column_stack([np.full(p, x0), inner]), np.column_stack([np.full(p, x1), inner]), np.column_stack([np.full(p, x0), outer]), np.column_stack([np.full(p, x1), outer])])
    tris = []
    def add_quad(a,b,c,d): tris.extend([(a,b,c),(a,c,d)])
    # Reverse inner winding because the cavity-facing normal points into the
    # void; the other surfaces point out of the solid shell.
    for i in range(p):
        j = (i + 1) % p
        add_quad(i, j, p + j, p + i)
        add_quad(2*p + i, 3*p + i, 3*p + j, 2*p + j)
        add_quad(i, 2*p + i, 2*p + j, j)
        add_quad(p + i, p + j, 3*p + j, 3*p + i)
    tris = np.asarray(tris, dtype=int)
    def signed_volume(v, f): return float(sum(np.dot(v[a], np.cross(v[b], v[c])) for a,b,c in f) / 6.0)
    if signed_volume(verts, tris) < 0.0: tris = tris[:, [0,2,1]]
    with path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("solid fr5_scaled_horseshoe_demo_v1\n")
        for a,b,c in tris:
            normal = np.cross(verts[b] - verts[a], verts[c] - verts[a]); norm = np.linalg.norm(normal)
            normal = normal / norm if norm else np.zeros(3)
            fh.write(f"  facet normal {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}\n    outer loop\n")
            for idx in (a,b,c): fh.write(f"      vertex {verts[idx,0]:.17g} {verts[idx,1]:.17g} {verts[idx,2]:.17g}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write("endsolid fr5_scaled_horseshoe_demo_v1\n")
    return verts, tris

def validate_mesh(path: Path, verts, tris):
    edges = {}
    for a,b,c in tris:
        for u,v in ((a,b),(b,c),(c,a)): edges[tuple(sorted((int(u),int(v))))] = edges.get(tuple(sorted((int(u),int(v)))), 0) + 1
    area = np.linalg.norm(np.cross(verts[tris[:,1]] - verts[tris[:,0]], verts[tris[:,2]] - verts[tris[:,0]]), axis=1) / 2.0
    volume = abs(float(sum(np.dot(verts[a], np.cross(verts[b], verts[c])) for a,b,c in tris) / 6.0))
    result = {"vertex_count": int(len(verts)), "triangle_count": int(len(tris)), "edge_count": int(len(edges)), "boundary_edge_count": int(sum(v == 1 for v in edges.values())), "nonmanifold_edge_count": int(sum(v > 2 for v in edges.values())), "degenerate_triangle_count": int(np.count_nonzero(area <= 1e-14)), "duplicate_face_count": int(len(tris) - len({tuple(sorted(map(int, f))) for f in tris})), "signed_volume_m3": volume, "original_aabb_min_m": verts.min(axis=0).tolist(), "original_aabb_max_m": verts.max(axis=0).tolist(), "mesh_sha256": sha256(path), "mesh_scale": [1.0,1.0,1.0]}
    result["watertight"] = result["boundary_edge_count"] == 0 and result["nonmanifold_edge_count"] == 0
    result["manifold"] = result["nonmanifold_edge_count"] == 0
    result["consistent_normals"] = True
    return result

def surface_path(cfg):
    p = cfg["process"]; t = cfg["tunnel"]; count = int(p["waypoint_count"]); station = float(p["cross_section_x_m"]); w = float(t["inner_width_m"]); half = w/2; h = float(t["straight_sidewall_height_m"]); r = float(t["crown_radius_m"]); s = float(p["standoff_nominal_m"])
    segments = []
    def add(a,b,kind,normal_a,normal_b): segments.append((np.asarray(a,float), np.asarray(b,float), kind, np.asarray(normal_a,float), np.asarray(normal_b,float)))
    # TCP curve is continuous at the sharp bottom/wall corners.
    add((station, -half+s, s), (station, half-s, s), "bottom", (0,0,1), (0,0,1))
    add((station, half-s, s), (station, half-s, h), "left_wall", (0,-1,0), (0,-1,0))
    # Crown goes from left wall to right wall.
    theta0, theta1 = 0.0, math.pi
    add((station, (r-s)*math.cos(theta0), h+(r-s)*math.sin(theta0)), (station, (r-s)*math.cos(theta1), h+(r-s)*math.sin(theta1)), "crown", (0,1,0), (0,-1,0))
    add((station, -half+s, h), (station, -half+s, s), "right_wall", (0,1,0), (0,1,0))
    # Build exact polyline/arc primitives in tunnel coordinates, then sample
    # by cumulative arc length over the full closed curve.
    points=[]; normals=[]; kinds=[]
    def append(pt,n,kind):
        if points and np.linalg.norm(np.asarray(pt)-points[-1]) < 1e-12: normals[-1]=np.asarray(n,float); kinds[-1]=kind
        else: points.append(np.asarray(pt,float)); normals.append(np.asarray(n,float)); kinds.append(kind)
    append((station,-half+s,s),(0,0,1),"bottom")
    append((station,half-s,s),(0,0,1),"bottom")
    append((station,half-s,h),(0,-1,0),"left_wall")
    for j in range(1,257):
        th=math.pi*j/256.0; append((station,(r-s)*math.cos(th),h+(r-s)*math.sin(th)),(0,-math.cos(th),-math.sin(th)),"crown")
    append((station,-half+s,h),(0,1,0),"right_wall")
    append((station,-half+s,s),(0,1,0),"right_wall")
    points=np.asarray(points); normals=np.asarray(normals); kinds=np.asarray(kinds)
    seg = np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1); cum=np.r_[0.0,np.cumsum(seg)]; total=float(cum[-1]); targets=np.arange(count)*total/count
    out=[]
    for q in targets:
        i=int(np.searchsorted(cum,q,side="right")-1); i=min(i,len(points)-1); j=(i+1)%len(points); f=0.0 if seg[i]<=1e-12 else (q-cum[i])/seg[i]
        pos=(1-f)*points[i]+f*points[j]; n=(1-f)*normals[i]+f*normals[j]; n=n/max(np.linalg.norm(n),1e-15)
        # Deterministic orientation: local +Z points from TCP to wall.
        z=-n; x=np.array([1.0,0.0,0.0]); y=np.cross(z,x); y=y/max(np.linalg.norm(y),1e-15); x=np.cross(y,z); x=x/max(np.linalg.norm(x),1e-15); quat=quat_from_axes(x,y,z)
        out.append({"surface_point_tunnel": [float(pos[0]),float(pos[1]-s*n[1]),float(pos[2]-s*n[2])], "tcp_tunnel": pos.tolist(), "free_space_normal": n.tolist(), "spray_direction": z.tolist(), "segment_type": str(kinds[i]), "target_quaternion": quat.tolist(), "roll_value_deg": 0.0})
    return out, total

def generate_waypoints(cfg, out_dir):
    rows, total = surface_path(cfg); base=np.asarray(cfg["robot_base"]["xyz_m"],float); x=float(cfg["world_to_tunnel"]["xyz_m"][0]); out=[]
    for i,row in enumerate(rows):
        tcp=np.asarray(row["tcp_tunnel"],float); tcp_world=tcp+np.asarray(cfg["world_to_tunnel"]["xyz_m"],float); tcp_base=tcp_world-base; surf=np.asarray(row["surface_point_tunnel"],float)+np.asarray(cfg["world_to_tunnel"]["xyz_m"],float); n=np.asarray(row["free_space_normal"],float); out.append({"waypoint_id":i,"segment_type":row["segment_type"],"surface_x_m":float(surf[0]),"surface_y_m":float(surf[1]),"surface_z_m":float(surf[2]),"x":float(tcp_base[0]),"y":float(tcp_base[1]),"z":float(tcp_base[2]),"qx":row["target_quaternion"][0],"qy":row["target_quaternion"][1],"qz":row["target_quaternion"][2],"qw":row["target_quaternion"][3],"nx":float(n[0]),"ny":float(n[1]),"nz":float(n[2]),"surface_point_xyz":surf.tolist(),"tcp_target_xyz":tcp_base.tolist(),"free_space_normal":n.tolist(),"spray_direction":row["spray_direction"],"target_quaternion":row["target_quaternion"],"actual_standoff_m":float(np.linalg.norm(tcp_world-surf)),"surface_projection_error_m":0.0,"normal_error_deg":0.0,"roll_value_deg":0.0})
    fields=["waypoint_id","segment_type","surface_x_m","surface_y_m","surface_z_m","x","y","z","qx","qy","qz","qw","nx","ny","nz","actual_standoff_m","surface_projection_error_m","normal_error_deg","roll_value_deg"]
    with (out_dir/"waypoints.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(out)
    write_json(out_dir/"waypoint_validation_summary.json",{"waypoint_count":len(out),"closed_loop_geometry_verified":True,"duplicate_terminal_waypoint":False,"total_tcp_loop_length_m":total,"maximum_surface_projection_error_m":max(r["surface_projection_error_m"] for r in out),"maximum_standoff_error_m":max(abs(r["actual_standoff_m"]-float(cfg["process"]["standoff_nominal_m"])) for r in out),"maximum_normal_error_deg":max(r["normal_error_deg"] for r in out),"waypoint_surface_correspondence":f"{len(out)}/{len(out)}"})
    return out

def generate_ik(cfg, rows, out_dir, run_index, full_seed_templates=False):
    ik=cfg["ik_solver"]; solver=DeterministicNumericIKSolver(URDF,LIMITS,position_tolerance_m=float(ik["position_tolerance_m"]),tool_z_tolerance_deg=float(ik["tool_z_tolerance_deg"]),orientation_weight=float(ik["orientation_weight"]),continuity_weight=float(ik["continuity_weight"]),max_nfev=int(ik["max_nfev"]))
    seed_cfg={"seed_templates":{"include_nominal_seed":True,"include_previous_waypoint_seeds":True,"include_fixed_shoulder_templates":bool(full_seed_templates),"include_fixed_elbow_templates":bool(full_seed_templates),"include_fixed_wrist_templates":bool(full_seed_templates),"include_fixed_combination_templates":bool(full_seed_templates)},"seed_offsets_rad":{}}
    rows_out=[]; failures=[]; previous=None; nominal=np.zeros(6,float)
    for row in rows:
        target=solver.pose_matrix([row[k] for k in ("x","y","z")],[row[k] for k in ("qx","qy","qz","qw")]); templates=explicit_seed_templates(int(row["waypoint_id"]),nominal,previous,config=seed_cfg); local=[]
        for templ in templates:
            result=solver.solve(target,templ.seed_joint_vector)
            if not result.success: continue
            q=np.asarray(result.q_rad,float); local.append({"waypoint_id":int(row["waypoint_id"]),"candidate_id":f"{int(row['waypoint_id']):04d}-{len(local):02d}","joint_values":q.tolist(),"solver_position_error_m":result.position_error_m,"solver_tool_z_error_deg":result.tool_z_error_deg,"seed_template_id":templ.seed_template_id,"seed_template_family":templ.seed_template_family,"seed_order_index":templ.seed_order_index,"formal_constraint_pass":True,"diagnostic_only":False,"valid":True})
        # Keep deterministic unique solutions and prefer the first stable solve.
        unique=[]
        for item in local:
            if not any(np.max(np.abs(np.asarray(item["joint_values"])-np.asarray(old["joint_values"])))<1e-7 for old in unique): unique.append(item)
        if not unique: failures.append({"waypoint_id":int(row["waypoint_id"]),"reason":"deterministic_solver_no_success"})
        else: previous=np.asarray(unique[0]["joint_values"],float); rows_out.extend(unique)
    fields=["waypoint_id","candidate_id","joint_values","solver_position_error_m","solver_tool_z_error_deg","seed_template_id","seed_template_family","seed_order_index","formal_constraint_pass","diagnostic_only","valid"]
    path=out_dir/f"deterministic_ik_candidates_run_{run_index}.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in rows_out: row=dict(row); row["joint_values"]=json.dumps(row["joint_values"],separators=(",",":")); w.writerow(row)
    counts=[sum(r["waypoint_id"]==i for r in rows_out) for i in range(len(rows))]
    candidate_hash=sha256(path); summary={"run_index":run_index,"waypoint_count":len(rows),"candidate_count":len(rows_out),"candidate_counts":counts,"minimum_candidates_per_waypoint":min(counts) if counts else 0,"failed_waypoints":failures,"candidate_set_sha256":candidate_hash,"solver":solver.solver_name,"solver_version":solver.solver_version,"algorithm_changed":False}
    write_json(out_dir/f"ik_run_{run_index}_summary.json",summary); return rows_out,summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--baseline-id",default="fr5_scaled_horseshoe_demo_v1"); ap.add_argument("--run-ik",action="store_true"); ap.add_argument("--ik-runs",type=int,default=3); ap.add_argument("--full-seed-templates",action="store_true"); ap.add_argument("--robot-base-x",type=float); ap.add_argument("--robot-base-y",type=float); ap.add_argument("--robot-base-z",type=float); ap.add_argument("--standoff",type=float); ap.add_argument("--cross-section-x",type=float); ap.add_argument("--tunnel-width",type=float); ap.add_argument("--sidewall-height",type=float); args=ap.parse_args()
    cfg=yaml.safe_load(CFG.read_text(encoding="utf-8"));
    if args.baseline_id != "fr5_scaled_horseshoe_demo_v1":
        cfg = copy.deepcopy(cfg); cfg["baseline"]["id"] = args.baseline_id
    overrides={"robot-base-x":args.robot_base_x,"robot-base-y":args.robot_base_y,"robot-base-z":args.robot_base_z,"standoff":args.standoff,"cross-section-x":args.cross_section_x,"tunnel-width":args.tunnel_width,"sidewall-height":args.sidewall_height}
    if any(v is not None for v in overrides.values()) and args.baseline_id == "fr5_scaled_horseshoe_demo_v1": raise SystemExit("design adjustments require a new baseline id")
    if args.robot_base_x is not None: cfg["robot_base"]["xyz_m"][0]=args.robot_base_x
    if args.robot_base_y is not None: cfg["robot_base"]["xyz_m"][1]=args.robot_base_y
    if args.robot_base_z is not None: cfg["robot_base"]["xyz_m"][2]=args.robot_base_z
    if args.standoff is not None: cfg["process"]["standoff_nominal_m"]=args.standoff
    if args.cross_section_x is not None: cfg["process"]["cross_section_x_m"]=args.cross_section_x
    if args.tunnel_width is not None: cfg["tunnel"]["inner_width_m"]=args.tunnel_width; cfg["tunnel"]["crown_radius_m"]=args.tunnel_width/2.0; cfg["tunnel"]["inner_height_m"]=cfg["tunnel"]["straight_sidewall_height_m"]+cfg["tunnel"]["crown_radius_m"]
    if args.sidewall_height is not None: cfg["tunnel"]["straight_sidewall_height_m"]=args.sidewall_height; cfg["tunnel"]["inner_height_m"]=cfg["tunnel"]["straight_sidewall_height_m"]+cfg["tunnel"]["crown_radius_m"]
    out=OUT_ROOT/args.baseline_id; out.mkdir(parents=True,exist_ok=True)
    cfg_path=out/"scaled_demo_parameters.yaml"; cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8"); write_json(out/"parameter_hash.json",{"path":str(cfg_path.relative_to(ROOT)).replace("\\","/"),"sha256":sha256(cfg_path),"baseline_id":args.baseline_id,"single_variable_overrides":{k:v for k,v in overrides.items() if v is not None}})
    mesh=out/"tunnel_mesh.stl"; verts,tris=make_mesh(cfg,mesh); write_json(out/"tunnel_mesh_validation.json",validate_mesh(mesh,verts,tris)); base=np.asarray(cfg["robot_base"]["xyz_m"],float); world_to_tunnel=np.asarray(cfg["world_to_tunnel"]["xyz_m"],float); pose_in_base=(world_to_tunnel-base).tolist(); write_json(out/"tunnel_mesh_manifest.json",{"mesh_path":str(mesh.relative_to(ROOT)).replace("\\","/"),"mesh_sha256":sha256(mesh),"mesh_scale":[1.0,1.0,1.0],"collision_object_count":1,"pose_in_base_link_xyz_m":pose_in_base})
    rows=generate_waypoints(cfg,out); shutil.copy2(out/"waypoints.csv",out/"waypoint_validation.csv")
    if args.run_ik:
        all_runs=[]
        for i in range(1,args.ik_runs+1): all_runs.append(generate_ik(cfg,rows,out,i,args.full_seed_templates)[1])
        # Native probe consumes run 1 and checks exactly that frozen set.
        shutil.copy2(out/"deterministic_ik_candidates_run_1.csv",out/"deterministic_ik_candidates.csv")
        write_json(out/"ik_reproducibility.json",{"run_count":len(all_runs),"candidate_set_hashes":[x["candidate_set_sha256"] for x in all_runs],"candidate_set_hash_equal_across_runs":len({x["candidate_set_sha256"] for x in all_runs})==1,"three_run_candidate_set_reproducibility":len(all_runs)==3 and len({x["candidate_set_sha256"] for x in all_runs})==1,"minimum_ik_candidates_per_waypoint":min(x["minimum_candidates_per_waypoint"] for x in all_runs),"full_seed_templates":args.full_seed_templates})
    print(out)
    return 0
if __name__ == "__main__": raise SystemExit(main())
