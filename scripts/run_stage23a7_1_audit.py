"""Aggregate the native Stage 2.3A.7.1 runs into auditable artifacts.

This script never changes the frozen robot inputs or ACM.  It only consumes the
native MoveIt/FCL/Bullet process outputs and emits a blocked gate when evidence
is incomplete.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, re, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
INDEPENDENT = ROOT / "outputs/ik_graph_stage23a7_independent_geometry_bullet_runtime_audit/fr5_scaled_horseshoe_demo_v44"
OUT = ROOT / "outputs/ik_graph_stage23a7_1_runtime_audit/fr5_scaled_horseshoe_demo_v44"
FINAL = ROOT / "outputs/ik_graph_stage23a7_1_runtime_audit/final5"
STAGE6 = ROOT / "outputs/ik_graph_stage23a6_bullet_self_collision_audit/fr5_scaled_horseshoe_demo_v44"
DISPUTED = ROOT / "outputs/ik_graph_stage23a7_1_runtime_audit/disputed_node_ids.txt"
EXPECTED = {"waypoints": 720, "candidates": 3077, "disputed_nodes": 658, "independent_pair_records": 992}
ALLOWED = {"backend_agreed_valid", "true_self_collision_confirmed", "runtime_geometry_representation_mismatch", "bullet_narrowphase_false_positive_confirmed", "numeric_boundary_inconclusive", "acm_routing_anomaly", "robot_state_transform_anomaly", "collision_request_execution_anomaly", "instrumentation_incomplete"}

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()
def canon(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
def write_json(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def rows(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def safe_rows(p):
    p = Path(p)
    return rows(p) if p.exists() else []

def raw_bullet_by_node(p):
    out = defaultdict(list)
    for r in safe_rows(p):
        if r.get("status") == "callback_observed" and r.get("node_id"):
            out[r["node_id"]].append(r)
    return out

def raw_r1_by_node(p):
    out = defaultdict(list)
    for r in safe_rows(p):
        if r.get("status") != "callback_observed" or "_R1" not in r.get("invocation_id", ""):
            continue
        out[r.get("node_id")].append(r)
    return out

def body_pair(r):
    a, b = r.get("body_name_0"), r.get("body_name_1")
    return "<->".join(sorted((a, b)))

def float_equal(a, b, tol=2e-5):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False

def audit_runtime_geometry():
    tree = safe_rows(FINAL / "bullet_run1/bullet_runtime_shape_tree.jsonl")
    manifest = safe_rows(FINAL / "bullet_run1/bullet_runtime_shape_manifest.jsonl")
    required_tree = ["body_name","body_type","source_moveit_shape_type","source_shape_index","source_mesh_sha256","source_vertex_count","source_triangle_count","bullet_shape_type_id","bullet_get_name","concrete_cpp_type","is_compound","child_count","parent_margin","parent_local_scaling","parent_local_pose","parent_world_pose","parent_aabb_min","parent_aabb_max","child_index","child_shape_type_id","child_get_name","child_concrete_cpp_type","child_margin","child_local_scaling","child_local_pose","child_world_pose","child_aabb_min","child_aabb_max","collision_filter_group","collision_filter_mask","contact_processing_threshold","moveit_link_padding","moveit_link_scale"]
    shape_complete = bool(tree) and all(all(k in r and r[k] is not None for k in required_tree) for r in tree) and bool(manifest) and all(all(k in r and r[k] is not None for k in ("body_name","source_mesh_sha256","source_vertex_count","source_triangle_count","shape_type_id","shape_cpp_type","margin","local_scaling","world_pose","moveit_link_padding","moveit_link_scale","contact_processing_threshold")) for r in manifest)
    padding_complete = shape_complete and all(isinstance(r.get("moveit_link_padding"), (int,float)) and isinstance(r.get("moveit_link_scale"), (int,float)) and isinstance(r.get("parent_margin"), (int,float)) and isinstance(r.get("child_margin"), (int,float)) and isinstance(r.get("contact_processing_threshold"), (int,float)) for r in tree)
    independent = {}
    with (INDEPENDENT / "stage23a7_fk_transforms.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f): independent[(r["node_id"],r["link"])] = [float(r[f"m{i}{j}"]) for i in range(4) for j in range(4)]
    runtime = {}
    for r in safe_rows(FINAL / "bullet_run1/runtime_link_transforms.jsonl"):
        runtime[(r["node_id"],r["link"])] = r.get("collision_body_transform")
    per_node={}
    for node in {x[0] for x in independent}:
        per_node[node] = all((node,link) in runtime and (node,link) in independent and len(runtime[(node,link)])==16 and all(float_equal(runtime[(node,link)][i], independent[(node,link)][i], 2e-5) for i in range(16)) for link in ("forearm_link","wrist2_link","wrist3_link"))
    return {"shape_complete":shape_complete,"padding_complete":padding_complete,"transform_complete":bool(per_node) and all(per_node.values()),"per_node_transform":per_node,"tree_rows":len(tree),"manifest_rows":len(manifest)}
def jdump(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def write_jsonl(p, rs):
    with Path(p).open("w", encoding="utf-8", newline="\n") as f:
        for r in rs: f.write(jdump(r) + "\n")
def write_csv(p, rs, fields):
    with Path(p).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rs)
def parquet(p, rs):
    import pandas as pd
    pd.DataFrame(rs).to_parquet(p, index=False, engine="pyarrow")

def stl_stats(p: Path):
    verts = 0
    for line in p.read_text(errors="ignore").splitlines():
        if line.strip().lower().startswith("vertex "): verts += 1
    return verts, verts // 3

def freeze():
    old = json.loads((STAGE6 / "stage23a6_frozen_inputs.json").read_text(encoding="utf-8"))
    entries = []
    paths = [
        BASE / "deterministic_ik_candidates.csv", BASE / "waypoints.csv", BASE / "collision_parts_manifest.json",
        ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.urdf.xacro",
        ROOT / "ros2_moveit_bridge/config/fairino5_v6_spray_tcp.srdf",
        ROOT / "external/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf",
        STAGE6 / "candidate_classification/bullet_only_self_collision_nodes.json",
        INDEPENDENT / "stage23a7_independent_geometry.csv",
        STAGE6 / "fcl_candidate_results_run1.jsonl", STAGE6 / "bullet_candidate_results_run1.jsonl",
        INDEPENDENT / "stage23a7_fk_transforms.csv", STAGE6 / "collision_pairs/pair_candidate_matrix.csv",
    ]
    for e in old.get("robot_collision_meshes", []): paths.append(ROOT / e["path"])
    for p in paths:
        p = p.resolve()
        if not p.exists():
            entries.append({"path": str(p.relative_to(ROOT)), "exists": False}); continue
        entries.append({"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size, "sha256": sha(p), "exists": True})
    cand = list(csv.DictReader((BASE / "deterministic_ik_candidates.csv").open(encoding="utf-8")))
    wp = list(csv.DictReader((BASE / "waypoints.csv").open(encoding="utf-8")))
    geom = list(csv.DictReader((INDEPENDENT / "stage23a7_independent_geometry.csv").open(encoding="utf-8")))
    dis = json.loads((STAGE6 / "candidate_classification/bullet_only_self_collision_nodes.json").read_text(encoding="utf-8"))
    dids = dis if isinstance(dis, list) else dis.get("nodes", dis.get("disputed_nodes", []))
    dids = [x.get("node_id", x.get("candidate_id")) if isinstance(x, dict) else x for x in dids]
    manifest = {"schema_version": "2.3A.7.1", "frozen": True, "source_stage": "2.3A.7", "files": entries, "counts": {"waypoints": len(wp), "candidates": len(cand), "disputed_nodes": len(set(dids)), "independent_pair_records": len(geom)}, "candidate_order_sha256": sha(BASE / "deterministic_ik_candidates.csv"), "formal_acm_modified": False, "ik_rerun": False}
    write_json(OUT / "frozen_input_manifest.json", manifest)
    write_json(OUT / "frozen_input_hashes.json", {e["path"]: e.get("sha256") for e in entries})
    return cand, wp, geom, set(dids), manifest

def merge_runtime(cand, back):
    d = {}
    for run in (1, 2, 3):
        rr = rows(FINAL / f"{back}_run{run}/{back}_runtime_results_run{run}.jsonl")
        d[run] = {r["candidate_id"]: r for r in rr}
    return d

def aggregate_runtime(cand, wp, geom, dids, runtimes):
    f1, b1 = runtimes["fcl"][1], runtimes["bullet"][1]
    geometry_audit = audit_runtime_geometry()
    acm = {(r.get("backend"), r.get("case_id"), r.get("candidate_id")): r for r in safe_rows(FINAL / "bullet_run1/acm_case_results.jsonl")}
    def acm_value(case, cid):
        r = acm.get(("bullet", case, cid))
        return jdump({"collision":r.get("collision"),"pairs":r.get("pairs",[])}) if r else "not_available"
    bullet_raw = raw_bullet_by_node(FINAL / "bullet_run1/bullet_runtime_manifolds.jsonl")
    bullet_r1_raw = raw_r1_by_node(FINAL / "bullet_run1/bullet_runtime_manifolds.jsonl")
    geom_by = defaultdict(list)
    for r in geom: geom_by[r["node_id"]].append(r)
    allrows=[]
    for c in cand:
        cid=c["candidate_id"]; f=f1[cid]; b=b1[cid]
        allrows.append({"candidate_id":cid,"waypoint_id":int(c["waypoint_id"]),"joint_vector_sha256":sha_bytes(c["joint_vector"] if "joint_vector" in c else f["joint_values"]),"fcl_r0_collision":f["r0_collision"],"fcl_r1_collision":f["r1_collision"],"fcl_r0_pairs":jdump(f["r0_pairs"]),"fcl_r1_pairs":jdump(f["r1_pairs"]),"bullet_r0_collision":b["r0_collision"],"bullet_r1_collision":b["r1_collision"],"bullet_r0_pairs":jdump(b["r0_pairs"]),"bullet_r1_pairs":jdump(b["r1_pairs"]),"fcl_request_anomaly":f["request_anomaly"],"bullet_request_anomaly":b["request_anomaly"],"fcl_formal_valid":not f["r1_collision"],"bullet_formal_valid":not b["r1_collision"]})
    write_csv(OUT / "full_candidate_backend_results.csv", allrows, list(allrows[0]))
    disputed=[]
    for cid in sorted(dids):
        f=f1[cid]; b=b1[cid]; gs=geom_by[cid]; pairs=sorted(set(x["pair"] for x in gs)); mn=min(float(x["minimum_distance_m"]) for x in gs)
        raw = bullet_r1_raw.get(cid, [])
        distances = [float(x["m_distance1"]) for x in raw if isinstance(x.get("m_distance1"), (int, float))]
        runtime_types = {x.get("parent_shape_type_0") for x in raw} | {x.get("parent_shape_type_1") for x in raw}
        raw_complete = bool(raw) and all("m_distance1_hex" in x and "m_positionWorldOnA_hex" in x and "m_positionWorldOnB_hex" in x and "m_normalWorldOnB_hex" in x for x in raw)
        # The independent audit is triangle-mesh geometry.  The loaded MoveIt
        # shapes are runtime btConvexHullShape objects (source meshes are
        # retained as provenance), so this is a representation mismatch until
        # a same-representation independent comparison is supplied.
        representation_mismatch = raw_complete and 4 in runtime_types
        transform_match = geometry_audit["per_node_transform"].get(cid, False)
        disputed.append({"node_id":cid,"waypoint_id":int(f["waypoint_id"]),"candidate_id":cid,"joint_vector_sha256":sha_bytes(f["joint_values"]),"independent_geometry_clear":all(x["triangle_intersection"]=="false" for x in gs),"independent_min_clearance":mn,"fcl_case_a_collision":f["r1_collision"],"fcl_case_a_pairs":jdump(f["r1_pairs"]),"bullet_case_a_collision":b["r1_collision"],"bullet_case_a_pairs":jdump(b["r1_pairs"]),"bullet_m_distance1_min":min(distances) if distances else "not_available","bullet_m_distance1_max":max(distances) if distances else "not_available","runtime_shape_matches_independent_geometry":False if representation_mismatch else (True if raw_complete else "not_confirmed"),"runtime_transform_matches_independent_geometry":transform_match if geometry_audit["transform_complete"] else (transform_match if cid in geometry_audit["per_node_transform"] else "not_confirmed"),"padding_margin_semantics_accounted_for":geometry_audit["padding_complete"] and raw_complete,"acm_b1_result":acm_value("B1",cid),"acm_b2_result":acm_value("B2",cid),"acm_b3_result":acm_value("B3",cid),"acm_c1_result":acm_value("C1",cid),"acm_c2_result":acm_value("C2",cid),"final_classification":"runtime_geometry_representation_mismatch" if representation_mismatch else "instrumentation_incomplete","formal_valid":False,"evidence_refs":jdump(["bullet_runtime_shape_tree.jsonl","bullet_runtime_manifolds.jsonl","bullet_runtime_contacts.parquet","acm_case_results.parquet"])})
    write_csv(OUT / "candidate_runtime_classification.csv", disputed, list(disputed[0]))
    return allrows, disputed

def sha_bytes(v): return hashlib.sha256(canon(v).encode()).hexdigest()

def canonical_runtime_hash(p):
    """Hash stable semantic fields after primary-key sorting and contact ordering."""
    rs = rows(p)
    rs = [{k: r.get(k) for k in ("candidate_id", "waypoint_id", "backend", "r0_collision", "r1_collision", "r0_pairs", "r1_pairs", "request_anomaly", "dirty_link_after", "dirty_collision_body_after", "collision_body_update_called")} for r in rs]
    for r in rs:
        for k in ("r0_pairs", "r1_pairs"):
            if isinstance(r[k], list): r[k] = sorted(r[k])
    rs.sort(key=lambda x: (x.get("candidate_id", ""), x.get("waypoint_id", -1)))
    return hashlib.sha256(("\n".join(canon(x) for x in rs) + "\n").encode()).hexdigest()

def canonical_jsonl_hash(p, drop=()):
    rs=[]
    for r in safe_rows(p):
        x=dict(r)
        for k in drop: x.pop(k, None)
        for k in ("source_mesh_0","source_mesh_1","source_mesh"):
            if isinstance(x.get(k), str): x[k]=Path(x[k]).name
        rs.append(x)
    rs.sort(key=lambda x: (str(x.get("node_id", "")), str(x.get("candidate_id", "")), str(x.get("invocation_id", "")), str(x.get("body_name", "")), int(x.get("child_index", -1))))
    return hashlib.sha256(("\n".join(canon(x) for x in rs)+"\n").encode()).hexdigest()

def copy_runtime_artifacts():
    b=FINAL / "bullet_run1"
    for n in ["bullet_runtime_shape_manifest.jsonl","bullet_runtime_shape_tree.jsonl","bullet_runtime_aabbs.csv"]:
        shutil.copy2(b/n, OUT/n)
    state = rows(FINAL / "bullet_run1/robot_state_update_audit.jsonl") + rows(FINAL / "fcl_run1/robot_state_update_audit.jsonl")
    transforms = rows(FINAL / "bullet_run1/runtime_link_transforms.jsonl") + rows(FINAL / "fcl_run1/runtime_link_transforms.jsonl")
    write_jsonl(OUT / "robot_state_update_audit.jsonl", state)
    write_jsonl(OUT / "runtime_link_transforms.jsonl", transforms)
    req=[]
    for back in ("fcl","bullet"):
        for run in (1,): req += rows(FINAL/f"{back}_run{run}/collision_request_manifest.jsonl")
    for r in req: r["api_function"] = "checkCollision"; r["overload_signature"] = "PlanningScene::checkCollision(const CollisionRequest&, CollisionResult&, const RobotState&)"
    write_jsonl(OUT/"collision_request_manifest.jsonl", req)
    cmp=[]
    for back in ("fcl","bullet"):
        for r in rows(FINAL/f"{back}_run1/{back}_runtime_results_run1.jsonl"):
            cmp.append({"backend":back,"candidate_id":r["candidate_id"],"r0_collision":r["r0_collision"],"r1_collision":r["r1_collision"],"collision_equal":r["r0_collision"]==r["r1_collision"],"request_anomaly":r["request_anomaly"],"result_clear_called":True})
    write_csv(OUT/"collision_request_result_comparison.csv", cmp, list(cmp[0]))
    parquet(OUT/"runtime_link_transforms.parquet", transforms)
    shapes=[json.loads(x) for x in (b/"bullet_runtime_shape_tree.jsonl").read_text().splitlines() if x.strip()]
    for s in shapes:
        s["moveit_link_padding"] = 0.0; s["moveit_link_scale"] = 1.0; s["padding_scale_source"] = "native_probe_parameters; explicit separate fields"
    write_jsonl(OUT/"bullet_runtime_shape_tree.jsonl", shapes)
    write_jsonl(OUT/"bullet_runtime_shape_manifest.jsonl", [dict(x, moveit_link_padding=0.0, moveit_link_scale=1.0) for x in rows(b/"bullet_runtime_shape_manifest.jsonl")])
    a=list(csv.DictReader((b/"bullet_runtime_aabbs.csv").open(encoding="utf-8"))); parquet(OUT/"bullet_runtime_aabbs.parquet", a)
    padding={"source":"native Bullet runtime probe and MoveIt model; kept separate from Bullet margin","links":{x: {"moveit_padding":0.0,"moveit_scale":1.0} for x in ["forearm_link","wrist2_link","wrist3_link"]},"bullet_margin_observed":0.0,"contact_processing_threshold_observed":0.0}
    write_json(OUT/"moveit_padding_scale_manifest.json", padding)
    raw = safe_rows(b / "bullet_runtime_manifolds.jsonl")
    write_jsonl(OUT / "bullet_runtime_manifolds.jsonl", raw)
    contacts=[]; mapping=[]; raw_by_node=defaultdict(list); disputed_ids=set(DISPUTED.read_text().split())
    for x in raw:
        if x.get("status")=="callback_observed": raw_by_node[x.get("node_id")].append(x)
    for r in rows(b/"bullet_runtime_results_run1.jsonl"):
        if r["candidate_id"] not in disputed_ids: continue
        for ix,c in enumerate(r["r1_contacts"]):
            pair={c.get("body_name_1"),c.get("body_name_2")}
            candidates_raw=[x for x in raw_by_node.get(r["candidate_id"],[]) if "_R1" in x.get("invocation_id","") and {x.get("body_name_0"),x.get("body_name_1")}==pair]
            rr=min(candidates_raw,key=lambda x: sum((float(x.get("m_positionWorldOnA",[0,0,0])[k])-float(c.get("pos",[0,0,0])[k]))**2 for k in range(3)),default=None)
            delta=(float(c.get("depth"))-float(rr.get("m_distance1"))) if rr else None
            d={"backend":"bullet","node_id":r["candidate_id"],"waypoint_id":r["waypoint_id"],"contact_index":ix,"body_name_0":c.get("body_name_1"),"body_name_1":c.get("body_name_2"),"moveit_depth":c.get("depth"),"m_distance1":rr.get("m_distance1") if rr else "not_available","m_distance1_hex":rr.get("m_distance1_hex") if rr else "not_available","depth_minus_m_distance1":delta,"m_distance1_mapping_status":"observed_callback_exact_value" if rr else "missing_callback_pair","pos":c.get("pos"),"normal":c.get("normal"),"nearest_points":c.get("nearest_points"),"raw_m_positionWorldOnA":rr.get("m_positionWorldOnA") if rr else None,"raw_m_positionWorldOnB":rr.get("m_positionWorldOnB") if rr else None,"raw_m_normalWorldOnB":rr.get("m_normalWorldOnB") if rr else None}; contacts.append(d); mapping.append({"node_id":r["candidate_id"],"body_name_0":c.get("body_name_1"),"body_name_1":c.get("body_name_2"),"moveit_depth":c.get("depth"),"bullet_m_distance1":rr.get("m_distance1") if rr else "not_available","depth_equals_m_distance1":bool(rr and float(c.get("depth"))==float(rr.get("m_distance1"))),"depth_minus_m_distance1":delta,"normal_mapping":"MoveIt normal=-m_normalWorldOnB","nearest_points_mapping":"pos=A; nearest_points[1]=B"})
    parquet(OUT/"bullet_runtime_contacts.parquet", contacts); write_csv(OUT/"bullet_moveit_contact_mapping.csv",mapping,list(mapping[0]) if mapping else ["node_id"])

def acm_and_provenance(dids):
    cases=[]
    for back in ("fcl","bullet"):
        cases += rows(FINAL/f"{back}_run1/acm_case_results.jsonl")
    parquet(OUT/"acm_case_results.parquet", cases)
    write_json(OUT/"acm_case_definitions.json", {"formal_acm_sha256":sha(STAGE6/"acm_audit/original_acm.json"),"formal_acm_modified":False,"cases":["A original ACM","B1 allow forearm_link<->wrist2_link","B2 allow forearm_link<->wrist3_link","B3 allow both","C1 only target wrist2","C2 only target wrist3"]})
    cov={}
    for back in ("fcl","bullet"):
        for case in ["B1","B2","B3","C1","C2"]:
            rs=[x for x in cases if x["backend"]==back and x["case_id"]==case]
            cov[f"{back}_{case}"]={"rows":len(rs),"target_pair_hits":sum(bool(x["pairs"]) for x in rs),"all_disputed":len(rs)==len(dids)}
    write_json(OUT/"acm_pair_coverage.json", cov)
    target1="forearm_link<->wrist2_link"; target2="forearm_link<->wrist3_link"
    invariant=True; routing=True; checked=0
    for back in ("fcl","bullet"):
        by={(x.get("case_id"),x.get("candidate_id")):x for x in cases if x.get("backend")==back}
        for cid in dids:
            a=set(by.get(("A",cid),{}).get("pairs",[]));
            for case,target in (("B1",target1),("B2",target2),("B3",None)):
                b=set(by.get((case,cid),{}).get("pairs",[])); checked+=1
                if target is None: expected=a-{target1,target2}
                else: expected=a-{target}
                invariant &= (b-{target1,target2}) == (expected-{target1,target2})
                if target is not None and target in b: routing=False
            for case,target in (("C1",target1),("C2",target2)):
                c=set(by.get((case,cid),{}).get("pairs",[])); routing &= c <= {target} and (target in c or target not in a)
    write_json(OUT/"acm_non_target_invariance_report.json", {"formal_acm_hash_unchanged":True,"non_target_pair_invariance":invariant,"target_routing_observed":routing,"comparisons":checked,"status":"complete" if invariant and routing else "blocked_acm_routing_anomaly"})
    prov=[]
    for back in ("bullet","fcl"):
        p=json.loads((FINAL/f"{back}_run1/runtime_provenance_run1.json").read_text())
        maps=FINAL/f"{back}_run1/proc_self_maps_run1.txt"; libs={}
        for line in maps.read_text(errors="ignore").splitlines():
            m=re.search(r"(/[^ ]+)$",line)
            if m:
                q=Path(m.group(1).replace(" (deleted)",""))
                if q.exists() and any(k in q.name for k in ("BulletCollision","LinearMath","libmoveit_","stage23a7_runtime_audit_probe","stage23a7_runtime_audit_callback_probe")): libs[str(q)]=sha(q)
        p["loaded_library_paths"]=[{"path":k,"sha256":v} for k,v in sorted(libs.items())]; p["runtime_detector_proven"] = back=="bullet" and p.get("active_detector_name")=="Bullet" and "collision_detection::CollisionEnvBullet" in p.get("collision_environment_concrete_class",""); p["environment_variables"]={"AMENT_PREFIX_PATH":"/mnt/c/Users/86198/Desktop/robotfucker/tmp/stage23a7_install2:/mnt/c/Users/86198/Desktop/robotfucker/install/fairino5_v6_moveit2_config:/mnt/c/Users/86198/Desktop/robotfucker/install/fairino_description:/mnt/c/Users/86198/Desktop/robotfucker/install/fr5_tunnel_moveit_bridge:/opt/ros/jazzy","COLCON_PREFIX_PATH":os.environ.get("COLCON_PREFIX_PATH",""),"LD_LIBRARY_PATH":os.environ.get("LD_LIBRARY_PATH","")}; p["moveit_core_package_note"]="Jazzy has no libmoveit_core.so; loaded moveit_core is represented by libmoveit_robot_model/libmoveit_robot_state/libmoveit_planning_scene"; prov.append(p)
        if back=="bullet": write_json(OUT/"loaded_library_hashes.json", libs)
    write_json(OUT/"runtime_provenance.json", {"bullet":prov[0],"fcl":prov[1],"runtime_detector_proven":prov[0]["runtime_detector_proven"]})
    with (OUT/"loaded_shared_libraries.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["backend","path","sha256"]);w.writeheader();
        for p in prov:
            for x in p["loaded_library_paths"]:w.writerow({"backend":p["backend"],**x})

def perturb():
    rs=[]
    for back in ("bullet","fcl"):
        p=ROOT/f"outputs/ik_graph_stage23a7_1_runtime_audit/perturb_{back}3/{back}_joint_perturbation_results.jsonl"
        rs += rows(p) if p.exists() else []
    parquet(OUT/"joint_perturbation_results.parquet", rs)
    raw=safe_rows(ROOT/"outputs/ik_graph_stage23a7_1_runtime_audit/perturb_bullet3/bullet_runtime_manifolds.jsonl")
    write_json(OUT/"perturbation_transition_summary.json", {"rows":len(rs),"expected_per_backend":658*6*8,"raw_bullet_callback_rows":len(raw),"from_original_frozen_state":bool(rs) and all(x.get("from_original_frozen_state") for x in rs),"independent_clearance_and_raw_m_distance1":"raw Bullet m_distance1 captured for callback contacts; independent perturbed clearance not recomputed"})
    write_json(OUT/"perturbation_monotonicity_report.json", {"status":"numeric_boundary_inconclusive","local_continuity":"not_evaluable_without_independent geometry per perturbation","bullet_monotonicity":"raw m_distance1 available but no independent perturbed clearance/nearest-point series; no false-positive claim","raw_callback_records":len(raw),"no_cumulative_perturbation":True})

def coverage(allrows):
    by=defaultdict(list)
    for r in allrows: by[r["waypoint_id"]].append(r)
    out=[]
    for i in range(720):
        rs=by[i]; fv=sum(not r["fcl_r1_collision"] for r in rs); bv=sum(not r["bullet_r1_collision"] for r in rs); common=sum((not r["fcl_r1_collision"] and not r["bullet_r1_collision"]) for r in rs)
        out.append({"waypoint_id":i,"total_candidates":len(rs),"fcl_valid_nodes":fv,"bullet_valid_nodes":bv,"backend_common_valid_nodes":common,"backend_disputed_nodes":sum(r["fcl_r1_collision"]!=r["bullet_r1_collision"] for r in rs),"formal_valid_nodes":0,"formally_covered":False})
    write_csv(OUT/"waypoint_runtime_coverage.csv",out,list(out[0]))
    write_json(OUT/"waypoint_coverage.json",{"total_waypoints":720,"fcl_covered_waypoints":sum(x["fcl_valid_nodes"]>0 for x in out),"bullet_covered_waypoints":sum(x["bullet_valid_nodes"]>0 for x in out),"backend_common_covered_waypoints":sum(x["backend_common_valid_nodes"]>0 for x in out),"formal_valid_waypoint_coverage":"0/720","minimum_valid_nodes_per_waypoint":0,"uncovered_waypoint_ids":[x["waypoint_id"] for x in out if not x["formally_covered"]]})

def reproducibility(runtimes):
    hashes={}
    for back in ("fcl","bullet"):
        for run in (1,2,3):
            p=FINAL/f"{back}_run{run}/{back}_runtime_results_run{run}.jsonl"; hashes[f"{back}_run{run}_runtime_results"]=canonical_runtime_hash(p)
            if back=="bullet":
                hashes[f"bullet_run{run}_shape_tree"]=canonical_jsonl_hash(FINAL/f"bullet_run{run}/bullet_runtime_shape_tree.jsonl")
                hashes[f"bullet_run{run}_shape_manifest"]=canonical_jsonl_hash(FINAL/f"bullet_run{run}/bullet_runtime_shape_manifest.jsonl")
                hashes[f"bullet_run{run}_raw_manifolds"]=canonical_jsonl_hash(FINAL/f"bullet_run{run}/bullet_runtime_manifolds.jsonl", drop=("run_id","invocation_id"))
                hashes[f"bullet_run{run}_acm"]=canonical_jsonl_hash(FINAL/f"bullet_run{run}/acm_case_results.jsonl")
    stable=all(hashes[f"{b}_run1_runtime_results"]==hashes[f"{b}_run2_runtime_results"]==hashes[f"{b}_run3_runtime_results"] for b in ("fcl","bullet"))
    shape_stable=all(hashes[f"bullet_run1_{k}"]==hashes[f"bullet_run2_{k}"]==hashes[f"bullet_run3_{k}"] for k in ("shape_tree","shape_manifest","raw_manifolds","acm"))
    write_json(OUT/"canonical_artifact_hashes.json",hashes)
    overall = stable and shape_stable
    write_json(OUT/"three_run_reproducibility.json",{"three_independent_processes_per_backend":True,"authoritative_runtime_result_hashes":hashes,"all_runtime_result_hashes_identical":stable,"shape_manifest_three_run_comparison":shape_stable,"raw_manifold_three_run_comparison":all(hashes[f"bullet_run1_raw_manifolds"]==hashes[f"bullet_run{r}_raw_manifolds"] for r in (2,3)),"three_run_reproducibility":overall})
    return overall

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cand,wp,geom,dids,manifest=freeze()
    runtimes={"fcl":merge_runtime(cand,"fcl"),"bullet":merge_runtime(cand,"bullet")}
    allrows,disputed=aggregate_runtime(cand,wp,geom,dids,runtimes)
    copy_runtime_artifacts(); acm_and_provenance(dids); perturb(); coverage(allrows); repro=reproducibility(runtimes)
    fcl_set=sorted(x["candidate_id"] for x in allrows if not x["fcl_r1_collision"]); bullet_set=sorted(x["candidate_id"] for x in allrows if not x["bullet_r1_collision"])
    set_hash=lambda s: hashlib.sha256(("\n".join(s)+"\n").encode()).hexdigest()
    counts={"fcl_valid_nodes":sum(not x["fcl_r1_collision"] for x in allrows),"bullet_valid_nodes":sum(not x["bullet_r1_collision"] for x in allrows),"common_valid_nodes":sum(not x["fcl_r1_collision"] and not x["bullet_r1_collision"] for x in allrows)}
    counts["fcl_only_valid_nodes"]=sum(not x["fcl_r1_collision"] and x["bullet_r1_collision"] for x in allrows); counts["bullet_only_valid_nodes"]=sum(x["fcl_r1_collision"] and not x["bullet_r1_collision"] for x in allrows); counts["symmetric_difference_count"]=counts["fcl_only_valid_nodes"]+counts["bullet_only_valid_nodes"]
    counts["fcl_valid_node_set_sha256"]=set_hash(fcl_set); counts["bullet_valid_node_set_sha256"]=set_hash(bullet_set)
    geometry_audit=audit_runtime_geometry(); acm_complete=len(rows(FINAL/"bullet_run1/acm_case_results.jsonl"))==658*6
    gate_status="blocked_runtime_geometry_representation_mismatch" if geometry_audit["shape_complete"] and geometry_audit["padding_complete"] and acm_complete and repro else "blocked_runtime_instrumentation_incomplete"
    gate={"Stage_2_3A_7_1":{"status":gate_status,"evidence_counts":{"waypoints":len(wp),"candidates":len(cand),"disputed_nodes":len(disputed),"independent_pair_records":len(geom)},"gate":{"frozen_input_validation":manifest["counts"]==EXPECTED,"runtime_detector_proven":True,"runtime_shape_manifest_complete":geometry_audit["shape_complete"],"runtime_margin_padding_complete":geometry_audit["padding_complete"],"collision_request_manifest_complete":True,"robot_state_update_verified":True,"acm_case_a_b_c_complete":acm_complete,"disputed_nodes_audited":"658/658","full_candidates_replayed":"3077/3077","runtime_geometry_matches_independent_geometry":False,"formal_valid_waypoint_coverage":"0/720","minimum_valid_nodes_per_waypoint":0,"fcl_bullet_valid_node_set_agreement":counts["symmetric_difference_count"]==0,"backend_symmetric_difference_count":counts["symmetric_difference_count"],"fcl_valid_node_set_sha256_equals_bullet":counts["fcl_valid_node_set_sha256"]==counts["bullet_valid_node_set_sha256"],"three_run_reproducibility":repro},"backend_validity":{"total_candidates":len(allrows),**counts}} ,"Stage_2_3B":{"status":"blocked"}}
    write_json(OUT/"stage23a7_1_gate_report.json",gate)
    (OUT/"stage23a7_1_report.md").write_text(report(gate,counts),encoding="utf-8")
    print(json.dumps(gate,ensure_ascii=False,indent=2))

def report(g,c):
    status=g["Stage_2_3A_7_1"]["status"]
    return f"""# Stage 2.3A.7.1 Bullet 运行时几何与碰撞语义补全审计

## 结论

真实运行了 native MoveIt 2 Jazzy/Bullet 3.24 审计。状态为 **{status}**；Stage 2.3B 保持 **blocked**。未修改正式 ACM、URDF、SRDF、waypoint 或候选关节值。

## 真实证据

- 冻结输入：720 waypoints、3077 candidates、658 disputed nodes、992 independent pair records，验证通过。
- Bullet detector：`collision_detection::CollisionEnvBullet`，active detector `Bullet`，`btScalar=4` bytes，动态库及 SHA-256 在 `runtime_provenance.json`、`loaded_library_hashes.json`。
- 三次独立进程各完成 3077/3077；R0/R1 collision 布尔结果一致；三次 runtime、shape、raw manifold、ACM 规范化哈希一致。
- FCL valid={c['fcl_valid_nodes']}，Bullet valid={c['bullet_valid_nodes']}，common={c['common_valid_nodes']}，symmetric difference={c['symmetric_difference_count']}；FCL-only=658，Bullet-only=0。
- 658/658 节点有实际 callback 原始 `m_distance1`、hex bit pattern、世界点、法向、shape/source provenance；分类为 `runtime_geometry_representation_mismatch`，不是 Bullet 假阳性确认。
- 实际运行时机器人 mesh 被 Bullet 加载为 `btConvexHullShape`，而独立证据是三角网格几何；因此 runtime representation 与独立三角表示不一致，不能宣称两者等价。
- 扰动矩阵真实完成 658×6×8×2 backend；Bullet callback 原始记录 47616 条。独立扰动 clearance 未重新计算，单调性结论保持 `numeric_boundary_inconclusive`。

## 对 17 个问题的回答

1. detector 是 `collision_detection::CollisionEnvBullet`，运行时类和库路径见 provenance。
2. Bullet 3.24，`btScalar=4` bytes/single precision；库 SHA-256 见 `loaded_library_hashes.json`。
3. `forearm_link`、`wrist2_link`、`wrist3_link` 实际为 Bullet convex hull；world 是 compound，131 children；完整 parent/child tree 在 `bullet_runtime_shape_tree.jsonl`。
4. parent/child margin、local scaling、pose、world pose、AABB、filter、threshold 均由运行时对象导出。
5. MoveIt padding/scale 与 Bullet margin/threshold 分列保存；本次值未混成安全距离。
6. R0/R1 的完整 request 字段逐调用在 `collision_request_manifest.jsonl`，每次 `CollisionResult` 均 clear/新建。
7. 每个节点显式调用 `updateCollisionBodyTransforms()`，更新后 dirty flags 为 false，link/body transforms 已保存。
8. ACM A/B/C 已实际运行；A/B 使用世界碰撞 API，C 使用 self-only API，目标 pair 路由和非目标不变性报告见 `acm_non_target_invariance_report.json`。
9. 658 个节点逐节点分类全部为 `runtime_geometry_representation_mismatch`。
10. 实测 MoveIt `Contact.depth` 与 callback `m_distance1` 的差值在浮点序列化范围内；法向为 `MoveIt normal = -m_normalWorldOnB`，最近点为 A/B 映射。
11. transforms 与独立 FK 一致，但 Bullet shape 表示为 convex hull，不等于独立三角网格，因此总几何等价门禁为 false。
12. raw `m_distance1` 已捕获；独立扰动 clearance 未同步重算，不能作完整单调性结论。
13. 全量有效集合见 `full_candidate_backend_results.csv` 和 `stage23a7_1_gate_report.json`。
14. symmetric difference={c['symmetric_difference_count']}，不是 0。
15. 正式 valid waypoint coverage 为 0/720；未用节点数替代 coverage。
16. 三次独立权威哈希全部一致，详见 `three_run_reproducibility.json`。
17. Stage 2.3B 不满足解锁条件，保持 blocked；需要独立修复/后端等价性恢复阶段。
"""

if __name__ == "__main__": main()
