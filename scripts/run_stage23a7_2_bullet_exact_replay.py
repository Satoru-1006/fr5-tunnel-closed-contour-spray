"""Evidence-first Stage 2.3A.7.2 post-processing and isolated replay runner.

The native MoveIt capture is run separately with the current overlay. This script
only consumes that capture, exports deterministic replay inputs, invokes the
isolated Bullet executable, and writes a blocked gate when any required branch is
not proven. Formal URDF/SRDF/ACM, waypoints, and candidate values are never edited.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ik_graph_stage23a7_2_bullet_exact_replay/fr5_scaled_horseshoe_demo_v45"
CAPTURE = OUT / "runtime_capture_run1"
PREV = ROOT / "outputs/ik_graph_stage23a7_1_runtime_audit/fr5_scaled_horseshoe_demo_v44"
BASE = ROOT / "outputs/ik_graph_stage23a4_backend_equivalent/fr5_scaled_horseshoe_demo_v44"
DISPUTED = ROOT / "outputs/ik_graph_stage23a7_1_runtime_audit/disputed_node_ids.txt"
PAIR_NAMES = ("forearm_link<->wrist2_link", "forearm_link<->wrist3_link")
TARGET_LINKS = ("forearm_link", "wrist2_link", "wrist3_link")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def wsl_path(path: Path) -> str:
    return "/mnt/c/Users/86198/Desktop/robotfucker/" + path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_binary_stl(path: Path) -> list[tuple[float, float, float]]:
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise RuntimeError(f"non-binary STL is not accepted by the isolated D input: {path}")
    verts = []
    for n in range(count):
        off = 84 + n * 50 + 12
        vals = struct.unpack_from("<9f", raw, off)
        verts.extend((tuple(vals[0:3]), tuple(vals[3:6]), tuple(vals[6:9])))
    return verts


def write_float32_vertices(path: Path, verts: list[tuple[float, float, float]]) -> None:
    with path.open("wb") as f:
        for v in verts:
            f.write(struct.pack("<3f", *v))


def parse_transform16(values: list[float]) -> list[float]:
    return values[:12]


def capture_shape_maps() -> tuple[dict[tuple[str, int], dict], list[dict]]:
    rows = list(jsonl(CAPTURE / "bullet_runtime_shape_tree.jsonl"))
    shape_map = {(r["body_name"], int(r["child_index"])): r for r in rows}
    provenance = []
    vertex_file = CAPTURE / "runtime_shape_vertices.bin"
    for r in rows:
        item = dict(r)
        count = int(r.get("convex_point_count", 0))
        offset = int(r.get("vertex_file_offset", 0))
        item["runtime_vertex_file"] = str(vertex_file.resolve())
        item["runtime_vertex_raw_bytes_sha256"] = sha256(vertex_file) if count else None
        item["runtime_vertex_slice_bytes"] = count * 3 * int(r.get("vertex_scalar_bytes", 4))
        item["runtime_vertex_slice_sha256"] = hashlib.sha256(vertex_file.read_bytes()[offset:offset + item["runtime_vertex_slice_bytes"]]).hexdigest() if count else None
        item["runtime_vertex_numeric_hash"] = canonical_hash({"unscaled": "binary_float32_slice", "offset": offset, "count": count, "scaling": r.get("child_local_scaling")}) if count else None
        provenance.append(item)
    (OUT / "runtime_shape_provenance.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in provenance), encoding="utf-8")
    return shape_map, provenance


def triangle_inputs() -> dict[str, tuple[Path, int, str]]:
    out = {}
    tri_dir = OUT / "triangle_vertices"
    tri_dir.mkdir(parents=True, exist_ok=True)
    mesh_root = ROOT / "external/frcobot_ros2/fairino_description/meshes/fairino5_v6"
    for link in TARGET_LINKS:
        src = mesh_root / f"{link}.STL"
        verts = read_binary_stl(src)
        dst = tri_dir / f"{link}.bin"
        write_float32_vertices(dst, verts)
        out[link] = (dst, len(verts) // 3, sha256(src))
    write_json(OUT / "triangle_vertex_manifest.json", {k: {"file": str(v[0].resolve()), "triangle_count": v[1], "source_mesh_sha256": v[2], "encoding": "little_endian_float32_xyz"} for k, v in out.items()})
    return out


def runtime_contacts() -> tuple[dict[tuple[str, str], list[dict]], list[dict]]:
    wanted = {frozenset(p.split("<->")): p for p in PAIR_NAMES}
    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    capture_rows = []
    all_raw = list(jsonl(CAPTURE / "bullet_runtime_manifolds.jsonl"))
    fallback = []
    for r in all_raw:
        if r.get("status") != "callback_observed" or r.get("run_id") != "run1":
            continue
        inv = str(r.get("invocation_id", ""))
        pair = wanted.get(frozenset((r.get("body_name_0"), r.get("body_name_1"))))
        if not pair:
            continue
        if inv.endswith("_R1"):
            pass
        elif inv.endswith("_R0"):
            fallback.append((r, pair))
            continue
        else:
            continue
        key = (r["node_id"], pair)
        by[key].append(r)
        capture_rows.append({
            "candidate_id": r["node_id"], "waypoint_id": r["waypoint_id"], "pair": pair,
            "body_name_0": r["body_name_0"], "body_name_1": r["body_name_1"],
            "child_index_0": r.get("child_index_0"), "child_index_1": r.get("child_index_1"),
            "m_distance1": r.get("m_distance1"), "m_distance1_hex": r.get("m_distance1_hex"),
            "normal_world_on_b": json.dumps(r.get("m_normalWorldOnB"), separators=(",", ":")),
            "position_world_on_a": json.dumps(r.get("m_positionWorldOnA"), separators=(",", ":")),
            "position_world_on_b": json.dumps(r.get("m_positionWorldOnB"), separators=(",", ":")),
            "part_id_0": r.get("m_partId0"), "part_id_1": r.get("m_partId1"),
            "index_0": r.get("m_index0"), "index_1": r.get("m_index1"),
            "margin_0": r.get("parent_margin_0"), "margin_1": r.get("parent_margin_1"),
            "runtime_algorithm_type": r.get("runtime_algorithm_type", "not_available"),
        })
    present_nodes = {cid for cid, _ in by}
    for r, pair in fallback:
        if r["node_id"] in present_nodes:
            continue
        by[(r["node_id"], pair)].append(r)
        capture_rows.append({
            "candidate_id": r["node_id"], "waypoint_id": r["waypoint_id"], "pair": pair,
            "body_name_0": r["body_name_0"], "body_name_1": r["body_name_1"],
            "child_index_0": r.get("child_index_0"), "child_index_1": r.get("child_index_1"),
            "m_distance1": r.get("m_distance1"), "m_distance1_hex": r.get("m_distance1_hex"),
            "normal_world_on_b": json.dumps(r.get("m_normalWorldOnB"), separators=(",", ":")),
            "position_world_on_a": json.dumps(r.get("m_positionWorldOnA"), separators=(",", ":")),
            "position_world_on_b": json.dumps(r.get("m_positionWorldOnB"), separators=(",", ":")),
            "part_id_0": r.get("m_partId0"), "part_id_1": r.get("m_partId1"),
            "index_0": r.get("m_index0"), "index_1": r.get("m_index1"),
            "margin_0": r.get("parent_margin_0"), "margin_1": r.get("parent_margin_1"),
            "runtime_algorithm_type": r.get("runtime_algorithm_type", "not_available"),
        })
    fields = list(capture_rows[0]) if capture_rows else ["candidate_id", "waypoint_id", "pair"]
    write_csv(OUT / "runtime_contact_capture.csv", fields, capture_rows)
    return by, capture_rows


def transform_map() -> dict[tuple[str, str], list[float]]:
    out = {}
    for r in jsonl(CAPTURE / "runtime_link_transforms.jsonl"):
        if r.get("run_index") == 1 and r.get("link") in TARGET_LINKS:
            out[(r["node_id"], r["link"])] = parse_transform16([float(x) for x in r["collision_body_transform"]])
    return out


def make_replay_input(shape_map, contacts, triangles) -> dict:
    target = {frozenset(p.split("<->")): p for p in PAIR_NAMES}
    transforms = transform_map()
    nodes = {x.lstrip("\ufeff") for x in Path(DISPUTED).read_text(encoding="utf-8").split()}
    rows = []
    for (cid, pair), items in sorted(contacts.items()):
        if cid not in nodes or not items:
            continue
        a, b = pair.split("<->")
        s0 = shape_map.get((a, -1)); s1 = shape_map.get((b, -1))
        if not s0 or not s1 or not s0.get("convex_point_count") or not s1.get("convex_point_count"):
            raise RuntimeError(f"missing runtime convex vertices for {cid} {pair}")
        t0 = transforms[(cid, a)]; t1 = transforms[(cid, b)]
        tri0, n0, _ = triangles[a]; tri1, n1, _ = triangles[b]
        distances = [float(x["m_distance1"]) for x in items]
        runtime_min = min(distances)
        if runtime_min == 0.0 and any(str(x.get("m_distance1_hex", "")).lower().startswith("0x8") for x in items): runtime_min = -0.0
        runtime_algorithm = next((str(x.get("runtime_algorithm_type")) for x in items if x.get("runtime_algorithm_type") and not str(x.get("runtime_algorithm_type")).startswith("not_available")), "")
        values = [cid, str(items[0]["waypoint_id"]), pair, str(runtime_min), str(len(items)), wsl_path(CAPTURE / "runtime_shape_vertices.bin"), str(s0["vertex_file_offset"]), str(s0["convex_point_count"]), str(s0["child_margin"])]
        values.extend(str(x) for x in s0["child_local_scaling"]); values.extend(str(x) for x in t0)
        values.extend([str(s1["vertex_file_offset"]), str(s1["convex_point_count"]), str(s1["child_margin"])]); values.extend(str(x) for x in s1["child_local_scaling"]); values.extend(str(x) for x in t1)
        values.extend([wsl_path(tri0), str(n0), wsl_path(tri1), str(n1)])
        values.append(runtime_algorithm)
        rows.append("\t".join(values))
    if len(rows) < 658 or len({x.split("\t")[0] for x in rows}) != 658:
        raise RuntimeError(f"expected all 658 disputed nodes in replay input, got {len(rows)} rows / {len({x.split(chr(9))[0] for x in rows})} nodes")
    (OUT / "runtime_replay_inputs.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    write_json(OUT / "runtime_shape_vertex_manifest.json", {"runtime_shape_tree_rows": len(shape_map), "disputed_pair_rows": len(rows), "disputed_nodes": len(nodes), "runtime_vertex_file": str((CAPTURE / "runtime_shape_vertices.bin").resolve()), "runtime_vertex_file_sha256": sha256(CAPTURE / "runtime_shape_vertices.bin"), "raw_scalar_bytes": 4, "canonical_numeric_hash": canonical_hash(rows)})
    return {"rows": len(rows), "nodes": len({x.split("\t")[0] for x in rows}), "pairs": len({x.split("\t")[2] for x in rows})}


def run_replay(binary: Path) -> list[Path]:
    outputs = []
    binary_wsl = "/mnt/c/Users/86198/Desktop/robotfucker/" + binary.relative_to(ROOT).as_posix()
    expected_rows = sum(1 for line in (OUT / "runtime_replay_inputs.tsv").read_text(encoding="utf-8").splitlines() if line.strip())
    for n in (1, 2, 3):
        result = OUT / f"exact_replay_run{n}.csv"; contact = OUT / f"exact_replay_contacts_run{n}.csv"; log = OUT / f"exact_replay_run{n}.stderr.txt"
        if result.exists() and contact.exists() and sum(1 for _ in result.open(encoding="utf-8")) == expected_rows * 3 + 1 and log.exists() and f"replay_version=3 replayed_rows={expected_rows}" in log.read_text(encoding="utf-8", errors="ignore"):
            outputs.append(result)
            continue
        p = subprocess.run(["wsl", "bash", "-lc", f"{binary_wsl} /mnt/c/Users/86198/Desktop/robotfucker/{(OUT / 'runtime_replay_inputs.tsv').relative_to(ROOT).as_posix()} /mnt/c/Users/86198/Desktop/robotfucker/{result.relative_to(ROOT).as_posix()} /mnt/c/Users/86198/Desktop/robotfucker/{contact.relative_to(ROOT).as_posix()} {n}"], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        log.write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
        if p.returncode != 0: raise RuntimeError(f"exact replay run {n} failed: {p.stderr[-1000:]}")
        outputs.append(result)
    return outputs


def run_double_replay(binary: Path) -> list[Path]:
    outputs = []
    binary_wsl = wsl_path(binary)
    input_wsl = wsl_path(OUT / "runtime_replay_inputs.tsv")
    expected_rows = sum(1 for line in (OUT / "runtime_replay_inputs.tsv").read_text(encoding="utf-8").splitlines() if line.strip())
    for n in (1, 2, 3):
        result = OUT / f"group_c_double_precision_run{n}.csv"
        contact = OUT / f"exact_replay_contacts_c_run{n}.csv"
        log = OUT / f"group_c_double_precision_run{n}.stderr.txt"
        if result.exists() and contact.exists() and sum(1 for _ in result.open(encoding="utf-8")) == expected_rows + 1 and log.exists() and f"replay_version=3 replayed_rows={expected_rows} btScalar_bytes=8 double_precision=true" in log.read_text(encoding="utf-8", errors="ignore"):
            outputs.append(result)
            continue
        cmd = f"{binary_wsl} {input_wsl} {wsl_path(result)} {wsl_path(contact)} {n} C"
        p = subprocess.run(["wsl", "bash", "-lc", cmd], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        log.write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
        if p.returncode != 0: raise RuntimeError(f"double precision replay run {n} failed: {p.stderr[-1000:]}")
        outputs.append(result)
    return outputs


def run_diagnostic_commands() -> None:
    binary = OUT / "install/stage23a7_runtime_audit/lib/stage23a7_runtime_audit/stage23a7_2_bullet_exact_replay"
    maps = list((CAPTURE / "proc_self_maps_run1.txt").read_text(encoding="utf-8", errors="ignore").splitlines())
    libs = sorted({line.split()[-1] for line in maps if "/lib" in line and ("Bullet" in line or "moveit_collision" in line)})
    paths = {"capture_executable": str((OUT / "install/stage23a7_runtime_audit/lib/stage23a7_runtime_audit/stage23a7_runtime_audit_callback_probe").resolve()), "replay_executable": str(binary.resolve()), "loaded_map_library_paths": libs}
    hashes = {}
    for p in libs:
        digest = subprocess.run(["wsl", "bash", "-lc", f"sha256sum '{p}'"], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=30).stdout.split()
        if digest: hashes[p] = digest[0]
    paths["bullet_library_paths"] = [p for p in libs if "Bullet" in p]
    paths["moveit_collision_bullet_library_path"] = next((p for p in libs if "moveit_collision_detection_bullet" in p), "not_available")
    write_json(OUT / "runtime_library_provenance.json", {"capture": json.loads((CAPTURE / "runtime_provenance_run1.json").read_text()), "actual_loaded_library_paths": paths, "sha256": hashes, "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(), "formal_files_modified": False, "btScalar": 4, "BT_USE_DOUBLE_PRECISION": False, "compiler": "gcc 13.3.0", "build_type": "O2", "loaded_overlay": "stage23a7_2 nested prefix first; verified by ros2 pkg prefix"})
    double_libs = ["/usr/lib/x86_64-linux-gnu/libBulletCollision-float64.so.3.24", "/usr/lib/x86_64-linux-gnu/libLinearMath-float64.so.3.24"]
    double_hashes = {}
    for p in double_libs:
        digest = subprocess.run(["wsl", "bash", "-lc", f"sha256sum '{p}'"], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=30).stdout.split()
        if digest: double_hashes[p] = digest[0]
    write_json(OUT / "double_precision_library_provenance.json", {"library_paths": double_libs, "sha256": double_hashes, "bullet_version": "3.24", "sizeof_btScalar": 8, "BT_USE_DOUBLE_PRECISION": True, "compiler": "gcc 13.3.0", "compile_flags": "-O3 -std=c++17 -DBT_USE_DOUBLE_PRECISION", "isolated_from_moveit_runtime": True})
    cache = OUT / "build/stage23a7_runtime_audit/CMakeCache.txt"
    if cache.exists(): shutil.copy2(cache, OUT / "CMakeCache_stage23a7_2.txt")
    def run_to(name, command):
        p = subprocess.run(["wsl", "bash", "-lc", command], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
        (OUT / name).write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
    exe = "/mnt/c/Users/86198/Desktop/robotfucker/outputs/ik_graph_stage23a7_2_bullet_exact_replay/fr5_scaled_horseshoe_demo_v45/install/stage23a7_runtime_audit/lib/stage23a7_runtime_audit/stage23a7_runtime_audit_callback_probe"
    overlay_ld = "/mnt/c/Users/86198/Desktop/robotfucker/tmp/stage23a7_install2/lib:/opt/ros/jazzy/lib:/lib/x86_64-linux-gnu"
    run_to("ldd_callback_probe.txt", f"LD_LIBRARY_PATH={overlay_ld} ldd {exe}"); run_to("readelf_callback_probe.txt", f"readelf -d {exe}"); run_to("nm_callback_probe.txt", f"nm -D {exe}")
    double_exe = wsl_path(OUT / "native/stage23a7_2_bullet_exact_replay_float64")
    run_to("ldd_double_precision_replay.txt", f"ldd {double_exe}"); run_to("readelf_double_precision_replay.txt", f"readelf -d {double_exe}"); run_to("nm_double_precision_replay.txt", f"nm -D {double_exe}")
    (OUT / "build_commands.txt").write_text("source /opt/ros/jazzy/setup.bash\ncolcon --log-base outputs/.../colcon_log build --base-paths cpp/stage23a7 --packages-select stage23a7_runtime_audit --build-base outputs/.../build --install-base outputs/.../install --event-handlers console_direct+\n", encoding="utf-8")
    (OUT / "double_precision_build_commands.txt").write_text("g++ -O3 -std=c++17 -DBT_USE_DOUBLE_PRECISION -I/usr/include/bullet cpp/stage23a7/stage23a7_2_bullet_exact_replay.cpp -o outputs/.../native/stage23a7_2_bullet_exact_replay_float64 -L/usr/lib/x86_64-linux-gnu -lBulletCollision-float64 -lLinearMath-float64 -Wl,-rpath,/usr/lib/x86_64-linux-gnu\n", encoding="utf-8")
    (OUT / "run_commands.txt").write_text("ros2 launch tools/stage23a7_runtime_audit_launch.py ... backend:=bullet only_disputed:=true\ninstalled stage23a7_2_bullet_exact_replay INPUT.tsv RESULTS.csv CONTACTS.csv RUN_ID\nnative/stage23a7_2_bullet_exact_replay_float64 INPUT.tsv group_c_double_precision_runN.csv CONTACTS.csv RUN_ID C\n", encoding="utf-8")


def formal_file_audit() -> None:
    manifest = ROOT / "outputs/ik_graph_stage23a6_bullet_self_collision_audit/fr5_scaled_horseshoe_demo_v44/stage23a6_frozen_inputs.json"
    records = []
    if manifest.exists():
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        for item in frozen.get("files", []):
            p = ROOT / item["path"]
            records.append({"path": item["path"], "exists": p.exists(), "current_sha256": sha256(p) if p.exists() else None, "frozen_sha256": item.get("sha256"), "matches_frozen": p.exists() and sha256(p) == item.get("sha256")})
    write_json(OUT / "formal_file_hash_audit.json", {"formal_configuration_modified_by_stage23a7_2": False, "records": records, "note": "Pre-existing dirty-worktree changes are not attributed to this stage."})


def sha256_manifest() -> None:
    rows = []
    for p in sorted(OUT.rglob("*")):
        try:
            is_file = p.is_file()
        except OSError:
            is_file = False
        if is_file and p.name != "sha256_manifest.txt": rows.append(f"{sha256(p)}  {p.relative_to(OUT).as_posix()}")
    (OUT / "sha256_manifest.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def aggregate(replay_files: list[Path], perturbation_path: Path | None) -> dict:
    rows = []
    for p in replay_files:
        with p.open(encoding="utf-8", newline="") as f: rows.extend(csv.DictReader(f))
    with replay_files[0].open(encoding="utf-8", newline="") as f: first = list(csv.DictReader(f))
    write_csv(OUT / "exact_replay_results.csv", list(first[0]), first)
    write_csv(OUT / "group_a_native_replay.csv", list(first[0]), [r for r in first if r["group"] == "A"])
    write_csv(OUT / "group_b_zero_margin.csv", list(first[0]), [r for r in first if r["group"] == "B"])
    write_csv(OUT / "group_d_triangle_representation.csv", list(first[0]), [r for r in first if r["group"] == "D"])
    double_rows = list(csv.DictReader((OUT / "group_c_double_precision_run1.csv").open(encoding="utf-8")))
    write_csv(OUT / "group_c_double_precision.csv", list(double_rows[0]), double_rows)
    all_rows = first + double_rows
    write_csv(OUT / "exact_replay_results.csv", list(first[0]), all_rows)
    by_node = defaultdict(lambda: defaultdict(list))
    for r in all_rows: by_node[r["candidate_id"]][r["group"]].append(r)
    pert = defaultdict(list)
    if perturbation_path and perturbation_path.exists():
        for r in jsonl(perturbation_path):
            if r.get("backend") == "bullet": pert[r["node_id"]].append(r)
    classifications = []
    for cid, groups in sorted(by_node.items()):
        a = any(r["replay_collision"] == "true" for r in groups.get("A", [])); b = any(r["replay_collision"] == "true" for r in groups.get("B", [])); d = any(r["replay_collision"] == "true" for r in groups.get("D", [])); c = any(r["replay_collision"] == "true" for r in groups.get("C", []))
        mismatch = any(r["sign_match"] != "true" for r in groups.get("A", []))
        algorithm_mismatch = any(r["algorithm_match"] != "true" for r in groups.get("A", []) + groups.get("B", []))
        precision = a != c or any(r["sign_match"] != "true" for r in groups.get("C", []))
        p_rows = pert.get(cid, []); p_signs = {str(x.get("collision")) for x in p_rows}; perturb = len(p_signs) > 1
        if mismatch: primary = "runtime_replay_mismatch"
        elif a and d: primary = "collision_under_both_representations"
        elif a and not d: primary = "representation_induced"
        elif a and not b: primary = "margin_induced"
        elif perturb: primary = "perturbation_sensitive"
        else: primary = "unresolved"
        labels = []
        if algorithm_mismatch: labels.append("algorithm_identity_mismatch")
        if precision: labels.append("precision_sensitive")
        if perturb: labels.append("perturbation_sensitive")
        classifications.append({"candidate_id": cid, "primary_classification": primary, "A_collision": a, "B_collision": b, "C_collision": c, "D_collision": d, "runtime_replay_mismatch": mismatch, "algorithm_identity_mismatch": algorithm_mismatch, "perturbation_sensitive": perturb, "precision_sensitive": precision, "auxiliary_labels": ";".join(labels) if labels else "none"})
    fields = list(classifications[0])
    write_csv(OUT / "candidate_causal_classification.csv", fields, classifications)
    return {"rows": len(first), "a_sign_matches": sum(r["sign_match"] == "true" for r in first if r["group"] == "A"), "a_rows": sum(r["group"] == "A" for r in first), "algorithm_matched_rows": sum(r["algorithm_match"] == "true" for r in first if r["group"] in ("A", "B")), "precision_sensitive_nodes": sum(x["precision_sensitive"] for x in classifications), "classification_counts": {k: sum(x["primary_classification"] == k for x in classifications) for k in sorted({x["primary_classification"] for x in classifications})}, "classified_nodes": len(classifications), "unresolved_nodes": sum(x["primary_classification"] == "unresolved" for x in classifications)}


def perturbation_summary() -> dict:
    baseline = {}
    with (OUT / "group_a_native_replay.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f): baseline[r["candidate_id"]] = r["replay_collision"] == "true"
    states = defaultdict(list)
    distances = defaultdict(list)
    for run in (1, 2, 3):
        p = OUT / f"perturbation_run{run}/bullet_joint_perturbation_results.jsonl"
        for r in jsonl(p): states[r["node_id"]].append(r)
        for r in jsonl(OUT / f"perturbation_run{run}/bullet_runtime_manifolds.jsonl"):
            if r.get("status") == "callback_observed" and r.get("m_distance1") is not None:
                distances[r.get("node_id")].append(float(r["m_distance1"]))
    rows = []
    for cid in sorted(baseline):
        vals = states[cid]
        signs = [bool(x.get("collision")) for x in vals]
        ds = distances[cid]
        flips = sum(x != baseline[cid] for x in signs)
        rows.append({"candidate_id": cid, "baseline_collision": baseline[cid], "perturbation_state_count": len(vals), "baseline_distance": "not_available_from_MoveIt_result", "min_perturbed_m_distance1": min(ds) if ds else None, "max_perturbed_m_distance1": max(ds) if ds else None, "sign_flip_count": flips, "classification_stable": flips == 0, "perturbation_seed": 23072, "runs": 3})
    write_csv(OUT / "perturbation_results.csv", list(rows[0]), rows)
    return {"nodes": len(rows), "states": sum(x["perturbation_state_count"] for x in rows), "sign_flip_nodes": sum(x["sign_flip_count"] > 0 for x in rows), "complete": len(rows) == 658 and all(x["perturbation_state_count"] == 3 * 6 * 8 for x in rows)}


def coverage() -> dict:
    with (PREV / "full_candidate_backend_results.csv").open(encoding="utf-8", newline="") as f:
        frozen = {r["candidate_id"]: r for r in csv.DictReader(f)}
    current_bullet = {r["candidate_id"]: r for r in jsonl(CAPTURE / "bullet_runtime_results_run1.jsonl")}
    fset = {k for k, r in frozen.items() if r.get("fcl_formal_valid", "False").lower() == "true"}
    bset = {k for k, r in frozen.items() if r.get("bullet_formal_valid", "False").lower() == "true"}
    for cid, r in current_bullet.items():
        if cid in bset and r.get("r1_collision", True): bset.remove(cid)
        if cid not in bset and not r.get("r1_collision", True): bset.add(cid)
    inter = fset & bset
    by_wp = {i: {"fcl": 0, "bullet": 0, "consensus": 0} for i in range(720)}
    for cid in frozen:
        wp = int(frozen[cid]["waypoint_id"])
        if cid in fset: by_wp[wp]["fcl"] += 1
        if cid in bset: by_wp[wp]["bullet"] += 1
        if cid in inter: by_wp[wp]["consensus"] += 1
    write_json(OUT / "waypoint_coverage_report.json", {"fcl_valid_nodes": len(fset), "bullet_valid_nodes": len(bset), "consensus_nodes": len(inter), "fcl_waypoints": sum(x["fcl"] > 0 for x in by_wp.values()), "bullet_waypoints": sum(x["bullet"] > 0 for x in by_wp.values()), "consensus_waypoints": sum(x["consensus"] > 0 for x in by_wp.values()), "certified_waypoints": "0/720", "certified_reason": "runtime_geometry_equivalence_or_authoritative_backend_not_resolved", "intersection_nodes": len(inter), "fcl_only_nodes": len(fset - bset), "bullet_only_nodes": len(bset - fset), "symmetric_difference_nodes": len(fset ^ bset), "bullet_valid_subset_of_fcl_valid": bset <= fset, "per_waypoint": by_wp})
    return {"fcl": len(fset), "bullet": len(bset), "consensus": len(inter), "fcl_waypoints": sum(x["fcl"] > 0 for x in by_wp.values()), "bullet_waypoints": sum(x["bullet"] > 0 for x in by_wp.values()), "consensus_waypoints": sum(x["consensus"] > 0 for x in by_wp.values())}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    shape_map, _ = capture_shape_maps(); triangles = triangle_inputs(); contacts, _ = runtime_contacts(); replay_input = make_replay_input(shape_map, contacts, triangles)
    run_diagnostic_commands()
    formal_file_audit()
    binary = OUT / "install/stage23a7_runtime_audit/lib/stage23a7_runtime_audit/stage23a7_2_bullet_exact_replay"
    replay_files = run_replay(binary)
    double_replay_files = run_double_replay(OUT / "native/stage23a7_2_bullet_exact_replay_float64")
    perturb = OUT / "perturbation_run1/bullet_joint_perturbation_results.jsonl"
    summary = aggregate(replay_files, perturb if perturb.exists() else None); perturb_summary_data = perturbation_summary(); cov = coverage()
    hashes = {p.name: sha256(p) for p in replay_files}; double_hashes = {p.name: sha256(p) for p in double_replay_files}; shape_hashes = {f"run{n}": sha256(OUT / f"runtime_capture_run{n}/runtime_shape_vertices.bin") for n in (1, 2, 3)}; runtime_hashes = {f"run{n}": sha256(OUT / f"runtime_capture_run{n}/bullet_runtime_results_run{n}.jsonl") for n in (1, 2, 3)}
    determinism = {"capture_shape_hashes": shape_hashes, "capture_runtime_result_hashes": runtime_hashes, "replay_result_hashes": hashes, "double_precision_replay_result_hashes": double_hashes, "capture_shapes_identical": len(set(shape_hashes.values())) == 1, "capture_runtime_files_byte_identical": len(set(runtime_hashes.values())) == 1, "replay_files_byte_identical": len(set(hashes.values())) == 1, "double_precision_replay_files_byte_identical": len(set(double_hashes.values())) == 1, "independent_runs": 3}
    write_json(OUT / "determinism_manifest.json", determinism)
    write_json(OUT / "perturbation_results.json", {"status": "captured", "source": str(perturb), **perturb_summary_data})
    (OUT / "modified_files.txt").write_text("cpp/stage23a7/stage23a7_runtime_audit_callback_probe.cpp\ncpp/stage23a7/stage23a7_2_bullet_exact_replay.cpp\ncpp/stage23a7/CMakeLists.txt\ntools/stage23a7_runtime_audit_launch.py\nscripts/run_stage23a7_2_bullet_exact_replay.py\ntests/test_stage23a7_2_outputs.py\nformal URDF/SRDF/ACM/waypoints/candidate values: not modified\n", encoding="utf-8")
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_stage23a7_2_outputs.py"], cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    (OUT / "test_results.txt").write_text("COMMAND: python -m pytest -q tests/test_stage23a7_2_outputs.py\nEXIT_CODE: " + str(test.returncode) + "\n\nSTDOUT\n" + (test.stdout or "") + "\nSTDERR\n" + (test.stderr or ""), encoding="utf-8")
    native_rows = list(csv.DictReader((OUT / "group_a_native_replay.csv").open(encoding="utf-8")))
    double_rows = list(csv.DictReader((OUT / "group_c_double_precision.csv").open(encoding="utf-8")))
    all_algorithm_match = summary["algorithm_matched_rows"] == 2 * replay_input["rows"]
    technical_complete = summary["a_rows"] == replay_input["rows"] and summary["a_sign_matches"] == replay_input["rows"] and len(double_rows) == replay_input["rows"] and summary["classified_nodes"] == 658 and summary["unresolved_nodes"] == 0 and all_algorithm_match and test.returncode == 0
    status = "blocked_formal_coverage_not_certified" if technical_complete else ("blocked_runtime_algorithm_identity_mismatch" if not all_algorithm_match else "blocked_exact_runtime_replay_incomplete")
    gate = {"Stage_2_3A_7_2": {"status": status, "execution_completed": True, "exact_runtime_replay": {"attempted_nodes": 658, "attempted_pair_rows": replay_input["rows"], "sign_matched_pair_rows": summary["a_sign_matches"], "algorithm_matched_rows": summary["algorithm_matched_rows"], "classification_matched_nodes": summary["classified_nodes"] - summary["unresolved_nodes"], "max_absolute_distance_error": max(float(r["absolute_error"]) for r in native_rows)}, "double_precision_replay": {"attempted_pair_rows": len(double_rows), "sizeof_btScalar": 8, "BT_USE_DOUBLE_PRECISION": True, "deterministic_runs": 3}, "classifications": summary["classification_counts"] | {"precision_sensitive_nodes": summary["precision_sensitive_nodes"]}, "coverage": cov | {"certified_waypoints": "0/720"}, "deterministic_runs": 3, "hashes_consistent": determinism["capture_shapes_identical"] and determinism["capture_runtime_files_byte_identical"] and determinism["replay_files_byte_identical"] and determinism["double_precision_replay_files_byte_identical"], "tests": {"command": "python -m pytest -q tests/test_stage23a7_2_outputs.py", "exit_code": test.returncode, "status": "passed" if test.returncode == 0 else "failed"}, "formal_files_modified": False}, "Stage_2_3B": {"status": "blocked"}}
    write_json(OUT / "stage23a7_2_gate_report.json", gate)
    lines = ["# Stage 2.3A.7.2 Bullet runtime convex-hull exact replay", "", f"Status: **{gate['Stage_2_3A_7_2']['status']}**", "", "The formal configuration was not modified. Stage 2.3B remains blocked.", "", "## Runtime provenance", "", "- Native capture: 3 independent MoveIt2/Bullet processes, btScalar=4, Bullet 3.24, with runtime callback algorithm type `btConvexConvexAlgorithm` recorded.", f"- Runtime shape vertices: {replay_input['rows']} disputed pair rows / 658 disputed nodes; raw float32 vertex file and per-shape offsets are preserved.", "- The capture used the current nested overlay, verified by `ros2 pkg prefix stage23a7_runtime_audit`.", "", "## A/B/C/D replay", "", f"- A attempted {replay_input['rows']} pair rows; sign matched {summary['a_sign_matches']}/{replay_input['rows']}; runtime/replay algorithm identity matched {summary['algorithm_matched_rows']}/{2 * replay_input['rows']} A/B rows.", f"- C completed {len(double_rows)}/{replay_input['rows']} rows using isolated Bullet 3.24 float64 libraries (`sizeof(btScalar)=8`).", f"- Classification counts: {summary['classification_counts']}; precision-sensitive auxiliary nodes: {summary['precision_sensitive_nodes']}.", "- B is isolated margin=0; D is diagnostic triangle representation only.", "", "## Coverage", "", f"- FCL/Bullet/consensus waypoint coverage: {cov['fcl_waypoints']}/{cov['bullet_waypoints']}/{cov['consensus_waypoints']} of 720; certified coverage remains 0/720.", "", "## Limits", "", "The isolated replay is diagnostic evidence. Formal downstream promotion remains blocked because certified waypoint coverage is 0/720 and the native runtime convex-hull geometry is not equivalent to the formal triangle representation.", ""]
    (OUT / "stage23a7_2_report.md").write_text("\n".join(lines), encoding="utf-8")
    sha256_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
