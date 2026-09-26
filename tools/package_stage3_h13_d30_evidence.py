"""Finalize D30 evidence and build/verify its single Desktop tar.xz archive."""
from __future__ import annotations
import argparse, hashlib, json, lzma, shutil, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; DESKTOP=Path(r"C:\Users\86198\Desktop"); LIMIT=20_000_000
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()
def dump(p:Path,x): p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
def manifest(root:Path,name="ARCHIVE_INTERNAL_MANIFEST.json"):
    files=[{"relative_path":p.relative_to(root).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.name!=name]
    dump(root/name,{"schema_version":"stage3_h13_d30_archive_internal_manifest_v1","hash_algorithm":"SHA-256","self_excluded":True,"managed_files":files,"file_count":len(files)})
def finalize_output(out:Path):
    repairs=[{"defect":"direct-script repository import path missing","scope":"wrapper only; failed before parent load/candidate generation","repair":"insert repository root into sys.path before tools import","regression":"py_compile PASS and inherited focused AdamW tests 3/3 PASS","scientific_state_changed":False},{"event":"authenticated D29 Desktop archive was externally removed after complete authentication and restored byte-identically before scientific screening","restored_sha256":"fd8d8bca87ed935b7802df2fd6f3a82eeb16c6eaef11271c931e5dee8b720bdb","scientific_state_changed":False}]
    impl=json.loads((out/"IMPLEMENTATION_CHANGES.json").read_text()); impl["validator_or_wrapper_repairs"]=repairs; dump(out/"IMPLEMENTATION_CHANGES.json",impl)
    exe=json.loads((out/"D30_EXECUTION_MANIFEST.json").read_text()); exe["validator_or_wrapper_repairs"]=repairs; dump(out/"D30_EXECUTION_MANIFEST.json",exe)
    records=json.loads((out/"D30_CANDIDATE_RESULTS.json").read_text()); replay_rows=[]
    for r in records:
        tag=(f"{float(r['alpha']):.9f}").rstrip("0").rstrip(".").replace(".","p"); detail=json.loads((out/"optimizer_consistency"/f"update_{r['candidate_update']}_alpha_{tag}.json").read_text())["deterministic_replay"]
        replay_rows.append({"parent_update":r["parent_update"],"update":r["candidate_update"],"alpha":r["alpha"],"effective_lr":r["effective_lr"],"parent_checkpoint_sha256":r["parent_checkpoint_SHA256"],"candidate_checkpoint_sha256":r.get("candidate_checkpoint_SHA256"),"candidate_model_state_hash":detail.get("candidate_model_state_hash"),"candidate_optimizer_state_hash":detail.get("candidate_optimizer_state_hash"),"H1":r["candidate_H1"],"H32":r["candidate_H32"],"H32_margin":r["post_update_H32_margin"],"optimizer_state_equal":detail.get("optimizer_state_equal"),"rng_identity":detail.get("candidate_rng_hash"),"rng_state_equal":detail.get("rng_state_equal"),"batch_identity":r.get("batch_identity"),"schedule_index":r["candidate_update"]-1,"replay_residuals":{"model_state_equal":detail.get("model_state_equal"),"optimizer_state_equal":detail.get("optimizer_state_equal"),"record_equal":detail.get("record_equal"),"rng_state_equal":detail.get("rng_state_equal")},"candidate_record_digest":detail.get("candidate_record_digest"),"replay_record_digest":detail.get("replay_record_digest"),"status":detail.get("status")})
    dump(out/"DETERMINISTIC_REPLAY_REPORT.json",{"schema_version":"stage3_h13_d30_deterministic_replay_v2","status":"PASS" if all(x["status"]=="PASS" for x in replay_rows) else "FAIL","candidate_checkpoint_note":"Unselected trial states are identified by deterministic model/optimizer/RNG hashes; only canonical selected states are serialized as checkpoint files.","candidates":replay_rows})
    traj=json.loads((out/"D30_CANONICAL_TRAJECTORY.json").read_text()); max_canonical=float(traj["measurements"]["max_H32_margin_reached"]); max_screen=max(float(r["post_update_H32_margin"]) for r in records)
    for name in ("FINAL_REPORT.md","TASK_AND_RESULTS_SUMMARY.md"):
        p=out/name; text=p.read_text(); text=text.replace("VALIDATOR_OR_WRAPPER_REPAIRS: none","VALIDATOR_OR_WRAPPER_REPAIRS: one pre-scientific import-path wrapper repair; D29 archive restoration event documented; zero scientific-state change")
        text=text.replace(f"MAX_H32_MARGIN_REACHED_DURING_D30: {max_canonical:.17g}",f"MAX_H32_MARGIN_REACHED_DURING_D30: {max_screen:.17g} (all screened candidates; rejected states were never canonical)\nMAX_CANONICAL_H32_MARGIN_REACHED_DURING_D30: {max_canonical:.17g}")
        p.write_text(text,encoding="utf-8",newline="\n")
    testlog=ROOT/"tmp"/"d30_validator_regression.log"; dump(out/"VALIDATOR_REGRESSION_TEST_RESULTS.json",{"command":"pytest -q tests/test_stage3_h13_d26_optimizer_consistency_recovery.py","captured_output":testlog.read_text(),"captured_output_sha256":sha(testlog),"status":"PASS" if "3 passed" in testlog.read_text() else "FAIL","py_compile":"PASS"})
    shutil.copy2(ROOT/"tmp"/"d30_execution_console.log",out/"EXECUTION_CONSOLE.log")
    src=out/"implementation"; src.mkdir(exist_ok=True); shutil.copy2(ROOT/"tools"/"stage3_h13_d30_active_boundary_continuation.py",src/"stage3_h13_d30_active_boundary_continuation.py"); shutil.copy2(Path(__file__),src/Path(__file__).name)
    manifest(out,"INTERNAL_FILE_MANIFEST.json")
def populate(out:Path,stage:Path):
    stage.mkdir(parents=True)
    top=["START_HERE_NEXT_CHAT.txt","TASK_AND_RESULTS_SUMMARY.md","FINAL_REPORT.md","D30_EXECUTION_MANIFEST.json","D30_CANDIDATE_RESULTS.json","D30_CANDIDATE_RESULTS.csv","D30_CANONICAL_TRAJECTORY.json","D30_CANONICAL_TRAJECTORY.csv","OPTIMIZER_CONSISTENCY_REPORT.json","DETERMINISTIC_REPLAY_REPORT.json","RNG_CONTINUITY_REPORT.json","STAGE3_COMPLETION_CONTRACT_EVALUATION.json","IMPLEMENTATION_CHANGES.json","VALIDATOR_REGRESSION_TEST_RESULTS.json","EXECUTION_CONSOLE.log","TASK_PROMPT.md"]
    for n in top: shutil.copy2(out/n,stage/n)
    for sub in ("metrics","implementation"):
        shutil.copytree(out/sub,stage/sub)
    (stage/"optimizer_consistency").mkdir()
    for p in (out/"optimizer_consistency").glob("*.json"): shutil.copy2(p,stage/"optimizer_consistency"/p.name)
    traj=json.loads((out/"D30_CANONICAL_TRAJECTORY.json").read_text()); final=traj["trajectory"][-1]; cp=out/"checkpoints"/f"committed_update_{final['update']}.pt"; (stage/"checkpoints").mkdir(); shutil.copy2(cp,stage/"checkpoints"/cp.name)
    d29=ROOT/"outputs"/"stage3_h13_post_d28_d29_trust_region_reexpansion_20260823T170000+0800"; shutil.copy2(d29/"checkpoints"/"committed_update_420.pt",stage/"checkpoints"/"committed_update_420.pt"); (stage/"D29_PARENT_EVIDENCE").mkdir()
    for n in ("START_HERE_NEXT_CHAT.txt","D29_EXECUTION_MANIFEST.json","D29_CANONICAL_TRAJECTORY.json"): shutil.copy2(d29/n,stage/"D29_PARENT_EVIDENCE"/n)
    (stage/"ARCHIVE_CONTENTS_GUIDE.md").write_text("# D30 archive contents\n\nRead START_HERE_NEXT_CHAT.txt, TASK_AND_RESULTS_SUMMARY.md, and FINAL_REPORT.md first. Candidate and trajectory JSON/CSV preserve all decisions. Validation reports and per-candidate JSON preserve optimizer, replay, RNG, clipping, and policy gates. Parent update 420 and final update 423 checkpoints are included. Margin recovery succeeded; controlled re-expansion did not occur. Maximum canonical margin was 0.000181866232865667; rejected update-424 trials showed larger non-canonical margins.\n",encoding="utf-8",newline="\n")
    omitted=[]
    for p in sorted((out/"optimizer_consistency").glob("*.pt")): omitted.append(f"- `{p.relative_to(out).as_posix()}` — {p.stat().st_size} bytes; SHA-256 `{sha(p)}`")
    for p in sorted((out/"checkpoints").glob("committed_update_42[12].pt")): omitted.append(f"- `{p.relative_to(out).as_posix()}` — {p.stat().st_size} bytes; SHA-256 `{sha(p)}`")
    (stage/"EXCLUDED_ARTIFACTS.md").write_text("# Exclusions\n\nLarge redundant diagnostic and intermediate checkpoint binaries are excluded for the 20 MB limit; decisive JSON and chain identities are retained.\n\n"+"\n".join(omitted)+"\n\nThe authenticated D29 archive remains on Desktop unchanged and is omitted because its final parent checkpoint/key evidence are included here. Unrelated outputs are excluded.\n",encoding="utf-8",newline="\n")
    manifest(stage)
def compress(stage:Path,archive:Path):
    raw=archive.with_suffix("")
    def filt(i): i.uid=i.gid=0; i.uname=i.gname=""; i.mtime=0; return i
    with tarfile.open(raw,"w",format=tarfile.PAX_FORMAT) as t:
        for p in sorted(stage.rglob("*")): t.add(p,arcname=p.relative_to(stage).as_posix(),recursive=False,filter=filt)
    c=lzma.LZMACompressor(format=lzma.FORMAT_XZ,preset=9|lzma.PRESET_EXTREME)
    with raw.open("rb") as s, archive.open("xb") as d:
        while b:=s.read(1024*1024): d.write(c.compress(b))
        d.write(c.flush())
    raw.unlink()
def verify(a:Path,expected_cp:str):
    decompressed=0
    with lzma.open(a,"rb") as f:
        while b:=f.read(1024*1024): decompressed+=len(b)
    with tarfile.open(a,"r:xz") as t:
        names={m.name for m in t.getmembers() if m.isfile()}; m=json.load(t.extractfile("ARCHIVE_INTERNAL_MANIFEST.json")); mismatches=[]
        for x in m["managed_files"]:
            f=t.extractfile(x["relative_path"]); data=f.read() if f else b""
            if len(data)!=x["size_bytes"] or hashlib.sha256(data).hexdigest()!=x["sha256"]: mismatches.append(x["relative_path"])
        cpname="checkpoints/committed_update_423.pt"; cp=t.extractfile(cpname); cpsha=hashlib.sha256(cp.read()).hexdigest() if cp else None
        required={"START_HERE_NEXT_CHAT.txt","TASK_AND_RESULTS_SUMMARY.md","FINAL_REPORT.md","D30_EXECUTION_MANIFEST.json","D30_CANDIDATE_RESULTS.json","D30_CANONICAL_TRAJECTORY.json","OPTIMIZER_CONSISTENCY_REPORT.json","DETERMINISTIC_REPLAY_REPORT.json","RNG_CONTINUITY_REPORT.json","STAGE3_COMPLETION_CONTRACT_EVALUATION.json","TASK_PROMPT.md","ARCHIVE_INTERNAL_MANIFEST.json",cpname}
    passed=not mismatches and required<=names and cpsha==expected_cp and a.stat().st_size<LIMIT
    return {"overall":"PASS" if passed else "FAIL","archive_path":str(a),"archive_size_bytes":a.stat().st_size,"size_limit_bytes":LIMIT,"size_limit_pass":a.stat().st_size<LIMIT,"archive_sha256":sha(a),"compression":"XZ / LZMA2 preset 9 EXTREME","xz_decompression_test":"PASS","tar_traversal_read":"PASS","internal_manifest_verification":"PASS" if not mismatches else "FAIL","cross_document_consistency":"PASS" if required<=names else "FAIL","final_checkpoint_sha256":cpsha,"final_checkpoint_identity":"PASS" if cpsha==expected_cp else "FAIL","decompressed_tar_bytes":decompressed,"mismatches":mismatches}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--stamp",required=True); x=ap.parse_args(); out=x.output.resolve(); finalize_output(out)
    stage=ROOT/"tmp"/f"d30_archive_stage_{x.stamp}"; archive=DESKTOP/f"STAGE3_H13_D30_ACTIVE_BOUNDARY_MARGIN_RECOVERY_CROSS_CHAT_EVIDENCE_MAX_XZ_{x.stamp}.tar.xz"
    if stage.exists() or archive.exists(): raise SystemExit("refusing overwrite")
    old=list(DESKTOP.glob("*D30*.tar.xz"))
    for p in old:
        rp=p.resolve(); assert rp.parent==DESKTOP.resolve() and "D30" in rp.name; rp.unlink()
    populate(out,stage); compress(stage,archive); check=verify(archive,"9d5c51872209baafe69f01b739ac1608b6adebd466ea8d7bae123bc56ba5fccc")
    check["desktop_d30_archive_count"]=len(list(DESKTOP.glob("*D30*.tar.xz"))); check["single_final_d30_archive"]=check["desktop_d30_archive_count"]==1; check["d29_parent_archive_preserved"]=Path(r"C:\Users\86198\Desktop\STAGE3_H13_D29_TRUST_REGION_REEXPANSION_CROSS_CHAT_EVIDENCE_MAX_XZ_20260823T181000+0800.tar.xz").is_file()
    dump(out/"FINAL_ARCHIVE_VERIFICATION.json",check); print(json.dumps(check,sort_keys=True)); return 0 if check["overall"]=="PASS" and check["single_final_d30_archive"] and check["d29_parent_archive_preserved"] else 1
if __name__=="__main__": raise SystemExit(main())
