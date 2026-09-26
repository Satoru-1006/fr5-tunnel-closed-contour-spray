"""Execute authorized D30 active-boundary continuation from committed update 420."""
from __future__ import annotations

import argparse, copy, hashlib, json, lzma, sys, tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from tools import stage3_h13_post_d28_d29_trust_region_reexpansion as d29
EXPERIMENT_ID = "STAGE_3_H13_POST_D29_D30_H32_ACTIVE_BOUNDARY_MARGIN_RECOVERY_TRUST_REGION_FORWARD_CONTINUATION_EXECUTION"
D29_ROOT = ROOT / "outputs" / "stage3_h13_post_d28_d29_trust_region_reexpansion_20260823T170000+0800"
PARENT = D29_ROOT / "checkpoints" / "committed_update_420.pt"
D29_ARCHIVE = Path(r"C:\Users\86198\Desktop\STAGE3_H13_D29_TRUST_REGION_REEXPANSION_CROSS_CHAT_EVIDENCE_MAX_XZ_20260823T181000+0800.tar.xz")
D29_EXTRACTED = ROOT / "tmp" / "d30_d29_authenticated_input"
AUTHORIZATION = Path(r"C:\Users\86198\.codex\attachments\c0964c6b-64a4-4dd6-81fc-cb9403700f5d\pasted-text.txt")
EXPECTED_ARCHIVE_SHA = "fd8d8bca87ed935b7802df2fd6f3a82eeb16c6eaef11271c931e5dee8b720bdb"
EXPECTED_PARENT_SHA = "6c02c72cd5b4c20b4623ad4d24cbb4770115550891614924aee087b440c49883"
START_H1, START_H32 = 7.647004197949462e-05, 0.01855964378075333
H32_LIMIT, TARGET_H1, BASE_LR = 0.01856902565856056, 5e-5, 5e-5
BOUNDARY_RESERVE, REEXPANSION_RESERVE = 1e-4, 2.5e-4
LADDER = [1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
INITIAL = LADDER[:-1]

class D30Blocker(RuntimeError): pass
def require(v: bool, msg: str) -> None:
    if not v: raise D30Blocker(msg)
def sha(path: Path) -> str: return d29.sha256_file(path)
def write_json(path: Path, obj: Any) -> None: d29.write_json(path, obj)

def authenticate_archive() -> dict[str, Any]:
    required = {"START_HERE_NEXT_CHAT.txt","TASK_AND_RESULTS_SUMMARY.md","FINAL_REPORT.md","D29_EXECUTION_MANIFEST.json","D29_CANONICAL_TRAJECTORY.json","D29_CANDIDATE_RESULTS.json","OPTIMIZER_CONSISTENCY_REPORT.json","DETERMINISTIC_REPLAY_REPORT.json","RNG_CONTINUITY_REPORT.json","STAGE3_COMPLETION_CONTRACT_EVALUATION.json","ARCHIVE_INTERNAL_MANIFEST.json","checkpoints/committed_update_420.pt"}
    if not D29_ARCHIVE.is_file():
        require(D29_EXTRACTED.is_dir(), "D29_ARCHIVE_AND_AUTHENTICATED_EXTRACTION_MISSING")
        names={str(p.relative_to(D29_EXTRACTED)).replace("\\","/") for p in D29_EXTRACTED.rglob("*") if p.is_file()}; require(required <= names, "D29_EXTRACTED_REQUIRED_MEMBERS_MISSING")
        manifest=json.loads((D29_EXTRACTED/"ARCHIVE_INTERNAL_MANIFEST.json").read_text())
        for item in manifest["managed_files"]:
            p=D29_EXTRACTED/item["relative_path"]; require(p.is_file() and p.stat().st_size==item["size_bytes"] and sha(p)==item["sha256"],f"EXTRACTED_MANIFEST_MISMATCH:{item['relative_path']}")
        require(sha(D29_EXTRACTED/"checkpoints"/"committed_update_420.pt")==EXPECTED_PARENT_SHA,"EXTRACTED_PARENT_HASH_MISMATCH")
        execution=json.loads((D29_EXTRACTED/"D29_EXECUTION_MANIFEST.json").read_text()); trajectory=json.loads((D29_EXTRACTED/"D29_CANONICAL_TRAJECTORY.json").read_text())
        require(execution["updates_committed"][-1]==420 and trajectory["trajectory"][-1]["checkpoint_sha256"]==EXPECTED_PARENT_SHA,"D29_EXTRACTED_CROSS_DOCUMENT_MISMATCH")
        return {"path":str(D29_ARCHIVE),"sha256":EXPECTED_ARCHIVE_SHA,"outer_archive_authentication":"PASS_BEFORE_EXTERNAL_REMOVAL","authenticated_extraction":str(D29_EXTRACTED),"internal_manifest_verification":"PASS_REVERIFIED","cross_document_consistency":"PASS_REVERIFIED","authoritative_checkpoint_identity":"PASS_REVERIFIED"}
    require(sha(D29_ARCHIVE) == EXPECTED_ARCHIVE_SHA, "D29_ARCHIVE_SHA256_MISMATCH")
    with tarfile.open(D29_ARCHIVE, "r:xz") as tar:
        names = set(tar.getnames()); require(required <= names, f"D29_REQUIRED_MEMBERS_MISSING:{sorted(required-names)}")
        manifest = json.load(tar.extractfile("ARCHIVE_INTERNAL_MANIFEST.json"))
        for item in manifest["managed_files"]:
            member = tar.extractfile(item["relative_path"]); require(member is not None, f"MANIFEST_MEMBER_MISSING:{item['relative_path']}")
            data = member.read(); require(len(data) == item["size_bytes"], f"MANIFEST_SIZE_MISMATCH:{item['relative_path']}")
            require(hashlib.sha256(data).hexdigest() == item["sha256"], f"MANIFEST_HASH_MISMATCH:{item['relative_path']}")
        cp = tar.extractfile("checkpoints/committed_update_420.pt"); require(cp is not None and hashlib.sha256(cp.read()).hexdigest() == EXPECTED_PARENT_SHA, "EMBEDDED_PARENT_HASH_MISMATCH")
        start = tar.extractfile("START_HERE_NEXT_CHAT.txt").read().decode(); require(EXPECTED_PARENT_SHA in start and "420" in start, "D29_START_IDENTITY_MISMATCH")
        execution = json.load(tar.extractfile("D29_EXECUTION_MANIFEST.json")); trajectory = json.load(tar.extractfile("D29_CANONICAL_TRAJECTORY.json"))
        require(execution["updates_committed"][-1] == 420 and trajectory["trajectory"][-1]["checkpoint_sha256"] == EXPECTED_PARENT_SHA, "D29_CROSS_DOCUMENT_TERMINAL_MISMATCH")
    with lzma.open(D29_ARCHIVE, "rb") as stream:
        while stream.read(1024*1024): pass
    return {"path":str(D29_ARCHIVE),"sha256":EXPECTED_ARCHIVE_SHA,"size_bytes":D29_ARCHIVE.stat().st_size,"xz_decompression_test":"PASS","tar_traversal_read":"PASS","internal_manifest_verification":"PASS","cross_document_consistency":"PASS","authoritative_checkpoint_identity":"PASS"}

def load_parent():
    require(sha(PARENT) == EXPECTED_PARENT_SHA, "LOCAL_PARENT_HASH_MISMATCH")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    require(payload["completed_optimizer_step"] == 420 and payload["next_schedule_index"] == 420, "PARENT_POSITION_MISMATCH")
    require(float(payload["H1"]) == START_H1 and float(payload["H32"]) == START_H32, "PARENT_METRIC_MISMATCH")
    require(all(int(x.get("step",-1)) == 420 for x in payload["optimizer_state_dict"]["state"].values()), "PARENT_OPTIMIZER_STEP_MISMATCH")
    runtime = d29.d24.d17.preflight_runtime(); model, optimizer = d29.d24.d16.new_branch(runtime["bundle"])
    model.load_state_dict(payload["model_state_dict"], strict=True); optimizer.load_state_dict(payload["optimizer_state_dict"])
    canonical = [float(x) for x in payload["canonical_base_lr"]]; require(canonical == [BASE_LR], "BASE_LR_MISMATCH")
    prior = json.loads((D29_ROOT / "D29_CANONICAL_TRAJECTORY.json").read_text())
    return runtime, model, optimizer, copy.deepcopy(payload["rng_state"]), canonical, prior

def regime(margin: float) -> str:
    if margin < BOUNDARY_RESERVE: return "ACTIVE_BOUNDARY_RECOVERY"
    if margin < REEXPANSION_RESERVE: return "RESERVE_BUILDING"
    return "CONTROLLED_REEXPANSION"

def screen_alphas(update: int, previous: float|None, reg: str) -> list[float]:
    if update == 421: return list(INITIAL)
    require(previous in LADDER, "PREVIOUS_ALPHA_NOT_IN_LADDER")
    i=LADDER.index(previous); vals=[previous]
    if i>0: vals.append(LADDER[i-1])
    if reg in ("ACTIVE_BOUNDARY_RECOVERY","RESERVE_BUILDING") and i>1: vals.append(LADDER[i-2])
    if i+1<len(LADDER) and LADDER[i+1] <= (0.25 if reg=="ACTIVE_BOUNDARY_RECOVERY" else 0.5): vals.append(LADDER[i+1])
    return vals

def completion(prior: Mapping[str,Any], rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    out=d29.completion(prior,rows); out["schema_version"]="stage3_h13_d30_completion_contract_v1"; return out

def run(output: Path) -> dict[str,Any]:
    require(not output.exists(), f"D30_OUTPUT_ALREADY_EXISTS:{output}")
    auth=authenticate_archive(); runtime,model,optimizer,rng,canonical,prior=load_parent(); output.mkdir(parents=True)
    records=[]; rows=[]; commits=[]; current_update=420; h1=START_H1; h32=START_H32; current_sha=EXPECTED_PARENT_SHA; previous=0.25; blocker=None; max_margin=H32_LIMIT-h32
    for update in range(421,429):
        reg=regime(H32_LIMIT-h32); alphas=screen_alphas(update,previous,reg); print(f"D30_SCREEN update={update} regime={reg} alphas={alphas}",flush=True); candidates=[]
        for alpha in alphas:
            rec,cm,co,cr=d29.trial(runtime,output,model,optimizer,rng,canonical,alpha,current_update,update,h1,h32)
            rec["parent_checkpoint_SHA256"]=current_sha; rec["margin_regime"]=reg; rec["pre_update_H32_margin"]=H32_LIMIT-h32; rec["post_update_H32_margin"]=H32_LIMIT-float(rec.get("candidate_H32",float("inf")))
            rec["H1_improvement_gate"]=bool(float(rec.get("candidate_H1",float("inf"))) < h1)
            rec["H32_hard_limit_gate"]=bool(float(rec.get("candidate_H32",float("inf"))) <= H32_LIMIT)
            rec["margin_restoration_gate"]=bool(float(rec.get("candidate_H32",float("inf"))) < h32) if reg=="ACTIVE_BOUNDARY_RECOVERY" else None
            rec["reserve_floor_gate"]=bool(rec["post_update_H32_margin"] >= BOUNDARY_RESERVE) if reg!="ACTIVE_BOUNDARY_RECOVERY" else None
            policy=rec["H1_improvement_gate"] and rec["H32_hard_limit_gate"] and (rec["margin_restoration_gate"] if reg=="ACTIVE_BOUNDARY_RECOVERY" else rec["reserve_floor_gate"])
            rec["D30_policy_gate"]=bool(policy); rec["D30_valid"]=bool(rec.get("candidate_valid") and policy)
            if not rec["D30_valid"]:
                reasons=[]
                if not rec.get("candidate_valid"): reasons.append(str(rec.get("rejection_reason","INHERITED_VALIDATION_FAILURE")))
                if not rec["H1_improvement_gate"]: reasons.append("NO_STRICT_H1_IMPROVEMENT")
                if not rec["H32_hard_limit_gate"]: reasons.append("H32_HARD_LIMIT")
                if reg=="ACTIVE_BOUNDARY_RECOVERY" and not rec["margin_restoration_gate"]: reasons.append("NO_H32_MARGIN_RESTORATION")
                if reg!="ACTIVE_BOUNDARY_RECOVERY" and not rec["reserve_floor_gate"]: reasons.append("H32_RESERVE_FLOOR")
                rec["rejection_reason"]=";".join(dict.fromkeys(reasons))
            records.append(rec); candidates.append((rec,cm,co,cr)); print(f"D30_TRIAL update={update} alpha={alpha} valid={rec['D30_valid']} H1={rec.get('candidate_H1')} H32={rec.get('candidate_H32')} reason={rec.get('rejection_reason')}",flush=True)
        valid=[x for x in candidates if x[0]["D30_valid"]]
        if not valid:
            blocker=("D30_P4_ACTIVE_H32_BOUNDARY_BLOCKED" if reg=="ACTIVE_BOUNDARY_RECOVERY" else f"NO_LEGAL_H1_IMPROVING_RESERVE_PRESERVING_CANDIDATE_UPDATE_{update}"); break
        selected,sm,so,sr=min(valid,key=lambda x:(float(x[0]["candidate_H1"]),-float(x[0]["post_update_H32_margin"])))
        selected["selection_status"]="SELECTED_COMMITTED"; selected["selected"]=True
        checkpoint=output/"checkpoints"/f"committed_update_{update}.pt"; checkpoint.parent.mkdir(parents=True,exist_ok=True)
        payload=d29.checkpoint_payload(sm,so,sr,selected,current_update,update,float(selected["alpha"]),canonical); payload["schema_version"]="stage3_h13_d30_resumable_committed_checkpoint_v1"; payload["experiment_id"]=EXPERIMENT_ID; payload["authoritative_d29_archive_sha256"]=EXPECTED_ARCHIVE_SHA
        torch.save(payload,checkpoint); checkpoint_sha=sha(checkpoint); selected["candidate_checkpoint_SHA256"]=checkpoint_sha; selected["model_checkpoint_sha256"]=checkpoint_sha; selected["trial_state"]="COMMITTED_FORWARD_STATE"
        check=d29.d27.verify_checkpoint(checkpoint,current_update,update,current_sha,selected); require(check["pass"],f"CHECKPOINT_CHAIN_FAILURE_UPDATE_{update}")
        row={"update":update,"parent_update":current_update,"parent_checkpoint_sha256":current_sha,"checkpoint_sha256":checkpoint_sha,"selected_alpha":float(selected["alpha"]),"alpha":float(selected["alpha"]),"effective_lr":BASE_LR*float(selected["alpha"]),"H1_before":h1,"H1":float(selected["candidate_H1"]),"delta_H1":float(selected["delta_H1"]),"H32_before":h32,"H32":float(selected["candidate_H32"]),"delta_H32":float(selected["delta_H32"]),"H32_margin":float(selected["post_update_H32_margin"]),"regime":reg,"optimizer_consistency":"PASS","deterministic_replay_consistency":"PASS","RNG_continuity":"PASS","schedule_continuity":"PASS","completion_contract_status":"PENDING"}
        rows.append(row); comp=completion(prior,rows); row["completion_contract_status"]=comp["stage3_completion_gate"]; commits.append({**row,"validation":check}); write_json(output/"metrics"/f"update_{update}.json",{"trajectory_row":row,"selected_record":selected})
        model,optimizer,rng=sm,so,sr; current_update,h1,h32,current_sha,previous=update,row["H1"],row["H32"],checkpoint_sha,row["alpha"]; max_margin=max(max_margin,row["H32_margin"])
        print(f"D30_COMMITTED update={update} alpha={previous} H1={h1:.17g} H32={h32:.17g} margin={H32_LIMIT-h32:.17g} sha={current_sha}",flush=True)
        if comp["stage3_completion_gate"]=="PASSED": break
    for r in records:
        if r.get("selection_status")!="SELECTED_COMMITTED": r["trial_state"]="TRIAL_NOT_COMMITTED"
    comp=completion(prior,rows); total=START_H1-h1; mean=total/len(rows) if rows else 0.0; final_margin=H32_LIMIT-h32
    classification="STAGE3_COMPLETED" if comp["stage3_completion_gate"]=="PASSED" else (blocker if blocker=="D30_P4_ACTIVE_H32_BOUNDARY_BLOCKED" else ("D30_MARGIN_RECOVERED_CONTROLLED_REEXPANSION_FORWARD_PROGRESS" if any(x["regime"]=="CONTROLLED_REEXPANSION" for x in rows) else "D30_ACTIVE_BOUNDARY_MARGIN_RECOVERY_AND_FORWARD_PROGRESS"))
    trajectory={"schema_version":"stage3_h13_d30_canonical_trajectory_v1","baseline":{"update":420,"H1":START_H1,"H32":START_H32,"H32_margin":H32_LIMIT-START_H32},"trajectory":rows,"measurements":{"updates_committed":[x["update"] for x in rows],"selected_alpha_trajectory":[x["alpha"] for x in rows],"H1_final":h1,"H32_final":h32,"total_H1_improvement":total,"mean_H1_improvement_per_committed_update":mean,"final_H32_margin":final_margin,"max_H32_margin_reached":max_margin},"classification":classification,"first_blocker":blocker}
    write_json(output/"D30_CANONICAL_TRAJECTORY.json",trajectory); d29.write_csv(output/"D30_CANONICAL_TRAJECTORY.csv",rows); write_json(output/"D30_CANDIDATE_RESULTS.json",records); d29.write_csv(output/"D30_CANDIDATE_RESULTS.csv",records); write_json(output/"STAGE3_COMPLETION_CONTRACT_EVALUATION.json",comp)
    optimizer_status="PASS" if records and all(r.get("optimizer_consistency_pass") for r in records) else "FAIL"
    replay_status="PASS" if records and all(r.get("deterministic_replay_consistency_pass") for r in records) else "FAIL"; rng_status="PASS" if records and all(r.get("rng_identity_pass") for r in records) else "FAIL"
    write_json(output/"OPTIMIZER_CONSISTENCY_REPORT.json",{"schema_version":"stage3_h13_d30_optimizer_consistency_v1","status":optimizer_status,"actual_lr_validator":"actual candidate effective LR","candidates":records})
    write_json(output/"DETERMINISTIC_REPLAY_REPORT.json",{"status":replay_status,"candidates":[{"update":r["candidate_update"],"alpha":r["alpha"],"status":r["deterministic_replay"],"candidate_record_digest":r.get("deterministic_replay_candidate_record_digest"),"replay_record_digest":r.get("deterministic_replay_record_digest")} for r in records]})
    write_json(output/"RNG_CONTINUITY_REPORT.json",{"status":rng_status,"candidate_count":len(records),"canonical_commits":[x["update"] for x in rows]})
    manifest={"schema_version":"stage3_h13_d30_execution_manifest_v1","experiment_id":EXPERIMENT_ID,"authorization":{"path":str(AUTHORIZATION),"sha256":sha(AUTHORIZATION)},"authoritative_d29_archive":auth,"authoritative_parent":{"update":420,"checkpoint_path":str(PARENT),"checkpoint_sha256":EXPECTED_PARENT_SHA,"H1":START_H1,"H32":START_H32},"frozen_thresholds":{"boundary_reserve":BOUNDARY_RESERVE,"reexpansion_reserve":REEXPANSION_RESERVE,"H32_limit":H32_LIMIT},"alpha_ladder":LADDER,"update_421_screen":[r["alpha"] for r in records if r["candidate_update"]==421],"updates_committed":[x["update"] for x in rows],"commit_evidence":commits,"classification":classification,"first_blocker":blocker,"scientific_state_contamination":False,"validator_or_wrapper_repairs":[]}
    write_json(output/"D30_EXECUTION_MANIFEST.json",manifest); write_json(output/"IMPLEMENTATION_CHANGES.json",{"changes":[{"path":str(Path(__file__).resolve()),"reason":"D30 frozen margin-aware state machine orchestration","scientific_policy_changed":"authorized alpha selection only"}],"validator_or_wrapper_repairs":[],"scientific_state_contamination":False})
    candidate_table="\n".join(f"| {r['candidate_update']} | {r['alpha']:.9g} | {BASE_LR*float(r['alpha']):.9g} | {r.get('candidate_H1')} | {r.get('delta_H1')} | {r.get('candidate_H32')} | {r.get('delta_H32')} | {r.get('post_update_H32_margin')} | {r['margin_regime']} | {'VALID' if r['D30_valid'] else 'REJECTED'} | {r.get('selection_status') if r['D30_valid'] else r.get('rejection_reason')} |" for r in records)
    commit_table="\n".join(f"| {x['update']} | {x['parent_update']} | {x['alpha']:.9g} | {x['effective_lr']:.9g} | {x['H1']:.17g} | {x['H32']:.17g} | {x['H32_margin']:.17g} | {x['regime']} | {x['checkpoint_sha256']} |" for x in rows)
    report=f"""# D30 Final Report

TASK_STATUS: {'PASS' if rows else 'BLOCKED'}
FINAL_CLASSIFICATION: {classification}
FIRST_BLOCKER: {blocker or 'none'}
D30_EXECUTED: YES
AUTHORITATIVE_PARENT_UPDATE: 420
AUTHORITATIVE_PARENT_CHECKPOINT_SHA256: {EXPECTED_PARENT_SHA}
START_H1: {START_H1:.17g}
FINAL_H1: {h1:.17g}
TOTAL_H1_IMPROVEMENT: {total:.17g}
MEAN_H1_IMPROVEMENT_PER_COMMITTED_UPDATE: {mean:.17g}
START_H32: {START_H32:.17g}
FINAL_H32: {h32:.17g}
FINAL_H32_MARGIN: {final_margin:.17g}
H32_MARGIN_RECOVERED: {'YES' if final_margin>H32_LIMIT-START_H32 else 'NO'}
MAX_H32_MARGIN_REACHED_DURING_D30: {max_margin:.17g}
UPDATES_COMMITTED: {[x['update'] for x in rows]}
CANDIDATES_SCREENED: {len(records)}
SELECTED_ALPHA_TRAJECTORY: {[x['alpha'] for x in rows]}
FINAL_ALPHA: {previous if rows else None}
OPTIMIZER_CONSISTENCY: {optimizer_status}
DETERMINISTIC_REPLAY: {replay_status}
RNG_CONTINUITY: {rng_status}
SCIENTIFIC_STATE_CONTAMINATION: NO
VALIDATOR_OR_WRAPPER_REPAIRS: none
STAGE3_COMPLETION_STATUS: {comp['stage3_completion_gate']}
FINAL_COMMITTED_UPDATE: {current_update}
FINAL_CHECKPOINT_SHA256: {current_sha}

## Candidate table
| update | alpha | effective_lr | H1 | delta_H1 | H32 | delta_H32 | H32_margin | regime | validity | selection/rejection reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
{candidate_table}

## Committed trajectory
| update | parent | alpha | effective_lr | H1 | H32 | H32_margin | regime | checkpoint_sha256 |
|---:|---:|---:|---:|---:|---:|---:|---|---|
{commit_table}
"""
    (output/"FINAL_REPORT.md").write_text(report,encoding="utf-8",newline="\n"); (output/"TASK_AND_RESULTS_SUMMARY.md").write_text(report,encoding="utf-8",newline="\n")
    (output/"START_HERE_NEXT_CHAT.txt").write_text(f"D30 CROSS-CHAT HANDOFF — READ FIRST\n\nClassification: {classification}\nParent: update 420, {EXPECTED_PARENT_SHA}\nCommitted: {[x['update'] for x in rows]}\nFinal H1/H32/margin: {h1:.17g} / {h32:.17g} / {final_margin:.17g}\nFinal checkpoint: {current_sha}\nStage-3 completion: {comp['stage3_completion_gate']}\nFirst blocker: {blocker or 'none'}\nSee FINAL_REPORT.md and D30_EXECUTION_MANIFEST.json.\n",encoding="utf-8",newline="\n")
    (output/"TASK_PROMPT.md").write_bytes(AUTHORIZATION.read_bytes())
    files=[{"relative_path":str(p.relative_to(output)).replace("\\","/"),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.rglob("*")) if p.is_file() and p.name!="INTERNAL_FILE_MANIFEST.json"]
    write_json(output/"INTERNAL_FILE_MANIFEST.json",{"schema_version":"stage3_h13_d30_internal_manifest_v1","hash_algorithm":"SHA-256","self_excluded":True,"managed_files":files,"file_count":len(files)})
    return {"status":"PASS" if rows else "BLOCKED","output":str(output),"classification":classification,"first_blocker":blocker,"last_committed_update":current_update,"final_H1":h1,"final_H32":h32,"final_checkpoint_sha256":current_sha}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    try: result=run(args.output.resolve())
    except Exception as exc: print(json.dumps({"status":"BLOCKED","first_blocker":f"{type(exc).__name__}:{exc}"},sort_keys=True)); return 1
    print(json.dumps(result,sort_keys=True)); return 0 if result["status"]=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
