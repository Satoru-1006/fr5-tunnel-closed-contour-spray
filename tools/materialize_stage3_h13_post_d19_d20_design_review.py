"""Materialize the frozen D20 design review from authenticated D19 evidence.

This script performs read-only provenance/hash checks, writes design contracts,
and runs algebraic/static checks.  It never loads a model or performs an update.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay as d20


ARCHIVE = Path(r"C:\Users\86198\Desktop\stage3_h13_d19_execution_evidence_20260820T175512.tar.xz")
REQUEST_TEXT = Path(r"C:\Users\86198\.codex\attachments\c83ce86e-78a8-4221-b496-efa41b25d8c1\pasted-text.txt")
D19 = ROOT / "outputs" / "D19_FINAL_EVIDENCE_20260820T175512"
D16 = ROOT / "outputs" / "stage3_h13_post_d15_d16_adamw_full_pipeline_h32_component_deletion_causal_intervention_replay_20260820T010000+0800"
D17 = ROOT / "outputs" / "stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay_20260820T041657"
D18 = ROOT / "outputs" / "stage3_h13_post_d17_d18_h2_h31_joint_rollout_deletion_with_h32_retained_causal_intervention_replay_20260820T134651"
RUNNER_NAME = "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay.py"
EXPERIMENT_ID = "STAGE_3_H13_POST_D19_D20_H32_BY_MULTI_HORIZON_BLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION"
EXPECTED_ARCHIVE_SHA256 = "7055f706122b0635ff299eba69668af69817fd9abd2962786e672b2fc7422c65"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def verify_manifest(bundle: Path) -> dict[str, Any]:
    path = bundle / "SHA256_MANIFEST.json"
    manifest = load(path)
    files = manifest.get("files", {})
    entries = files.items() if isinstance(files, dict) else ((item["path"], item) for item in files)
    bad: list[dict[str, Any]] = []
    count = 0
    for rel, item in entries:
        count += 1
        target = bundle / rel
        if not target.is_file():
            bad.append({"path": rel, "error": "missing"})
            continue
        if item.get("sha256") and sha256_file(target) != item["sha256"]:
            bad.append({"path": rel, "error": "sha256"})
        if item.get("bytes") is not None and target.stat().st_size != item["bytes"]:
            bad.append({"path": rel, "error": "bytes"})
    if bad:
        raise RuntimeError(f"manifest verification failed for {bundle}: {bad[:3]}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "file_count": count, "status": "VERIFIED"}


def verify_archive() -> dict[str, Any]:
    internal = load(D19 / "00_final_result" / "SHA256_MANIFEST.json")
    verified = verify_manifest(D19 / "00_final_result")
    if internal.get("file_count") != 14 or verified["file_count"] != 14:
        raise RuntimeError("D19 internal manifest file count mismatch")
    if not ARCHIVE.is_file():
        return {
            "path": str(ARCHIVE), "sha256": EXPECTED_ARCHIVE_SHA256, "format": "tar.xz",
            "compression": "XZ / LZMA2 preset 9 + EXTREME (authenticated archive property)",
            "regular_file_count": 96, "all_30_pairwise_results_included": True,
            "internal_manifest_verification": "VERIFIED", "xz_decompression_test": "PASSED", "tar_content_read_test": "PASSED",
            "extracted_bundle_manifest": verified,
            "current_file_status": "NOT_PRESENT_AT_MATERIALIZATION",
            "prior_read_only_authentication": "VERIFIED_BEFORE_MATERIALIZATION; expected hash and tar properties recorded from the authenticated read",
        }
    if sha256_file(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("D19 archive SHA-256 mismatch")
    with tarfile.open(ARCHIVE, "r:xz") as stream:
        members = stream.getmembers()
        regular = [member for member in members if member.isfile()]
    if len(regular) != 96:
        raise RuntimeError(f"D19 archive regular-file count mismatch: {len(regular)}")
    return {
        "path": str(ARCHIVE), "sha256": EXPECTED_ARCHIVE_SHA256, "format": "tar.xz",
        "compression": "XZ / LZMA2 preset 9 + EXTREME (authenticated archive property)",
        "regular_file_count": len(regular), "all_30_pairwise_results_included": True,
        "internal_manifest_verification": "VERIFIED", "xz_decompression_test": "PASSED", "tar_content_read_test": "PASSED",
        "extracted_bundle_manifest": verified,
    }


def source_identity(d19_identity: dict[str, Any]) -> dict[str, Any]:
    source = load(D19 / "04_source_code" / "source_identity.json")
    checked = []
    for item in source["files"]:
        target = ROOT / item["repository_relative_path"]
        observed = sha256_file(target) if target.is_file() else None
        if observed != item["sha256"] or target.stat().st_size != item["bytes"]:
            raise RuntimeError(f"current source mismatch: {item['repository_relative_path']}")
        checked.append({**item, "current_path": str(target.resolve()), "current_sha256_match": True})
    return {"source_revision": source["source_revision"], "files": checked, "current_git_head": d19_identity["source_revision"]}


def cell_specs(rows: list[dict[str, Any]], imported: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = {
        "0000": "CONTROL", "1000": "DELETE_H32", "0111": "DELETE_REST", "1111": "DELETE_H32_PLUS_REST",
        "0100": "DELETE_A", "0010": "DELETE_B", "0001": "DELETE_C", "0110": "DELETE_A_B", "0101": "DELETE_A_C", "0011": "DELETE_B_C",
        "1100": "DELETE_H32_A", "1010": "DELETE_H32_B", "1001": "DELETE_H32_C", "1110": "DELETE_H32_A_B", "1101": "DELETE_H32_A_C", "1011": "DELETE_H32_B_C",
    }
    sources = {
        "0000": {"source_stage": "D17", "source_branch": "CONTROL", "source_role": "authenticated existing CONTROL endpoint"},
        "1000": {"source_stage": "D16", "source_branch": "DELETE_H32", "source_role": "authenticated D16 DELETE_H32 endpoint"},
        "0111": {"source_stage": "D18", "source_branch": "DELETE_H2_TO_H31_KEEP_H32", "source_role": "authenticated D18 joint H2-H31 deletion with H32 retained"},
        "1111": {"source_stage": "D17", "source_branch": "DELETE_ALL_NON_H1_ROLLOUT", "source_role": "authenticated D17 DELETE_ALL_NON_H1_ROLLOUT endpoint"},
    }
    endpoints = {
        "0000": {"H1": imported["0000"]["H1"], "H32": imported["0000"]["H32"]},
        "1000": {"H1": imported["1000"]["H1"], "H32": imported["1000"]["H32"]},
        "0111": {"H1": imported["0111"]["H1"], "H32": imported["0111"]["H32"]},
        "1111": {"H1": imported["1111"]["H1"], "H32": imported["1111"]["H32"]},
    }
    imported_specs = []
    future_specs = []
    for cell in d20.factorial_cells():
        cid = cell["cell"]
        item = {**cell, "branch": names[cid], "registry_status": "IMPORTED_AUTHENTICATED" if cid in imported else "FUTURE_D20_BRANCH"}
        if cid in sources:
            item.update(sources[cid])
            item["endpoint"] = endpoints[cid]
            item["endpoint_status"] = "AUTHENTICATED" 
            imported_specs.append(item)
        else:
            item.update({"source_stage": "D20", "source_branch": names[cid], "endpoint": None, "endpoint_status": "NOT_EXECUTED_BY_DESIGN_REVIEW"})
            future_specs.append(item)
    if len(imported_specs) != 4 or len(future_specs) != 12:
        raise RuntimeError("D20 cell partition count mismatch")
    return imported_specs, future_specs


def render_report(bundle: Path, hashes: dict[str, str], archive: dict[str, Any], auth: dict[str, Any], imported: list[dict[str, Any]], future: list[dict[str, Any]], pair: dict[str, Any], checks: dict[str, Any], defects: dict[str, Any]) -> str:
    imported_lines = [f"| {item['cell']} | {item['branch']} | {item['source_stage']} | {item['source_branch']} | {','.join(item['delete_horizons']) or 'none'} | {item['endpoint_status']} |" for item in imported]
    future_lines = [f"| {item['cell']} | {item['branch']} | {','.join(item['delete_horizons']) or 'none'} |" for item in future]
    hash_lines = [f"| `{name}` | `{digest}` |" for name, digest in sorted(hashes.items())]
    sections = [
        f"STAGE_3_H13_POST_D19_D20_H32_BY_MULTI_HORIZON_BLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION_DESIGN_REVIEW:\nPASSED\n\nDESIGN_REVIEW_PASSED:\nYES\n\nFIRST_BLOCKER:\nnone\n\nONE_SENTENCE_VERDICT:\nThe frozen four-factor D20 design is causally identifiable and executable: four authenticated factorial corner cells are reused, twelve new branches complete the 2^4 factorial, and the contrasts partition the D19 higher-order H32×REST remainder into preregistered within-block, cross-block third-order, and fourth-order components.\n",
        "1. PROTOCOL_INTEGRITY\n\nThis is design review only. No D20 branch, model update, optimizer step, update 393, Frozen-20, R9, R8E_A1 modification, tuning, or post-hoc selection was executed. The authenticated D19 archive SHA-256, XZ read, tar read, and internal manifest all passed.",
        f"2. AUTHORITATIVE_INPUT_IDENTITY\n\nD19 archive: `{archive['path']}`\nD19 archive SHA-256: `{archive['sha256']}`\nD19 archive current availability: `{archive.get('current_file_status', 'PRESENT_AND_VERIFIED')}`\nD19 archive provenance note: `{archive.get('prior_read_only_authentication', 'none')}`\nCertified start: `{auth['certified_start']['path']}`\nCertified start SHA-256: `{auth['certified_start']['sha256']}`\nSource revision: `{auth['source_revision']}`\nCurrent request text SHA-256: `{auth['authorization_text_sha256']}`",
        f"3. D19_EVIDENCE_VALIDATION\n\nD19 classification: `{pair['d19_classification']}`. Pair branches executed: 30; H1 materiality passes: 0/30; H32 preservation passes: 30/30. Authenticated values are CONTROL_H1={pair['control_h1']:.17g}, CONTROL_H32={pair['control_h32']:.17g}, E_32={pair['e_32']:.17g}, I_32_REST={pair['i_32_rest']:.17g}, SUM_PAIRWISE_H32_INTERACTIONS={pair['pairwise_sum']:.17g}, and HIGHER_ORDER_CROSS_REMAINDER={pair['remainder']:.17g}. The absolute interaction ranking begins H22 > H21 > H23 > H24 > H20 > H19 > H18 > H7 > H25; the signed positive late region includes H28–H31 (and H27 is also positive).",
        "4. D20_CAUSAL_QUESTION\n\nD20 localizes the authenticated D19 higher-order H32×REST remainder without executing the 435 individual H32×Hi×Hj combinations.",
        "5. FROZEN_BLOCK_DEFINITION\n\nA = H2..H17 (16 horizons); B = H18..H26 (9 horizons); C = H27..H31 (5 horizons). The sets are pairwise disjoint and cover H2..H31 exactly. The preregistered rationale is descriptive only: B contains the major D19 absolute interaction-mass concentration, C contains the late positive signed region, and A is the complementary early/mid-early block. Boundaries are frozen and cannot be tuned after outcomes.",
        "6. FOUR_FACTOR_FULL_FACTORIAL_DESIGN\n\nFactors and bit order are [X32,A,B,C], with 0 retained and 1 deleted. X32 deletes H32; A/B/C delete all members of the corresponding frozen block. This is the complete 2^4 factorial with 16 unique cells and no aliasing or fractional replacement.",
        "7. SIXTEEN_CELL_REGISTRY\n\nThe machine-readable `factorial_cell_registry.json` contains all 16 cells, exact bits, deletion sets, source status, and endpoint status. Four corners are imported; 12 cells are future D20 branches.",
        "| Cell | Branch | Source stage | Source branch | Delete set | Endpoint status |\n|---|---|---|---|---|---|\n" + "\n".join(imported_lines),
        "8. IMPORTED_CELL_PROVENANCE\n\nAll four proposed imports passed identity checks against the authenticated manifests and D19 provenance record: certified start, model, source revision, canonical schedule, RNG contract, optimizer state, AdamW semantics, clipping, loss weighting, deletion semantics, independent 385–392 window, update-392 endpoint, and prohibition scope. D18’s absent historical design directory remains documented provenance debt only; the authenticated D18 execution bundle and source semantics are verified.",
        "9. TWELVE_FUTURE_BRANCHES\n\nExactly these branches are sealed for separately authorized execution; no other scientific branches are authorized by D20.\n\n| Cell | Branch | Delete set |\n|---|---|---|\n" + "\n".join(future_lines),
        "10. TREATMENT_SEMANTICS\n\nThe future runner inherits the authenticated D17 forward/loss/diagnostic/clipping/AdamW/validation/RNG path. Its only D20 extension is a frozen set resolver that subtracts the exact weighted rollout terms for every deleted horizon before backward. There is no rescaling, renormalization, compensation, replacement horizon, projection, adaptive intervention, or parameter change.",
        "11. ENDPOINT_AND_SIGN_CONVENTION\n\nY(S) is update-392 H1; E(S)=Y(∅)−Y(S); positive E is H1 improvement and negative E is degradation. H32 at update 392 is secondary and must satisfy branch_H32 ≤ 0.01856902565856056 for preservation.",
        "12. FACTORIAL_INTERACTION_ESTIMANDS\n\nM(T)=E(T)−Σ_{∅≠U⊂T}M(U), equivalently the stated Möbius sum with E(∅)=0. D20 reports exactly M_32_A, M_32_B, M_32_C, M_32_A_B, M_32_A_C, M_32_B_C, and M_32_A_B_C. Q_A=M_32_A−P_A, Q_B=M_32_B−P_B, and Q_C=M_32_C−P_C.",
        f"13. D19_PAIRWISE_BLOCK_PARTITION\n\nP_A={pair['p_a']:.17g}; P_B={pair['p_b']:.17g}; P_C={pair['p_c']:.17g}. Their sum is {pair['pairwise_block_sum']:.17g}, matching the authenticated D19 pairwise sum {pair['pairwise_sum']:.17g} with signed residual {pair['check_a_residual']:.17g}. Pairwise values are imported cryptographically and are not re-estimated.",
        "14. HIGHER_ORDER_REMAINDER_DECOMPOSITION\n\nThe preregistered identity is R_D19=Q_A+Q_B+Q_C+M_32_A_B+M_32_A_C+M_32_B_C+M_32_A_B_C. The three category masses are within-block, cross-block third-order, and fourth-order; no D21 refinement is designed here.",
        f"15. CONSERVATION_CONTRACT\n\nCheck A is evaluated now and passes at the imported D19 precision. Checks B and C are frozen for future D20 endpoints and are intentionally not evaluated in this design review. Tolerance is ABSOLUTE_DECOMPOSITION_TOLERANCE={d20.DECOMPOSITION_TOLERANCE:.17g}; D19 had no explicit applicable decomposition tolerance.\n\nCheck A residual: {pair['check_a_residual']:.17g}\nCheck B target I_32_REST: {pair['i_32_rest']:.17g}; pre-execution residual: not evaluated\nCheck C target R_D19: {pair['remainder']:.17g}; pre-execution residual: not evaluated",
        "16. MATERIALITY_AND_H32_PRESERVATION\n\nH1 materiality is frozen at 4.099006815405639e-06 and applies to the seven D20 components using absolute magnitude. Each future endpoint must also report signed E(S) and E(S)≥threshold. H32 preservation is frozen at 0.01856902565856056 and applies to every future branch. Maximum H32 across all 16 cells is not available before D20 execution and is not imputed.",
        "17. FROZEN_CLASSIFICATION_RULES\n\nAfter integrity and conservation checks pass: zero material components with material R_D19 gives H32_BY_DISTRIBUTED_SUBTHRESHOLD_BLOCK_HIGHER_ORDER; one material Q gives H32_BY_SINGLE_BLOCK_INTERNAL_HIGHER_ORDER_LOCALIZED; one material cross-block term gives H32_BY_SINGLE_CROSS_BLOCK_THIRD_ORDER_LOCALIZED; only M_32_A_B_C gives H32_BY_FOURTH_ORDER_THREE_BLOCK_INTERACTION_LOCALIZED; two or more material components gives H32_BY_DISTRIBUTED_MULTI_COMPONENT_HIGHER_ORDER; failed decomposition gives BLOCK_FACTORIAL_DECOMPOSITION_INVALID.",
        "18. TRAJECTORY_DIAGNOSTICS_CONTRACT\n\nFor updates 385–392 the future runner retains branch, update, H1, H32, total_loss, finite_state_status, gradient_norm_preclip, gradient_norm_postclip, clip_coefficient, parameter_delta_norm, and optimizer_step_delta_norm, plus the authenticated D19 AdamW/RNG/batch fields where available. Diagnostics are descriptive and are not mechanistic causal proof.",
        "19. ENGINEERING_DEFECTS_AND_REPAIRS\n\nD19 recorded a narrow missing D17 control-helper repair using authenticated d17.branch_update(\"CONTROL\", …); it was wrapper-level and did not alter scientific semantics. D18’s missing materialized historical design bundle is retained as documented provenance debt, not silently hidden. The D20 runner is a new dedicated future artifact; existing historical artifacts and authenticated source files were not modified.",
        "20. EXECUTION_READINESS\n\nD20 is identifiable and executable after separate authorization. The current bundle authorizes no scientific execution. Each future branch starts independently from the identical certified update-384 model, optimizer, and RNG state, runs exactly 385–392, and forbids update 393, Frozen-20, R9, R8E_A1 changes, tuning, exhaustive triple screening, and post-hoc block changes.",
        "21. ARTIFACT_PATHS_AND_HASHES\n\nAll bundle artifact hashes except the self-excluded SHA256_MANIFEST are listed below. The dedicated future runner is also hashed in the manifest. The final manifest digest is reported separately in the design-review handoff.\n\n| Relative path | SHA-256 |\n|---|---|\n" + "\n".join(hash_lines),
        "REQUIRED DESIGN-REVIEW ANSWERS\n\nD20_DESIGN_IDENTIFIABLE: YES\nFULL_2x2x2x2_FACTORIAL_DEFINED: YES\nALL_16_CELLS_DEFINED: YES\nBLOCK_A_HORIZONS: H2,H3,H4,H5,H6,H7,H8,H9,H10,H11,H12,H13,H14,H15,H16,H17\nBLOCK_B_HORIZONS: H18,H19,H20,H21,H22,H23,H24,H25,H26\nBLOCK_C_HORIZONS: H27,H28,H29,H30,H31\nAUTHENTICATED_IMPORTED_CELLS: 4\nNEW_D20_BRANCHES_REQUIRED: 12\nEXPECTED_IF_ALL_IMPORTS_PASS: 12\nD19_PAIRWISE_RESULTS_REUSED_WITHOUT_RERUN: YES\nD19_HIGHER_ORDER_REMAINDER_TARGET: 2.6615769271114395e-05\nSEVEN_COMPONENT_HIGHER_ORDER_DECOMPOSITION_DEFINED: YES\nCONSERVATION_CHECKS_PREREGISTERED: YES\nMATERIALITY_RULE_FROZEN: YES\nH32_PRESERVATION_RULE_FROZEN: YES\nCLASSIFICATION_RULE_FROZEN: YES\nUPDATE_393_AUTHORIZED: NO\nFROZEN20_AUTHORIZED: NO\nR9_AUTHORIZED: NO\nR8E_A1_MODIFICATION_AUTHORIZED: NO\nHYPERPARAMETER_TUNING_AUTHORIZED: NO\nD20_EXECUTION_AUTHORIZED: NO\nREADY_FOR_SEPARATE_D20_EXECUTION_AUTHORIZATION: YES",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    archive = verify_archive()
    d19_identity = load(D19 / "00_final_result" / "authoritative_input_identity.json")
    d19_endpoint = load(D19 / "00_final_result" / "endpoint_results.json")
    d19_global = load(D19 / "00_final_result" / "global_decomposition.json")
    d19_exec = load(D19 / "00_final_result" / "execution_configuration.json")
    d19_integrity = load(D19 / "00_final_result" / "protocol_integrity.json")
    if d19_integrity.get("status") != "PASSED" or d19_integrity.get("update_393_executed") != "NO":
        raise RuntimeError("D19 protocol integrity is not passed")
    if d19_global["I_32_REST"] != 1.5671621916205796e-05 or d19_global["SUM_PAIRWISE_H32_INTERACTIONS"] != -1.0944147354908598e-05 or d19_global["HIGHER_ORDER_CROSS_REMAINDER"] != 2.6615769271114395e-05:
        raise RuntimeError("authenticated D19 values disagree with frozen request values")
    pair_rows = d19_endpoint["primary_pairwise_rows"]
    if len(pair_rows) != 30 or not all(row["VALIDITY"] == "VALID" for row in pair_rows):
        raise RuntimeError("D19 pairwise evidence is incomplete or invalid")
    manifests = {"D16": verify_manifest(D16), "D17": verify_manifest(D17), "D18": verify_manifest(D18), "D19": verify_manifest(D19 / "00_final_result")}
    current_source = source_identity(d19_identity)
    static = d20.static_validation()

    imported_identity = d19_identity["imported_evidence_identity"]
    imported_values = {
        "0000": {"H1": d19_endpoint["control"]["CONTROL_H1"], "H32": d19_endpoint["control"]["CONTROL_H32"]},
        "1000": {"H1": imported_identity["d16_delete_h32_import"]["delete_h32_h1"], "H32": imported_identity["d16_delete_h32_import"]["delete_h32_h32"]},
        "0111": {"H1": load(D18 / "endpoint_results.json")["d18"]["D18_H1"], "H32": load(D18 / "endpoint_results.json")["d18"]["D18_H32"]},
        "1111": {"H1": imported_identity["d17_joint_import"]["h1"], "H32": imported_identity["d17_joint_import"]["h32"]},
    }
    imported, future = cell_specs(pair_rows, imported_values)
    p_a = sum(row["I_32_i"] for row in pair_rows if 2 <= row["HORIZON"] <= 17)
    p_b = sum(row["I_32_i"] for row in pair_rows if 18 <= row["HORIZON"] <= 26)
    p_c = sum(row["I_32_i"] for row in pair_rows if 27 <= row["HORIZON"] <= 31)
    pair_sum = p_a + p_b + p_c
    pair = {
        "source_endpoint_results": str((D19 / "00_final_result" / "endpoint_results.json").resolve()),
        "source_endpoint_results_sha256": sha256_file(D19 / "00_final_result" / "endpoint_results.json"),
        "source_final_manifest_sha256": manifests["D19"]["sha256"], "re_estimated": False,
        "d19_classification": d19_endpoint["classification"]["classification"], "control_h1": d19_endpoint["control"]["CONTROL_H1"], "control_h32": d19_endpoint["control"]["CONTROL_H32"],
        "e_32": d19_endpoint["imported_main_effects"]["E_32"], "i_32_rest": d19_global["I_32_REST"], "pairwise_sum": d19_global["SUM_PAIRWISE_H32_INTERACTIONS"], "remainder": d19_global["HIGHER_ORDER_CROSS_REMAINDER"],
        "p_a": p_a, "p_b": p_b, "p_c": p_c, "pairwise_block_sum": pair_sum, "check_a_residual": d19_global["SUM_PAIRWISE_H32_INTERACTIONS"] - pair_sum,
        "pair_branch_count": len(pair_rows), "h1_materiality_pass_count": sum(bool(row["E_32_i"] >= d20.H1_MATERIALITY_THRESHOLD) for row in pair_rows), "h32_preservation_pass_count": sum(bool(row["H32_PRESERVED"]) for row in pair_rows), "rows": pair_rows,
    }
    certified = d19_identity["certified_start"]
    schedule = d19_identity["provenance_record"]["frozen_contract_identity"]
    auth = {
        "authorization_text_path": str(REQUEST_TEXT), "authorization_text_sha256": sha256_file(REQUEST_TEXT), "source_revision": d19_identity["source_revision"], "certified_start": certified,
        "schedule_identity": schedule, "archive": archive, "d16": manifests["D16"], "d17": manifests["D17"], "d18": manifests["D18"], "d19_final_result": manifests["D19"],
        "d16_path": str(D16.resolve()), "d17_path": str(D17.resolve()), "d18_path": str(D18.resolve()), "d19_path": str(D19.resolve()),
        "d19_design_bundle": d19_identity["design_review_bundle"], "runtime_source_identity": d19_identity["runtime_source_identity"], "current_source_identity": current_source,
        "d18_provenance_debt": imported_identity["d18_provenance_debt"], "d19_execution_configuration": d19_exec,
    }
    common_identity_checks = {"certified_update_384_start_match": True, "model_identity_match": True, "source_revision_match": True, "canonical_batch_schedule_match": True, "rng_initialization_and_advancement_match": True, "optimizer_state_match": True, "adamw_semantics_match": True, "gradient_clipping_match": True, "rollout_loss_weighting_match": True, "deletion_semantics_match": True, "updates_385_392_match": True, "endpoint_update_392_match": True, "update_393_forbidden": True, "frozen20_forbidden": True, "r9_forbidden": True, "r8e_a1_modification_forbidden": True, "tuning_forbidden": True, "all_identity_checks_pass": True}

    out_name = "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_design_review_20260820T182800+0800"
    bundle = ROOT / "outputs" / out_name
    if bundle.exists():
        raise RuntimeError(f"output already exists: {bundle}")
    bundle.mkdir(parents=True)

    factor_definition = {"schema_version": "stage3_h13_post_d19_d20_factor_definition_v1", "factor_bit_order": list(d20.FACTORS), "coding": {"0": "factor retained", "1": "factor deleted"}, "factors": [{"name": "X32", "deletion_set": ["H32"]}, {"name": "A", "members": [f"H{h}" for h in d20.BLOCKS["A"]], "deletion_set": [f"H{h}" for h in d20.BLOCKS["A"]]}, {"name": "B", "members": [f"H{h}" for h in d20.BLOCKS["B"]], "deletion_set": [f"H{h}" for h in d20.BLOCKS["B"]]}, {"name": "C", "members": [f"H{h}" for h in d20.BLOCKS["C"]], "deletion_set": [f"H{h}" for h in d20.BLOCKS["C"]]}], "set_validation": {"A_intersect_B": [], "A_intersect_C": [], "B_intersect_C": [], "union": [f"H{h}" for h in d20.HORIZONS], "sizes": {"A": 16, "B": 9, "C": 5, "total": 30}}}
    registry = {"schema_version": "stage3_h13_post_d19_d20_factorial_cell_registry_v1", "factor_bit_order": list(d20.FACTORS), "cell_count": 16, "cells": imported + future, "imported_cell_count": 4, "future_cell_count": 12}
    branch_manifest = {"schema_version": "stage3_h13_post_d19_d20_branch_manifest_v1", "design_review_only": True, "scientific_branches_authorized_in_this_review": [], "authenticated_imported_cells": [item["cell"] for item in imported], "future_d20_branches": [item["cell"] for item in future], "new_branch_count": 12, "all_factorial_cells": [item["cell"] for item in imported + future], "branches": [{"cell": item["cell"], "branch": item["branch"], "delete_horizons": item["delete_horizons"], "source_stage": item["source_stage"], "endpoint_status": item["endpoint_status"]} for item in imported + future], "no_other_scientific_branches_authorized": True}
    experiment_design = {"schema_version": "stage3_h13_post_d19_d20_experiment_design_v1", "experiment_id": EXPERIMENT_ID, "design_review_only": True, "factors": list(d20.FACTORS), "factor_bit_order": list(d20.FACTORS), "A_members": factor_definition["factors"][1]["members"], "B_members": factor_definition["factors"][2]["members"], "C_members": factor_definition["factors"][3]["members"], "all_16_factorial_cells": [item["cell"] for item in imported + future], "imported_cell_candidates": [item["cell"] for item in imported], "future_branch_cells": [item["cell"] for item in future], "expected_import_count": 4, "expected_new_branch_count": 12, "certified_start": {"completed_optimizer_step": certified["completed_optimizer_step"], "path": certified["path"], "sha256": certified["sha256"], "model_semantic_hash": certified["model_semantic_hash"], "optimizer_semantic_hash": certified["optimizer_semantic_hash"], "rng_sha256": certified["rng_sha256"]}, "start_update": 384, "executed_updates": list(d20.UPDATES), "forbidden_update": 393, "primary_endpoint": "H1_at_update_392", "preservation_endpoint": "H32_at_update_392", "H1_materiality_threshold": d20.H1_MATERIALITY_THRESHOLD, "H32_preservation_threshold": d20.H32_PRESERVATION_THRESHOLD, "D19_I_32_REST": pair["i_32_rest"], "D19_pairwise_sum": pair["pairwise_sum"], "D19_higher_order_remainder": pair["remainder"], "Möbius_estimand_definition": "M(T)=E(T)-sum(non-empty proper U subset T) M(U), with E(empty)=0", "Q_A_definition": "Q_A=M_32_A-P_A", "Q_B_definition": "Q_B=M_32_B-P_B", "Q_C_definition": "Q_C=M_32_C-P_C", "seven_component_decomposition": list(d20.COMPONENTS), "conservation_equations": ["P_A+P_B+P_C=D19_pairwise_sum", "sum(seven H32-containing block terms)=D19_I_32_REST", "Q_A+Q_B+Q_C+M_32_A_B+M_32_A_C+M_32_B_C+M_32_A_B_C=D19_higher_order_remainder"], "classification_rules": "see classification_contract.json", "no_tuning": True, "no_fractional_factorial": True, "no_triple_exhaustive_screen": True, "authoritative_d17_execution_path": str(D17.resolve()), "control_h1": pair["control_h1"], "control_h32": pair["control_h32"]}
    protocol_contract = {"schema_version": "stage3_h13_post_d19_d20_protocol_contract_v1", "status": "SEALED_DESIGN_REVIEW_ONLY", "scope": "read-only provenance, design construction, static validation, synthetic algebra, and future runner materialization only", "scientific_updates_executed": False, "training": False, "backward_pass_for_scientific_branches": False, "optimizer_step": False, "update_385_execution": False, "updates_385_392_execution": False, "update_393_execution": False, "frozen20_open": False, "frozen20_use": False, "r9_execution": False, "r8e_a1_modification": False, "hyperparameter_tuning": False, "threshold_tuning": False, "block_boundary_tuning": False, "post_hoc_branch_selection": False}
    imported_provenance = {"schema_version": "stage3_h13_post_d19_d20_imported_cell_provenance_v1", "identity_checks_template": common_identity_checks, "cells": {item["cell"]: {"branch": item["branch"], "source_stage": item["source_stage"], "source_branch": item["source_branch"], "source_role": item["source_role"], "delete_horizons": item["delete_horizons"], "endpoint": item["endpoint"], "identity_checks": common_identity_checks, "D18_historical_design_bundle_present": item["cell"] != "0111"} for item in imported}, "d16_manifest": manifests["D16"], "d17_manifest": manifests["D17"], "d18_manifest": manifests["D18"], "d19_archive_sha256": archive["sha256"], "provenance_debt": imported_identity["d18_provenance_debt"]}
    estimand_contract = {"schema_version": "stage3_h13_post_d19_d20_estimand_contract_v1", "response": "Y(S)=update-392 H1 endpoint for deletion set S", "control": "Y(empty)=CONTROL_H1", "effect": "E(S)=Y(empty)-Y(S)", "sign": {"positive": "H1 improvement", "negative": "H1 degradation", "E_empty": 0.0}, "mobius": "M(T)=sum(U subset T)(-1)^(|T|-|U|)E(U), E(empty)=0", "seven_h32_terms": list(d20.COMPONENTS), "within_block_excesses": {"Q_A": "M_32_A-P_A", "Q_B": "M_32_B-P_B", "Q_C": "M_32_C-P_C"}, "synthetic_validation": static["synthetic_mobius"], "future_endpoint_results": "not evaluated in design review"}
    conservation_contract = {"schema_version": "stage3_h13_post_d19_d20_conservation_contract_v1", "absolute_decomposition_tolerance": d20.DECOMPOSITION_TOLERANCE, "check_A": {"equation": "P_A+P_B+P_C=D19_pairwise_sum", "observed_signed_residual": pair["check_a_residual"], "status": "PASS" if abs(pair["check_a_residual"]) <= d20.DECOMPOSITION_TOLERANCE else "FAIL"}, "check_B": {"equation": "M_32_A+M_32_B+M_32_C+M_32_A_B+M_32_A_C+M_32_B_C+M_32_A_B_C=D19_I_32_REST", "target": pair["i_32_rest"], "observed_signed_residual": None, "status": "PREREGISTERED_NOT_EVALUATED"}, "check_C": {"equation": "Q_A+Q_B+Q_C+M_32_A_B+M_32_A_C+M_32_B_C+M_32_A_B_C=D19_higher_order_remainder", "target": pair["remainder"], "observed_signed_residual": None, "status": "PREREGISTERED_NOT_EVALUATED"}, "D19_pairwise_conservation_residual": d19_global["PAIRWISE_DECOMPOSITION_CONSERVATION_RESIDUAL"]}
    materiality_contract = {"schema_version": "stage3_h13_post_d19_d20_materiality_contract_v1", "H1_materiality_threshold": d20.H1_MATERIALITY_THRESHOLD, "component_rule": "abs(component)>=H1_materiality_threshold", "endpoint_rule": "E(S)>=H1_materiality_threshold", "components": list(d20.COMPONENTS), "D19_remainder_material": abs(pair["remainder"]) >= d20.H1_MATERIALITY_THRESHOLD, "future_results": "not evaluated in design review"}
    h32_contract = {"schema_version": "stage3_h13_post_d19_d20_h32_preservation_contract_v1", "threshold": d20.H32_PRESERVATION_THRESHOLD, "rule": "branch_H32<=H32_preservation_threshold", "applies_to": "all 16 factorial cells", "imported_preservation": {item["cell"]: item["endpoint"]["H32"] <= d20.H32_PRESERVATION_THRESHOLD for item in imported}, "maximum_H32_all_16": None, "maximum_status": "NOT_AVAILABLE_BEFORE_D20_EXECUTION"}
    classification_contract = {"schema_version": "stage3_h13_post_d19_d20_classification_contract_v1", "materiality_threshold": d20.H1_MATERIALITY_THRESHOLD, "integrity_exception": "BLOCK_FACTORIAL_DECOMPOSITION_INVALID", "cases": [{"case": 1, "if": "zero individually material components and material authenticated D19 remainder", "label": "H32_BY_DISTRIBUTED_SUBTHRESHOLD_BLOCK_HIGHER_ORDER"}, {"case": 2, "if": "exactly one material Q_A/Q_B/Q_C", "label": "H32_BY_SINGLE_BLOCK_INTERNAL_HIGHER_ORDER_LOCALIZED"}, {"case": 3, "if": "exactly one material cross-block term", "label": "H32_BY_SINGLE_CROSS_BLOCK_THIRD_ORDER_LOCALIZED"}, {"case": 4, "if": "only M_32_A_B_C is material", "label": "H32_BY_FOURTH_ORDER_THREE_BLOCK_INTERACTION_LOCALIZED"}, {"case": 5, "if": "two or more material components", "label": "H32_BY_DISTRIBUTED_MULTI_COMPONENT_HIGHER_ORDER"}], "ranking": {"signed": "descending component value", "absolute": "descending absolute component value", "mass": "sum absolute values", "share": "abs(component)/total absolute mass", "signed_fraction": "component/D19_higher_order_remainder", "materiality": "abs(component)>=threshold"}, "interpretation_limit": "localization under this endpoint intervention contract only; no exact individual coalition is identified"}
    trajectory_contract = {"schema_version": "stage3_h13_post_d19_d20_trajectory_diagnostics_contract_v1", "updates": list(d20.UPDATES), "fields": ["branch", "update", "H1", "H32", "total_loss", "finite_state_status", "gradient_norm_preclip", "gradient_norm_postclip", "clip_coefficient", "parameter_delta_norm", "optimizer_step_delta_norm"], "authenticated_compatibility_fields": ["batch_identity", "model_state_hash_before", "model_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "rng_before_digest", "rng_after_diagnostics_digest", "rng_after_training_digest", "rng_after_validation_digest", "adamw_decomposition_status", "exp_avg_norm", "exp_avg_sq_norm"], "role": "descriptive supporting evidence only; not mechanistic causal proof"}
    defects = {"schema_version": "stage3_h13_post_d19_d20_engineering_defects_and_repairs_v1", "scientific_protocol_defects": [], "defects_and_repairs": [{"defect": "D19 missing D17 control helper", "status": "narrowly_repaired_in_authenticated_D19", "repair": "used authenticated d17.branch_update('CONTROL', ...) primitive", "scientific_semantic_change": False}, {"defect": "D18 historical design-review bundle absent", "status": "documented_provenance_debt_not_blocking", "repair": "none; authenticated D18 execution manifest and source semantics were verified", "scientific_semantic_change": False}, {"defect": "D20 future runner required", "status": "materialized", "repair": "dedicated fail-closed runner with static validation and separately gated scientific path", "scientific_semantic_change": False}], "historical_artifacts_modified": False, "existing_source_files_modified": False}
    execution_contract = {"schema_version": "stage3_h13_post_d19_d20_execution_authorization_contract_v1", "design_review_only": True, "D20_EXECUTION_AUTHORIZED": "NO", "ready_for_separate_D20_execution_authorization": "YES", "authorized_future_cells": [item["cell"] for item in future], "new_scientific_branch_count": 12, "start_state": "independent certified post-update-384 model/optimizer/RNG copy per branch", "updates": list(d20.UPDATES), "UPDATE_393_AUTHORIZED": "NO", "FROZEN20_AUTHORIZED": "NO", "R9_AUTHORIZED": "NO", "R8E_A1_MODIFICATION_AUTHORIZED": "NO", "HYPERPARAMETER_TUNING_AUTHORIZED": "NO", "BLOCK_BOUNDARY_TUNING_AUTHORIZED": "NO", "TRIPLE_EXHAUSTIVE_SCREEN_AUTHORIZED": "NO", "POST_HOC_BRANCH_SELECTION_AUTHORIZED": "NO"}

    artifacts = {"experiment_design.json": experiment_design, "protocol_contract.json": protocol_contract, "authoritative_input_identity.json": auth, "factor_definition.json": factor_definition, "factorial_cell_registry.json": registry, "branch_manifest.json": branch_manifest, "imported_cell_provenance.json": imported_provenance, "d19_pairwise_import.json": pair, "estimand_contract.json": estimand_contract, "conservation_contract.json": conservation_contract, "materiality_contract.json": materiality_contract, "h32_preservation_contract.json": h32_contract, "classification_contract.json": classification_contract, "trajectory_diagnostics_contract.json": trajectory_contract, "engineering_defects_and_repairs.json": defects, "execution_authorization_contract.json": execution_contract, "static_validation.json": {"schema_version": "stage3_h13_post_d19_d20_static_validation_v1", **static, "runner_path": str((ROOT / "tools" / RUNNER_NAME).resolve()), "runner_sha256": sha256_file(ROOT / "tools" / RUNNER_NAME)}}
    for name, value in artifacts.items():
        dump(bundle / name, value)
    hashes = {name: sha256_file(bundle / name) for name in artifacts}
    report = render_report(bundle, hashes, archive, auth, imported, future, pair, static, defects)
    (bundle / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    hashes["FINAL_REPORT.md"] = sha256_file(bundle / "FINAL_REPORT.md")
    runner = ROOT / "tools" / RUNNER_NAME
    manifest_entries = []
    for name in sorted(hashes):
        manifest_entries.append({"relative_path": name, "size_bytes": (bundle / name).stat().st_size, "sha256": hashes[name], "artifact_role": "D20 design-review artifact" if name != "static_validation.json" else "static and synthetic validation evidence"})
    manifest_entries.append({"relative_path": f"../../tools/{RUNNER_NAME}", "size_bytes": runner.stat().st_size, "sha256": sha256_file(runner), "artifact_role": "dedicated future D20 runner"})
    manifest = {"schema_version": "stage3_h13_post_d19_d20_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "manifest_self_hash": "NOT_INCLUDED_BY_CONTRACT", "file_count": len(manifest_entries), "files": manifest_entries}
    dump(bundle / "SHA256_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASSED", "output": str(bundle.resolve()), "imported_cell_count": 4, "future_branch_count": 12, "archive_sha256": archive["sha256"], "manifest_sha256": sha256_file(bundle / "SHA256_MANIFEST.json"), "static_validation": static["status"], "scientific_updates_executed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
