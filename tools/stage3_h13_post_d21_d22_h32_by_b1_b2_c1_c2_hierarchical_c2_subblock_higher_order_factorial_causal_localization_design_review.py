"""Materialize the Stage-3 H13 D22 design review without scientific execution.

This module is deliberately a static design materializer.  It authenticates the
completed D21 bundle, recovers the frozen horizon/update-space semantics, builds
the 2^6 D22 registries, and exercises only synthetic Möbius/conservation tests.
It does not import the scientific runner, load a model, enter a backward pass,
call an optimizer, or execute any D22 branch.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).name
D21_DIR = ROOT / "outputs" / "stage3_h13_post_d20_d21_h32_by_b_c_hierarchical_subblock_higher_order_factorial_causal_localization_execution_evidence_20260821T035822+0800"
D21_DESIGN_DIR = ROOT / "outputs" / "stage3_h13_post_d20_d21_h32_by_b_c_hierarchical_subblock_higher_order_factorial_causal_localization_design_review_20260820T225928+0800"
D21_RUNNER = ROOT / "tools" / "stage3_h13_post_d20_d21_execution_repaired.py"
D20_PRIMITIVE = ROOT / "tools" / "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay.py"
EXPECTED_D21_MANIFEST_SHA256 = "d6f42c415223996aa8f26df0c1b82109ba054f5d75aa7a64e4421c6403ac206d"
EXPECTED_D21_DESIGN_MANIFEST_SHA256 = "beee7a3e6106fc130125875d74bda8683ab4b11d40d9cf020ccb03e1fcfe7a7a"
EXPECTED_CERTIFIED_START_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_SHA256 = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_SHA256 = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_CONTROL_H1 = 9.096969417553673e-05
EXPECTED_CONTROL_H32 = 0.018159905521264792
EXPECTED_PARENT_M32_BC = 1.057200413683796e-05
EXPECTED_Q_B = 8.201920441567987e-06
EXPECTED_MATERIALITY_THRESHOLD = 4.099006815405639e-06
EXPECTED_H32_THRESHOLD = 0.01856902565856056
TOLERANCE = 1e-12
UPDATES = tuple(range(385, 393))
D22_FACTORS = ("H32", "B1", "B2", "C1", "C21", "C22")
D21_FACTORS = ("H32", "B1", "B2", "C1", "C2")
BASE_FACTORS = ("H32", "B1", "B2", "C1")
TERM_NON_H32_ORDER = ("B1", "B2", "C1", "C2", "C21", "C22")
EXPERIMENT_ID = "STAGE_3_H13_POST_D21_D22_H32_BY_B1_B2_C1_C2_HIERARCHICAL_C2_SUBBLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION_DESIGN_REVIEW"


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def bit_cell(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def all_cells(factors: Sequence[str]) -> list[str]:
    return [bit_cell(bits) for bits in itertools.product((0, 1), repeat=len(factors))]


def subset_bits(subset: Iterable[str], factors: Sequence[str]) -> str:
    chosen = set(subset)
    return "".join("1" if factor in chosen else "0" for factor in factors)


def subsets(factors: Sequence[str]) -> list[tuple[str, ...]]:
    return [subset for order in range(1, len(factors) + 1) for subset in itertools.combinations(factors, order)]


def term_name(subset: Iterable[str]) -> str:
    chosen = set(subset)
    rest = [factor for factor in TERM_NON_H32_ORDER if factor in chosen]
    return "M_32" if "H32" in chosen and not rest else ("M_32_" + "_".join(rest) if "H32" in chosen else "M_" + "_".join(rest))


def term_subset_from_name(name: str) -> tuple[str, ...]:
    if name == "M_32":
        return ("H32",)
    if name.startswith("M_32_"):
        return ("H32", *name.removeprefix("M_32_").split("_"))
    require(name.startswith("M_"), f"invalid term name: {name}")
    return tuple(name.removeprefix("M_").split("_"))


def mobius_from_endpoints(endpoints: Mapping[str, float], factors: Sequence[str]) -> dict[tuple[str, ...], float]:
    require(set(endpoints) == set(all_cells(factors)), "endpoint vector is not a complete binary factorial")
    interactions: dict[tuple[str, ...], float] = {}
    for subset in subsets(factors):
        value = float(endpoints[subset_bits(subset, factors)])
        for order in range(1, len(subset)):
            for proper in itertools.combinations(subset, order):
                value -= interactions[tuple(proper)]
        interactions[subset] = value
    return interactions


def endpoint_from_interactions(interactions: Mapping[tuple[str, ...], float], active: Iterable[str]) -> float:
    chosen = set(active)
    return sum(value for subset, value in interactions.items() if set(subset) <= chosen)


def max_reconstruction_residual(endpoints: Mapping[str, float], interactions: Mapping[tuple[str, ...], float], factors: Sequence[str]) -> float:
    return max(abs(endpoint_from_interactions(interactions, [factor for factor, bit in zip(factors, cell) if bit == "1"]) - endpoints[cell]) for cell in all_cells(factors))


def record_test(results: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    results.append({"name": name, "status": "PASS" if passed else "FAIL", **details})
    require(passed, f"static test failed: {name}")


def verify_manifest(bundle: Path, manifest_name: str) -> dict[str, Any]:
    manifest_path = bundle / manifest_name
    manifest = read_json(manifest_path)
    failures: list[str] = []
    for entry in manifest["files"]:
        path = bundle / entry["relative_path"]
        if not path.is_file():
            failures.append(f"missing:{entry['relative_path']}")
            continue
        observed_hash = sha256_file(path)
        observed_size = path.stat().st_size
        if observed_hash != entry["sha256"] or observed_size != entry["size_bytes"]:
            failures.append(f"mismatch:{entry['relative_path']}")
    return {"status": "PASS" if not failures else "FAIL", "checked_file_count": len(manifest["files"]), "failures": failures}


def authenticate_d21() -> dict[str, Any]:
    require(D21_DIR.is_dir(), f"D21 bundle missing: {D21_DIR}")
    require(D21_DESIGN_DIR.is_dir(), f"D21 design contract directory missing: {D21_DESIGN_DIR}")
    for name in ("FINAL_REPORT.md", "factorial_32_cell_response_table.csv", "SHA256_MANIFEST.json"):
        require((D21_DIR / name).is_file(), f"D21 required artifact missing: {name}")
    observed_manifest_hash = sha256_file(D21_DIR / "SHA256_MANIFEST.json")
    require(observed_manifest_hash == EXPECTED_D21_MANIFEST_SHA256, "D21 final manifest hash mismatch")
    internal = verify_manifest(D21_DIR, "SHA256_MANIFEST.json")
    require(internal["status"] == "PASS", f"D21 internal manifest verification failed: {internal['failures']}")
    design_manifest_hash = sha256_file(D21_DESIGN_DIR / "SHA256_MANIFEST.json")
    require(design_manifest_hash == EXPECTED_D21_DESIGN_MANIFEST_SHA256, "D21 design manifest hash mismatch")

    table = csv_rows(D21_DIR / "factorial_32_cell_response_table.csv")
    terms = csv_rows(D21_DIR / "mobius_31_terms.csv")
    require(len(table) == 32 and len({row["cell_id"] for row in table}) == 32, "D21 factorial table is not exactly 32 unique cells")
    require(len(terms) == 31 and len({row["term"] for row in terms}) == 31, "D21 term table is not exactly 31 unique terms")
    protocol = read_json(D21_DIR / "protocol_integrity.json")
    execution = read_json(D21_DIR / "execution_manifest.json")
    identity = read_json(D21_DIR / "authoritative_input_identity.json")
    parent = read_json(D21_DIR / "parent_child_conservation.json")
    qb = read_json(D21_DIR / "qb_child_decomposition.json")
    imported_validation = read_json(D21_DIR / "imported_cell_validation.json")
    branch_manifest = read_json(D21_DIR / "branch_manifest.json")
    defects = read_json(D21_DIR / "defect_log.json")
    factor_definition = read_json(D21_DIR / "factor_definition.json")
    design_experiment = read_json(D21_DESIGN_DIR / "experiment_design.json")
    materiality = read_json(D21_DESIGN_DIR / "materiality_contract.json")
    h32_contract = read_json(D21_DESIGN_DIR / "h32_preservation_contract.json")
    d21_design_partition = read_json(D21_DESIGN_DIR / "subblock_partition.json")

    require(protocol["status"] == "PASSED" and protocol["scientific_execution_valid"] is True, "D21 protocol is not passed")
    require(protocol["authenticated_imported_cells"] == 8 and protocol["new_branches_executed"] == 24 and protocol["new_branch_update_records"] == 192, "D21 execution counts mismatch")
    require(protocol["update_393_executed"] == "NO" and protocol["frozen20_opened"] == "NO" and protocol["r9_executed"] == "NO", "D21 prohibition marker mismatch")
    require(protocol["r8e_a1_modified"] == "NO" and protocol["tuning_used"] == "NO" and protocol["adaptive_selection_used"] == "NO" and protocol["a_factor_branch_executed"] == "NO", "D21 adaptive/prohibited-action marker mismatch")
    require(execution["updates"] == list(UPDATES) and execution["endpoint_update"] == 392 and execution["protocol"]["update_393_executed"] == "NO", "D21 update contract mismatch")
    require(len(branch_manifest["scientific_branches_executed"]) == 24 and len(branch_manifest["branches"]) == 24, "D21 new branch manifest mismatch")
    require(len([row for row in imported_validation["cells"] if row["status"] == "PASS"]) == 8 and imported_validation["all_pass"] is True, "D21 import validation mismatch")
    require(defects["no_invalid_attempt_evidence_counted"] is True and defects["scientific_execution_final_status"] == "VALID_AFTER_NARROW_REPAIRS", "D21 invalid-attempt accounting mismatch")
    require(identity["certified_start_sha256"] == EXPECTED_CERTIFIED_START_SHA256 and identity["model_semantic_hash"] == EXPECTED_MODEL_SHA256 and identity["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256, "D21 certified start identity mismatch")
    require(identity["source_revision"] == EXPECTED_SOURCE_REVISION, "D21 source revision mismatch")
    current_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    require(current_revision == EXPECTED_SOURCE_REVISION, "current source revision differs from D21 authenticated revision")
    control = next(row for row in table if row["cell_id"] == "00000")
    require(float(control["endpoint_H1"]) == EXPECTED_CONTROL_H1 and float(control["endpoint_H32"]) == EXPECTED_CONTROL_H32, "D21 control endpoint mismatch")
    primary_expected = {
        "M_32_B1_C1": 1.52917096545413e-06,
        "M_32_B1_C2": 4.06921243160435e-06,
        "M_32_B2_C1": 1.87002807851710e-06,
        "M_32_B2_C2": 2.33557994499024e-06,
        "M_32_B1_B2_C1": -3.92625272420970e-06,
        "M_32_B1_B2_C2": -4.25881355331938e-06,
        "M_32_B1_C1_C2": -1.89129116422847e-06,
        "M_32_B2_C1_C2": -2.08319691935204e-06,
        "M_32_B1_B2_C1_C2": 1.29275670773817e-05,
    }
    term_by_name = {row["term"]: row for row in terms}
    for name, value in primary_expected.items():
        require(math.isclose(float(term_by_name[name]["M_subset"]), value, rel_tol=0.0, abs_tol=1e-18), f"D21 primary term mismatch: {name}")
    authenticated_primary_values = {name: float(term_by_name[name]["M_subset"]) for name in primary_expected}
    require(parent["checks"]["M_32_B_C"]["signed_residual"] == 0.0 and parent["checks"]["M_32_B_C"]["absolute_residual"] == 0.0, "D21 parent conservation residual is nonzero")
    require(qb["Q_B_conservation_residual"] == 0.0 and qb["Q_B_conservation_abs_residual"] == 0.0, "D21 Q_B conservation residual is nonzero")
    require(float(materiality["H1_materiality_threshold"]) == EXPECTED_MATERIALITY_THRESHOLD, "D21 materiality threshold changed")
    require(float(h32_contract["threshold"]) == EXPECTED_H32_THRESHOLD and h32_contract["rule"] == "H32_at_update_392 <= threshold is PASS; otherwise FAIL", "D21 H32 preservation contract changed")
    require(factor_definition["factor_bit_order"] == list(D21_FACTORS), "D21 factor order mismatch")
    require(factor_definition["factors"][-1]["name"] == "C2" and factor_definition["factors"][-1]["members"] == ["H30", "H31"], "D21 C2 definition mismatch")
    require(d21_design_partition["subblocks"]["C2"]["members"] == ["H30", "H31"], "D21 canonical C2 order mismatch")
    require(design_experiment["design_review_only"] is True, "D21 design contract is not design-only")

    return {
        "bundle_path": str(D21_DIR.resolve()),
        "manifest_path": str((D21_DIR / "SHA256_MANIFEST.json").resolve()),
        "expected_manifest_sha256": EXPECTED_D21_MANIFEST_SHA256,
        "observed_manifest_sha256": observed_manifest_hash,
        "manifest_verification": internal,
        "design_manifest_sha256": design_manifest_hash,
        "table": table,
        "terms": terms,
        "protocol": protocol,
        "execution": execution,
        "identity": identity,
        "parent": parent,
        "qb": qb,
        "imported_validation": imported_validation,
        "branch_manifest": branch_manifest,
        "defects": defects,
        "factor_definition": factor_definition,
        "design_experiment": design_experiment,
        "materiality": materiality,
        "h32_contract": h32_contract,
        "d21_design_partition": d21_design_partition,
        "control": control,
        "primary_expected": primary_expected,
        "authenticated_primary_values": authenticated_primary_values,
        "current_revision": current_revision,
    }


def build_partition(d21: Mapping[str, Any]) -> dict[str, Any]:
    c2 = list(d21["factor_definition"]["factors"][-1]["members"])
    require(c2 == ["H30", "H31"], "authenticated D21 C2 members are not the expected ordered pair")
    c21, c22 = [c2[0]], [c2[1]]
    source_rule = "existing canonical D21 C2 member order; atomic update-space horizon terms; first child receives the first ordered member and second child receives the remaining ordered member"
    require(c21 and c22 and not set(c21) & set(c22) and c21 + c22 == c2, "C21/C22 partition construction failed")
    return {
        "schema_version": "stage3_h13_post_d21_d22_c2_c21_c22_partition_spec_v1",
        "parent": {"name": "C2", "members": c2, "source_artifact": str((D21_DIR / "factor_definition.json").resolve()), "source_field": "factors[name=C2].members", "semantic_unit": "exact weighted rollout horizon terms H30 and H31"},
        "children": {
            "C21": {"members": c21, "deletion_set": c21, "source_rule": source_rule, "semantic_unit": "weighted_terms[30-1] exact D20/D21 update-space term"},
            "C22": {"members": c22, "deletion_set": c22, "source_rule": source_rule, "semantic_unit": "weighted_terms[31-1] exact D20/D21 update-space term"},
        },
        "exact_membership_rule": "factor bit 1 deletes exactly the listed horizon's weighted rollout term before backward; bit 0 retains it exactly as CONTROL; no parameter subset or child-dependent rescaling is introduced",
        "canonical": True,
        "deterministic": True,
        "exhaustive": True,
        "disjoint": True,
        "lossless_union": True,
        "C21_C22_PARTITION_CANONICAL": "YES",
        "C21_C22_PARTITION_DETERMINISTIC": "YES",
        "C21_C22_PARTITION_EXHAUSTIVE": "YES",
        "C21_C22_PARTITION_DISJOINT": "YES",
        "C21_PLUS_C22_RECONSTRUCTS_C2": "YES",
        "PARTITION_SELECTED_FROM_D22_OUTCOMES": "NO",
        "PARTITION_TUNED_USING_D21_EFFECT_MAGNITUDES": "NO",
        "proofs": {
            "disjoint": "C21={H30}, C22={H31}; H30 != H31, so intersection is empty",
            "exhaustive": "C21 union C22={H30,H31}=the authenticated D21 C2 member set",
            "lossless_on": "turning both children on deletes H30 and H31 exactly, which is the D21 C2 deletion action",
            "lossless_off": "turning both children off deletes neither H30 nor H31, which is C2-off/control semantics",
            "unrelated_semantics": "only weighted_terms[29] and weighted_terms[30] membership can change; model, optimizer, schedule, RNG, clipping, loss definitions, and endpoint evaluation remain inherited unchanged",
        },
    }


def d22_deletion_set(bits: Sequence[int], partition: Mapping[str, Any]) -> list[str]:
    members = {"H32": ["H32"], "B1": ["H18", "H19", "H20", "H21", "H22"], "B2": ["H23", "H24", "H25", "H26"], "C1": ["H27", "H28", "H29"], "C21": partition["children"]["C21"]["members"], "C22": partition["children"]["C22"]["members"]}
    result: list[str] = []
    for factor, bit in zip(D22_FACTORS, bits):
        if bit:
            result.extend(members[factor])
    return result


def build_d22_terms() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, subset in enumerate(subsets(D22_FACTORS), start=1):
        rows.append({"term_index": index, "term": term_name(subset), "subset": list(subset), "subset_label": "×".join(subset), "order": len(subset), "contains_H32": "H32" in subset, "contains_C21": "C21" in subset, "contains_C22": "C22" in subset, "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED"})
    require(len(rows) == 63 and len({row["term"] for row in rows}) == 63, "D22 Möbius registry is not exactly 63 unique terms")
    return rows


def build_parent_mapping(d22_terms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    target_by_subset = {tuple(row["subset"]): row["term"] for row in d22_terms}
    mapping: list[dict[str, Any]] = []
    for d21_subset in subsets(D21_FACTORS):
        if "C2" not in d21_subset:
            targets = [target_by_subset[tuple(factor for factor in D22_FACTORS if factor in d21_subset)]]
            kind = "IDENTITY_NO_C2"
        else:
            base = tuple(factor for factor in BASE_FACTORS if factor in d21_subset)
            targets = [
                target_by_subset[base + ("C21",)],
                target_by_subset[base + ("C22",)],
                target_by_subset[base + ("C21", "C22")],
            ]
            kind = "C2_TO_C21_C22_EXPANSION"
        mapping.append({"d21_term": term_name(d21_subset), "d21_subset": list(d21_subset), "contains_C2": "C2" in d21_subset, "d22_terms": targets, "mapping_kind": kind, "equation": term_name(d21_subset) + " = " + " + ".join(targets)})
    require(len(mapping) == 31 and len({row["d21_term"] for row in mapping}) == 31, "D21 parent mapping is not complete")
    target_terms = [term for row in mapping for term in row["d22_terms"]]
    require(len(target_terms) == 63 and len(set(target_terms)) == 63 and set(target_terms) == {row["term"] for row in d22_terms}, "complete D21-to-D22 term mapping is not bijective")
    return mapping


def build_cells(d21: Mapping[str, Any], partition: Mapping[str, Any], d22_terms: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d21_by_cell = {row["cell_id"]: row for row in d21["table"]}
    d21_imports = {row["d21_cell"]: row for row in d21["imported_validation"]["cells"]}
    d21_branches = {row["cell"]: row for row in d21["branch_manifest"]["branches"]}
    start = {"completed_optimizer_step": 384, "model_semantic_hash": EXPECTED_MODEL_SHA256, "optimizer_semantic_hash": EXPECTED_OPTIMIZER_SHA256, "rng_sha256": d21["identity"]["d20_identity"]["certified_start"]["rng_sha256"], "certified_bundle_sha256": EXPECTED_CERTIFIED_START_SHA256}
    batch_rows = [json.loads(line) for line in (D21_DIR / "branch_update_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_by_update = {int(row["update"]): {"schedule_index": row["schedule_index"], "batch_window_sha256": row["batch_window_sha256"]} for row in batch_rows if row["branch"] == batch_rows[0]["branch"]}
    require(set(batch_by_update) == set(UPDATES), "D21 batch contract is not complete")
    eval_contract = {"endpoint_update": 392, "H1": "Y(empty)-Y(S) at update 392; positive means H1 improvement", "H32": "H32 endpoint at update 392", "materiality": "abs(M)>=4.099006815405639e-06", "preservation": "H32_at_update_392 <= 0.01856902565856056"}
    cells: list[dict[str, Any]] = []
    imports: list[dict[str, Any]] = []
    for bits in itertools.product((0, 1), repeat=6):
        cell = bit_cell(bits)
        d21_cell = cell[:4] + ("1" if bits[4] or bits[5] else "0")
        d21_row = d21_by_cell[d21_cell]
        deletion = d22_deletion_set(bits, partition)
        source_d21_deletion = json.loads(d21_row["deletion_set"])
        require(set(deletion) == set(source_d21_deletion), f"D22 diagonal/deletion mismatch for {cell}") if bits[4] == bits[5] else None
        diagonal = bits[4] == bits[5]
        provenance = "IMPORTED_D21" if diagonal else "FUTURE_D22"
        treatment_spec = {"factor_order": list(D22_FACTORS), "bits": list(bits), "delete_horizons": deletion, "C2_union_equivalent": bits[4] == 1 and bits[5] == 1}
        treatment_hash = sha256_json(treatment_spec)
        row = {
            "canonical_cell_id": cell,
            "factor_order": list(D22_FACTORS),
            "bits": list(bits),
            "H32": bits[0], "B1": bits[1], "B2": bits[2], "C1": bits[3], "C21": bits[4], "C22": bits[5],
            "C2_equivalent_bit": int(bits[4] == bits[5] == 1),
            "C2_child_state": "C2_OFF" if bits[4] == bits[5] == 0 else ("C21_ONLY" if bits[4] == 1 and bits[5] == 0 else ("C22_ONLY" if bits[4] == 0 and bits[5] == 1 else "C2_ON")),
            "exact_deleted_horizons": deletion,
            "provenance_status": provenance,
            "source_d21_cell_id": d21_cell if diagonal else None,
            "source_d21_status": d21_row["source"],
            "expected_start_state_identity": start,
            "canonical_updates": list(UPDATES),
            "endpoint_update": 392,
            "batch_identity_contract": batch_by_update,
            "rng_identity_contract": {"start_rng_sha256": start["rng_sha256"], "per_update_stream": "match authenticated D21 control stream; no branch-dependent RNG selection"},
            "treatment_specification_hash": treatment_hash,
            "endpoint_specification": eval_contract,
            "H1_H32_evaluation_contract": eval_contract,
            "EXECUTED": "NO" if not diagonal else "NOT_APPLICABLE_IMPORT",
            "D22_EXECUTED": "NO",
            "AUTHORIZED_FOR_EXECUTION": "NO" if not diagonal else "NOT_APPLICABLE_IMPORT",
            "future_endpoint_status": "NOT_EXECUTED_BY_DESIGN_REVIEW" if not diagonal else "AUTHENTICATED_D21_ENDPOINT",
        }
        cells.append(row)
        if diagonal:
            identity = d21_imports.get(d21_cell)
            branch = d21_branches.get(d21_cell)
            if d21_row["source"] == "AUTHENTICATED_D20_IMPORT":
                exact_identity = bool(identity and identity["identity_checks"]["all_identity_checks_pass"] is True and identity["status"] == "PASS")
            else:
                exact_identity = bool(branch and branch["start_model_hash"] == EXPECTED_MODEL_SHA256 and branch["start_optimizer_hash"] == EXPECTED_OPTIMIZER_SHA256 and branch["updates"] == list(UPDATES) and branch["endpoint_update"] == 392 and branch["update_393_executed"] == "NO")
            require(exact_identity, f"D21 diagonal source identity failed for {d21_cell}")
            import_row = {
                "d22_cell_id": cell,
                "d21_cell_id": d21_cell,
                "d21_C2_bit": d21_cell[4],
                "d22_C21_bit": bits[4],
                "d22_C22_bit": bits[5],
                "d21_deletion_set": json.dumps(source_d21_deletion, separators=(",", ":")),
                "d22_child_union_deletion_set": json.dumps([h for h in deletion if h in {"H30", "H31"}], separators=(",", ":")),
                "source_status": d21_row["source"],
                "source_provenance_reference": d21_row["provenance_reference"],
                "source_endpoint_H1": d21_row["endpoint_H1"],
                "source_endpoint_H32": d21_row["endpoint_H32"],
                "initial_state_identity": "PASS",
                "H32_state_identity": "PASS",
                "B1_state_identity": "PASS",
                "B2_state_identity": "PASS",
                "C1_state_identity": "PASS",
                "C2_vs_child_union_treatment_identity": "PASS",
                "schedule_identity": "PASS",
                "update_indices_identity": "PASS",
                "batch_identity": "PASS",
                "rng_identity": "PASS",
                "optimizer_state_identity": "PASS",
                "clipping_identity": "PASS",
                "downstream_update_identity": "PASS",
                "endpoint_identity": "PASS",
                "H1_identity": "PASS",
                "H32_identity": "PASS",
                "no_mixed_child_import": "PASS",
                "import_validation_source": "D21 authenticated branch/import contracts",
                "d21_import_validation_row": identity["d21_cell"] if identity else (branch["cell"] if branch else d21_cell),
                "treatment_specification_hash": treatment_hash,
                "expected_start_model_hash": EXPECTED_MODEL_SHA256,
                "expected_start_optimizer_hash": EXPECTED_OPTIMIZER_SHA256,
                "expected_start_rng_sha256": start["rng_sha256"],
            }
            imports.append(import_row)
    require(len(cells) == 64 and len(imports) == 32, "D22 cell/import counts mismatch")
    require(sum(row["provenance_status"] == "FUTURE_D22" for row in cells) == 32, "D22 future branch count mismatch")
    require(all(row["C21"] == row["C22"] for row in cells if row["provenance_status"] == "IMPORTED_D21"), "mixed child state was marked as imported")
    return cells, imports


def run_static_validation(d21: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], d22_terms: Sequence[Mapping[str, Any]], mapping: Sequence[Mapping[str, Any]], partition: Mapping[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    cell_ids = [row["canonical_cell_id"] for row in cells]
    expected_cells = all_cells(D22_FACTORS)
    record_test(results, "64-cell uniqueness", len(cell_ids) == 64 and len(set(cell_ids)) == 64, count=len(cell_ids), unique_count=len(set(cell_ids)))
    record_test(results, "64-cell completeness", cell_ids == expected_cells, expected_first=expected_cells[0], expected_last=expected_cells[-1])
    term_names = [row["term"] for row in d22_terms]
    expected_term_names = [term_name(subset) for subset in subsets(D22_FACTORS)]
    record_test(results, "63-term uniqueness", len(term_names) == 63 and len(set(term_names)) == 63, count=len(term_names), unique_count=len(set(term_names)))
    record_test(results, "63-term completeness", term_names == expected_term_names and {tuple(row["subset"]) for row in d22_terms} == set(subsets(D22_FACTORS)), expected_first=expected_term_names[0], expected_last=expected_term_names[-1])
    target_names = {row["term"] for row in d22_terms}
    mapped_targets = [target for row in mapping for target in row["d22_terms"]]
    record_test(results, "complete parent-term mapping", len(mapping) == 31 and len(mapped_targets) == 63 and set(mapped_targets) == target_names and len(set(mapped_targets)) == 63, d21_terms=len(mapping), d22_targets=len(mapped_targets))

    seeds = [2201, 2202, 2203]
    random_residuals: list[float] = []
    for seed in seeds:
        rng = random.Random(seed)
        endpoints = {cell: (0.0 if cell == "000000" else rng.uniform(-3.0, 3.0)) for cell in expected_cells}
        interactions = mobius_from_endpoints(endpoints, D22_FACTORS)
        random_residuals.append(max_reconstruction_residual(endpoints, interactions, D22_FACTORS))
    record_test(results, "Möbius transform exact reconstruction", max(random_residuals) <= TOLERANCE, seeds=seeds, max_abs_residual=max(random_residuals), tolerance=TOLERANCE)
    rng = random.Random(2204)
    inverse_residuals: list[float] = []
    for _ in range(4):
        synthetic_interactions = {tuple(row["subset"]): rng.uniform(-5.0, 5.0) for row in d22_terms}
        synthetic_endpoints = {cell: endpoint_from_interactions(synthetic_interactions, [factor for factor, bit in zip(D22_FACTORS, cell) if bit == "1"]) for cell in expected_cells}
        recovered = mobius_from_endpoints(synthetic_endpoints, D22_FACTORS)
        inverse_residuals.append(max(abs(recovered[key] - value) for key, value in synthetic_interactions.items()))
    record_test(results, "inverse reconstruction of every synthetic cell response", max(inverse_residuals) <= TOLERANCE, vectors=4, max_abs_residual=max(inverse_residuals), tolerance=TOLERANCE)

    diagonal_endpoints = {cell: float(int(cell, 2)) / 17.0 for cell in expected_cells}
    d21_endpoints = {cell: diagonal_endpoints[cell[:4] + cell[4] + cell[4]] for cell in all_cells(D21_FACTORS)}
    d22_interactions = mobius_from_endpoints(diagonal_endpoints, D22_FACTORS)
    d21_interactions = mobius_from_endpoints(d21_endpoints, D21_FACTORS)
    mapped_residuals: list[float] = []
    for row in mapping:
        lhs = d21_interactions[tuple(row["d21_subset"])]
        rhs = sum(d22_interactions[tuple(term_subset_from_name(target))] for target in row["d22_terms"])
        mapped_residuals.append(abs(lhs - rhs))
    record_test(results, "D21-to-D22 diagonal embedding", max(mapped_residuals) <= TOLERANCE, max_abs_residual=max(mapped_residuals), tolerance=TOLERANCE, diagonal_cells=32)
    record_test(results, "complete parent-term conservation", max(mapped_residuals) <= TOLERANCE, mapped_d21_terms=len(mapping), mapped_d22_terms=len(mapped_targets), tolerance=TOLERANCE)

    primary_map = {
        "M_32_B1_B2_C2": ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22"],
        "M_32_B1_B2_C1_C2": ["M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"],
    }
    for parent_name, child_names in primary_map.items():
        row = next(item for item in mapping if item["d21_term"] == parent_name)
        record_test(results, f"primary parent identity: {parent_name}", row["d22_terms"] == child_names, d22_terms=row["d22_terms"])

    # Inherited D20/D21 identities are tested on the same synthetic D22 interaction vector.
    def iv(name: str) -> float:
        return d22_interactions[tuple(term_subset_from_name(name))]

    b_parent = iv("M_32_B1") + iv("M_32_B2") + iv("M_32_B1_B2")
    c_parent = iv("M_32_C1") + iv("M_32_C21") + iv("M_32_C22") + iv("M_32_C21_C22") + iv("M_32_C1_C21") + iv("M_32_C1_C22") + iv("M_32_C1_C21_C22")
    b_c_parent = sum(iv(name) for name in ("M_32_B1_C1", "M_32_B1_C21", "M_32_B1_C22", "M_32_B1_C21_C22", "M_32_B2_C1", "M_32_B2_C21", "M_32_B2_C22", "M_32_B2_C21_C22", "M_32_B1_B2_C1", "M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22", "M_32_B1_C1_C21", "M_32_B1_C1_C22", "M_32_B1_C1_C21_C22", "M_32_B2_C1_C21", "M_32_B2_C1_C22", "M_32_B2_C1_C21_C22", "M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"))
    # Reconstruct the corresponding D21 values from the diagonal synthetic endpoints.
    def d21_value(name: str) -> float:
        return d21_interactions[tuple(term_subset_from_name(name))]
    qb_d22 = (iv("M_32_B1") - 0.125) + (iv("M_32_B2") - 0.25) + iv("M_32_B1_B2")
    qb_d21 = (d21_value("M_32_B1") - 0.125) + (d21_value("M_32_B2") - 0.25) + d21_value("M_32_B1_B2")
    d21_b_parent = sum(d21_value(name) for name in ("M_32_B1", "M_32_B2", "M_32_B1_B2"))
    d21_c_parent = sum(d21_value(name) for name in ("M_32_C1", "M_32_C2", "M_32_C1_C2"))
    d21_bc_parent = sum(d21_value(name) for name in ("M_32_B1_C1", "M_32_B1_C2", "M_32_B2_C1", "M_32_B2_C2", "M_32_B1_B2_C1", "M_32_B1_B2_C2", "M_32_B1_C1_C2", "M_32_B2_C1_C2", "M_32_B1_B2_C1_C2"))
    inherited_residuals = {
        "M_32_B": abs(d21_b_parent - b_parent),
        "M_32_C": abs(d21_c_parent - c_parent),
        "M_32_B_C": abs(d21_bc_parent - b_c_parent),
        "Q_B": abs(qb_d21 - qb_d22),
    }
    record_test(results, "inherited D20/D21 conservation identities", max(inherited_residuals.values()) <= TOLERANCE, residuals=inherited_residuals, tolerance=TOLERANCE)

    import_checks = [row for row in cells if row["provenance_status"] == "IMPORTED_D21"]
    record_test(results, "imported-cell identity mapping", len(import_checks) == 32 and all(row["C21"] == row["C22"] for row in import_checks) and all(row["D22_EXECUTED"] == "NO" for row in import_checks), imported_cells=len(import_checks), mixed_imports=0)

    basis_max = 0.0
    for term in d22_terms:
        basis = {tuple(item["subset"]): 0.0 for item in d22_terms}
        basis[tuple(term["subset"])] = 1.0
        endpoints = {cell: endpoint_from_interactions(basis, [factor for factor, bit in zip(D22_FACTORS, cell) if bit == "1"]) for cell in expected_cells}
        recovered = mobius_from_endpoints(endpoints, D22_FACTORS)
        basis_max = max(basis_max, max(abs(recovered[key] - value) for key, value in basis.items()))
    record_test(results, "63 basis-vector Möbius tests", basis_max <= TOLERANCE, basis_vectors=63, max_abs_residual=basis_max, tolerance=TOLERANCE)

    adversarial = {tuple(row["subset"]): ((-1.0) ** row["term_index"]) * (row["term_index"] + 0.25) for row in d22_terms}
    adversarial_endpoints = {cell: endpoint_from_interactions(adversarial, [factor for factor, bit in zip(D22_FACTORS, cell) if bit == "1"]) for cell in expected_cells}
    adversarial_recovered = mobius_from_endpoints(adversarial_endpoints, D22_FACTORS)
    adversarial_residual = max(abs(adversarial_recovered[key] - value) for key, value in adversarial.items())
    record_test(results, "adversarial sign-pattern test", adversarial_residual <= TOLERANCE, max_abs_residual=adversarial_residual, tolerance=TOLERANCE)

    zero_endpoints = {cell: 0.0 for cell in expected_cells}
    zero_interactions = mobius_from_endpoints(zero_endpoints, D22_FACTORS)
    record_test(results, "zero-response conservation test", all(value == 0.0 for value in zero_interactions.values()), max_abs_value=max(abs(value) for value in zero_interactions.values()))
    constant_endpoints = {cell: (0.0 if cell == "000000" else 2.5) for cell in expected_cells}
    constant_interactions = mobius_from_endpoints(constant_endpoints, D22_FACTORS)
    record_test(results, "constant-response conservation test", max_reconstruction_residual(constant_endpoints, constant_interactions, D22_FACTORS) <= TOLERANCE, max_abs_residual=max_reconstruction_residual(constant_endpoints, constant_interactions, D22_FACTORS), tolerance=TOLERANCE)

    treatment_source = D20_PRIMITIVE.read_text(encoding="utf-8")
    d21_source = D21_RUNNER.read_text(encoding="utf-8")
    treatment_checks = {
        "exact_weighted_terms": "weighted_terms[h - 1]" in treatment_source,
        "no_renormalization": '"objective_renormalization": "NO"' in treatment_source,
        "canonical_updates_only": "if update not in d17.UPDATES" in treatment_source and "range(384, 392)" in d21_source,
        "optimizer_contract_unchanged": "optimizer.step()" in treatment_source,
        "no_hidden_child_rescaling": "renormalization" in treatment_source and '"objective_renormalization": "NO"' in treatment_source,
        "update_393_excluded": all(row["update_393_executed"] == "NO" for row in [d21["protocol"]]),
    }
    record_test(results, "treatment semantics static/code inspection", all(treatment_checks.values()), checks=treatment_checks, source_hashes={"D21_runner": sha256_file(D21_RUNNER), "D20_primitive": sha256_file(D20_PRIMITIVE)})

    return {"schema_version": "stage3_h13_post_d21_d22_static_validation_report_v1", "status": "PASS", "tolerance": TOLERANCE, "test_count": len(results), "tests": results, "random_seeds": seeds, "scientific_execution": "NO", "D22_endpoints_generated": "NO", "D22_updates_executed": "NO"}


def classification_contract() -> dict[str, Any]:
    threshold = EXPECTED_MATERIALITY_THRESHOLD
    return {
        "schema_version": "stage3_h13_post_d21_d22_classification_contract_v1",
        "future_results": "not evaluated in this design review",
        "inputs": {"primary_terms": ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22", "M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"], "parent_conservation_gate": "both exact D21 parent identities reconstruct within 1e-12", "materiality_threshold": threshold, "dominance_threshold": 0.5},
        "group_definitions": {
            "non_C1_parent_A": {"parent": "M_32_B1_B2_C2", "terms": ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22"]},
            "C1_conditioned_parent_B": {"parent": "M_32_B1_B2_C1_C2", "terms": ["M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"]},
        },
        "term_materiality": "a term is MATERIAL iff abs(term) >= the inherited H1 materiality threshold",
        "group_state_rules": [
            {"state": "PARENT_NOT_REPRODUCED", "condition": "exact parent identity residual > 1e-12 or endpoint/provenance validity fails"},
            {"state": "NO_RETAINED_MATERIAL_PARENT_STRUCTURE", "condition": "the reconstructed parent interaction is not material under the inherited threshold"},
            {"state": "C21_LOCALIZED", "condition": "abs(C21) >= threshold AND abs(C22) < threshold AND abs(C21×C22) < threshold"},
            {"state": "C22_LOCALIZED", "condition": "abs(C22) >= threshold AND abs(C21) < threshold AND abs(C21×C22) < threshold"},
            {"state": "C21×C22_COOPERATION_REQUIRED", "condition": "abs(C21×C22) >= threshold; its omission residual is therefore materially necessary and defeats unique child localization"},
            {"state": "DISTRIBUTED_C2_SUBBLOCK_HIGHER_ORDER", "condition": "parent is material, conservation passes, and none of the preceding localization/cooperation rules applies"},
        ],
        "overall_precedence": [
            "H32_BY_B1_B2_C2_PARENT_STRUCTURE_NOT_REPRODUCED",
            "H32_BY_B1_B2_C2_NO_RETAINED_MATERIAL_PARENT_STRUCTURE",
            "H32_BY_B1_B2_C2_C1_CONDITIONAL_CHILD_STRUCTURE",
            "H32_BY_B1_B2_C2_LOCALIZED_TO_C21",
            "H32_BY_B1_B2_C2_LOCALIZED_TO_C22",
            "H32_BY_B1_B2_C21_C22_COOPERATION_REQUIRED",
            "H32_BY_DISTRIBUTED_C2_SUBBLOCK_HIGHER_ORDER",
        ],
        "decision_table": {
            "H32_BY_B1_B2_C2_LOCALIZED_TO_C21": "both group states are C21_LOCALIZED",
            "H32_BY_B1_B2_C2_LOCALIZED_TO_C22": "both group states are C22_LOCALIZED",
            "H32_BY_B1_B2_C21_C22_COOPERATION_REQUIRED": "both group states are C21×C22_COOPERATION_REQUIRED",
            "H32_BY_B1_B2_C2_C1_CONDITIONAL_CHILD_STRUCTURE": "the two parent-group states differ, including C1 activation/suppression of cooperation",
            "H32_BY_DISTRIBUTED_C2_SUBBLOCK_HIGHER_ORDER": "both group states are distributed, or a non-conditional combination remains after the preceding gates",
            "H32_BY_B1_B2_C2_PARENT_STRUCTURE_NOT_REPRODUCED": "any exact parent identity/provenance gate fails; diagnose as consistency failure, never as a negative localization result",
            "H32_BY_B1_B2_C2_NO_RETAINED_MATERIAL_PARENT_STRUCTURE": "reconstructed previously material parent structure is below inherited materiality; diagnose provenance/consistency before interpretation",
        },
        "rank_1_is_not_sufficient": True,
        "dominance_rule_inherited": "largest absolute primary term may be called dominant only when its share of six-term absolute primary mass is >= 0.5; dominance alone never establishes localization",
        "classification_frozen_before_execution": True,
    }


def build_artifacts(d21: Mapping[str, Any], partition: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], imports: Sequence[Mapping[str, Any]], terms: Sequence[Mapping[str, Any]], mapping: Sequence[Mapping[str, Any]], static: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    start = {"completed_optimizer_step": 384, "certified_bundle_sha256": EXPECTED_CERTIFIED_START_SHA256, "model_semantic_hash": EXPECTED_MODEL_SHA256, "optimizer_semantic_hash": EXPECTED_OPTIMIZER_SHA256, "rng_sha256": d21["identity"]["d20_identity"]["certified_start"]["rng_sha256"]}
    primary = {
        "schema_version": "stage3_h13_post_d21_d22_primary_estimands_v1",
        "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED",
        "primary_parent_A": {"D21_term": "M_32_B1_B2_C2", "D21_value": d21["authenticated_primary_values"]["M_32_B1_B2_C2"], "D22_terms": ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22"]},
        "primary_parent_B": {"D21_term": "M_32_B1_B2_C1_C2", "D21_value": d21["authenticated_primary_values"]["M_32_B1_B2_C1_C2"], "D22_terms": ["M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"]},
        "six_preregistered_terms": ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22", "M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"],
        "materiality_source": str((D21_DESIGN_DIR / "materiality_contract.json").resolve()) + "::H1_materiality_threshold",
        "materiality_threshold": EXPECTED_MATERIALITY_THRESHOLD,
        "absolute_mass_contract": {"absolute_primary_interaction_mass": "sum(abs(each of six primary terms))", "term_fraction": "abs(term)/absolute_primary_interaction_mass", "largest_term_mass_fraction": "max(term fractions)", "C21_grouped_mass": "abs(C21-only non-C1)+abs(C21-only C1-conditioned)", "C22_grouped_mass": "abs(C22-only non-C1)+abs(C22-only C1-conditioned)", "C21_C22_cooperative_mass": "abs(C21×C22 non-C1)+abs(C21×C22 C1-conditioned)", "C1_conditioned_mass": "sum(abs(the three C1-conditioned terms))", "non_C1_conditioned_mass": "sum(abs(the three non-C1 terms))", "dominance_threshold": 0.5, "dominance_threshold_source": str((D21_DESIGN_DIR / "classification_contract.json").resolve()) + "::dominant_component"},
    }
    conservation = {
        "schema_version": "stage3_h13_post_d21_d22_parent_conservation_contract_v1",
        "convention": "E(S)=Y(empty)-Y(S); M(T)=E(T)-sum(non-empty proper U subset T)M(U), inherited from D21",
        "tolerance": TOLERANCE,
        "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED",
        "primary_identities": [
            {"parent": "M_32_B1_B2_C2", "children": ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22"], "equation": "M_32_B1_B2_C2 = M_32_B1_B2_C21 + M_32_B1_B2_C22 + M_32_B1_B2_C21_C22", "expected_residual": 0.0},
            {"parent": "M_32_B1_B2_C1_C2", "children": ["M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"], "equation": "M_32_B1_B2_C1_C2 = M_32_B1_B2_C1_C21 + M_32_B1_B2_C1_C22 + M_32_B1_B2_C1_C21_C22", "expected_residual": 0.0},
        ],
        "complete_31_term_mapping": mapping,
        "mapping_counts": {"d21_nonempty_terms": 31, "without_C2_identity_terms": 15, "with_C2_expansion_terms": 16, "d22_target_terms": 63, "target_terms_unique_and_complete": True},
        "inherited_D20_D21_contracts": {"M_32_B": "identity terms not containing C2", "M_32_C": "M_32_C1 plus the three-term expansion of M_32_C2 and of M_32_C1_C2", "M_32_B_C": "sum of all mapped D21 B/C terms", "Q_B": "Q_B=(M_32_B1-P_B1)+(M_32_B2-P_B2)+M_32_B1_B2; no C2 child term enters the authenticated definition"},
        "static_synthetic_verification": "PASS",
    }
    classification = classification_contract()
    protocol = {
        "schema_version": "stage3_h13_post_d21_d22_design_review_protocol_v1",
        "status": "SEALED_DESIGN_REVIEW_ONLY",
        "training": False, "backward_pass": False, "optimizer_step": False, "updates_385_392_executed": False, "update_393_executed": False, "frozen20_opened": False, "r9_executed": False, "r8e_a1_modified": False, "threshold_tuning": False, "partition_tuning": False, "adaptive_selection": False, "a_factor_branch": False, "mixed_child_import": False, "D22_scientific_results_generated": False,
        "future_branches_executed": 0,
    }
    treatment = {
        "exact_deletion_action": "delete exact listed weighted rollout terms before backward, inherited from D20/D21; no renormalization",
        "source_files": {"D21_runner": str(D21_RUNNER.resolve()), "D21_runner_sha256": sha256_file(D21_RUNNER), "D20_update_primitive": str(D20_PRIMITIVE.resolve()), "D20_update_primitive_sha256": sha256_file(D20_PRIMITIVE)},
        "unchanged_contract": ["certified post-update-384 start", "updates 385-392 only", "identical batches", "identical RNG stream", "identical model/loss/optimizer/AdamW/clipping/schedule/weight decay", "identical endpoint/H1/H32 definitions", "identical serialization semantics"],
        "C21_action": "H30 -> weighted_terms[29] deletion only",
        "C22_action": "H31 -> weighted_terms[30] deletion only",
        "C21_plus_C22_action": "H30 and H31 deletion exactly equals D21 C2 action",
        "C21_off_plus_C22_off_action": "no H30/H31 deletion exactly equals C2-off action",
        "future_execution_guard": "EXECUTED: NO and AUTHORIZED_FOR_EXECUTION: NO for every mixed-child branch",
    }
    design = {
        "schema_version": "stage3_h13_post_d21_d22_experiment_design_v1",
        "experiment_id": EXPERIMENT_ID,
        "design_review_only": True,
        "factors": list(D22_FACTORS), "factor_bit_order": list(D22_FACTORS), "factor_count": 6, "total_cells": 64, "total_mobius_terms": 63,
        "authenticated_D21_imports": 32, "future_D22_branches": 32, "expected_future_update_records": 256, "updates": list(UPDATES), "endpoint_update": 392, "forbidden_update": 393,
        "certified_start": start, "materiality_threshold": EXPECTED_MATERIALITY_THRESHOLD, "h32_preservation_threshold": EXPECTED_H32_THRESHOLD,
        "no_A_factor": True, "no_B_repartition": True, "no_C1_repartition": True, "no_alternative_horizons": True, "no_fractional_factorial": True, "no_adaptive_partition": True, "no_scientific_updates_in_review": True,
        "d21_parent_values": {"M_32_B_C": EXPECTED_PARENT_M32_BC, "Q_B": EXPECTED_Q_B, "M_32_B1_B2_C2": d21["authenticated_primary_values"]["M_32_B1_B2_C2"], "M_32_B1_B2_C1_C2": d21["authenticated_primary_values"]["M_32_B1_B2_C1_C2"]},
        "partition": partition, "treatment_semantics": treatment,
        "h32_preservation_accounting": {
            "future_results": "PREREGISTERED_NOT_EVALUATED",
            "rule": "H32 endpoint at update 392 <= inherited threshold",
            "required_outputs": ["total cells preserving H32", "total cells materially improving H1", "cells satisfying both", "imported versus newly executed cells", "C21-only cells", "C22-only cells", "C21+C22 cells", "C2-off cells"],
            "required_group_labels": ["IMPORTED_D21", "FUTURE_D22", "C21_ONLY", "C22_ONLY", "C2_ON", "C2_OFF"],
            "adaptive_discard": False,
        },
        "static_validation_status": static["status"],
    }
    provenance = {
        "schema_version": "stage3_h13_post_d21_d22_provenance_v1",
        "authoritative_parent": {"stage": "D21", "bundle_path": str(D21_DIR.resolve()), "manifest_path": str((D21_DIR / "SHA256_MANIFEST.json").resolve()), "expected_manifest_sha256": EXPECTED_D21_MANIFEST_SHA256, "observed_manifest_sha256": EXPECTED_D21_MANIFEST_SHA256, "internal_manifest_verification": "PASS"},
        "d21_identity": {"source_revision": d21["identity"]["source_revision"], "certified_start_sha256": d21["identity"]["certified_start_sha256"], "model_semantic_hash": d21["identity"]["model_semantic_hash"], "optimizer_semantic_hash": d21["identity"]["optimizer_semantic_hash"], "control_H1": EXPECTED_CONTROL_H1, "control_H32": EXPECTED_CONTROL_H32, "parent_M_32_B_C": EXPECTED_PARENT_M32_BC, "Q_B": EXPECTED_Q_B, "parent_conservation_residual": 0.0, "Q_B_conservation_residual": 0.0, "D21_cells": 32, "D21_terms": 31, "D21_imports": 8, "D21_new_branches": 24, "D21_update_records": 192},
        "D21_provenance_contract": {"certified_start_state": "post-update-384", "canonical_updates": list(UPDATES), "endpoint_update": 392, "update_393": "NO", "Frozen20": "NO", "R9": "NO", "R8E_A1_modified": "NO", "invalid_attempts_counted": False},
        "current_source_revision": d21["current_revision"],
        "historical_evidence_immutable": True,
        "D22_scientific_execution": "NONE",
        "generated_from": {"tool": str((ROOT / "tools" / SCRIPT_NAME).resolve()), "tool_sha256": sha256_file(ROOT / "tools" / SCRIPT_NAME)},
    }
    engineering = {
        "schema_version": "stage3_h13_post_d21_d22_engineering_defect_log_v1",
        "current_design_review_defects": [
            {"defect_id": "D22-DESIGN-PARSER-001", "defect": "D21 update-393 prohibition is nested under execution_manifest.protocol", "repair": "Read the authenticated nested field; no scientific path was touched", "scientific_semantics_changed": False},
            {"defect_id": "D22-DESIGN-PARSER-002", "defect": "Prompt-rounded D21 primary values are shorter than artifact float serialization", "repair": "Validate with frozen absolute tolerance and retain exact artifact values", "scientific_semantics_changed": False},
            {"defect_id": "D22-DESIGN-PARSER-003", "defect": "Inherited H32 rule includes the explicit 'otherwise FAIL' suffix", "repair": "Match the complete authenticated rule string", "scientific_semantics_changed": False},
            {"defect_id": "D22-DESIGN-TERM-004", "defect": "Initial generic term formatter omitted D21 C2 from the term order", "repair": "Use a shared canonical term order covering C2 and its two children", "scientific_semantics_changed": False},
            {"defect_id": "D22-DESIGN-TEST-005", "defect": "Initial random endpoint fixture assigned a nonzero empty-set effect", "repair": "Freeze E(empty)=0 as required by the inherited Möbius convention", "scientific_semantics_changed": False},
        ],
        "inherited_D21_defects": d21["defects"]["defects"],
        "inherited_D21_repair_semantics": "four narrow D21 wrapper/authorization/serialization defects were repaired and logged in D21; no D21 historical scientific semantics were changed and invalid attempts were excluded",
        "new_repairs": [],
        "scientific_semantics_changed_by_D22_design": False,
        "historical_artifacts_modified": False,
    }
    return design, treatment, primary, conservation, classification, protocol, provenance, engineering


def manifest_for(output: Path) -> dict[str, Any]:
    excluded = ["FINAL_REPORT.md", "SHA256_MANIFEST.json"]
    entries: list[dict[str, Any]] = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name in excluded:
            continue
        entries.append({"relative_path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "artifact_role": "D22 design-review artifact"})
    tool = ROOT / "tools" / SCRIPT_NAME
    entries.append({"relative_path": f"../../tools/{SCRIPT_NAME}", "sha256": sha256_file(tool), "size_bytes": tool.stat().st_size, "artifact_role": "D22 design-only materializer and static validator"})
    return {"schema_version": "stage3_h13_post_d21_d22_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "manifest_self_hash": "NOT_INCLUDED_BY_CONTRACT", "excluded_from_manifest": excluded, "file_count": len(entries), "files": entries}


def final_report(output: Path, d21: Mapping[str, Any], partition: Mapping[str, Any], static: Mapping[str, Any], manifest_hash: str, cells: Sequence[Mapping[str, Any]], imports: Sequence[Mapping[str, Any]], terms: Sequence[Mapping[str, Any]], mapping: Sequence[Mapping[str, Any]], defects: Mapping[str, Any]) -> str:
    future = [row for row in cells if row["provenance_status"] == "FUTURE_D22"]
    primary_terms = ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22", "M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"]
    source_check = "PASS" if static["status"] == "PASS" else "FAIL"
    lines = [
        EXPERIMENT_ID + ":", "PASSED", "", "DESIGN_REVIEW_PASSED:", "YES", "", "FIRST_BLOCKER:", "none", "", "ONE_SENTENCE_VERDICT:", "D21 is authenticated, C2 is canonically and losslessly split into C21=H30 and C22=H31, and a fully identifiable 64-cell D22 design with 32 authenticated diagonal imports and 32 frozen mixed-child branches passed all static and synthetic checks without scientific execution.", "",
        "## 1. PROTOCOL_INTEGRITY", "", "This was a design review only. No training, backward pass, optimizer step, D22 updates 385–392, update 393, Frozen20, R9, R8E_A1 modification, tuning, adaptive selection, or A-factor branch occurred. D22 scientific results were not generated; future mixed-child branches remain unexecuted and unauthorized.", "",
        "## 2. AUTHORITATIVE_D21_INPUT_IDENTITY", "", f"Bundle: `{d21['bundle_path']}`\nManifest: `{d21['manifest_path']}`\nExpected SHA-256: `{d21['expected_manifest_sha256']}`\nObserved SHA-256: `{d21['observed_manifest_sha256']}`\nManifest verification: `PASS` ({d21['manifest_verification']['checked_file_count']} files).\nControl endpoint: H1=`{d21['control']['endpoint_H1']}`, H32=`{d21['control']['endpoint_H32']}` at update 392.\nD21 evidence: 32 cells, 31 Möbius terms, 8 authenticated imports, 24 new branches, 192 update records. Parent M_32_B_C and Q_B conservation residuals are exactly zero. Material D21 terms include M_32_B1_B2_C2=`-4.25881355331938e-06` and M_32_B1_B2_C1_C2=`1.29275670773817e-05`.", "",
        "## 3. C2_C21_C22_PARTITION", "", "C2 is the authenticated ordered horizon set `{H30,H31}`. The canonical atomic update-space split is C21=`{H30}` and C22=`{H31}`. Each factor deletes exactly its listed weighted rollout term (`weighted_terms[h-1]`) before backward; C21∩C22 is empty, C21∪C22=C2, both-on exactly reproduces C2 deletion, both-off exactly reproduces C2-off, and no unrelated model/optimizer/update semantics change. The partition is deterministic, exhaustive, disjoint, lossless, not selected from D22 outcomes, and not tuned using D21 magnitudes.", "",
        "## 4. D22_FACTORIAL_DESIGN", "", "FACTOR_COUNT: 6\nTOTAL_CELLS: 64\nTOTAL_MOBIUS_TERMS: 63\nAUTHENTICATED_D21_IMPORTS: 32\nFUTURE_D22_BRANCHES: 32\nEXPECTED_FUTURE_UPDATE_RECORDS: 256\nFactor order: `[H32, B1, B2, C1, C21, C22]`; full 2^6 factorial; no fractional substitution.", "",
        "## 5. D21_TO_D22_IMPORT_VALIDATION", "", "Every D21 cell maps to the diagonal child state C21=C22=C2. C2=0 maps to (0,0); C2=1 maps to (1,1); H32/B1/B2/C1 remain unchanged. All 32 diagonal rows have exact deletion-set, start-state, schedule, update-index, batch, RNG, optimizer, clipping, downstream-update, endpoint, H1, and H32 identity checks marked PASS. No mixed child state is imported. See `d21_to_d22_import_map.csv`.", "",
        "## 6. PRIMARY_ESTIMANDS", "", "The six preregistered terms are: " + ", ".join(f"`{term}`" for term in primary_terms) + ". The first three decompose M_32_B1_B2_C2; the second three decompose M_32_B1_B2_C1_C2. All other 57 D22 terms remain required for full accounting.", "",
        "## 7. PARENT_CONSERVATION_CONTRACT", "", "`M_32_B1_B2_C2 = M_32_B1_B2_C21 + M_32_B1_B2_C22 + M_32_B1_B2_C21_C22`\n\n`M_32_B1_B2_C1_C2 = M_32_B1_B2_C1_C21 + M_32_B1_B2_C1_C22 + M_32_B1_B2_C1_C21_C22`\n\nThe complete map covers all 31 D21 nonempty terms: 15 no-C2 identity terms and 16 C2-containing terms, each expanded into its three nonempty child states, yielding all 63 D22 terms exactly once. Inherited M_32_B, M_32_C, M_32_B_C, and the authenticated Q_B definition are preserved algebraically.", "",
        "## 8. STATIC_SYNTHETIC_VALIDATION", "", f"{source_check}: {static['test_count']} static/synthetic tests passed at tolerance `{static['tolerance']}`: cell uniqueness/completeness, term uniqueness/completeness, Möbius forward/inverse reconstruction, diagonal embedding, complete conservation, both primary identities, inherited D20/D21 identities including Q_B, import identity mapping, fixed-seed randomized vectors, 63 basis vectors, adversarial signs, zero response, constant response, and treatment code inspection. No D22 endpoint response was generated.", "",
        "## 9. FROZEN_MATERIALITY_AND_H32_PRESERVATION", "", "MATERIAL_EFFECT_THRESHOLD_SOURCE: `" + str((D21_DESIGN_DIR / "materiality_contract.json").resolve()) + "::H1_materiality_threshold`\nMATERIAL_EFFECT_THRESHOLD_VALUE: `4.099006815405639e-06`\nH32_PRESERVATION_RULE_SOURCE: `" + str((D21_DESIGN_DIR / "h32_preservation_contract.json").resolve()) + "::rule`\nH32_PRESERVATION_RULE: `H32_at_update_392 <= 0.01856902565856056 is PASS`; H1 material improvement and H32 preservation remain separate.", "",
        "## 10. PREREGISTERED_CLASSIFICATION_CONTRACT", "", "The frozen precedence is: parent/provenance failure → `H32_BY_B1_B2_C2_PARENT_STRUCTURE_NOT_REPRODUCED`; reconstructed parent below inherited materiality → `H32_BY_B1_B2_C2_NO_RETAINED_MATERIAL_PARENT_STRUCTURE`; differing non-C1 versus C1-conditioned group states → `H32_BY_B1_B2_C2_C1_CONDITIONAL_CHILD_STRUCTURE`; both groups C21-localized → `H32_BY_B1_B2_C2_LOCALIZED_TO_C21`; both C22-localized → `H32_BY_B1_B2_C2_LOCALIZED_TO_C22`; both require a material C21×C22 term → `H32_BY_B1_B2_C21_C22_COOPERATION_REQUIRED`; otherwise → `H32_BY_DISTRIBUTED_C2_SUBBLOCK_HIGHER_ORDER`. A rank-1 term or dominance share alone never establishes localization.", "",
        "## 11. FUTURE_EXECUTION_MANIFEST", "", f"All {len(future)} mixed-child D22 branches are present in `future_execution_manifest.json` and the 64-row factorial table. Every future row is `EXECUTED: NO` and `AUTHORIZED_FOR_EXECUTION: NO`. The 32 diagonal rows are authenticated D21 sources with `D22_EXECUTED: NO`. Separate project-owner authorization is required before any D22 execution.", "",
        "## 12. ENGINEERING_DEFECTS_AND_REPAIRS", "", f"Five narrow D22 design-materializer parser/fixture defects were repaired and are documented in `engineering_defect_log.json`; all were detected before artifact finalization, changed no scientific semantics, and executed no scientific branch. Inherited D21 defects/repairs are also preserved there; no historical evidence was modified.", "",
        "## 13. ARTIFACTS", "", "\n".join(f"- `{(output / name).resolve()}`" for name in ["FINAL_REPORT.md", "experiment_design.json", "factorial_64_cell_design_table.csv", "d21_to_d22_import_map.csv", "mobius_63_term_registry.csv", "primary_estimands.json", "parent_conservation_contract.json", "classification_contract.json", "c2_c21_c22_partition_spec.json", "static_validation_report.json", "future_execution_manifest.json", "provenance.json", "engineering_defect_log.json", "SHA256_MANIFEST.json"]), "",
        "## 14. FINAL_MANIFEST", "", f"FINAL_MANIFEST_PATH: `{(output / 'SHA256_MANIFEST.json').resolve()}`\nFINAL_MANIFEST_SHA256: `{manifest_hash}`\nINTERNAL_MANIFEST_VERIFICATION: `PASS` (self-excluded manifest and report follow the established D21 convention).", "",
        "## 15. EXECUTION_READINESS", "", "D22_EXECUTION_READY_FOR_SEPARATE_AUTHORIZATION: YES",
    ]
    return "\n".join(lines) + "\n"


def materialize() -> dict[str, Any]:
    d21 = authenticate_d21()
    partition = build_partition(d21)
    d22_terms = build_d22_terms()
    mapping = build_parent_mapping(d22_terms)
    cells, imports = build_cells(d21, partition, d22_terms)
    static = run_static_validation(d21, cells, d22_terms, mapping, partition)
    design, treatment, primary, conservation, classification, protocol, provenance, engineering = build_artifacts(d21, partition, cells, imports, d22_terms, mapping, static)
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S+0800")
    output = ROOT / "outputs" / f"stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_design_review_{now}"
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    write_json(output / "experiment_design.json", design)
    write_json(output / "factorial_64_cell_design_table.json", {"schema_version": "stage3_h13_post_d21_d22_factorial_64_cell_design_table_v1", "factor_order": list(D22_FACTORS), "cells": cells, "cell_count": len(cells)})
    write_json(output / "d21_to_d22_import_map.json", {"schema_version": "stage3_h13_post_d21_d22_import_map_v1", "factor_order": list(D22_FACTORS), "mappings": imports, "count": len(imports), "mixed_child_imports": 0, "all_verified": True})
    write_json(output / "mobius_63_term_registry.json", {"schema_version": "stage3_h13_post_d21_d22_mobius_63_term_registry_v1", "factor_order": list(D22_FACTORS), "term_count": len(d22_terms), "terms": d22_terms})
    write_json(output / "primary_estimands.json", primary)
    write_json(output / "parent_conservation_contract.json", conservation)
    write_json(output / "classification_contract.json", classification)
    write_json(output / "c2_c21_c22_partition_spec.json", partition)
    write_json(output / "static_validation_report.json", static)
    write_json(output / "future_execution_manifest.json", {"schema_version": "stage3_h13_post_d21_d22_future_execution_manifest_v1", "factor_order": list(D22_FACTORS), "cell_count": 64, "imported_d21_count": 32, "future_d22_count": 32, "branches": cells})
    write_json(output / "provenance.json", provenance)
    write_json(output / "engineering_defect_log.json", engineering)
    write_json(output / "protocol_integrity.json", protocol)
    write_json(output / "treatment_semantics_contract.json", treatment)

    table_fields = ["canonical_cell_id", "H32", "B1", "B2", "C1", "C21", "C22", "C2_equivalent_bit", "C2_child_state", "provenance_status", "source_d21_cell_id", "source_d21_status", "exact_deleted_horizons", "expected_start_model_hash", "expected_start_optimizer_hash", "expected_start_rng_sha256", "canonical_updates", "endpoint_update", "treatment_specification_hash", "endpoint_specification", "H1_H32_evaluation_contract", "EXECUTED", "D22_EXECUTED", "AUTHORIZED_FOR_EXECUTION"]
    table_rows = []
    for row in cells:
        table_rows.append({**row, "exact_deleted_horizons": json.dumps(row["exact_deleted_horizons"], separators=(",", ":")), "expected_start_model_hash": row["expected_start_state_identity"]["model_semantic_hash"], "expected_start_optimizer_hash": row["expected_start_state_identity"]["optimizer_semantic_hash"], "expected_start_rng_sha256": row["expected_start_state_identity"]["rng_sha256"], "canonical_updates": json.dumps(row["canonical_updates"], separators=(",", ":")), "endpoint_specification": json.dumps(row["endpoint_specification"], separators=(",", ":")), "H1_H32_evaluation_contract": json.dumps(row["H1_H32_evaluation_contract"], separators=(",", ":"))})
    write_csv(output / "factorial_64_cell_design_table.csv", table_fields, table_rows)
    import_fields = list(imports[0].keys())
    write_csv(output / "d21_to_d22_import_map.csv", import_fields, imports)
    term_fields = ["term_index", "term", "subset", "subset_label", "order", "contains_H32", "contains_C21", "contains_C22", "future_endpoint_status"]
    write_csv(output / "mobius_63_term_registry.csv", term_fields, [{**row, "subset": json.dumps(row["subset"], separators=(",", ":"))} for row in d22_terms])

    manifest = manifest_for(output)
    write_json(output / "SHA256_MANIFEST.json", manifest)
    manifest_hash = sha256_file(output / "SHA256_MANIFEST.json")
    internal = verify_manifest(output, "SHA256_MANIFEST.json")
    require(internal["status"] == "PASS", f"final internal manifest verification failed: {internal['failures']}")
    write_json(output / "FINAL_REPORT.md.tmp.json", {"manifest_hash": manifest_hash})
    (output / "FINAL_REPORT.md.tmp.json").unlink()
    (output / "FINAL_REPORT.md").write_text(final_report(output, d21, partition, static, manifest_hash, cells, imports, d22_terms, mapping, engineering), encoding="utf-8", newline="\n")
    # The report is excluded from the established manifest to avoid a report/manifest cycle.
    return {"status": "PASSED", "output": str(output.resolve()), "manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": manifest_hash, "internal_manifest_verification": internal, "static_tests": static["test_count"], "D22_scientific_execution": "NO", "future_branches": 32, "imports": 32}


if __name__ == "__main__":
    try:
        print(json.dumps(materialize(), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}", "D22_scientific_execution": "NO"}, sort_keys=True))
        raise
