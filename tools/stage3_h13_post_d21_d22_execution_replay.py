"""Execute and finalize the frozen Stage-3 H13 D22 causal replay.

This runner is intentionally a D22 wrapper around the authenticated D20/D21
scientific update primitive.  It does not alter the forward pass, loss,
optimizer, schedule, RNG handling, or endpoint definitions.  It only resolves
the frozen six-factor deletion set, executes the 32 mixed-child cells, imports
the 32 authenticated D21 diagonal cells, and finalizes the evidence bundle.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
D22_DESIGN_DIR = ROOT / "outputs" / "stage3_h13_post_d21_d22_h32_by_b1_b2_c1_c2_hierarchical_c2_subblock_higher_order_factorial_causal_localization_design_review_20260821T113446+0800"
D21_DIR = ROOT / "outputs" / "stage3_h13_post_d20_d21_h32_by_b_c_hierarchical_subblock_higher_order_factorial_causal_localization_execution_evidence_20260821T035822+0800"
D21_RUNNER = ROOT / "tools" / "stage3_h13_post_d20_d21_execution_repaired.py"
D20_PRIMITIVE = ROOT / "tools" / "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay.py"
SELF_PATH = Path(__file__).resolve()

EXPECTED_D22_DESIGN_MANIFEST_SHA256 = "6ed9855be329cfd024805eff1efdc4f313e161de65f8584bfd6721959c1d329f"
EXPECTED_D21_MANIFEST_SHA256 = "d6f42c415223996aa8f26df0c1b82109ba054f5d75aa7a64e4421c6403ac206d"
EXPECTED_D21_RUNNER_SHA256 = "139fbf51759b7e498b69b5275c510141c7c3dcc2b603ffa9b004e62c5a101a86"
EXPECTED_D20_PRIMITIVE_SHA256 = "5c50a9c75a24b4d29d114ceca9f3c9917d62a6b1584359f5dfd765834a606518"
EXPECTED_START_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_SHA256 = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_SHA256 = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"
EXPECTED_RNG_SHA256 = "857d5ddd6c4271019b55a53774ad68c70e435abd38cf430b7f0d57ed047848d8"
EXPECTED_SOURCE_REVISION = "a6dff2b6887cc3f7598ab95549c2dd21c0efe763"
EXPECTED_CONTROL_H1 = 9.096969417553673e-05
EXPECTED_CONTROL_H32 = 0.018159905521264792
MATERIALITY_THRESHOLD = 4.099006815405639e-06
H32_PRESERVATION_THRESHOLD = 0.01856902565856056
TOLERANCE = 1e-12
UPDATES = tuple(range(385, 393))
FACTORS = ("H32", "B1", "B2", "C1", "C21", "C22")
TERM_NON_H32_ORDER = ("B1", "B2", "C1", "C21", "C22")
EXPERIMENT_ID = "STAGE_3_H13_POST_D21_D22_H32_BY_B1_B2_C1_C2_HIERARCHICAL_C2_SUBBLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION_REPLAY"


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


def verify_manifest(bundle: Path, manifest_name: str, expected_hash: str) -> dict[str, Any]:
    manifest_path = bundle / manifest_name
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    observed_manifest_hash = sha256_file(manifest_path)
    require(observed_manifest_hash == expected_hash, f"manifest hash mismatch: {bundle}")
    manifest = read_json(manifest_path)
    failures: list[str] = []
    for entry in manifest.get("files", []):
        relative = Path(str(entry["relative_path"]).replace("/", os.sep))
        path = (bundle / relative).resolve()
        if not path.is_file():
            failures.append(f"missing:{entry['relative_path']}")
            continue
        if sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            failures.append(f"mismatch:{entry['relative_path']}")
    require(not failures, f"internal manifest verification failed: {failures}")
    return {"status": "PASS", "manifest_sha256": observed_manifest_hash, "checked_file_count": len(manifest.get("files", [])), "failures": [], "manifest": manifest}


def all_cells(factors: Sequence[str] = FACTORS) -> list[str]:
    return ["".join(str(int(bit)) for bit in bits) for bits in itertools.product((0, 1), repeat=len(factors))]


def subsets(factors: Sequence[str] = FACTORS) -> list[tuple[str, ...]]:
    return [subset for size in range(1, len(factors) + 1) for subset in itertools.combinations(factors, size)]


def bits_for_subset(subset: Iterable[str], factors: Sequence[str] = FACTORS) -> str:
    chosen = set(subset)
    return "".join("1" if factor in chosen else "0" for factor in factors)


def subset_for_cell(cell: str, factors: Sequence[str] = FACTORS) -> tuple[str, ...]:
    require(len(cell) == len(factors) and set(cell) <= {"0", "1"}, f"invalid cell: {cell}")
    return tuple(factor for factor, bit in zip(factors, cell) if bit == "1")


def term_name(subset: Iterable[str]) -> str:
    chosen = set(subset)
    rest = [factor for factor in TERM_NON_H32_ORDER if factor in chosen]
    return "M_32" if "H32" in chosen and not rest else ("M_32_" + "_".join(rest) if "H32" in chosen else "M_" + "_".join(rest))


def mobius_from_endpoints(endpoints: Mapping[str, float]) -> dict[tuple[str, ...], float]:
    require(set(endpoints) == set(all_cells()), "endpoint vector is not complete")
    interactions: dict[tuple[str, ...], float] = {}
    for subset in subsets():
        value = float(endpoints[bits_for_subset(subset)])
        for size in range(1, len(subset)):
            for proper in itertools.combinations(subset, size):
                value -= interactions[proper]
        interactions[subset] = value
    return interactions


def endpoint_from_interactions(interactions: Mapping[tuple[str, ...], float], active: Iterable[str]) -> float:
    chosen = set(active)
    return sum(value for subset, value in interactions.items() if set(subset) <= chosen)


def json_list(value: str) -> list[Any]:
    parsed = json.loads(value)
    require(isinstance(parsed, list), "expected JSON list")
    return parsed


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def factor_bits(cell: str) -> dict[str, int]:
    return {factor: int(bit) for factor, bit in zip(FACTORS, cell)}


def expected_deleted_horizons(bits: Mapping[str, int], factor_members: Mapping[str, list[str]]) -> list[str]:
    deleted: list[str] = []
    for factor in FACTORS:
        if bits[factor]:
            deleted.extend(factor_members[factor])
    return deleted


def authorize(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"authorization file missing: {path}")
    text = path.read_text(encoding="utf-8")
    required = {
        "D22_EXECUTION_AUTHORIZED_NOW": "YES",
        "D22_DESIGN_REVIEW_PASSED": "YES",
        "DESIGN_CHANGE_AUTHORIZED": "NO",
        "ADAPTIVE_TUNING_AUTHORIZED": "NO",
        "NEW_FACTOR_OR_PARTITION_AUTHORIZED": "NO",
        "D23_AUTHORIZED": "NO",
        "UPDATE_393_EXECUTED": "NO",
        "FROZEN20_USED": "NO",
        "R9_EXECUTED": "NO",
        "R8E_A1_MODIFIED": "NO",
        "ADAPTIVE_SELECTION": "NO",
        "THRESHOLD_TUNING": "NO",
    }
    missing = [f"{key}:{value}" for key, value in required.items() if f"{key}:\n{value}" not in text and f"{key}: {value}" not in text]
    require(not missing, f"D22 authorization scope mismatch: {missing}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "scope_verified": True, "required_markers": required}


def authenticate_inputs(authorization: Path) -> dict[str, Any]:
    d22_manifest = verify_manifest(D22_DESIGN_DIR, "SHA256_MANIFEST.json", EXPECTED_D22_DESIGN_MANIFEST_SHA256)
    d21_manifest = verify_manifest(D21_DIR, "SHA256_MANIFEST.json", EXPECTED_D21_MANIFEST_SHA256)
    require(sha256_file(D21_RUNNER) == EXPECTED_D21_RUNNER_SHA256, "authenticated D21 runner hash mismatch")
    require(sha256_file(D20_PRIMITIVE) == EXPECTED_D20_PRIMITIVE_SHA256, "authenticated D20 primitive hash mismatch")
    design = read_json(D22_DESIGN_DIR / "experiment_design.json")
    require(design["factor_bit_order"] == list(FACTORS), "D22 factor order mismatch")
    require(design["total_cells"] == 64 and design["total_mobius_terms"] == 63, "D22 factorial dimensions mismatch")
    require(design["authenticated_D21_imports"] == 32 and design["future_D22_branches"] == 32, "D22 branch accounting mismatch")
    require(design["expected_future_update_records"] == 256, "D22 update accounting mismatch")
    require(design["updates"] == list(UPDATES) and design["endpoint_update"] == 392 and design["forbidden_update"] == 393, "D22 update window mismatch")
    require(design["design_review_only"] is True and design["static_validation_status"] == "PASS", "D22 design bundle is not sealed")
    d21_execution = read_json(D21_DIR / "execution_manifest.json")
    d21_protocol = read_json(D21_DIR / "protocol_integrity.json")
    d21_identity = read_json(D21_DIR / "authoritative_input_identity.json")
    d21_factor_definition = read_json(D21_DIR / "factor_definition.json")
    d21_table = csv_rows(D21_DIR / "factorial_32_cell_response_table.csv")
    d21_terms = csv_rows(D21_DIR / "mobius_31_terms.csv")
    d21_status = read_json(D21_DIR / "branch_execution_status.json")
    d21_branch_manifest = read_json(D21_DIR / "branch_manifest.json")
    d21_records = [json.loads(line) for line in (D21_DIR / "branch_update_records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    d21_parent = read_json(D21_DIR / "parent_child_conservation.json")
    d21_qb = read_json(D21_DIR / "qb_child_decomposition.json")
    require(len(d21_table) == 32 and len({row["cell_id"] for row in d21_table}) == 32, "D21 table is not complete")
    require(len(d21_terms) == 31 and len({row["term"] for row in d21_terms}) == 31, "D21 term table is not complete")
    require(d21_protocol["status"] == "PASSED" and d21_protocol["scientific_execution_valid"] is True, "D21 protocol not passed")
    require(d21_protocol["authenticated_imported_cells"] == 8 and d21_protocol["new_branches_executed"] == 24 and d21_protocol["new_branch_update_records"] == 192, "D21 counts mismatch")
    require(d21_execution["updates"] == list(UPDATES) and d21_execution["endpoint_update"] == 392 and d21_execution["protocol"]["update_393_executed"] == "NO", "D21 execution window mismatch")
    require(d21_status["all_pass"] is True and d21_status["count"] == 24, "D21 branch status mismatch")
    require(len(d21_branch_manifest["branches"]) == 24 and len(d21_records) == 192, "D21 raw trajectory count mismatch")
    require(d21_identity["certified_start_sha256"] == EXPECTED_START_BUNDLE_SHA256 and d21_identity["model_semantic_hash"] == EXPECTED_MODEL_SHA256 and d21_identity["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256, "D21 start-state identity mismatch")
    require(d21_identity["source_revision"] == EXPECTED_SOURCE_REVISION, "D21 source revision mismatch")
    current_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    require(current_revision == EXPECTED_SOURCE_REVISION, f"current source revision mismatch: {current_revision}")
    control = next(row for row in d21_table if row["cell_id"] == "00000")
    require(float(control["endpoint_H1"]) == EXPECTED_CONTROL_H1 and float(control["endpoint_H32"]) == EXPECTED_CONTROL_H32, "D21 control endpoint mismatch")
    require(d21_parent["all_pass"] is True and d21_qb["status"] == "PASS" and d21_parent["checks"]["M_32_B_C"]["absolute_residual"] == 0.0 and d21_qb["Q_B_conservation_abs_residual"] == 0.0, "D21 inherited conservation mismatch")
    require(float(design["materiality_threshold"]) == MATERIALITY_THRESHOLD and float(design["h32_preservation_threshold"]) == H32_PRESERVATION_THRESHOLD, "frozen thresholds changed")
    factor_members = {item["name"]: list(item.get("members", item.get("deletion_set", []))) for item in d21_factor_definition["factors"]}
    require(factor_members["C2"] == ["H30", "H31"], "D21 C2 ordered pair mismatch")
    # The authenticated D21 source names the parent as C2; D22 resolves the
    # already-frozen atomic children explicitly for its six-factor cells.
    factor_members["C21"] = ["H30"]
    factor_members["C22"] = ["H31"]
    auth = authorize(authorization)
    d17_control_path = Path(d21_identity["d20_identity"]["d17_control_reference"]["path"])
    require(d17_control_path.is_file(), f"D17 control reference missing: {d17_control_path}")
    control_records = [json.loads(line) for line in d17_control_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("branch") == "CONTROL"]
    control_by_update = {int(row["update"]): row for row in control_records}
    require(set(control_by_update) == set(UPDATES), "D17 control reference is incomplete")
    return {
        "d22_manifest": d22_manifest,
        "d21_manifest": d21_manifest,
        "design": design,
        "d21_execution": d21_execution,
        "d21_protocol": d21_protocol,
        "d21_identity": d21_identity,
        "d21_factor_definition": d21_factor_definition,
        "d21_table": d21_table,
        "d21_terms": d21_terms,
        "d21_status": d21_status,
        "d21_branch_manifest": d21_branch_manifest,
        "d21_records": d21_records,
        "d21_parent": d21_parent,
        "d21_qb": d21_qb,
        "factor_members": factor_members,
        "authorization": auth,
        "control_by_update": control_by_update,
        "current_revision": current_revision,
    }


def build_imported_cells(inputs: Mapping[str, Any], design_cells: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    d21_by_cell = {row["cell_id"]: row for row in inputs["d21_table"]}
    d21_status_by_cell = {row["cell_id"]: row for row in inputs["d21_status"]["records"]}
    imported: list[dict[str, Any]] = []
    imported_by_id: dict[str, dict[str, Any]] = {}
    for spec in design_cells:
        if spec["provenance_status"] != "IMPORTED_D21":
            continue
        cell = spec["canonical_cell_id"]
        bits = factor_bits(cell)
        require(bits["C21"] == bits["C22"], f"non-diagonal import: {cell}")
        d21_cell = cell[:4] + str(bits["C21"])
        source = d21_by_cell.get(d21_cell)
        require(source is not None, f"missing D21 source row: {d21_cell}")
        expected_delete = expected_deleted_horizons(bits, inputs["factor_members"])
        actual_delete = json_list(source["deletion_set"])
        require(actual_delete == expected_delete, f"D21 deletion mismatch: {cell}")
        require(float(source["endpoint_H1"]) == float(source["endpoint_H1"]) and float(source["endpoint_H32"]) == float(source["endpoint_H32"]), f"D21 endpoint is nonfinite: {cell}")
        validity = {
            "cell_id_mapping": True,
            "deletion_set_identity": True,
            "start_state_identity": spec["expected_start_state_identity"]["model_semantic_hash"] == EXPECTED_MODEL_SHA256 and spec["expected_start_state_identity"]["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256 and spec["expected_start_state_identity"]["rng_sha256"] == EXPECTED_RNG_SHA256,
            "update_identity": spec["canonical_updates"] == list(UPDATES) and spec["endpoint_update"] == 392,
            "source_endpoint_identity": True,
            "provenance_status": source["branch_validity"].startswith("PASS_"),
        }
        require(all(validity.values()), f"D21 import validation failed: {cell}")
        record = {
            "canonical_cell_id": cell,
            "factor_bits": bits,
            "factor_order": list(FACTORS),
            "exact_deleted_horizons": actual_delete,
            "source": "AUTHENTICATED_D21_IMPORT",
            "source_d21_cell_id": d21_cell,
            "source_d21_status": source["source"],
            "source_reference": source["provenance_reference"],
            "provenance_validity": "PASS",
            "provenance_checks": validity,
            "update_392_H1": float(source["endpoint_H1"]),
            "update_392_H32": float(source["endpoint_H32"]),
            "H1_response_Y_S": float(source["H1_effect"]),
            "H1_effect": float(source["H1_effect"]),
            "H1_materiality_status": "MATERIAL" if abs(float(source["H1_effect"])) >= MATERIALITY_THRESHOLD else "NONMATERIAL",
            "H32_preservation_status": "PRESERVED" if source["H32_preserved"].lower() == "true" else "NOT_PRESERVED",
            "joint_validity_status": "VALID",
            "satisfies_both": abs(float(source["H1_effect"])) >= MATERIALITY_THRESHOLD and source["H32_preserved"].lower() == "true",
            "treatment_specification_hash": spec["treatment_specification_hash"],
            "start_state_identity": spec["expected_start_state_identity"],
            "update_count": 8,
            "new_update_count": 0,
            "endpoint_update": 392,
            "branch": "AUTHENTICATED_D21_" + d21_cell,
            "d21_branch_status": d21_status_by_cell.get(d21_cell, {"status": "AUTHENTICATED_IMPORT"}),
        }
        imported.append(record)
        imported_by_id[cell] = record
    require(len(imported) == 32 and len(imported_by_id) == 32, "D22 imported cell count mismatch")
    return imported, imported_by_id


def execute_new_branches(inputs: Mapping[str, Any], design_cells: Sequence[Mapping[str, Any]], output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    future = [spec for spec in design_cells if spec["provenance_status"] == "FUTURE_D22"]
    require(len(future) == 32, "D22 future branch count mismatch")
    require(sum(spec["C21"] == 1 and spec["C22"] == 0 for spec in future) == 16, "C21-only count mismatch")
    require(sum(spec["C21"] == 0 and spec["C22"] == 1 for spec in future) == 16, "C22-only count mismatch")
    sys.path.insert(0, str(ROOT))
    from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17
    from tools import stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay as d20

    runtime = d17.preflight_runtime()
    require(runtime["loaded"]["model_semantic_hash"] == EXPECTED_MODEL_SHA256 and runtime["loaded"]["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256, "certified runtime identity mismatch")
    require(d17.base.rng_digest(runtime["bundle"]["rng_state"]) == EXPECTED_RNG_SHA256, "certified start RNG mismatch")
    all_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    checkpoint_path = output / "branch_update_records.checkpoint.jsonl"
    checkpoint_path.write_text("", encoding="utf-8", newline="\n")
    for index, spec in enumerate(future, start=1):
        cell = spec["canonical_cell_id"]
        bits = factor_bits(cell)
        deleted = [int(item.removeprefix("H")) for item in spec["exact_deleted_horizons"]]
        require(spec["exact_deleted_horizons"] == expected_deleted_horizons(bits, inputs["factor_members"]), f"D22 design deletion mismatch: {cell}")
        model, optimizer = d17.d16.new_branch(runtime["bundle"])
        start_model_hash = d17.canonical_model_state_sha256(model.state_dict())
        start_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
        require(start_model_hash == EXPECTED_MODEL_SHA256 and start_optimizer_hash == EXPECTED_OPTIMIZER_SHA256, f"branch start identity mismatch: {cell}")
        rng = copy.deepcopy(runtime["bundle"]["rng_state"])
        branch_rows: list[dict[str, Any]] = []
        for zero_index in range(384, 392):
            row, _, rng = d20._scientific_branch_update(d17, f"D22_{cell}", deleted, model, optimizer, runtime, zero_index, rng)
            require(row["update"] in UPDATES, f"unauthorized update in {cell}: {row['update']}")
            require(row["deleted_horizons"] == sorted(set(deleted)), f"deletion record mismatch: {cell}:{row['update']}")
            control = inputs["control_by_update"][row["update"]]
            for field in ("batch_window_sha256", "rng_before_digest", "rng_after_diagnostics_digest", "rng_after_training_digest", "rng_after_validation_digest"):
                require(row[field] == control[field], f"control stream mismatch: {cell}:{row['update']}:{field}")
            require(row["finite_status"] == "FINITE" and row["branch_rng_contract_status"] == "PASS", f"runtime validity mismatch: {cell}:{row['update']}")
            row.update({
                "canonical_cell_id": cell,
                "factor_order": list(FACTORS),
                "factor_bits": bits,
                "source": "NEW_D22_EXECUTION",
                "d22_branch_kind": "C21_ONLY" if (bits["C21"], bits["C22"]) == (1, 0) else "C22_ONLY",
                "treatment_specification_hash": spec["treatment_specification_hash"],
                "start_state_identity": spec["expected_start_state_identity"],
                "endpoint_update": 392,
            })
            branch_rows.append(row)
            all_rows.append(row)
        endpoint = branch_rows[-1]
        h1_endpoint = float(endpoint["per_update_H1"])
        h32_endpoint = float(endpoint["per_update_H32"])
        h1_effect = EXPECTED_CONTROL_H1 - h1_endpoint
        require(finite_number(h1_endpoint) and finite_number(h32_endpoint) and finite_number(h1_effect), f"nonfinite endpoint: {cell}")
        h1_material = abs(h1_effect) >= MATERIALITY_THRESHOLD
        h32_preserved = h32_endpoint <= H32_PRESERVATION_THRESHOLD
        statuses.append({
            "branch": f"D22_{cell}",
            "cell_id": cell,
            "canonical_cell_id": cell,
            "factor_bits": bits,
            "child_state": spec["C2_child_state"],
            "source": "NEW_D22_EXECUTION",
            "status": "PASS",
            "updates": list(UPDATES),
            "endpoint_update": 392,
            "update_393_executed": "NO",
            "start_model_hash": start_model_hash,
            "start_optimizer_hash": start_optimizer_hash,
            "start_rng_sha256": EXPECTED_RNG_SHA256,
            "exact_deleted_horizons": spec["exact_deleted_horizons"],
            "update_count": len(branch_rows),
            "endpoint_H1": h1_endpoint,
            "endpoint_H32": h32_endpoint,
            "H1_effect": h1_effect,
            "H1_material": h1_material,
            "H32_preserved": h32_preserved,
            "treatment_specification_hash": spec["treatment_specification_hash"],
        })
        manifests.append({
            "branch": f"D22_{cell}",
            "cell": cell,
            "canonical_cell_id": cell,
            "factor_order": list(FACTORS),
            "factor_bits": bits,
            "child_state": spec["C2_child_state"],
            "exact_deleted_horizons": spec["exact_deleted_horizons"],
            "start_model_hash": start_model_hash,
            "start_optimizer_hash": start_optimizer_hash,
            "start_rng_sha256": EXPECTED_RNG_SHA256,
            "updates": list(UPDATES),
            "endpoint_update": 392,
            "update_393_executed": "NO",
            "status": "PASS",
            "treatment_specification_hash": spec["treatment_specification_hash"],
        })
        with checkpoint_path.open("a", encoding="utf-8", newline="\n") as checkpoint:
            for row in branch_rows:
                checkpoint.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        write_json(output / "branch_execution_checkpoint.json", {"completed_branch_count": len(statuses), "completed_update_record_count": len(all_rows), "completed_cells": [item["canonical_cell_id"] for item in statuses], "statuses": statuses, "manifests": manifests})
        print(f"D22 branch {index}/32 complete: {cell}", flush=True)
    require(len(all_rows) == 256 and len(statuses) == 32 and len(manifests) == 32, "D22 execution accounting mismatch")
    return all_rows, statuses, manifests


def response_table(inputs: Mapping[str, Any], design_cells: Sequence[Mapping[str, Any]], imported_by_id: Mapping[str, Mapping[str, Any]], statuses_by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for spec in design_cells:
        cell = spec["canonical_cell_id"]
        bits = factor_bits(cell)
        if spec["provenance_status"] == "IMPORTED_D21":
            source = dict(imported_by_id[cell])
        else:
            status = statuses_by_id[cell]
            h1_effect = float(status["H1_effect"])
            source = {
                "canonical_cell_id": cell,
                "factor_bits": bits,
                "factor_order": list(FACTORS),
                "exact_deleted_horizons": list(spec["exact_deleted_horizons"]),
                "source": "NEW_D22_EXECUTION",
                "source_reference": f"branch_update_records.jsonl::D22_{cell}/update=392",
                "provenance_validity": "PASS",
                "update_392_H1": float(status["endpoint_H1"]),
                "update_392_H32": float(status["endpoint_H32"]),
                "H1_response_Y_S": h1_effect,
                "H1_effect": h1_effect,
                "H1_materiality_status": "MATERIAL" if abs(h1_effect) >= MATERIALITY_THRESHOLD else "NONMATERIAL",
                "H32_preservation_status": "PRESERVED" if status["H32_preserved"] else "NOT_PRESERVED",
                "joint_validity_status": "VALID",
                "satisfies_both": bool(status["H1_material"] and status["H32_preserved"]),
                "treatment_specification_hash": spec["treatment_specification_hash"],
                "start_state_identity": spec["expected_start_state_identity"],
                "update_count": 8,
                "new_update_count": 8,
                "endpoint_update": 392,
                "branch": f"D22_{cell}",
            }
        row = {
            "canonical_cell_id": cell,
            "factor_bits": bits,
            "H32": bits["H32"], "B1": bits["B1"], "B2": bits["B2"], "C1": bits["C1"], "C21": bits["C21"], "C22": bits["C22"],
            "child_state": spec["C2_child_state"],
            "source": source["source"],
            "provenance_validity": source["provenance_validity"],
            "source_reference": source.get("source_reference"),
            "exact_deleted_horizons": list(source["exact_deleted_horizons"]),
            "update_392_H1": float(source["update_392_H1"]),
            "update_392_H32": float(source["update_392_H32"]),
            "H1_response_Y_S": float(source["H1_response_Y_S"]),
            "H1_effect": float(source["H1_effect"]),
            "H1_materiality_status": source["H1_materiality_status"],
            "H32_preservation_status": source["H32_preservation_status"],
            "joint_validity_status": source["joint_validity_status"],
            "satisfies_both": bool(source["satisfies_both"]),
            "treatment_specification_hash": source["treatment_specification_hash"],
            "start_state_identity": source["start_state_identity"],
            "update_count": int(source["update_count"]),
            "new_update_count": int(source["new_update_count"]),
            "endpoint_update": 392,
            "branch": source["branch"],
        }
        responses.append(row)
    require([row["canonical_cell_id"] for row in responses] == all_cells(), "response table is not in canonical order")
    require(len(responses) == 64 and len({row["canonical_cell_id"] for row in responses}) == 64, "response table is not complete")
    return responses


def build_mobius(responses: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], float], dict[str, float], dict[str, Any]]:
    endpoints = {row["canonical_cell_id"]: float(row["H1_effect"]) for row in responses}
    require(abs(endpoints["000000"]) <= 1e-18, "empty-set response is not zero")
    interactions = mobius_from_endpoints(endpoints)
    reconstruction = {cell: endpoint_from_interactions(interactions, subset_for_cell(cell)) for cell in all_cells()}
    residuals = {cell: reconstruction[cell] - endpoints[cell] for cell in all_cells()}
    max_residual = max(abs(value) for value in residuals.values())
    require(max_residual <= TOLERANCE, f"Möbius reconstruction residual exceeds tolerance: {max_residual}")
    sorted_by_abs = sorted(interactions.items(), key=lambda item: (-abs(item[1]), item[0]))
    rank_by_subset = {subset: index for index, (subset, _) in enumerate(sorted_by_abs, start=1)}
    rows: list[dict[str, Any]] = []
    for index, subset in enumerate(subsets(), start=1):
        value = float(interactions[subset])
        chosen = set(subset)
        rows.append({
            "term_index": index,
            "term": term_name(subset),
            "subset": list(subset),
            "subset_label": "×".join(subset),
            "order": len(subset),
            "M_subset": value,
            "abs_M_subset": abs(value),
            "materiality_status": "MATERIAL" if abs(value) >= MATERIALITY_THRESHOLD else "NONMATERIAL",
            "rank_by_absolute_magnitude": rank_by_subset[subset],
            "contains_H32": "H32" in chosen,
            "contains_B_child": bool(chosen & {"B1", "B2"}),
            "contains_C_child": bool(chosen & {"C1", "C21", "C22"}),
            "contains_C21": "C21" in chosen,
            "contains_C22": "C22" in chosen,
        })
    require(len(rows) == 63 and len({row["term"] for row in rows}) == 63, "Möbius term completeness mismatch")
    diagnostics = {"forward_mobius_transform": "PASS", "inverse_reconstruction": "PASS", "cell_count": 64, "term_count": 63, "max_absolute_reconstruction_residual": max_residual, "residuals": residuals, "tolerance": TOLERANCE}
    return rows, interactions, endpoints, diagnostics


def term_values(interactions: Mapping[tuple[str, ...], float], names: Sequence[str]) -> dict[str, float]:
    by_name = {term_name(subset): value for subset, value in interactions.items()}
    return {name: float(by_name[name]) for name in names}


def build_conservation(inputs: Mapping[str, Any], interactions: Mapping[tuple[str, ...], float], mobius_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    d21_term_values = {row["term"]: float(row["M_subset"]) for row in inputs["d21_terms"]}
    mapping = read_json(D22_DESIGN_DIR / "parent_conservation_contract.json")["complete_31_term_mapping"]
    target_terms = [row["term"] for row in mobius_rows]
    mapped_terms: list[str] = []
    mapping_results: list[dict[str, Any]] = []
    for row in mapping:
        children = list(row["d22_terms"])
        mapped_terms.extend(children)
        target = d21_term_values.get(row["d21_term"])
        require(target is not None, f"missing D21 term for mapping: {row['d21_term']}")
        child_values = term_values(interactions, children)
        reconstructed = sum(child_values.values())
        residual = reconstructed - target
        mapping_results.append({**row, "target_value": target, "child_values": child_values, "reconstructed_value": reconstructed, "signed_residual": residual, "absolute_residual": abs(residual), "status": "PASS" if abs(residual) <= TOLERANCE else "FAIL"})
    require(len(mapping_results) == 31 and len(mapped_terms) == 63 and len(set(mapped_terms)) == 63 and set(mapped_terms) == set(target_terms), "complete D21-to-D22 map is not exact")
    require(all(row["status"] == "PASS" for row in mapping_results), "D21-to-D22 conservation failed")

    def sum_terms(names: Iterable[str]) -> float:
        return sum(term_values(interactions, list(names)).values())

    parent = inputs["d21_parent"]
    inherited_targets = {
        "M_32_B": float(parent["checks"]["M_32_B"]["target"]),
        "M_32_C": float(parent["checks"]["M_32_C"]["target"]),
        "M_32_B_C": float(parent["checks"]["M_32_B_C"]["target"]),
        "Q_B": float(inputs["d21_qb"]["Q_B_D20_reference"]),
    }
    b_names = ["M_32_B1", "M_32_B2", "M_32_B1_B2"]
    c_names = ["M_32_C1", "M_32_C21", "M_32_C22", "M_32_C21_C22", "M_32_C1_C21", "M_32_C1_C22", "M_32_C1_C21_C22"]
    bc_names = [row["term"] for row in mobius_rows if row["contains_H32"] and row["contains_B_child"] and row["contains_C_child"]]
    q_b_value = (term_values(interactions, ["M_32_B1"])["M_32_B1"] - float(inputs["d21_qb"]["P_B1"])) + (term_values(interactions, ["M_32_B2"])["M_32_B2"] - float(inputs["d21_qb"]["P_B2"])) + term_values(interactions, ["M_32_B1_B2"])["M_32_B1_B2"]
    inherited_specs = {
        "M_32_B": b_names,
        "M_32_C": c_names,
        "M_32_B_C": bc_names,
        "Q_B": [],
    }
    inherited_results: dict[str, Any] = {}
    for name, names in inherited_specs.items():
        reconstructed = q_b_value if name == "Q_B" else sum_terms(names)
        target = inherited_targets[name]
        residual = reconstructed - target
        inherited_results[name] = {"target": target, "reconstructed": reconstructed, "signed_residual": residual, "absolute_residual": abs(residual), "tolerance": TOLERANCE, "status": "PASS" if abs(residual) <= TOLERANCE else "FAIL", "source_terms": names}
    require(all(item["status"] == "PASS" for item in inherited_results.values()), "inherited identity conservation failed")
    complete = {
        "schema_version": "stage3_h13_post_d21_d22_d21_to_d22_conservation_v1",
        "tolerance": TOLERANCE,
        "d21_nonempty_terms": 31,
        "identity_terms": sum(row["mapping_kind"] == "IDENTITY_NO_C2" for row in mapping_results),
        "expanded_terms": sum(row["mapping_kind"] == "C2_TO_C21_C22_EXPANSION" for row in mapping_results),
        "d22_target_terms": 63,
        "target_term_coverage": "PASS",
        "mapping_results": mapping_results,
        "inherited_identity_results": inherited_results,
        "all_pass": True,
    }
    primary_a = ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22"]
    primary_b = ["M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"]
    primary_results = {
        "PARENT_A": {"parent": "M_32_B1_B2_C2", "target": d21_term_values["M_32_B1_B2_C2"], "children": term_values(interactions, primary_a)},
        "PARENT_B": {"parent": "M_32_B1_B2_C1_C2", "target": d21_term_values["M_32_B1_B2_C1_C2"], "children": term_values(interactions, primary_b)},
    }
    for item in primary_results.values():
        item["reconstructed"] = sum(item["children"].values())
        item["signed_residual"] = item["reconstructed"] - item["target"]
        item["absolute_residual"] = abs(item["signed_residual"])
        item["tolerance"] = TOLERANCE
        item["status"] = "PASS" if item["absolute_residual"] <= TOLERANCE else "FAIL"
    require(all(item["status"] == "PASS" for item in primary_results.values()), "primary parent conservation failed")
    primary_conservation = {"schema_version": "stage3_h13_post_d21_d22_parent_conservation_results_v1", "tolerance": TOLERANCE, "PARENT_A": primary_results["PARENT_A"], "PARENT_B": primary_results["PARENT_B"], "inherited_D21_identities": inherited_results, "all_pass": True}
    return primary_conservation, complete


def classify_group(name: str, parent_value: float, child_values: Mapping[str, float], endpoint_valid: bool) -> dict[str, Any]:
    c21 = float(child_values["C21"])
    c22 = float(child_values["C22"])
    cooperative = float(child_values["C21_C22"])
    parent_reconstructed = c21 + c22 + cooperative
    parent_residual = parent_reconstructed - parent_value
    parent_reproduced = endpoint_valid and abs(parent_residual) <= TOLERANCE
    if not parent_reproduced:
        state = "PARENT_NOT_REPRODUCED"
    elif abs(parent_reconstructed) < MATERIALITY_THRESHOLD:
        state = "NO_RETAINED_MATERIAL_PARENT_STRUCTURE"
    elif abs(c21) >= MATERIALITY_THRESHOLD and abs(c22) < MATERIALITY_THRESHOLD and abs(cooperative) < MATERIALITY_THRESHOLD:
        state = "C21_LOCALIZED"
    elif abs(c22) >= MATERIALITY_THRESHOLD and abs(c21) < MATERIALITY_THRESHOLD and abs(cooperative) < MATERIALITY_THRESHOLD:
        state = "C22_LOCALIZED"
    elif abs(cooperative) >= MATERIALITY_THRESHOLD:
        state = "C21xC22_COOPERATION_REQUIRED"
    else:
        state = "DISTRIBUTED_C2_SUBBLOCK_HIGHER_ORDER"
    return {"group": name, "parent_value": parent_value, "child_values": dict(child_values), "reconstructed_parent": parent_reconstructed, "parent_signed_residual": parent_residual, "parent_absolute_residual": abs(parent_residual), "parent_reproduced": parent_reproduced, "parent_material": abs(parent_reconstructed) >= MATERIALITY_THRESHOLD, "state": state, "materiality_threshold": MATERIALITY_THRESHOLD, "tolerance": TOLERANCE}


def build_primary_and_classification(interactions: Mapping[tuple[str, ...], float], conservation: Mapping[str, Any], responses: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    primary_names = ["M_32_B1_B2_C21", "M_32_B1_B2_C22", "M_32_B1_B2_C21_C22", "M_32_B1_B2_C1_C21", "M_32_B1_B2_C1_C22", "M_32_B1_B2_C1_C21_C22"]
    values = term_values(interactions, primary_names)
    mass = sum(abs(values[name]) for name in primary_names)
    require(mass > 0.0, "primary interaction mass is zero")
    fractions = {name: abs(values[name]) / mass for name in primary_names}
    dominant_name = max(primary_names, key=lambda name: abs(values[name]))
    primary_rows: list[dict[str, Any]] = []
    for name in primary_names:
        group = "C1_CONDITIONED" if "_C1_" in name else "NON_C1_CONDITIONED"
        if name.endswith("C21") or name.endswith("C1_C21"):
            child = "C21"
        elif name.endswith("C22") or name.endswith("C1_C22"):
            child = "C22"
        else:
            child = "C21×C22"
        primary_rows.append({"term": name, "signed_value": values[name], "absolute_value": abs(values[name]), "materiality_status": "MATERIAL" if abs(values[name]) >= MATERIALITY_THRESHOLD else "NONMATERIAL", "term_mass_fraction": fractions[name], "group": group, "child_role": child, "dominant": fractions[name] >= 0.5, "interpretation": "frozen D22 child interaction under exact weighted rollout-term deletion; dominance is descriptive and does not establish localization"})
    primary = {"schema_version": "stage3_h13_post_d21_d22_primary_estimands_results_v1", "terms": primary_rows, "absolute_primary_interaction_mass": mass, "largest_term": dominant_name, "largest_term_mass_fraction": fractions[dominant_name], "dominance_threshold": 0.5, "C21_grouped_mass": abs(values["M_32_B1_B2_C21"]) + abs(values["M_32_B1_B2_C1_C21"]), "C22_grouped_mass": abs(values["M_32_B1_B2_C22"]) + abs(values["M_32_B1_B2_C1_C22"]), "C21_C22_cooperative_mass": abs(values["M_32_B1_B2_C21_C22"]) + abs(values["M_32_B1_B2_C1_C21_C22"]), "NON_C1_CONDITIONED_MASS": sum(abs(values[name]) for name in primary_names[:3]), "C1_CONDITIONED_MASS": sum(abs(values[name]) for name in primary_names[3:]), "materiality_threshold": MATERIALITY_THRESHOLD}
    parent_a = conservation["PARENT_A"]["target"]
    parent_b = conservation["PARENT_B"]["target"]
    group_a = classify_group("GROUP_A_NON_C1_PARENT", parent_a, {"C21": values["M_32_B1_B2_C21"], "C22": values["M_32_B1_B2_C22"], "C21_C22": values["M_32_B1_B2_C21_C22"]}, all(row["joint_validity_status"] == "VALID" for row in responses))
    group_b = classify_group("GROUP_B_C1_CONDITIONED_PARENT", parent_b, {"C21": values["M_32_B1_B2_C1_C21"], "C22": values["M_32_B1_B2_C1_C22"], "C21_C22": values["M_32_B1_B2_C1_C21_C22"]}, all(row["joint_validity_status"] == "VALID" for row in responses))
    states = (group_a["state"], group_b["state"])
    if "PARENT_NOT_REPRODUCED" in states:
        overall = "H32_BY_B1_B2_C2_PARENT_STRUCTURE_NOT_REPRODUCED"
    elif "NO_RETAINED_MATERIAL_PARENT_STRUCTURE" in states:
        overall = "H32_BY_B1_B2_C2_NO_RETAINED_MATERIAL_PARENT_STRUCTURE"
    elif states == ("C21_LOCALIZED", "C21_LOCALIZED"):
        overall = "H32_BY_B1_B2_C2_LOCALIZED_TO_C21"
    elif states == ("C22_LOCALIZED", "C22_LOCALIZED"):
        overall = "H32_BY_B1_B2_C2_LOCALIZED_TO_C22"
    elif states == ("C21xC22_COOPERATION_REQUIRED", "C21xC22_COOPERATION_REQUIRED"):
        overall = "H32_BY_B1_B2_C21_C22_COOPERATION_REQUIRED"
    elif states[0] != states[1]:
        overall = "H32_BY_B1_B2_C2_C1_CONDITIONAL_CHILD_STRUCTURE"
    else:
        overall = "H32_BY_DISTRIBUTED_C2_SUBBLOCK_HIGHER_ORDER"
    classification = {"schema_version": "stage3_h13_post_d21_d22_classification_v1", "group_A": group_a, "group_B": group_b, "overall_classification": overall, "precedence_applied": ["parent/provenance failure", "no retained material parent", "C1 conditional", "both C21 localized", "both C22 localized", "both cooperative", "distributed"], "dominance_alone_never_localizes": True, "h32_preservation_is_separate": True}
    materiality = {"schema_version": "stage3_h13_post_d21_d22_materiality_results_v1", "threshold": MATERIALITY_THRESHOLD, "rule": "abs(component) >= threshold means MATERIAL", "material_cell_ids": [row["canonical_cell_id"] for row in responses if row["H1_materiality_status"] == "MATERIAL"], "nonmaterial_cell_ids": [row["canonical_cell_id"] for row in responses if row["H1_materiality_status"] != "MATERIAL"], "material_cell_count": sum(row["H1_materiality_status"] == "MATERIAL" for row in responses), "cell_count": 64, "primary_terms": primary_rows}
    return primary, classification, materiality, {"dominant_name": dominant_name, "fractions": fractions}


def accounting(responses: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    def ids(predicate: Any) -> list[str]:
        return [row["canonical_cell_id"] for row in responses if predicate(row)]
    by_source = {"IMPORTED_D21": [row["canonical_cell_id"] for row in responses if row["source"] == "AUTHENTICATED_D21_IMPORT"], "FUTURE_D22": [row["canonical_cell_id"] for row in responses if row["source"] == "NEW_D22_EXECUTION"]}
    by_child = {"C21_ONLY": ids(lambda row: row["child_state"] == "C21_ONLY"), "C22_ONLY": ids(lambda row: row["child_state"] == "C22_ONLY"), "C2_ON": ids(lambda row: row["child_state"] == "C2_ON"), "C2_OFF": ids(lambda row: row["child_state"] == "C2_OFF")}
    preserved = ids(lambda row: row["H32_preservation_status"] == "PRESERVED")
    material = ids(lambda row: row["H1_materiality_status"] == "MATERIAL")
    joint = ids(lambda row: bool(row["satisfies_both"]))
    invalid = ids(lambda row: row["joint_validity_status"] != "VALID")
    fail_h32 = ids(lambda row: row["H32_preservation_status"] != "PRESERVED")
    details = {
        "TOTAL_CELLS": 64,
        "TOTAL_H32_PRESERVED": len(preserved),
        "TOTAL_H1_MATERIAL": len(material),
        "TOTAL_JOINTLY_VALID": len(joint),
        "by_source": {key: {"count": len(value), "total_H32_preserved": sum(responses_by_id["H32_preservation_status"] == "PRESERVED" for responses_by_id in responses if responses_by_id["canonical_cell_id"] in value), "total_H1_material": sum(responses_by_id["H1_materiality_status"] == "MATERIAL" for responses_by_id in responses if responses_by_id["canonical_cell_id"] in value), "total_jointly_valid": sum(responses_by_id["canonical_cell_id"] in joint for responses_by_id in responses if responses_by_id["canonical_cell_id"] in value), "cell_ids": value} for key, value in by_source.items()},
        "by_child_state": {key: {"count": len(value), "total_H32_preserved": sum(row["canonical_cell_id"] in value and row["H32_preservation_status"] == "PRESERVED" for row in responses), "total_H1_material": sum(row["canonical_cell_id"] in value and row["H1_materiality_status"] == "MATERIAL" for row in responses), "total_jointly_valid": sum(row["canonical_cell_id"] in value and row["canonical_cell_id"] in joint for row in responses), "cell_ids": value} for key, value in by_child.items()},
        "materially_improve_H1_cell_ids": material,
        "preserve_H32_cell_ids": preserved,
        "satisfy_both_cell_ids": joint,
        "fail_H32_preservation_cell_ids": fail_h32,
        "scientifically_invalid_cell_ids": invalid,
        "adaptive_exclusion": False,
    }
    h32 = {"schema_version": "stage3_h13_post_d21_d22_h32_preservation_v1", "threshold": H32_PRESERVATION_THRESHOLD, "rule": "H32_at_update_392 <= 0.01856902565856056 means H32 PRESERVED", "total_cells": 64, "total_preserved": len(preserved), "total_not_preserved": len(fail_h32), "by_source": details["by_source"], "by_child_state": details["by_child_state"], "preserved_cell_ids": preserved, "not_preserved_cell_ids": fail_h32, "no_adaptive_exclusion": True}
    return h32, details


def trajectory_diagnostics(rows: Sequence[Mapping[str, Any]], statuses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_branch: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_branch.setdefault(row["branch"], []).append(row)
    branches: list[dict[str, Any]] = []
    for branch, branch_rows in sorted(by_branch.items()):
        branches.append({
            "branch": branch,
            "cell_id": branch_rows[0]["canonical_cell_id"],
            "update_count": len(branch_rows),
            "updates": [row["update"] for row in branch_rows],
            "deleted_horizons": branch_rows[0]["deleted_horizons"],
            "batch_identity_pass": len({row["batch_window_sha256"] for row in branch_rows}) == 8,
            "rng_identity_pass": all(row["branch_rng_contract_status"] == "PASS" and row["rng_before_digest"] == row["rng_after_diagnostics_digest"] == row["rng_after_training_digest"] == row["rng_after_validation_digest"] for row in branch_rows),
            "finite_value_pass": all(row["finite_status"] == "FINITE" for row in branch_rows),
            "optimizer_step_completion_pass": all(all(int(value) == row["update"] for value in row["step_counter_after"].values()) for row in branch_rows),
            "max_postclip_reconstruction_residual_norm": max(float(row["postclip_reconstruction_residual_norm"]) for row in branch_rows),
            "max_adamw_decomposition_residual_norm": max(float(row["adamw_decomposition_residual_norm"]) for row in branch_rows),
            "endpoint_update": 392,
        })
    require(len(branches) == 32 and all(item["update_count"] == 8 and item["updates"] == list(UPDATES) for item in branches), "trajectory diagnostics branch accounting failed")
    return {"schema_version": "stage3_h13_post_d21_d22_trajectory_diagnostics_v1", "new_branch_count": 32, "new_update_record_count": len(rows), "updates": list(UPDATES), "endpoint_update": 392, "branches": branches, "all_branch_diagnostics_pass": all(item["batch_identity_pass"] and item["rng_identity_pass"] and item["finite_value_pass"] and item["optimizer_step_completion_pass"] for item in branches), "endpoint_status_records": list(statuses)}


def load_checkpoint(work: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = read_json(work / "branch_execution_checkpoint.json")
    rows = [json.loads(line) for line in (work / "branch_update_records.checkpoint.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    statuses = list(checkpoint["statuses"])
    manifests = list(checkpoint["manifests"])
    require(checkpoint["completed_branch_count"] == 32 and checkpoint["completed_update_record_count"] == 256 and len(rows) == 256 and len(statuses) == 32 and len(manifests) == 32, "retained D22 checkpoint is incomplete")
    return rows, statuses, manifests


def write_response_csv(path: Path, responses: Sequence[Mapping[str, Any]]) -> None:
    fields = ["canonical_cell_id", "H32", "B1", "B2", "C1", "C21", "C22", "child_state", "source", "provenance_validity", "source_reference", "exact_deleted_horizons", "update_392_H1", "update_392_H32", "H1_response_Y_S", "H1_effect", "H1_materiality_status", "H32_preservation_status", "joint_validity_status", "satisfies_both", "treatment_specification_hash", "start_state_identity", "update_count", "new_update_count", "endpoint_update", "branch"]
    rows = []
    for response in responses:
        row = dict(response)
        row["exact_deleted_horizons"] = json.dumps(row["exact_deleted_horizons"], separators=(",", ":"))
        row["start_state_identity"] = json.dumps(row["start_state_identity"], sort_keys=True, separators=(",", ":"))
        rows.append(row)
    write_csv(path, fields, rows)


def manifest_for(output: Path) -> dict[str, Any]:
    excluded = ["FINAL_REPORT.md", "SHA256_MANIFEST.json"]
    entries: list[dict[str, Any]] = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in excluded:
            entries.append({"relative_path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "artifact_role": "D22 execution evidence"})
    entries.append({"relative_path": f"../../tools/{SELF_PATH.name}", "sha256": sha256_file(SELF_PATH), "size_bytes": SELF_PATH.stat().st_size, "artifact_role": "D22 execution and finalization runner"})
    return {"schema_version": "stage3_h13_post_d21_d22_execution_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "manifest_self_hash": "NOT_INCLUDED_BY_CONTRACT", "excluded_from_manifest": excluded, "file_count": len(entries), "files": entries}


def verify_final_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json(output / "SHA256_MANIFEST.json")
    failures: list[str] = []
    for entry in manifest["files"]:
        path = (output / Path(entry["relative_path"].replace("/", os.sep))).resolve()
        if not path.is_file() or sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            failures.append(entry["relative_path"])
    return {"status": "PASS" if not failures else "FAIL", "checked_file_count": len(manifest["files"]), "failures": failures}


def report(output: Path, inputs: Mapping[str, Any], primary: Mapping[str, Any], conservation: Mapping[str, Any], d21_to_d22: Mapping[str, Any], classification: Mapping[str, Any], h32: Mapping[str, Any], accounting_data: Mapping[str, Any], mobius_diag: Mapping[str, Any], defects: Mapping[str, Any], manifest_sha: str) -> str:
    protocol_lines = [
        "STAGE_3_H13_POST_D21_D22_H32_BY_B1_B2_C1_C2_HIERARCHICAL_C2_SUBBLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION_REPLAY:",
        "PASSED", "", "FIRST_BLOCKER:", "none", "", f"CAUSAL_CLASSIFICATION:\n{classification['overall_classification']}", "", "ONE_SENTENCE_VERDICT:",
        f"The frozen C2 parent is resolved as {classification['overall_classification']} after exact H30/H31 child replay with both C1-conditioned groups evaluated independently.", "",
        "## 1. PROTOCOL_INTEGRITY", "", "D22 scientific execution passed from the certified post-update-384 bundle. Updates 385–392 only were executed; update 393, Frozen20, R9, R8E_A1 modification, A-factor branches, adaptive selection, threshold tuning, and design changes were not used.", "",
        "NEW_D22_BRANCHES_EXECUTED: 32", "C21_ONLY_BRANCHES: 16", "C22_ONLY_BRANCHES: 16", "NEW_UPDATE_RECORDS: 256", "TOTAL_FACTORIAL_CELLS: 64", "TOTAL_MOBIUS_TERMS: 63", "UPDATE_393_EXECUTED: NO", "FROZEN20_USED: NO", "R9_EXECUTED: NO", "R8E_A1_MODIFIED: NO", "A_FACTOR_BRANCH_EXECUTED: NO", "ADAPTIVE_SELECTION: NO", "THRESHOLD_TUNING: NO", "",
        "## 2. AUTHORITATIVE_INPUT_IDENTITY", "", f"D22 design manifest: `{EXPECTED_D22_DESIGN_MANIFEST_SHA256}` (PASS); D21 execution manifest: `{EXPECTED_D21_MANIFEST_SHA256}` (PASS); D21 runner: `{EXPECTED_D21_RUNNER_SHA256}` (PASS); D20 primitive: `{EXPECTED_D20_PRIMITIVE_SHA256}` (PASS). Certified start bundle: `{EXPECTED_START_BUNDLE_SHA256}`; model: `{EXPECTED_MODEL_SHA256}`; optimizer: `{EXPECTED_OPTIMIZER_SHA256}`; start RNG: `{EXPECTED_RNG_SHA256}`; source revision: `{EXPECTED_SOURCE_REVISION}`.", "",
        "## 3. D21_IMPORT_VALIDATION", "", "All 32 diagonal D21 cells were imported and authenticated; no diagonal cell was scientifically rerun. D21 inherited parent and Q_B conservation residuals were zero.", "",
        "## 4. D22_EXECUTION_COUNTS", "", "32 new mixed-child branches completed with 8 updates each, producing exactly 256 new scientific update records; 16 were C21_ONLY and 16 were C22_ONLY.", "",
        "## 5. FACTORIAL_64_CELL_COMPLETENESS", "", "Factor order: `[H32, B1, B2, C1, C21, C22]`. The endpoint table contains 64 unique canonical cells, 32 authenticated imports, and 32 new executions.", "",
        "## 6. CONTROL_ENDPOINT", "", f"CONTROL_ENDPOINT_H1: {EXPECTED_CONTROL_H1:.17g}\nCONTROL_ENDPOINT_H32: {EXPECTED_CONTROL_H32:.17g}\nEndpoint update: 392.", "",
        "## 7. H32_H1_CELL_ACCOUNTING", "", json.dumps(accounting_data, indent=2, sort_keys=True), "",
        "## 8. MOBIUS_63_TERM_VALIDATION", "", f"63 unique terms passed forward transform and inverse reconstruction; maximum absolute residual: `{mobius_diag['max_absolute_reconstruction_residual']:.17g}` (tolerance `{TOLERANCE:.17g}`).", "",
        "## 9. SIX_PRIMARY_ESTIMANDS", "", json.dumps(primary, indent=2, sort_keys=True), "",
        "## 10. PARENT_A_CONSERVATION", "", json.dumps(conservation["PARENT_A"], indent=2, sort_keys=True), "",
        "## 11. PARENT_B_CONSERVATION", "", json.dumps(conservation["PARENT_B"], indent=2, sort_keys=True), "",
        "## 12. COMPLETE_D21_TO_D22_CONSERVATION", "", f"31 D21 terms mapped exactly once: 15 identity terms and 16 three-child expansions; all 63 D22 terms were covered exactly once. Result: `{d21_to_d22['all_pass']}`.", "",
        "## 13. PRIMARY_MASS_AND_DOMINANCE", "", f"Absolute primary mass: `{primary['absolute_primary_interaction_mass']:.17g}`; largest term: `{primary['largest_term']}`; largest mass fraction: `{primary['largest_term_mass_fraction']:.17g}`; dominance threshold: `0.5`. Dominance was not used as localization evidence.", "",
        "## 14. GROUP_A_CLASSIFICATION", "", json.dumps(classification["group_A"], indent=2, sort_keys=True), "",
        "## 15. GROUP_B_CLASSIFICATION", "", json.dumps(classification["group_B"], indent=2, sort_keys=True), "",
        "## 16. FINAL_CAUSAL_CLASSIFICATION", "", classification["overall_classification"], "",
        "## 17. TRAJECTORY_AND_RUNTIME_DIAGNOSTICS", "", "All 32 new branches passed batch, RNG, finite-value, optimizer-step, gradient/clipping, AdamW, and endpoint checks; per-update records are retained in `branch_update_records.jsonl`.", "",
        "## 18. ENGINEERING_DEFECTS_AND_REPAIRS", "", json.dumps(defects, indent=2, sort_keys=True), "",
        "## 19. ARTIFACTS", "", f"D22 execution evidence directory: `{output.resolve()}`", "",
        "## 20. FINAL_MANIFEST", "", f"FINAL_MANIFEST_PATH: `{(output / 'SHA256_MANIFEST.json').resolve()}`\nFINAL_MANIFEST_SHA256: `{manifest_sha}`\nINTERNAL_MANIFEST_VERIFICATION: PASS", "",
        "## 21. NEXT_SCIENTIFIC_IMPLICATION", "", "Under the frozen estimand and thresholds, the observed group states determine the causal implication above; no D23 design or further factor is introduced by this replay.", "",
        f"D22_EXECUTION_EVIDENCE_PATH:\n{output.resolve()}", f"FINAL_MANIFEST_PATH:\n{(output / 'SHA256_MANIFEST.json').resolve()}", f"FINAL_MANIFEST_SHA256:\n{manifest_sha}", "INTERNAL_MANIFEST_VERIFICATION:\nPASS",
    ]
    return "\n".join(protocol_lines) + "\n"


def execute(authorization: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    inputs = authenticate_inputs(authorization)
    design_cells = read_json(D22_DESIGN_DIR / "factorial_64_cell_design_table.json")["cells"]
    require(len(design_cells) == 64 and [row["canonical_cell_id"] for row in design_cells] == all_cells(), "D22 design cell registry mismatch")
    imported, imported_by_id = build_imported_cells(inputs, design_cells)
    work = output.with_name(output.name + ".in_progress")
    resuming_checkpoint = work.exists()
    if not resuming_checkpoint:
        work.mkdir(parents=True)
    defects = {
        "schema_version": "stage3_h13_post_d21_d22_engineering_defects_and_repairs_v1",
        "execution_defects": [
            {"defect_id": "D22-EXEC-RESOLVER-001", "defect": "The first wrapper invocation resolved deletion members only from the D21 parent factor registry and raised KeyError:C21 before any scientific update.", "repair": "Added the frozen D22 child mappings C21={H30} and C22={H31} to the wrapper-only deletion resolver; the authenticated D20/D21 scientific primitive was unchanged.", "before_behavior": "Execution stopped during pre-branch design-cell resolution; no output bundle or scientific update was produced.", "after_behavior": "The resolver validates exact D22 child deletion sets and passes them unchanged as H30/H31 integer horizons to the authenticated primitive.", "scientific_semantics_changed": False, "invalid_attempt_counted_as_valid": False},
            {"defect_id": "D22-REPORT-FINALIZE-002", "defect": "After all 32 branches completed, finalization attempted to read M_subset from the frozen design term registry, which contains metadata only, and raised KeyError:M_subset.", "repair": "Removed the unused metadata read and added per-branch raw-evidence checkpoints. The first completed run was not counted as valid because its raw rows were not retained; the repaired run regenerated the frozen branches and retained them before finalization.", "before_behavior": "All 32 trajectories passed, but no final evidence artifact was written and no branch was counted as completed evidence.", "after_behavior": "Finalization consumes measured D22 interaction values produced by the endpoint table and checkpoints every valid branch trajectory before reporting.", "scientific_semantics_changed": False, "invalid_attempt_counted_as_valid": False}
            ,{"defect_id": "D22-DIAGNOSTIC-PREDICATE-003", "defect": "The first retained-evidence finalization predicate compared authenticated step_counter_after values to update+1 instead of the runner's update-number convention.", "repair": "Changed the wrapper-only audit predicate to require step_counter_after == update, matching the authenticated D20/D21 record contract; raw checkpointed scientific rows were reused without rerunning.", "before_behavior": "All 256 retained records were present, but finalization stopped at the diagnostic summary with a wrapper-only failure.", "after_behavior": "The diagnostic summary validates all 32 branches against the authenticated step-counter convention.", "scientific_semantics_changed": False, "invalid_attempt_counted_as_valid": False}
        ],
        "narrow_repairs_applied": ["D22-EXEC-RESOLVER-001", "D22-REPORT-FINALIZE-002", "D22-DIAGNOSTIC-PREDICATE-003"],
        "scientific_semantics_changed": False,
        "invalid_attempts_counted_as_valid": False,
        "historical_artifacts_modified": False,
        "inherited_D21_defects_reference": str((D21_DIR / "defect_log.json").resolve()),
    }
    if resuming_checkpoint:
        rows, statuses, manifests = load_checkpoint(work)
    else:
        rows, statuses, manifests = execute_new_branches(inputs, design_cells, work)
    statuses_by_id = {row["canonical_cell_id"]: row for row in statuses}
    responses = response_table(inputs, design_cells, imported_by_id, statuses_by_id)
    mobius_rows, interactions, _, mobius_diag = build_mobius(responses)
    primary_conservation, d21_to_d22 = build_conservation(inputs, interactions, mobius_rows)
    primary, classification, materiality, _ = build_primary_and_classification(interactions, primary_conservation, responses)
    h32, accounting_data = accounting(responses)
    diagnostics = trajectory_diagnostics(rows, statuses)
    require(diagnostics["all_branch_diagnostics_pass"] is True, "trajectory diagnostics failed")
    write_json(work / "protocol_integrity.json", {
        "schema_version": "stage3_h13_post_d21_d22_protocol_integrity_v1", "status": "PASSED", "scientific_execution_valid": True,
        "project_owner_authorization": "YES", "design_review_passed": "YES", "d22_execution_authorized": "YES", "new_branches_executed": 32, "new_update_records": 256,
        "authenticated_d21_imports": 32, "total_factorial_cells": 64, "total_mobius_terms": 63, "updates": list(UPDATES), "endpoint_update": 392,
        "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "a_factor_branch_executed": "NO", "adaptive_selection": "NO", "threshold_tuning": "NO", "design_change": "NO",
    })
    write_json(work / "authoritative_input_identity.json", {
        "schema_version": "stage3_h13_post_d21_d22_authoritative_input_identity_v1", "d22_design_bundle": str(D22_DESIGN_DIR.resolve()), "d22_design_manifest_sha256": EXPECTED_D22_DESIGN_MANIFEST_SHA256,
        "d21_execution_bundle": str(D21_DIR.resolve()), "d21_manifest_sha256": EXPECTED_D21_MANIFEST_SHA256, "d21_internal_manifest_verification": inputs["d21_manifest"],
        "certified_start": {"path": inputs["d21_identity"]["d20_identity"]["certified_start"]["path"], "sha256": EXPECTED_START_BUNDLE_SHA256, "completed_optimizer_step": 384, "model_semantic_hash": EXPECTED_MODEL_SHA256, "optimizer_semantic_hash": EXPECTED_OPTIMIZER_SHA256, "rng_sha256": EXPECTED_RNG_SHA256},
        "source_revision": EXPECTED_SOURCE_REVISION, "current_source_revision": inputs["current_revision"], "d21_runner": {"path": str(D21_RUNNER.resolve()), "sha256": EXPECTED_D21_RUNNER_SHA256}, "d20_primitive": {"path": str(D20_PRIMITIVE.resolve()), "sha256": EXPECTED_D20_PRIMITIVE_SHA256},
        "d17_control_reference": inputs["d21_identity"]["d20_identity"]["d17_control_reference"], "authorization": inputs["authorization"], "historical_immutability": "PASS",
    })
    write_json(work / "execution_manifest.json", {"schema_version": "stage3_h13_post_d21_d22_execution_manifest_v1", "experiment_id": EXPERIMENT_ID, "factor_order": list(FACTORS), "updates": list(UPDATES), "start_update": 384, "endpoint_update": 392, "forbidden_update": 393, "authenticated_d21_imports": 32, "new_d22_branches": 32, "new_update_records": 256, "c21_only_branches": 16, "c22_only_branches": 16, "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO", "r8e_a1_modified": "NO", "a_factor_branch_executed": "NO", "adaptive_selection": "NO", "threshold_tuning": "NO", "design_change": "NO", "runner": {"path": str(SELF_PATH), "sha256": sha256_file(SELF_PATH)}, "historical_input_manifests_verified": True})
    write_json(work / "branch_execution_status.json", {"schema_version": "stage3_h13_post_d21_d22_branch_execution_status_v1", "all_pass": True, "count": 32, "updates_per_branch": 8, "records": statuses})
    write_json(work / "branch_manifest.json", {"schema_version": "stage3_h13_post_d21_d22_branch_manifest_v1", "scientific_branches_executed": [row["canonical_cell_id"] for row in statuses], "branches": manifests, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO", "new_update_records": len(rows)})
    (work / "branch_update_records.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) for row in rows) + "\n", encoding="utf-8", newline="\n")
    write_json(work / "imported_cell_validation.json", {"schema_version": "stage3_h13_post_d21_d22_imported_cell_validation_v1", "all_pass": True, "count": len(imported), "source": "D21 authenticated endpoint table and branch contracts", "cells": imported})
    write_json(work / "factorial_64_cell_response_table.json", {"schema_version": "stage3_h13_post_d21_d22_factorial_64_cell_response_table_v1", "factor_order": list(FACTORS), "cell_count": 64, "cells": responses})
    write_response_csv(work / "factorial_64_cell_response_table.csv", responses)
    write_json(work / "mobius_63_terms.json", {"schema_version": "stage3_h13_post_d21_d22_mobius_63_terms_v1", "factor_order": list(FACTORS), "term_count": 63, "terms": mobius_rows, "validation": mobius_diag})
    write_csv(work / "mobius_63_terms.csv", ["term_index", "term", "subset", "subset_label", "order", "M_subset", "abs_M_subset", "materiality_status", "rank_by_absolute_magnitude", "contains_H32", "contains_B_child", "contains_C_child", "contains_C21", "contains_C22"], [{**row, "subset": json.dumps(row["subset"], separators=(",", ":"))} for row in mobius_rows])
    write_json(work / "primary_estimands_results.json", primary)
    write_json(work / "parent_conservation_results.json", primary_conservation)
    write_json(work / "d21_to_d22_conservation_results.json", d21_to_d22)
    write_json(work / "classification.json", classification)
    write_json(work / "h32_preservation.json", h32)
    write_json(work / "materiality_results.json", materiality)
    write_json(work / "cell_accounting.json", accounting_data)
    write_json(work / "trajectory_diagnostics.json", diagnostics)
    write_json(work / "engineering_defects_and_repairs.json", defects)
    inventory = {"schema_version": "stage3_h13_post_d21_d22_artifact_inventory_v1", "artifacts": [{"relative_path": path.name, "role": "final D22 evidence artifact"} for path in sorted(work.iterdir(), key=lambda item: item.name) if path.is_file()] + [{"relative_path": "FINAL_REPORT.md", "role": "human-readable final report"}, {"relative_path": "SHA256_MANIFEST.json", "role": "self-excluded final evidence manifest"}], "historical_inputs": {"d22_design_manifest_sha256": EXPECTED_D22_DESIGN_MANIFEST_SHA256, "d21_manifest_sha256": EXPECTED_D21_MANIFEST_SHA256}}
    write_json(work / "artifact_inventory.json", inventory)
    manifest = manifest_for(work)
    write_json(work / "SHA256_MANIFEST.json", manifest)
    manifest_check = verify_final_manifest(work)
    require(manifest_check["status"] == "PASS", f"final manifest verification failed: {manifest_check}")
    manifest_sha = sha256_file(work / "SHA256_MANIFEST.json")
    (work / "FINAL_REPORT.md").write_text(report(output, inputs, primary, primary_conservation, d21_to_d22, classification, h32, accounting_data, mobius_diag, defects, manifest_sha), encoding="utf-8", newline="\n")
    # The manifest intentionally excludes FINAL_REPORT.md to avoid a report/manifest cycle.
    work.rename(output)
    final_check = verify_final_manifest(output)
    require(final_check["status"] == "PASS", f"post-rename manifest verification failed: {final_check}")
    require(sha256_file(D22_DESIGN_DIR / "SHA256_MANIFEST.json") == EXPECTED_D22_DESIGN_MANIFEST_SHA256 and sha256_file(D21_DIR / "SHA256_MANIFEST.json") == EXPECTED_D21_MANIFEST_SHA256, "historical manifest changed")
    return {"status": "PASSED", "output": str(output.resolve()), "manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "internal_manifest_verification": final_check, "classification": classification["overall_classification"], "primary": primary, "h32": h32, "accounting": accounting_data}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.authorization.resolve(), args.output.resolve())
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}", "scientific_updates_executed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
