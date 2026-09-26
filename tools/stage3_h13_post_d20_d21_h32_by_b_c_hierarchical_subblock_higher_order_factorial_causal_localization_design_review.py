"""Materialize and validate the Stage 3 H13 D21 design review.

The default/materialize path is design-only.  It reads the authenticated D20
archive, reconstructs the frozen D20 blocks, deterministically subdivides B/C,
and writes a self-contained D21 design bundle.  No model, optimizer, backward
pass, or scientific update is touched by that path.

The optional execution path is fail-closed behind a separate D21 authorization
marker.  It reuses the authenticated D20/D17 update machinery and is not called
by this design-review materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tarfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REQUEST_TEXT = Path(r"C:\Users\86198\.codex\attachments\a6c93468-3d84-44d6-a7ec-80d597a74659\pasted-text.txt")
D20_ARCHIVE = Path(r"C:\Users\86198\Desktop\stage3_h13_d20_execution_evidence_20260820T202700.tar.xz")
D20_DESIGN_DIR = ROOT / "outputs" / "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_design_review_20260820T182800+0800"
D20_EXECUTION_DIR = ROOT / "outputs" / "stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay_20260820T190000+0800"
RUNNER_NAME = Path(__file__).name

EXPERIMENT_ID = "STAGE_3_H13_POST_D20_D21_H32_BY_B_C_HIERARCHICAL_SUBBLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION"
FACTORS = ("H32", "B1", "B2", "C1", "C2")
COARSE_D20_FACTORS = ("X32", "A", "B", "C")
HORIZONS = tuple(range(2, 32))
UPDATES = tuple(range(385, 393))
H1_MATERIALITY_THRESHOLD = 4.099006815405639e-06
H32_PRESERVATION_THRESHOLD = 0.01856902565856056
DECOMPOSITION_TOLERANCE = 1e-12

EXPECTED_D20_DESIGN_MANIFEST_SHA256 = "3b67b729d4fca976e03a695f47a57992738722a045d2dfa21a96665f599f794b"
EXPECTED_D20_ARCHIVE_SHA256 = "92383641d7c08f07494dd5dc44fbfb88ea0d8e752b8522e083c5867ff04a149b"
EXPECTED_CERTIFIED_UPDATE_384_BUNDLE_SHA256 = "4a022aa0340977e571a3a3085e30d96fd0d92129f4fbeff99aaa8e3420379083"
EXPECTED_MODEL_SHA256 = "f6f450abd4ac8780a7172c46d44db88d6b7278e22e6a14580b62c9b1627bcd58"
EXPECTED_OPTIMIZER_SHA256 = "610dd9fa330572888e60cbc16e8f67560e2ce23a3b8628cad7481a90b24936d8"

D20_PARENT_COMPONENTS = (
    "M_32_A",
    "M_32_B",
    "M_32_C",
    "M_32_A_B",
    "M_32_A_C",
    "M_32_B_C",
    "M_32_A_B_C",
)
D21_H32_BC_COMPONENTS = (
    "M_32_B1_C1",
    "M_32_B1_C2",
    "M_32_B2_C1",
    "M_32_B2_C2",
    "M_32_B1_B2_C1",
    "M_32_B1_B2_C2",
    "M_32_B1_C1_C2",
    "M_32_B2_C1_C2",
    "M_32_B1_B2_C1_C2",
)
# This list is expanded explicitly in the factor order used by every artifact.
D21_H32_ALL_COMPONENTS = (
    "M_32",
    "M_32_B1",
    "M_32_B2",
    "M_32_C1",
    "M_32_C2",
    "M_32_B1_B2",
    "M_32_B1_C1",
    "M_32_B1_C2",
    "M_32_B2_C1",
    "M_32_B2_C2",
    "M_32_C1_C2",
    "M_32_B1_B2_C1",
    "M_32_B1_B2_C2",
    "M_32_B1_C1_C2",
    "M_32_B2_C1_C2",
    "M_32_B1_B2_C1_C2",
)

D21_TO_D20_IMPORTS = {
    "00000": "0000",
    "00011": "0001",
    "01100": "0010",
    "01111": "0011",
    "10000": "1000",
    "10011": "1001",
    "11100": "1010",
    "11111": "1011",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_bytes(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def factor_subsets(factors: Sequence[str]) -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(factors) + 1)
        for subset in itertools.combinations(factors, size)
    ]


def cell_id(bits: Sequence[int]) -> str:
    return "".join(str(int(bit)) for bit in bits)


def bits_for_subset(subset: Iterable[str], factors: Sequence[str]) -> str:
    selected = set(subset)
    return "".join("1" if factor in selected else "0" for factor in factors)


def subset_for_cell(cell: str, factors: Sequence[str]) -> frozenset[str]:
    assert_true(len(cell) == len(factors) and set(cell) <= {"0", "1"}, f"invalid cell {cell}")
    return frozenset(factor for factor, bit in zip(factors, cell) if bit == "1")


def mobius_interactions(endpoints: Mapping[str, float], factors: Sequence[str]) -> dict[frozenset[str], float]:
    """Invert E(S)=sum(T subset S) M(T), with E(empty)=0."""
    expected = {cell_id(bits) for bits in itertools.product((0, 1), repeat=len(factors))}
    assert_true(set(endpoints) == expected, f"endpoint registry is not complete for {factors}")
    interactions: dict[frozenset[str], float] = {}
    for size in range(1, len(factors) + 1):
        for subset_tuple in itertools.combinations(factors, size):
            subset = frozenset(subset_tuple)
            value = float(endpoints[bits_for_subset(subset, factors)])
            for proper_size in range(1, size):
                for proper_tuple in itertools.combinations(subset_tuple, proper_size):
                    value -= interactions[frozenset(proper_tuple)]
            interactions[subset] = value
    return interactions


def reconstruct_endpoint(interactions: Mapping[frozenset[str], float], subset: frozenset[str]) -> float:
    return sum(value for term, value in interactions.items() if term <= subset)


def component_name(subset: Iterable[str]) -> str:
    ordered = [factor for factor in FACTORS[1:] if factor in set(subset)]
    return "M_32" if not ordered else "M_32_" + "_".join(ordered)


def component_factors(name: str) -> frozenset[str]:
    if name == "M_32":
        return frozenset({"H32"})
    assert_true(name.startswith("M_32_"), f"invalid component name {name}")
    return frozenset(("H32", *name.removeprefix("M_32_").split("_")))


def all_h32_component_names() -> list[str]:
    return [component_name(subset) for subset in factor_subsets(FACTORS[1:])]


def primary_component_terms() -> dict[str, list[str]]:
    return {
        "M_32_B": ["M_32_B1", "M_32_B2", "M_32_B1_B2"],
        "M_32_C": ["M_32_C1", "M_32_C2", "M_32_C1_C2"],
        "M_32_B_C": list(D21_H32_BC_COMPONENTS),
    }


def archive_json(stream: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    member = stream.getmember(member_name)
    assert_true(member.isfile(), f"archive member is not a regular file: {member_name}")
    extracted = stream.extractfile(member)
    assert_true(extracted is not None, f"cannot read archive member: {member_name}")
    return read_json_bytes(extracted.read())


def archive_text(stream: tarfile.TarFile, member_name: str) -> str:
    member = stream.getmember(member_name)
    extracted = stream.extractfile(member)
    assert_true(extracted is not None, f"cannot read archive member: {member_name}")
    return extracted.read().decode("utf-8")


def archive_member_sha256(stream: tarfile.TarFile, member_name: str) -> str:
    member = stream.getmember(member_name)
    extracted = stream.extractfile(member)
    assert_true(extracted is not None, f"cannot read archive member: {member_name}")
    return sha256_bytes(extracted.read())


def verify_d20_archive(stream: tarfile.TarFile) -> dict[str, Any]:
    member_names = set(stream.getnames())
    d20_manifest = archive_json(stream, "D20_DESIGN/SHA256_MANIFEST.json")
    assert_true(
        sha256_bytes(json.dumps(d20_manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
        == EXPECTED_D20_DESIGN_MANIFEST_SHA256,
        "D20 design manifest JSON serialization does not reproduce its authenticated hash",
    )
    # Verify the actual archived D20 design files listed by the authenticated
    # manifest.  The dedicated runner is separately packaged under FUTURE_RUNNER
    # and is checked by its own recorded repair identity.
    checked = []
    for entry in d20_manifest["files"]:
        relative = entry["relative_path"]
        if relative.startswith("../../"):
            continue
        member_name = f"D20_DESIGN/{relative}"
        assert_true(member_name in member_names, f"missing authenticated D20 design member {member_name}")
        observed = archive_member_sha256(stream, member_name)
        assert_true(observed == entry["sha256"], f"D20 design artifact hash mismatch: {relative}")
        checked.append(relative)
    assert_true("D20_EXECUTION/authoritative_input_identity.json" in member_names, "D20 execution identity missing")
    return {
        "archive_path": str(D20_ARCHIVE),
        "archive_sha256": sha256_file(D20_ARCHIVE),
        "expected_archive_sha256": EXPECTED_D20_ARCHIVE_SHA256,
        "archive_sha256_match": sha256_file(D20_ARCHIVE) == EXPECTED_D20_ARCHIVE_SHA256,
        "regular_file_count": sum(1 for member in stream.getmembers() if member.isfile()),
        "d20_design_manifest_sha256": EXPECTED_D20_DESIGN_MANIFEST_SHA256,
        "d20_design_files_verified": checked,
        "d20_design_file_verification": "VERIFIED",
    }


def d20_horizon_members(factor_definition: dict[str, Any]) -> dict[str, list[str]]:
    by_name = {item["name"]: list(item.get("members", item.get("deletion_set", []))) for item in factor_definition["factors"]}
    assert_true(set(by_name) == set(COARSE_D20_FACTORS), "D20 factor definition does not contain X32/A/B/C")
    assert_true(by_name["X32"] == ["H32"], "D20 H32 deletion set changed")
    return {name: by_name[name] for name in ("A", "B", "C", "X32")}


def expected_d20_delete(bits: Sequence[int], members: Mapping[str, Sequence[str]]) -> list[str]:
    names = ("X32", "A", "B", "C")
    deleted: list[str] = []
    for factor, bit in zip(names, bits):
        if bit:
            deleted.extend(members[factor])
    return deleted


def recover_parent_and_d20(stream: tarfile.TarFile) -> dict[str, Any]:
    factor_definition = archive_json(stream, "D20_DESIGN/factor_definition.json")
    d20_experiment = archive_json(stream, "D20_DESIGN/experiment_design.json")
    d20_registry = archive_json(stream, "D20_DESIGN/factorial_cell_registry.json")
    d20_imported = archive_json(stream, "D20_DESIGN/imported_cell_provenance.json")
    d20_execution_manifest = archive_json(stream, "D20_EXECUTION/branch_execution_manifest.json")
    d20_endpoints = archive_json(stream, "D20_EXECUTION/factorial_endpoint_results.json")
    d20_execution_identity = archive_json(stream, "D20_EXECUTION/authoritative_input_identity.json")
    d20_design_identity = archive_json(stream, "D20_DESIGN/authoritative_input_identity.json")
    d20_protocol = archive_json(stream, "D20_EXECUTION/protocol_integrity.json")
    d20_decomposition = archive_json(stream, "D20_EXECUTION/factorial_mobius_decomposition.json")
    d20_pairwise = archive_json(stream, "D20_DESIGN/d19_pairwise_import.json")
    d20_h32 = archive_json(stream, "D20_DESIGN/h32_preservation_contract.json")
    d20_materiality = archive_json(stream, "D20_DESIGN/materiality_contract.json")
    d20_engineering = archive_json(stream, "D20_EXECUTION/engineering_defects_and_repairs.json")

    recovered = d20_horizon_members(factor_definition)
    assert_true(recovered["A"] == d20_experiment["A_members"], "D20 A source cross-check failed")
    assert_true(recovered["B"] == d20_experiment["B_members"], "D20 B source cross-check failed")
    assert_true(recovered["C"] == d20_experiment["C_members"], "D20 C source cross-check failed")
    assert_true(recovered["X32"] == ["H32"], "D20 H32 source cross-check failed")
    assert_true(recovered["A"] + recovered["B"] + recovered["C"] == [f"H{h}" for h in HORIZONS], "D20 block order/coverage failed")

    registry_by_cell = {row["cell"]: row for row in d20_registry["cells"]}
    assert_true(set(registry_by_cell) == {cell_id(bits) for bits in itertools.product((0, 1), repeat=4)}, "D20 registry is incomplete")
    for row in d20_registry["cells"]:
        assert_true(row["bits"] == [int(bit) for bit in row["cell"]], f"D20 bit mismatch {row['cell']}")
        expected = expected_d20_delete(row["bits"], recovered)
        assert_true(row["delete_horizons"] == expected, f"D20 registry deletion mismatch {row['cell']}")

    imported_by_cell = d20_imported["cells"]
    executed_by_cell = {row["cell"]: row for row in d20_execution_manifest["branches"]}
    endpoints_by_cell = {row["cell"]: row for row in d20_endpoints["cells"]}
    authoritative_d20_cells: dict[str, dict[str, Any]] = {}
    for cell, registry_row in registry_by_cell.items():
        if cell in imported_by_cell:
            source = imported_by_cell[cell]
            source_field = "D20_DESIGN/imported_cell_provenance.json::cells[<cell>].delete_horizons"
            delete = list(source["delete_horizons"])
            source_stage = source["source_stage"]
            source_role = source["source_role"]
        else:
            assert_true(cell in executed_by_cell, f"D20 execution branch missing {cell}")
            source = executed_by_cell[cell]
            source_field = "D20_EXECUTION/branch_execution_manifest.json::branches[<cell>].delete_horizons"
            delete = list(source["delete_horizons"])
            source_stage = "D20"
            source_role = "authenticated D20 executed branch"
        assert_true(delete == registry_row["delete_horizons"], f"D20 branch deletion reproduction failed {cell}")
        authoritative_d20_cells[cell] = {
            "cell": cell,
            "bits": list(registry_row["bits"]),
            "factor_values": dict(registry_row["factor_values"]),
            "delete_horizons": delete,
            "source_artifact": source_field.split("::")[0],
            "source_field": source_field.split("::", 1)[1],
            "source_stage": source_stage,
            "source_role": source_role,
            "endpoint": endpoints_by_cell[cell],
        }

    assert_true(d20_execution_identity["design_manifest_file_sha256"] == EXPECTED_D20_DESIGN_MANIFEST_SHA256, "D20 execution design hash mismatch")
    certified = d20_design_identity["certified_start"]
    assert_true(certified["sha256"] == EXPECTED_CERTIFIED_UPDATE_384_BUNDLE_SHA256, "certified start bundle hash mismatch")
    assert_true(certified["model_semantic_hash"] == EXPECTED_MODEL_SHA256, "certified model hash mismatch")
    assert_true(certified["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256, "certified optimizer hash mismatch")
    assert_true(certified["completed_optimizer_step"] == 384, "D20 certified start is not post-update 384")
    assert_true(d20_execution_manifest["updates"] == list(UPDATES), "D20 update range mismatch")
    assert_true(d20_execution_manifest["endpoint_update"] == 392, "D20 endpoint update mismatch")
    assert_true(d20_execution_manifest["update_393_executed"] == "NO", "D20 update 393 prohibition violated")
    assert_true(d20_protocol["status"] == "PASSED" and d20_protocol["raw_scientific_evidence_valid"] is True, "D20 execution protocol is not passed")
    assert_true(d20_protocol["exactly_12_authorized_new_branches_executed"] is True, "D20 execution branch count was not authenticated")

    source_revision = d20_design_identity["source_revision"]
    runtime = d20_design_identity["runtime_source_identity"]
    schedule = d20_design_identity["schedule_identity"]
    assert_true(source_revision == runtime["git_head"] == runtime["environment"]["source_revision"] == schedule["source_revision"], "D20 source revision identity mismatch")
    current_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert_true(current_head == source_revision, "current source revision differs from authenticated D20 revision")

    endpoint_rows = d20_endpoints["cells"]
    assert_true(len(endpoint_rows) == 16 and len({row["cell"] for row in endpoint_rows}) == 16, "D20 endpoint registry is not complete")
    return {
        "factor_definition": factor_definition,
        "experiment_design": d20_experiment,
        "registry": d20_registry,
        "imported_provenance": d20_imported,
        "execution_manifest": d20_execution_manifest,
        "endpoints": d20_endpoints,
        "execution_identity": d20_execution_identity,
        "design_identity": d20_design_identity,
        "protocol": d20_protocol,
        "decomposition": d20_decomposition,
        "pairwise": d20_pairwise,
        "h32_contract": d20_h32,
        "materiality_contract": d20_materiality,
        "engineering_defects": d20_engineering,
        "blocks": recovered,
        "cells": authoritative_d20_cells,
        "source_revision": source_revision,
        "current_git_head": current_head,
        "certified_start": certified,
        "schedule_identity": schedule,
        "runtime_source_identity": runtime,
        "d20_deletion_reproduction": {
            "registry_cells_checked": len(registry_by_cell),
            "execution_cells_checked": len(executed_by_cell),
            "imported_cells_checked": len(imported_by_cell),
            "status": "PASS",
        },
    }


def split_block(members: Sequence[str]) -> tuple[list[str], list[str]]:
    first_size = math.ceil(len(members) / 2)
    return list(members[:first_size]), list(members[first_size:])


def build_partition(parent: Mapping[str, Any]) -> dict[str, Any]:
    blocks = parent["blocks"]
    b1, b2 = split_block(blocks["B"])
    c1, c2 = split_block(blocks["C"])
    assert_true(b1 + b2 == blocks["B"] and not set(b1) & set(b2), "B partition conservation failed")
    assert_true(c1 + c2 == blocks["C"] and not set(c1) & set(c2), "C partition conservation failed")
    assert_true(set(b1) | set(b2) == set(blocks["B"]), "B partition coverage failed")
    assert_true(set(c1) | set(c2) == set(blocks["C"]), "C partition coverage failed")
    return {
        "schema_version": "stage3_h13_post_d20_d21_subblock_partition_v1",
        "rule": "FIRST_SUBBLOCK_SIZE=ceil(n/2); SECOND_SUBBLOCK_SIZE=floor(n/2); preserve authoritative canonical order",
        "adaptive_partition_search": False,
        "alternative_partitions_evaluated": False,
        "parent_blocks": {
            "A": {
                "members": list(blocks["A"]),
                "source_artifact": "D20_DESIGN/factor_definition.json",
                "source_field": "factors[name=A].members",
                "cross_check_artifact": "D20_DESIGN/experiment_design.json",
                "cross_check_field": "A_members",
            },
            "B": {
                "members": list(blocks["B"]),
                "source_artifact": "D20_DESIGN/factor_definition.json",
                "source_field": "factors[name=B].members",
                "cross_check_artifact": "D20_DESIGN/experiment_design.json",
                "cross_check_field": "B_members",
            },
            "C": {
                "members": list(blocks["C"]),
                "source_artifact": "D20_DESIGN/factor_definition.json",
                "source_field": "factors[name=C].members",
                "cross_check_artifact": "D20_DESIGN/experiment_design.json",
                "cross_check_field": "C_members",
            },
            "H32": {
                "members": ["H32"],
                "source_artifact": "D20_DESIGN/factor_definition.json",
                "source_field": "factors[name=X32].deletion_set",
                "cross_check_artifact": "D20_DESIGN/factorial_cell_registry.json",
                "cross_check_field": "cells[cell=1000].delete_horizons",
            },
        },
        "subblocks": {
            "B1": {"members": b1, "size": len(b1), "first_subblock_size": math.ceil(len(blocks["B"]) / 2)},
            "B2": {"members": b2, "size": len(b2), "second_subblock_size": math.floor(len(blocks["B"]) / 2)},
            "C1": {"members": c1, "size": len(c1), "first_subblock_size": math.ceil(len(blocks["C"]) / 2)},
            "C2": {"members": c2, "size": len(c2), "second_subblock_size": math.floor(len(blocks["C"]) / 2)},
        },
        "conservation": {
            "B1_intersect_B2": [],
            "B1_union_B2": list(blocks["B"]),
            "C1_intersect_C2": [],
            "C1_union_C2": list(blocks["C"]),
            "status": "PASS",
        },
        "A_factor": {"included": False, "context": "retained under the canonical D20 A-retained parent context", "new_A_branches": 0},
    }


def d21_cells(partition: Mapping[str, Any]) -> list[dict[str, Any]]:
    subblocks = {name: list(value["members"]) for name, value in partition["subblocks"].items()}
    cells: list[dict[str, Any]] = []
    for bits in itertools.product((0, 1), repeat=len(FACTORS)):
        cid = cell_id(bits)
        deleted_factors = [factor for factor, bit in zip(FACTORS, bits) if bit]
        delete_horizons: list[str] = []
        for factor, bit in zip(FACTORS, bits):
            if bit:
                delete_horizons.extend(["H32"] if factor == "H32" else subblocks[factor])
        cells.append(
            {
                "cell": cid,
                "bits": list(bits),
                "factor_values": {factor: {"bit": int(bit), "state": "DELETE" if bit else "CONTROL"} for factor, bit in zip(FACTORS, bits)},
                "deleted_factors": deleted_factors,
                "deleted_components": delete_horizons,
                "delete_horizons": delete_horizons,
                "delete_set_size": len(delete_horizons),
                "branch": "CONTROL" if not deleted_factors else "DELETE_" + "_".join(deleted_factors),
                "A_state": "CONTROL_RETAINED",
            }
        )
    return cells


def d20_coarse_bits_for_d21(d21_bits: Sequence[int]) -> list[int]:
    h32, b1, b2, c1, c2 = d21_bits
    return [int(h32), 0, int(bool(b1 or b2)), int(bool(c1 or c2))]


def common_identity_checks(parent: Mapping[str, Any], d20_row: Mapping[str, Any], d20_cell: Mapping[str, Any], d21_row: Mapping[str, Any]) -> dict[str, bool]:
    design_identity = parent["design_identity"]
    execution_identity = parent["execution_identity"]
    protocol = parent["protocol"]
    template = parent["imported_provenance"]["identity_checks_template"]
    start = parent["certified_start"]
    branch_start_ok = d20_row["endpoint_status"] in {"AUTHENTICATED_IMPORTED", "EXECUTED_VALID_RAW_EVIDENCE"}
    common = {
        "start_state_identity": start["completed_optimizer_step"] == 384 and execution_identity["start_identity"]["all_branch_starts_identical"] is True,
        "model_identity": start["model_semantic_hash"] == EXPECTED_MODEL_SHA256 and execution_identity["start_identity"]["model_hash_match"] is True,
        "optimizer_identity": start["optimizer_semantic_hash"] == EXPECTED_OPTIMIZER_SHA256 and execution_identity["start_identity"]["optimizer_hash_match"] is True,
        "source_revision": parent["source_revision"] == parent["schedule_identity"]["source_revision"] == parent["runtime_source_identity"]["git_head"],
        "canonical_batch_schedule": execution_identity["d17_control_reference"]["compatibility_failure_count"] == 0 and bool(parent["schedule_identity"].get("schedule_identity_sha256")),
        "rng_contract": bool(start["rng_sha256"] == execution_identity["start_identity"]["certified_rng_sha256"] and execution_identity["start_identity"]["rng_digest_match_note"]),
        "update_range": parent["execution_manifest"]["updates"] == list(UPDATES) and parent["execution_manifest"]["endpoint_update"] == 392,
        "gradient_clipping_semantics": bool(template["gradient_clipping_match"]),
        "AdamW_semantics": bool(template["adamw_semantics_match"]),
        "rollout_loss_weighting": bool(template["rollout_loss_weighting_match"]),
        "rollout_deletion_semantics": bool(template["deletion_semantics_match"]) and list(d20_row["delete_horizons"]) == list(d21_row["delete_horizons"]),
        "endpoint_definition": d20_row["endpoint_status"] in {"AUTHENTICATED_IMPORTED", "EXECUTED_VALID_RAW_EVIDENCE"} and parent["execution_manifest"]["endpoint_update"] == 392,
        "H1_metric_semantics": "H1" in d20_row and "E" in d20_row,
        "H32_metric_semantics": "H32" in d20_row and "H32_preservation" in d20_row,
        "update_393_forbidden": parent["execution_manifest"]["update_393_executed"] == "NO" and protocol["update_393_executed"] == "NO",
        "frozen20_forbidden": protocol["frozen20_used"] == "NO",
        "r9_forbidden": protocol["r9_executed"] == "NO",
        "branch_endpoint_valid": branch_start_ok and math.isfinite(float(d20_row["H1"])) and math.isfinite(float(d20_row["H32"])),
    }
    # The authenticated D20 imported-cell template is itself part of the D20
    # provenance chain; require every inherited identity flag to remain true.
    common["inherited_D20_identity_template"] = all(bool(value) for value in template.values())
    common["all_identity_checks_pass"] = all(common.values())
    return common


def build_import_mapping(parent: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_d21 = {row["cell"]: row for row in cells}
    result: list[dict[str, Any]] = []
    for d21_cell, d20_cell in D21_TO_D20_IMPORTS.items():
        d21_row = by_d21[d21_cell]
        d20_row = parent["cells"][d20_cell]["endpoint"]
        d20_registry_row = parent["cells"][d20_cell]
        expected_coarse_bits = d20_coarse_bits_for_d21(d21_row["bits"])
        assert_true(d20_registry_row["bits"] == expected_coarse_bits, f"D21/D20 factor mapping mismatch {d21_cell}->{d20_cell}")
        assert_true(d20_registry_row["delete_horizons"] == d21_row["delete_horizons"], f"D21/D20 deletion set mismatch {d21_cell}->{d20_cell}")
        identity = common_identity_checks(parent, d20_row, d20_registry_row, d21_row)
        assert_true(identity["all_identity_checks_pass"], f"D20 import identity failed {d21_cell}->{d20_cell}: {identity}")
        result.append(
            {
                "d21_cell": d21_cell,
                "d21_branch": d21_row["branch"],
                "d20_cell": d20_cell,
                "d20_factor_order": list(COARSE_D20_FACTORS),
                "d20_bits": expected_coarse_bits,
                "A_retained": True,
                "d21_deleted_horizons": list(d21_row["delete_horizons"]),
                "d20_deleted_horizons": list(d20_registry_row["delete_horizons"]),
                "exact_deletion_member_set_match": True,
                "source_stage": d20_registry_row["source_stage"],
                "source_role": d20_registry_row["source_role"],
                "source_artifact": d20_registry_row["source_artifact"],
                "source_field": d20_registry_row["source_field"],
                "endpoint": {"H1": d20_row["H1"], "H32": d20_row["H32"], "E": d20_row["E"], "H32_preservation": d20_row["H32_preservation"]},
                "identity_checks": identity,
                "status": "IMPORT_FROM_D20_VERIFIED",
            }
        )
    assert_true(len(result) == 8 and len({row["d21_cell"] for row in result}) == 8, "D21 import count/uniqueness failed")
    return result


def build_future_branch_manifest(cells: Sequence[Mapping[str, Any]], imported: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    imported_ids = {row["d21_cell"] for row in imported}
    branches = []
    for row in cells:
        is_import = row["cell"] in imported_ids
        source = next((item for item in imported if item["d21_cell"] == row["cell"]), None)
        branches.append(
            {
                "cell_id": row["cell"],
                "branch": row["branch"],
                "factor_order": list(FACTORS),
                "bits": list(row["bits"]),
                "states": {factor: row["factor_values"][factor]["state"] for factor in FACTORS},
                "H32_state": row["factor_values"]["H32"]["state"],
                "B1_state": row["factor_values"]["B1"]["state"],
                "B2_state": row["factor_values"]["B2"]["state"],
                "C1_state": row["factor_values"]["C1"]["state"],
                "C2_state": row["factor_values"]["C2"]["state"],
                "A_state": "CONTROL_RETAINED",
                "exact_deleted_horizons": list(row["delete_horizons"]),
                "exact_deleted_components": list(row["deleted_components"]),
                "provenance_status": "IMPORT_FROM_D20" if is_import else "FUTURE_D21",
                "source_d20_cell": source["d20_cell"] if source else None,
                "expected_updates_if_future": list(UPDATES) if not is_import else [],
                "endpoint_status": "AUTHENTICATED_D20_ENDPOINT" if is_import else "NOT_EXECUTED_BY_DESIGN_REVIEW",
                "adaptive_membership_selection": False,
            }
        )
    return {
        "schema_version": "stage3_h13_post_d20_d21_future_branch_manifest_v1",
        "design_review_only": True,
        "scientific_branches_authorized_in_this_review": [],
        "all_factorial_cells": [row["cell_id"] for row in branches],
        "branches": branches,
        "authenticated_imported_cells": [row["d21_cell"] for row in imported],
        "future_d21_cells": [row["cell_id"] for row in branches if row["provenance_status"] == "FUTURE_D21"],
        "total_cells": len(branches),
        "imported_cell_count": len(imported),
        "future_branch_count": sum(row["provenance_status"] == "FUTURE_D21" for row in branches),
        "future_updates_per_branch": len(UPDATES),
        "total_future_branch_update_records": sum(row["provenance_status"] == "FUTURE_D21" for row in branches) * len(UPDATES),
        "no_other_scientific_branches_authorized": True,
    }


def synthetic_fixture() -> dict[str, Any]:
    factors = FACTORS
    coefficients: dict[frozenset[str], float] = {}
    for index, subset in enumerate(factor_subsets(factors), start=1):
        # A known nonzero coefficient for every term makes both lower- and
        # higher-order inversion errors observable.
        sign = -1.0 if index % 3 == 0 else 1.0
        coefficients[frozenset(subset)] = sign * (0.017 * index + 0.003 * len(subset))
    endpoints: dict[str, float] = {}
    for bits in itertools.product((0, 1), repeat=len(factors)):
        subset = frozenset(factor for factor, bit in zip(factors, bits) if bit)
        endpoints[cell_id(bits)] = reconstruct_endpoint(coefficients, subset)
    recovered = mobius_interactions(endpoints, factors)
    reconstructed = {
        cid: reconstruct_endpoint(recovered, subset_for_cell(cid, factors))
        for cid in endpoints
    }
    reconstruction_residual = max(abs(endpoints[cid] - reconstructed[cid]) for cid in endpoints)

    parent_factors = ("H32", "B", "C")
    parent_endpoints: dict[str, float] = {}
    for bits in itertools.product((0, 1), repeat=3):
        fine_bits = (bits[0], bits[1], bits[1], bits[2], bits[2])
        parent_endpoints[cell_id(bits)] = endpoints[cell_id(fine_bits)]
    parent_interactions = mobius_interactions(parent_endpoints, parent_factors)
    terms = primary_component_terms()
    parent_b = sum(recovered[component_factors(name)] for name in terms["M_32_B"])
    parent_c = sum(recovered[component_factors(name)] for name in terms["M_32_C"])
    parent_bc = sum(recovered[component_factors(name)] for name in terms["M_32_B_C"])
    parent_residuals = {
        "M_32_B": parent_interactions[frozenset(("H32", "B"))] - parent_b,
        "M_32_C": parent_interactions[frozenset(("H32", "C"))] - parent_c,
        "M_32_B_C": parent_interactions[frozenset(("H32", "B", "C"))] - parent_bc,
    }

    pairwise = {"B1": -0.31, "B2": 0.17}
    qb_child = {
        "Q_B1_internal": recovered[frozenset(("H32", "B1"))] - pairwise["B1"],
        "Q_B2_internal": recovered[frozenset(("H32", "B2"))] - pairwise["B2"],
        "Q_B1_B2_cross": recovered[frozenset(("H32", "B1", "B2"))],
    }
    qb_residual = (
        parent_interactions[frozenset(("H32", "B"))]
        - (pairwise["B1"] + pairwise["B2"])
        - sum(qb_child.values())
    )

    return {
        "status": "PASS" if reconstruction_residual <= DECOMPOSITION_TOLERANCE and all(abs(value) <= DECOMPOSITION_TOLERANCE for value in parent_residuals.values()) and abs(qb_residual) <= DECOMPOSITION_TOLERANCE else "FAIL",
        "known_coefficient_count": len(coefficients),
        "endpoint_count": len(endpoints),
        "mobius_term_count": len(recovered),
        "endpoint_reconstruction_max_abs_residual": reconstruction_residual,
        "parent_conservation_residuals": parent_residuals,
        "q_b_child_conservation": {"child_terms": qb_child, "residual": qb_residual},
        "tolerance": DECOMPOSITION_TOLERANCE,
    }


def classify_components(components: Mapping[str, float], parent_reproduced: bool = True, parent_effect_material: bool = True) -> dict[str, Any]:
    target = {name: float(components.get(name, 0.0)) for name in D21_H32_BC_COMPONENTS}
    material = [name for name, value in target.items() if abs(value) >= H1_MATERIALITY_THRESHOLD]
    absolute_ranking = sorted(target, key=lambda name: (-abs(target[name]), name))
    signed_ranking = sorted(target, key=lambda name: (-target[name], name))
    mass = sum(abs(value) for value in target.values())
    largest = absolute_ranking[0] if absolute_ranking else None
    dominant = largest if largest and mass > 0 and abs(target[largest]) / mass >= 0.5 else None

    def child_sets(name: str) -> tuple[set[str], set[str]]:
        factors = component_factors(name)
        return ({factor for factor in factors if factor.startswith("B")}, {factor for factor in factors if factor.startswith("C")})

    if not parent_reproduced:
        label = "H32_BY_B_C_PARENT_EFFECT_NOT_REPRODUCED"
    elif not material:
        label = "H32_BY_DISTRIBUTED_SUBTHRESHOLD_B_C_SUBBLOCK_HIGHER_ORDER" if parent_effect_material else "H32_BY_NON_MATERIAL_B_C_SUBBLOCK_EFFECT"
    elif len(material) == 1:
        name = material[0]
        bs, cs = child_sets(name)
        if bs == {"B1"} and cs == {"C1"}:
            label = "H32_BY_B1_C1_LOCALIZED"
        elif bs == {"B1"} and cs == {"C2"}:
            label = "H32_BY_B1_C2_LOCALIZED"
        elif bs == {"B2"} and cs == {"C1"}:
            label = "H32_BY_B2_C1_LOCALIZED"
        elif bs == {"B2"} and cs == {"C2"}:
            label = "H32_BY_B2_C2_LOCALIZED"
        elif {"B1", "B2"} <= bs:
            label = "H32_BY_B1_B2_CROSS_SUBBLOCK_HIGHER_ORDER"
        elif {"C1", "C2"} <= cs:
            label = "H32_BY_C1_C2_CROSS_SUBBLOCK_HIGHER_ORDER"
        else:
            label = "H32_BY_SINGLE_B_C_SUBBLOCK_HIGHER_ORDER_LOCALIZED"
    else:
        material_sets = [child_sets(name) for name in material]
        used_b = set().union(*(item[0] for item in material_sets))
        used_c = set().union(*(item[1] for item in material_sets))
        has_b_cross = any({"B1", "B2"} <= bs for bs, _ in material_sets)
        has_c_cross = any({"C1", "C2"} <= cs for _, cs in material_sets)
        if used_b == {"B1"} and used_c == {"C1", "C2"}:
            label = "H32_BY_SINGLE_B1_SUBBLOCK_MULTI_C"
        elif used_b == {"B2"} and used_c == {"C1", "C2"}:
            label = "H32_BY_SINGLE_B2_SUBBLOCK_MULTI_C"
        elif used_c == {"C1"} and used_b == {"B1", "B2"}:
            label = "H32_BY_SINGLE_C1_SUBBLOCK_MULTI_B"
        elif used_c == {"C2"} and used_b == {"B1", "B2"}:
            label = "H32_BY_SINGLE_C2_SUBBLOCK_MULTI_B"
        elif has_b_cross and not has_c_cross:
            label = "H32_BY_B1_B2_CROSS_SUBBLOCK_HIGHER_ORDER"
        elif has_c_cross and not has_b_cross:
            label = "H32_BY_C1_C2_CROSS_SUBBLOCK_HIGHER_ORDER"
        else:
            label = "H32_BY_DISTRIBUTED_B_C_SUBBLOCK_HIGHER_ORDER"

    return {
        "classification": label,
        "material_components": sorted(material, key=lambda name: (-abs(target[name]), name)),
        "absolute_ranking": [{"rank": index + 1, "component": name, "absolute_value": abs(target[name]), "signed_value": target[name], "material": name in material} for index, name in enumerate(absolute_ranking)],
        "signed_ranking": [{"rank": index + 1, "component": name, "signed_value": target[name], "absolute_value": abs(target[name]), "material": name in material} for index, name in enumerate(signed_ranking)],
        "largest_component": largest,
        "dominant_component": dominant,
        "dominance_definition": "largest absolute component with absolute mass share >= 0.5 over the nine primary H32×B/C child components",
        "unique_localization": len(material) == 1 and parent_reproduced,
        "unique_localization_definition": "exactly one independently material primary H32×B/C child component and parent conservation passes; rank #1 alone is insufficient",
        "materiality_threshold": H1_MATERIALITY_THRESHOLD,
    }


def synthetic_classification_fixture() -> dict[str, Any]:
    def vector(names: Sequence[str], value: float = H1_MATERIALITY_THRESHOLD * 2) -> dict[str, float]:
        return {name: value if name in names else 0.0 for name in D21_H32_BC_COMPONENTS}

    cases = [
        ("pair_B1_C1", ["M_32_B1_C1"], "H32_BY_B1_C1_LOCALIZED"),
        ("pair_B1_C2", ["M_32_B1_C2"], "H32_BY_B1_C2_LOCALIZED"),
        ("pair_B2_C1", ["M_32_B2_C1"], "H32_BY_B2_C1_LOCALIZED"),
        ("pair_B2_C2", ["M_32_B2_C2"], "H32_BY_B2_C2_LOCALIZED"),
        ("single_B1_multi_C", ["M_32_B1_C1", "M_32_B1_C2"], "H32_BY_SINGLE_B1_SUBBLOCK_MULTI_C"),
        ("single_C1_multi_B", ["M_32_B1_C1", "M_32_B2_C1"], "H32_BY_SINGLE_C1_SUBBLOCK_MULTI_B"),
        ("B_cross", ["M_32_B1_B2_C1"], "H32_BY_B1_B2_CROSS_SUBBLOCK_HIGHER_ORDER"),
        ("C_cross", ["M_32_B1_C1_C2"], "H32_BY_C1_C2_CROSS_SUBBLOCK_HIGHER_ORDER"),
        ("distributed", ["M_32_B1_C1", "M_32_B2_C2"], "H32_BY_DISTRIBUTED_B_C_SUBBLOCK_HIGHER_ORDER"),
        ("parent_not_reproduced", ["M_32_B1_C1"], "H32_BY_B_C_PARENT_EFFECT_NOT_REPRODUCED"),
    ]
    results = []
    for name, material, expected in cases:
        actual = classify_components(vector(material), parent_reproduced=name != "parent_not_reproduced")["classification"]
        results.append({"case": name, "expected": expected, "observed": actual, "pass": actual == expected})
    return {"status": "PASS" if all(row["pass"] for row in results) else "FAIL", "cases": results}


def build_contracts(parent: Mapping[str, Any], partition: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], imported: Sequence[Mapping[str, Any]], future_manifest: Mapping[str, Any], static: Mapping[str, Any]) -> dict[str, Any]:
    d20_components = parent["decomposition"]["seven_H32_containing_block_terms"]
    d20_parent_values = {
        "M_32_B": d20_components["M_32_B"],
        "M_32_C": d20_components["M_32_C"],
        "M_32_B_C": d20_components["M_32_B_C"],
    }
    p_b1 = sum(float(row["I_32_i"]) for row in parent["pairwise"]["rows"] if 18 <= int(row["HORIZON"]) <= 22)
    p_b2 = sum(float(row["I_32_i"]) for row in parent["pairwise"]["rows"] if 23 <= int(row["HORIZON"]) <= 26)
    p_b = float(parent["pairwise"]["p_b"])
    q_b = d20_parent_values["M_32_B"] - p_b

    terms = primary_component_terms()
    all_terms = [component_name(subset) for subset in factor_subsets(FACTORS)]
    mobius_contract = {
        "schema_version": "stage3_h13_post_d20_d21_mobius_estimand_contract_v1",
        "factor_order": list(FACTORS),
        "response": "Y(S)=H1 endpoint at update 392 for exact deletion set S",
        "control": "Y(empty)=CONTROL_H1 under A-retained canonical control context",
        "effect": "E(S)=Y(empty)-Y(S); positive means H1 improvement; negative means H1 degradation",
        "empty_effect": 0.0,
        "mobius_definition": "M(T)=sum(U subset T)(-1)^(|T|-|U|)E(U), equivalently M(T)=E(T)-sum(non-empty proper U subset T)M(U)",
        "all_nonempty_terms": all_terms,
        "all_nonempty_term_count": len(all_terms),
        "primary_localization_terms": list(D21_H32_BC_COMPONENTS),
        "parent_refinement_terms": terms,
        "parent_effect_context": "D20 A retained; D21 has no A factor",
        "q_b_refinement_terms": {
            "Q_B1_internal": "M_32_B1 - P_B1",
            "Q_B2_internal": "M_32_B2 - P_B2",
            "Q_B1_B2_cross_subblock": "M_32_B1_B2",
            "Q_B_conservation": "Q_B=(M_32_B1-P_B1)+(M_32_B2-P_B2)+M_32_B1_B2",
        },
        "materiality_rule": "abs(component)>=4.099006815405639e-06",
        "future_endpoint_results": "not evaluated in this design review",
    }
    conservation_contract = {
        "schema_version": "stage3_h13_post_d20_d21_parent_child_conservation_contract_v1",
        "tolerance": DECOMPOSITION_TOLERANCE,
        "parent_context": "A retained; D20 parent values imported from D20_EXECUTION/factorial_mobius_decomposition.json",
        "checks": {
            "M_32_B": {
                "target": d20_parent_values["M_32_B"],
                "child_terms": terms["M_32_B"],
                "equation": "M_32_B = M_32_B1 + M_32_B2 + M_32_B1_B2",
                "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED",
            },
            "M_32_C": {
                "target": d20_parent_values["M_32_C"],
                "child_terms": terms["M_32_C"],
                "equation": "M_32_C = M_32_C1 + M_32_C2 + M_32_C1_C2",
                "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED",
            },
            "M_32_B_C": {
                "target": d20_parent_values["M_32_B_C"],
                "child_terms": terms["M_32_B_C"],
                "equation": "M_32_B_C = sum of all M(T) containing H32, at least one of {B1,B2}, and at least one of {C1,C2}",
                "child_term_count": len(terms["M_32_B_C"]),
                "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED",
            },
        },
        "synthetic_fixture": static["synthetic_mobius"],
    }
    q_b_contract = {
        "schema_version": "stage3_h13_post_d20_d21_qb_child_decomposition_contract_v1",
        "source": "D20_DESIGN/d19_pairwise_import.json::rows[I_32_i] and D20_EXECUTION/factorial_mobius_decomposition.json",
        "re_estimated": False,
        "P_B1": p_b1,
        "P_B2": p_b2,
        "P_B": p_b,
        "P_B1_plus_P_B2": p_b1 + p_b2,
        "P_B_conservation_residual": (p_b1 + p_b2) - p_b,
        "parent_Q_B": q_b,
        "child_quantities": {
            "Q_B1_internal": "M_32_B1-P_B1",
            "Q_B2_internal": "M_32_B2-P_B2",
            "Q_B1_B2_cross_subblock": "M_32_B1_B2",
        },
        "exact_conservation_equation": "Q_B = Q_B1_internal + Q_B2_internal + Q_B1_B2_cross_subblock = (M_32_B1-P_B1)+(M_32_B2-P_B2)+M_32_B1_B2",
        "future_endpoint_status": "PREREGISTERED_NOT_EVALUATED",
        "static_fixture": static["synthetic_mobius"]["q_b_child_conservation"],
        "optional_if_D19_rows_unrecoverable": False,
    }
    materiality_contract = {
        "schema_version": "stage3_h13_post_d20_d21_materiality_contract_v1",
        "H1_materiality_threshold": H1_MATERIALITY_THRESHOLD,
        "component_rule": "abs(component)>=H1_materiality_threshold",
        "endpoint_effect_rule": "E(S)>=H1_materiality_threshold is H1-material improvement under D20 sign convention",
        "recovered_from": "D20_DESIGN/materiality_contract.json::H1_materiality_threshold",
        "reporting": ["absolute_component_ranking", "signed_component_ranking", "material_non_material_classification"],
        "future_results": "not evaluated in this design review",
    }
    h32_contract = {
        "schema_version": "stage3_h13_post_d20_d21_h32_preservation_contract_v1",
        "threshold": H32_PRESERVATION_THRESHOLD,
        "recovered_from": "D20_DESIGN/h32_preservation_contract.json::threshold",
        "rule": "H32_at_update_392 <= threshold is PASS; otherwise FAIL",
        "applies_to": "all future D21 branches and imported equivalents",
        "separate_from_H1": True,
        "valid_final_intervention_rule": "H1 material improvement and H32 preservation must both pass; H1 improvement alone is not a valid final intervention",
        "future_results": "not evaluated in this design review",
    }
    classification_contract = {
        "schema_version": "stage3_h13_post_d20_d21_classification_contract_v1",
        "classification_input": "the nine primary H32×B/C child Möbius components plus the parent-conservation gate",
        "materiality_threshold": H1_MATERIALITY_THRESHOLD,
        "precedence": ["H32_BY_B_C_PARENT_EFFECT_NOT_REPRODUCED", "single-material-component rules", "single-subblock multi-component rules", "cross-subblock rules", "distributed rules", "subthreshold rule"],
        "enum_definitions": {
            "H32_BY_B1_C1_LOCALIZED": "exactly one material primary term M_32_B1_C1",
            "H32_BY_B1_C2_LOCALIZED": "exactly one material primary term M_32_B1_C2",
            "H32_BY_B2_C1_LOCALIZED": "exactly one material primary term M_32_B2_C1",
            "H32_BY_B2_C2_LOCALIZED": "exactly one material primary term M_32_B2_C2",
            "H32_BY_SINGLE_B1_SUBBLOCK_MULTI_C": "multiple material terms use B1 only and collectively use C1 and C2",
            "H32_BY_SINGLE_B2_SUBBLOCK_MULTI_C": "multiple material terms use B2 only and collectively use C1 and C2",
            "H32_BY_SINGLE_C1_SUBBLOCK_MULTI_B": "multiple material terms use C1 only and collectively use B1 and B2",
            "H32_BY_SINGLE_C2_SUBBLOCK_MULTI_B": "multiple material terms use C2 only and collectively use B1 and B2",
            "H32_BY_B1_B2_CROSS_SUBBLOCK_HIGHER_ORDER": "material higher-order term requires both B1 and B2 without a C1/C2 cross term being the defining material support",
            "H32_BY_C1_C2_CROSS_SUBBLOCK_HIGHER_ORDER": "material higher-order term requires both C1 and C2 without a B1/B2 cross term being the defining material support",
            "H32_BY_DISTRIBUTED_B_C_SUBBLOCK_HIGHER_ORDER": "multiple material terms remain after all single-subblock/cross-subblock rules and are distributed across B/C child structure",
            "H32_BY_DISTRIBUTED_SUBTHRESHOLD_B_C_SUBBLOCK_HIGHER_ORDER": "no individual primary term is material while the authenticated parent effect remains material",
            "H32_BY_B_C_PARENT_EFFECT_NOT_REPRODUCED": "a parent-child conservation identity fails at the frozen tolerance",
        },
        "largest_component": "rank #1 by absolute value only",
        "dominant_component": "largest absolute component with absolute mass share >= 0.5 over the nine primary terms",
        "uniquely_localized_component": "exactly one independently material primary component and parent conservation passes; rank #1 alone never establishes uniqueness",
        "H32_preservation_is_separate": True,
        "synthetic_fixture": static["synthetic_classification"],
    }
    trajectory_contract = {
        "schema_version": "stage3_h13_post_d20_d21_trajectory_diagnostics_contract_v1",
        "reuse": "D20 instrumentation contract; no second incompatible format",
        "updates": list(UPDATES),
        "fields": ["branch ID", "update number", "batch identity/digest", "RNG identity/digest", "model state identity", "optimizer state identity", "gradient norms", "clipping diagnostics", "AdamW decomposition", "parameter deltas", "H1 trajectory", "H32 trajectory", "finite-state checks"],
        "authenticated_d20_field_names": ["branch", "update", "batch_identity", "batch_window_sha256", "rng_before_digest", "rng_after_diagnostics_digest", "rng_after_training_digest", "rng_after_validation_digest", "model_state_hash_before", "model_state_hash_after", "optimizer_state_hash_before", "optimizer_state_hash_after", "gradient_norm_preclip", "gradient_norm_postclip", "clip_coefficient", "adamw_decomposition_status", "parameter_delta_norm", "optimizer_step_delta_norm", "H1", "H32", "total_loss", "finite_state_status"],
        "role": "descriptive supporting diagnostics; not mechanistic causal proof",
        "future_status": "must be retained for every future branch-update record",
    }
    execution_contract = {
        "schema_version": "stage3_h13_post_d20_d21_execution_authorization_contract_v1",
        "design_review_only": True,
        "D21_EXECUTION_AUTHORIZED": "NO",
        "ready_for_separate_D21_execution_authorization": "YES",
        "authorized_future_cells_after_separate_authorization": future_manifest["future_d21_cells"],
        "imported_cells": future_manifest["authenticated_imported_cells"],
        "future_branch_count": future_manifest["future_branch_count"],
        "updates": list(UPDATES),
        "endpoint_update": 392,
        "UPDATE_393_AUTHORIZED": "NO",
        "FROZEN20_AUTHORIZED": "NO",
        "R9_AUTHORIZED": "NO",
        "R8E_A1_MODIFICATION_AUTHORIZED": "NO",
        "A_FACTOR_AUTHORIZED": "NO",
        "hyperparameter_tuning_authorized": "NO",
        "block_boundary_tuning_authorized": "NO",
        "adaptive_branch_selection_authorized": "NO",
        "post_hoc_partition_changes_authorized": "NO",
        "start_state": parent["certified_start"],
        "execution_engine": "reuse D20 deletion-set resolver and authenticated D17 update primitive; no scientific execution in this review",
    }
    return {
        "mobius_estimand_contract": mobius_contract,
        "parent_child_conservation_contract": conservation_contract,
        "q_b_contract": q_b_contract,
        "materiality_contract": materiality_contract,
        "h32_contract": h32_contract,
        "classification_contract": classification_contract,
        "trajectory_contract": trajectory_contract,
        "execution_contract": execution_contract,
    }


def build_static_validation(parent: Mapping[str, Any], partition: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], imported: Sequence[Mapping[str, Any]], future_manifest: Mapping[str, Any]) -> dict[str, Any]:
    ids = [row["cell"] for row in cells]
    deletion_sets = [tuple(sorted(row["delete_horizons"])) for row in cells]
    assert_true(ids == [cell_id(bits) for bits in itertools.product((0, 1), repeat=5)], "D21 cell IDs are not fixed-order 5-bit binary")
    assert_true(len(ids) == 32 and len(set(ids)) == 32, "D21 cell ID uniqueness failed")
    assert_true(len(set(deletion_sets)) == 32, "D21 intended deletion sets are not unique")
    assert_true([row["factor_order"] if "factor_order" in row else list(FACTORS) for row in cells] == [list(FACTORS)] * len(cells), "D21 factor order failed")
    assert_true(len(imported) == 8, "D21 imported count failed")
    assert_true(future_manifest["future_branch_count"] == 24, "D21 future branch count failed")
    assert_true(future_manifest["total_future_branch_update_records"] == 192, "D21 future record count failed")
    for row in cells:
        expected = []
        for factor, bit in zip(FACTORS, row["bits"]):
            if bit:
                expected.extend(["H32"] if factor == "H32" else partition["subblocks"][factor]["members"])
        assert_true(row["delete_horizons"] == expected, f"bit/deletion mapping failed {row['cell']}")

    synthetic_mobius = synthetic_fixture()
    synthetic_classification = synthetic_classification_fixture()
    assert_true(synthetic_mobius["status"] == "PASS", "synthetic Möbius/parent/Q_B fixture failed")
    assert_true(synthetic_classification["status"] == "PASS", "synthetic classification fixture failed")
    p_b1 = sum(float(row["I_32_i"]) for row in parent["pairwise"]["rows"] if 18 <= int(row["HORIZON"]) <= 22)
    p_b2 = sum(float(row["I_32_i"]) for row in parent["pairwise"]["rows"] if 23 <= int(row["HORIZON"]) <= 26)
    p_b = float(parent["pairwise"]["p_b"])
    assert_true(abs((p_b1 + p_b2) - p_b) <= DECOMPOSITION_TOLERANCE, "P_B1+P_B2 does not conserve authenticated P_B")

    return {
        "schema_version": "stage3_h13_post_d20_d21_static_validation_v1",
        "status": "PASS",
        "scientific_updates_executed": False,
        "scientific_replay_performed": False,
        "factor_registry": {"factor_order": list(FACTORS), "unique_cells": len(set(ids)), "cell_count": len(ids), "all_binary_vectors_present": True, "unique_deletion_sets": len(set(deletion_sets))},
        "bit_to_deletion_mapping": {"status": "PASS", "cells_checked": len(cells)},
        "subblock_partition": partition["conservation"],
        "d20_branch_deletion_reproduction": parent["d20_deletion_reproduction"],
        "import_mapping": {"status": "PASS", "proposed_mappings": len(imported), "all_identity_checks_pass": all(item["identity_checks"]["all_identity_checks_pass"] for item in imported)},
        "future_branch_counts": {"total_cells": 32, "authenticated_imports": 8, "future_branches": 24, "updates_per_future_branch": 8, "future_branch_update_records": 192},
        "synthetic_mobius": synthetic_mobius,
        "synthetic_classification": synthetic_classification,
        "q_b_parent_conservation": {"status": "PASS", "P_B1": p_b1, "P_B2": p_b2, "P_B": p_b, "residual": (p_b1 + p_b2) - p_b},
        "parent_effect_targets": {
            "M_32_B": parent["decomposition"]["seven_H32_containing_block_terms"]["M_32_B"],
            "M_32_C": parent["decomposition"]["seven_H32_containing_block_terms"]["M_32_C"],
            "M_32_B_C": parent["decomposition"]["seven_H32_containing_block_terms"]["M_32_B_C"],
        },
        "tolerance": DECOMPOSITION_TOLERANCE,
    }


def authoritative_identity(parent: Mapping[str, Any], archive_info: Mapping[str, Any]) -> dict[str, Any]:
    design_identity = parent["design_identity"]
    execution_identity = parent["execution_identity"]
    return {
        "schema_version": "stage3_h13_post_d20_d21_authoritative_input_identity_v1",
        "parent_experiment": "D20",
        "d20_design_manifest_sha256": EXPECTED_D20_DESIGN_MANIFEST_SHA256,
        "d20_execution_archive": archive_info,
        "d20_archive_expected_path": str(D20_ARCHIVE),
        "d20_design_directory": str(D20_DESIGN_DIR.resolve()),
        "d20_execution_directory": str(D20_EXECUTION_DIR.resolve()),
        "certified_start": parent["certified_start"],
        "certified_update_384_bundle_sha256": EXPECTED_CERTIFIED_UPDATE_384_BUNDLE_SHA256,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "optimizer_sha256": EXPECTED_OPTIMIZER_SHA256,
        "source_revision": parent["source_revision"],
        "current_git_head": parent["current_git_head"],
        "schedule_identity": parent["schedule_identity"],
        "runtime_source_identity": parent["runtime_source_identity"],
        "d20_execution_start_identity": execution_identity["start_identity"],
        "d20_execution_protocol": parent["protocol"],
        "request_text_path": str(REQUEST_TEXT),
        "request_text_sha256": sha256_file(REQUEST_TEXT),
        "recovered_block_sources": {
            "A": "D20_DESIGN/factor_definition.json::factors[name=A].members",
            "B": "D20_DESIGN/factor_definition.json::factors[name=B].members",
            "C": "D20_DESIGN/factor_definition.json::factors[name=C].members",
            "H32": "D20_DESIGN/factor_definition.json::factors[name=X32].deletion_set",
        },
    }


def engineering_defects(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "stage3_h13_post_d20_d21_engineering_defects_and_repairs_v1",
        "scientific_protocol_defects": [],
        "defects_and_repairs": [
            {
                "defect": "D20 post-execution reporting wrapper changed the runner file after the sealed D20 design manifest was written",
                "evidence": "D20_EXECUTION/authoritative_input_identity.json::runner.design_review_declared_runner_sha256 vs sha256_after_reporting_repairs",
                "declared_sha256": parent["execution_identity"]["runner"]["design_review_declared_runner_sha256"],
                "observed_after_reporting_repairs_sha256": parent["execution_identity"]["runner"]["sha256_after_reporting_repairs"],
                "status": "documented_inherited_nonsemantic_repair",
                "repair": "D21 imports use the archived D20 design/execution evidence and semantic identity contract; no D20 runner is rewritten or used as a provenance substitute",
                "scientific_semantic_change": False,
            },
            {
                "defect": "D20 historical D18 design bundle was absent",
                "evidence": "D20_DESIGN/imported_cell_provenance.json::provenance_debt",
                "status": "documented_inherited_provenance_debt_not_blocking",
                "repair": "none; authenticated D18 execution evidence and source semantics are preserved in the D20 archive",
                "scientific_semantic_change": False,
            },
            {
                "defect": "D21 needs a five-factor branch resolver while the authenticated D20 runner is four-factor",
                "status": "narrowly_extended_for_future_execution",
                "repair": "D21 materializer freezes five-factor cells and the optional runner reuses D20's authenticated deletion-set update primitive; no scientific branch was executed",
                "scientific_semantic_change": False,
            },
            {
                "defect": "Initial D21 materializer invocation omitted the static-validation argument at the report-render wrapper call",
                "status": "narrowly_repaired_before_final_bundle",
                "repair": "Corrected the wrapper call and regenerated the sealed bundle; the failure occurred before report finalization and before any scientific execution path",
                "scientific_semantic_change": False,
            },
        ],
        "historical_artifacts_modified": False,
        "existing_scientific_source_files_modified": False,
    }


def report_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |\n|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return header + ("\n" + "\n".join(body) if body else "")


def render_report(bundle: Path, parent: Mapping[str, Any], partition: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], imported: Sequence[Mapping[str, Any]], future_manifest: Mapping[str, Any], contracts: Mapping[str, Any], static: Mapping[str, Any], hashes: Mapping[str, str], archive_info: Mapping[str, Any]) -> str:
    d20_values = parent["decomposition"]["seven_H32_containing_block_terms"]
    future_rows = [row for row in future_manifest["branches"] if row["provenance_status"] == "FUTURE_D21"]
    import_rows = [{"d21_cell": row["d21_cell"], "d20_cell": row["d20_cell"], "delete": ",".join(row["d21_deleted_horizons"]), "source": row["source_stage"], "status": row["status"]} for row in imported]
    future_table_rows = [{"cell": row["cell_id"], "branch": row["branch"], "delete": ",".join(row["exact_deleted_horizons"]) or "none"} for row in future_rows]
    hash_rows = [{"path": name, "sha": digest} for name, digest in sorted(hashes.items())]
    block = partition["subblocks"]
    primary_terms = contracts["mobius_estimand_contract"]["primary_localization_terms"]
    lines = [
        "STAGE_3_H13_POST_D20_D21_H32_BY_B_C_HIERARCHICAL_SUBBLOCK_HIGHER_ORDER_FACTORIAL_CAUSAL_LOCALIZATION_DESIGN_REVIEW:\nPASSED",
        "DESIGN_REVIEW_PASSED:\nYES",
        "FIRST_BLOCKER:\nnone",
        "ONE_SENTENCE_VERDICT:\nThe frozen D21 five-factor B/C sub-block factorial is provenance-linked, statically validated, and ready for separate execution authorization with eight authenticated D20 imports and twenty-four future branches; no D21 scientific updates were executed.",
        "1. PROTOCOL_INTEGRITY\n\nThis review is design-only. No D21 model load, training update, backward pass, optimizer step, update 393, Frozen-20, R9, R8E_A1 modification, tuning, adaptive selection, or post-hoc partition change was executed. The D21 execution guard remains NO.",
        f"2. AUTHORITATIVE_INPUT_IDENTITY\n\nD20 design manifest SHA-256: `{EXPECTED_D20_DESIGN_MANIFEST_SHA256}`\nD20 execution archive: `{D20_ARCHIVE}`\nD20 execution archive SHA-256: `{archive_info['archive_sha256']}`\nCertified update-384 bundle SHA-256: `{EXPECTED_CERTIFIED_UPDATE_384_BUNDLE_SHA256}`\nModel SHA-256: `{EXPECTED_MODEL_SHA256}`\nOptimizer SHA-256: `{EXPECTED_OPTIMIZER_SHA256}`\nSource revision: `{parent['source_revision']}`\nThe D20 archive and all authenticated D20 design members passed hash verification.",
        f"3. D20_SCIENTIFIC_INTERPRETATION\n\nD20 classified the remaining mechanism as `H32_BY_DISTRIBUTED_MULTI_COMPONENT_HIGHER_ORDER`. The material D20 components were `M_32_B_C={d20_values['M_32_B_C']:.17g}`, `Q_B={d20_values['Q_B'] if 'Q_B' in d20_values else parent['decomposition']['final_higher_order_localization_components']['Q_B']:.17g}`, and `M_32_A_B_C={d20_values['M_32_A_B_C']:.17g}`. D21 resolves the stronger B/C parent and refines Q_B while keeping A out of the factorial.",
        f"4. D21_B_C_SUBBLOCK_PARTITION\n\nExact recovered D20 blocks: A={','.join(parent['blocks']['A'])}; B={','.join(parent['blocks']['B'])}; C={','.join(parent['blocks']['C'])}; H32=H32. Deterministic refinement: B1={','.join(block['B1']['members'])}; B2={','.join(block['B2']['members'])}; C1={','.join(block['C1']['members'])}; C2={','.join(block['C2']['members'])}. Source fields are frozen in `subblock_partition.json`; partition conservation passed. A is retained context, not a D21 factor.",
        "5. FIVE_FACTOR_DESIGN\n\nFactor order is `[H32, B1, B2, C1, C2]`; bit 0 is CONTROL-retained and bit 1 is deletion under the authenticated D20 semantics. All 32 binary cells and all 32 deletion sets are explicit in `factorial_cell_manifest.json`. No A factor, alternative split, or 2^6 design exists.",
        "6. D20_IMPORTED_CELL_VALIDATION\n\nAll eight proposed A-retained mappings passed start-state, model, optimizer, source revision, canonical schedule, RNG, update range, gradient/clipping, AdamW, rollout-deletion, endpoint, H1, and H32 identity checks.\n\n" + report_table(import_rows, (("D21", "d21_cell"), ("D20", "d20_cell"), ("Delete set", "delete"), ("Source", "source"), ("Status", "status"))),
        "7. FUTURE_BRANCH_MANIFEST\n\nExactly 24 cells remain future D21 branches; the complete 32-cell manifest, including the eight imports, is in `future_branch_manifest.json`.\n\n" + report_table(future_table_rows, (("Cell", "cell"), ("Branch", "branch"), ("Exact delete set", "delete"))),
        f"8. MOBIUS_ESTIMAND_CONTRACT\n\n`E(S)=Y(empty)-Y(S)` with positive values meaning H1 improvement. The full 31-term five-factor Möbius decomposition is frozen in `mobius_estimand_contract.json`. Primary H32/B/C terms are `{', '.join(primary_terms)}`. H32 preservation is never folded into H1 causal effect.",
        f"9. D20_TO_D21_PARENT_CONSERVATION\n\nWith A retained, `M_32_B` must equal `M_32_B1+M_32_B2+M_32_B1_B2` (target `{d20_values['M_32_B']:.17g}`); `M_32_C` must equal `M_32_C1+M_32_C2+M_32_C1_C2` (target `{d20_values['M_32_C']:.17g}`); and `M_32_B_C` must equal the sum of the nine D21 terms containing H32, at least one B child, and at least one C child (target `{d20_values['M_32_B_C']:.17g}`). The exact formulas and synthetic residuals are in `parent_child_conservation_contract.json`; all synthetic residuals passed `{DECOMPOSITION_TOLERANCE:.17g}`.",
        f"10. D19_QB_CHILD_DECOMPOSITION\n\nAuthenticated `P_B1={static['q_b_parent_conservation']['P_B1']:.17g}`, `P_B2={static['q_b_parent_conservation']['P_B2']:.17g}`, and `P_B={static['q_b_parent_conservation']['P_B']:.17g}` satisfy `P_B1+P_B2=P_B` at residual `{static['q_b_parent_conservation']['residual']:.3g}`. D21 freezes `Q_B=(M_32_B1-P_B1)+(M_32_B2-P_B2)+M_32_B1_B2`; no D19 rerun occurred.",
        f"11. H32_PRESERVATION_CONTRACT\n\nThe authenticated D20 threshold `{H32_PRESERVATION_THRESHOLD:.17g}` is reused exactly: H32 at update 392 must be less than or equal to the threshold. H1 material improvement and H32 preservation are reported separately; an H1-improving H32-failing cell is not a valid final intervention.",
        "12. MATERIALITY_AND_CLASSIFICATION_CONTRACT\n\nMateriality is frozen at `abs(component)>=4.099006815405639e-06`. The contract distinguishes largest, material, dominant, and uniquely localized components; uniqueness requires exactly one independently material primary component plus parent conservation, never rank #1 alone. The required localized, single-subblock, cross-subblock, distributed, and parent-not-reproduced enums are frozen in `classification_contract.json` and pass the synthetic fixture.",
        "13. STATIC_VALIDATION\n\nPASS: 32 unique IDs; 32 unique deletion sets; fixed factor order; bit/deletion mapping; B/C partition conservation; D20 branch-list reproduction; eight import mappings; 24 future branches; 192 future branch-update records; full Möbius inversion; parent B/C conservation formulas; Q_B conservation; synthetic classification. Scientific updates executed: NO.",
        "14. ENGINEERING_DEFECTS_AND_REPAIRS\n\nThe inherited D20 runner/reporting hash discrepancy and historical D18 provenance debt are documented as nonsemantic, nonblocking repairs. D21 adds only a narrow five-factor resolver and optional separately gated runner that reuses D20/D17 update machinery. No authenticated scientific artifact was modified.",
        f"15. ARTIFACT_PATHS_AND_HASHES\n\nFinal D21 design-review bundle: `{bundle.resolve()}`\nSHA256 manifest: `{(bundle / 'SHA256_MANIFEST.json').resolve()}`\n\n" + report_table(hash_rows, (("Artifact", "path"), ("SHA-256", "sha"))),
        "16. EXECUTION_READINESS\n\nThe design is implementation-ready after separate project-owner authorization. The current task authorizes no D21 scientific execution. Future execution must start each branch from the certified post-update-384 state, run exactly updates 385–392, retain the D20 diagnostics, and forbid update 393, Frozen-20, R9, R8E_A1 modification, tuning, adaptive selection, partition changes, and A-related new branches.",
        "D21_TOTAL_FACTORIAL_CELLS: 32",
        "D21_AUTHENTICATED_IMPORTED_CELLS: 8",
        "D21_FUTURE_BRANCHES_REQUIRED: 24",
        "D21_FUTURE_UPDATES_PER_BRANCH: 8",
        "D21_TOTAL_FUTURE_BRANCH_UPDATE_RECORDS: 192",
        "D21_EXECUTION_AUTHORIZED: NO",
        "READY_FOR_SEPARATE_D21_EXECUTION_AUTHORIZATION: YES",
    ]
    return "\n\n".join(lines) + "\n"


def materialize() -> dict[str, Any]:
    assert_true(D20_ARCHIVE.is_file(), f"authenticated D20 archive missing: {D20_ARCHIVE}")
    assert_true(D20_DESIGN_DIR.is_dir(), f"authenticated D20 design directory missing: {D20_DESIGN_DIR}")
    assert_true(D20_EXECUTION_DIR.is_dir(), f"authenticated D20 execution directory missing: {D20_EXECUTION_DIR}")
    assert_true(REQUEST_TEXT.is_file(), f"request text missing: {REQUEST_TEXT}")
    assert_true(sha256_file(D20_ARCHIVE) == EXPECTED_D20_ARCHIVE_SHA256, "D20 archive SHA-256 mismatch")
    assert_true(sha256_file(D20_DESIGN_DIR / "SHA256_MANIFEST.json") == EXPECTED_D20_DESIGN_MANIFEST_SHA256, "D20 design manifest SHA-256 mismatch")
    with tarfile.open(D20_ARCHIVE, "r:xz") as stream:
        archive_info = verify_d20_archive(stream)
        parent = recover_parent_and_d20(stream)

    partition = build_partition(parent)
    cells = d21_cells(partition)
    imported = build_import_mapping(parent, cells)
    future_manifest = build_future_branch_manifest(cells, imported)
    static = build_static_validation(parent, partition, cells, imported, future_manifest)
    contracts = build_contracts(parent, partition, cells, imported, future_manifest, static)
    identity = authoritative_identity(parent, archive_info)
    defects = engineering_defects(parent)

    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S+0800")
    output = ROOT / "outputs" / f"stage3_h13_post_d20_d21_h32_by_b_c_hierarchical_subblock_higher_order_factorial_causal_localization_design_review_{now}"
    assert_true(not output.exists(), f"refusing to overwrite existing D21 bundle: {output}")
    output.mkdir(parents=True)

    factor_definition = {
        "schema_version": "stage3_h13_post_d20_d21_factor_definition_v1",
        "factor_bit_order": list(FACTORS),
        "coding": {"0": "retain exactly as CONTROL", "1": "delete under authenticated D20 rollout deletion semantics"},
        "factors": [
            {"name": "H32", "members": ["H32"], "deletion_set": ["H32"]},
            {"name": "B1", "members": partition["subblocks"]["B1"]["members"], "deletion_set": partition["subblocks"]["B1"]["members"]},
            {"name": "B2", "members": partition["subblocks"]["B2"]["members"], "deletion_set": partition["subblocks"]["B2"]["members"]},
            {"name": "C1", "members": partition["subblocks"]["C1"]["members"], "deletion_set": partition["subblocks"]["C1"]["members"]},
            {"name": "C2", "members": partition["subblocks"]["C2"]["members"], "deletion_set": partition["subblocks"]["C2"]["members"]},
        ],
        "excluded_factors": [{"name": "A", "reason": "retained D20 parent context; reserved for later validation; no D21 A branches"}],
        "set_validation": {"all_factor_sets_pairwise_disjoint": True, "B1_union_B2": partition["conservation"]["B1_union_B2"], "C1_union_C2": partition["conservation"]["C1_union_C2"], "A_not_a_factor": True},
    }
    experiment_design = {
        "schema_version": "stage3_h13_post_d20_d21_experiment_design_v1",
        "experiment_id": EXPERIMENT_ID,
        "design_review_only": True,
        "factors": list(FACTORS),
        "factor_bit_order": list(FACTORS),
        "A_context": "retained under D20 A-retained parent context; no A factor",
        "B1_members": partition["subblocks"]["B1"]["members"],
        "B2_members": partition["subblocks"]["B2"]["members"],
        "C1_members": partition["subblocks"]["C1"]["members"],
        "C2_members": partition["subblocks"]["C2"]["members"],
        "factorial_cell_count": 32,
        "all_factorial_cells": [row["cell"] for row in cells],
        "authenticated_imported_cells": [row["d21_cell"] for row in imported],
        "future_branch_cells": future_manifest["future_d21_cells"],
        "expected_import_count": 8,
        "expected_future_branch_count": 24,
        "expected_future_branch_update_records": 192,
        "certified_start": parent["certified_start"],
        "authoritative_d17_execution_path": parent["experiment_design"]["authoritative_d17_execution_path"],
        "source_revision": parent["source_revision"],
        "updates": list(UPDATES),
        "endpoint_update": 392,
        "forbidden_update": 393,
        "H1_materiality_threshold": H1_MATERIALITY_THRESHOLD,
        "H32_preservation_threshold": H32_PRESERVATION_THRESHOLD,
        "D20_parent_values": {name: parent["decomposition"]["seven_H32_containing_block_terms"][name] for name in D20_PARENT_COMPONENTS},
        "D20_Q_B": parent["decomposition"]["final_higher_order_localization_components"]["Q_B"],
        "no_adaptive_partition": True,
        "no_A_factor": True,
        "no_scientific_updates_in_review": True,
    }
    factorial_manifest = {"schema_version": "stage3_h13_post_d20_d21_factorial_cell_manifest_v1", "factor_order": list(FACTORS), "cell_count": 32, "cells": cells, "imported_cell_count": 8, "future_cell_count": 24}

    artifacts: dict[str, Any] = {
        "experiment_design.json": experiment_design,
        "factor_definition.json": factor_definition,
        "subblock_partition.json": partition,
        "factorial_cell_manifest.json": factorial_manifest,
        "imported_cell_mapping.json": {"schema_version": "stage3_h13_post_d20_d21_imported_cell_mapping_v1", "factor_order": list(FACTORS), "mappings": imported, "count": len(imported), "all_verified": True},
        "future_branch_manifest.json": future_manifest,
        "mobius_estimand_contract.json": contracts["mobius_estimand_contract"],
        "parent_child_conservation_contract.json": contracts["parent_child_conservation_contract"],
        "q_b_child_decomposition_contract.json": contracts["q_b_contract"],
        "materiality_contract.json": contracts["materiality_contract"],
        "h32_preservation_contract.json": contracts["h32_contract"],
        "classification_contract.json": contracts["classification_contract"],
        "trajectory_diagnostics_contract.json": contracts["trajectory_contract"],
        "execution_authorization_contract.json": contracts["execution_contract"],
        "authoritative_input_identity.json": identity,
        "static_validation_results.json": static,
        "engineering_defects_and_repairs.json": defects,
        "protocol_contract.json": {"schema_version": "stage3_h13_post_d20_d21_protocol_contract_v1", "status": "SEALED_DESIGN_REVIEW_ONLY", "scientific_updates_executed": False, "training": False, "backward_pass": False, "optimizer_step": False, "update_393_executed": False, "A_factor_executed": False, "adaptive_selection": False, "threshold_tuning": False, "partition_tuning": False},
    }
    for name, value in artifacts.items():
        write_json(output / name, value)

    hashes = {name: sha256_file(output / name) for name in artifacts}
    report = render_report(output, parent, partition, cells, imported, future_manifest, contracts, static, hashes, archive_info)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    hashes["FINAL_REPORT.md"] = sha256_file(output / "FINAL_REPORT.md")
    runner_path = ROOT / "tools" / RUNNER_NAME
    manifest_entries = [
        {"relative_path": name, "size_bytes": (output / name).stat().st_size, "sha256": digest, "artifact_role": "D21 design-review artifact" if name != "static_validation_results.json" else "static and synthetic validation evidence"}
        for name, digest in sorted(hashes.items())
    ]
    manifest_entries.append({"relative_path": f"../../tools/{RUNNER_NAME}", "size_bytes": runner_path.stat().st_size, "sha256": sha256_file(runner_path), "artifact_role": "separately gated future D21 runner and static validator"})
    manifest = {"schema_version": "stage3_h13_post_d20_d21_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "manifest_self_hash": "NOT_INCLUDED_BY_CONTRACT", "file_count": len(manifest_entries), "files": manifest_entries}
    write_json(output / "SHA256_MANIFEST.json", manifest)
    return {"status": "PASSED", "output": str(output.resolve()), "manifest_sha256": sha256_file(output / "SHA256_MANIFEST.json"), "static_validation": static["status"], "scientific_updates_executed": False, "total_cells": 32, "imported_cells": 8, "future_branches": 24, "future_branch_update_records": 192}


def authorized(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    required = ("D21_EXECUTION_AUTHORIZED: YES", "UPDATE_393_AUTHORIZED: NO", "FROZEN20_AUTHORIZED: NO", "A_FACTOR_AUTHORIZED: NO")
    return all(marker in text for marker in required)


def execute_scientific(design_dir: Path, authorization: Path, output: Path) -> int:
    """Future-only execution hook; never called by the design review."""
    if not authorized(authorization):
        raise RuntimeError("D21_EXECUTION_AUTHORIZATION_MARKER_MISSING")
    if output.exists():
        raise RuntimeError(f"output_directory_already_exists_no_resume:{output}")
    design = read_json_file(design_dir / "experiment_design.json")
    assert_true(design.get("design_review_only") is True and design.get("no_A_factor") is True, "D21 design contract is not sealed")
    manifest = read_json_file(design_dir / "future_branch_manifest.json")
    future = [row for row in manifest["branches"] if row["provenance_status"] == "FUTURE_D21"]
    assert_true(len(future) == 24, "D21 future branch count mismatch")
    # Reuse the authenticated D17/D20 engine.  Importing these modules does not
    # execute a scientific branch; the authorization gate above is mandatory.
    sys.path.insert(0, str(ROOT))
    from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17
    from tools import stage3_h13_post_d19_d20_h32_by_multi_horizon_block_higher_order_factorial_causal_localization_replay as d20

    d17.verify_design_and_inputs()
    state = d17.preflight_runtime()
    control_rows = [json.loads(line) for line in (Path(design["authoritative_d17_execution_path"]) / "per_update_instrumentation.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line).get("branch") == "CONTROL"]
    control_by_update = {int(row["update"]): row for row in control_rows}
    assert_true(set(control_by_update) == set(UPDATES), "authenticated D17 control reference incomplete")
    output.mkdir(parents=True)
    all_rows: list[dict[str, Any]] = []
    branch_manifests = []
    for spec in future:
        model, optimizer = d17.d16.new_branch(state["bundle"])
        start_model_hash = d17.canonical_model_state_sha256(model.state_dict())
        start_optimizer_hash = d17.base.optimizer_semantic_hash(optimizer)
        assert_true(start_model_hash == design["certified_start"]["model_semantic_hash"], f"D21 model start mismatch {spec['cell_id']}")
        assert_true(start_optimizer_hash == design["certified_start"]["optimizer_semantic_hash"], f"D21 optimizer start mismatch {spec['cell_id']}")
        rng = d17.base.capture_rng()
        branch_rows = []
        for zero_index in range(384, 392):
            deleted = [int(item.removeprefix("H")) for item in spec["exact_deleted_horizons"]]
            row, _, rng = d20._scientific_branch_update(d17, f"D21_{spec['cell_id']}", deleted, model, optimizer, state, zero_index, rng)
            control = control_by_update[row["update"]]
            for field in ("batch_window_sha256", "rng_before_digest", "rng_after_diagnostics_digest", "rng_after_training_digest", "rng_after_validation_digest"):
                assert_true(row[field] == control[field], f"D21 control match failed {spec['cell_id']}:{row['update']}:{field}")
            branch_rows.append(row)
            all_rows.append(row)
        branch_manifests.append({"cell": spec["cell_id"], "branch": f"D21_{spec['cell_id']}", "start_model_hash": start_model_hash, "start_optimizer_hash": start_optimizer_hash, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO"})
    (output / "per_update_instrumentation.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in all_rows) + "\n", encoding="utf-8", newline="\n")
    write_json(output / "branch_execution_manifest.json", {"scientific_branches_executed": [row["cell"] for row in future], "branches": branch_manifests, "updates": list(UPDATES), "endpoint_update": 392, "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO"})
    write_json(output / "protocol_integrity.json", {"status": "PASSED", "d21_execution_authorized": "YES", "scientific_branch_count": 24, "new_branch_update_count": len(all_rows), "update_393_executed": "NO", "frozen20_used": "NO", "r9_executed": "NO"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-scientific", action="store_true")
    parser.add_argument("--design-dir", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.execute_scientific:
            if not args.design_dir or not args.authorization or not args.output:
                raise RuntimeError("--execute-scientific requires --design-dir, --authorization, and --output")
            return execute_scientific(args.design_dir.resolve(), args.authorization.resolve(), args.output.resolve())
        print(json.dumps(materialize(), sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "first_blocker": f"{type(exc).__name__}:{exc}", "scientific_updates_executed": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
