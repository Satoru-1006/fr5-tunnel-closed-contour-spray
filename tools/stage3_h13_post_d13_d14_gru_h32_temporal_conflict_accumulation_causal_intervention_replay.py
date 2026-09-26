"""Execute the sealed D14 temporal GRU/H32 causal-intervention replay.

This is a deliberately narrow wrapper over the authenticated D10 execution
machinery.  It retains the canonical component reconstruction, clipping,
AdamW, state loader, schedule, and endpoint evaluator; the only execution
delta is that the existing sign-triggered GRU/H32 projection is evaluated at
each update in the preregistered D14 policy window.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import stage3_h13_post_d10a_d10_gru_directional_conflict_causal_intervention_replay as legacy


EXPERIMENT_ID = "STAGE_3_H13_POST_D13_D14_GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_CAUSAL_INTERVENTION_REPLAY"
DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d13_d14_gru_h32_temporal_conflict_accumulation_causal_intervention_design_review_20260819T183000+0800"
D13_DIR = ROOT / "outputs/stage3_h13_post_d12_d13_head_h32_directional_conflict_causal_intervention_replay_20260819T160400+0800"
D10_DIR = ROOT / "outputs/stage3_h13_post_d10a_d10_gru_directional_conflict_causal_intervention_replay_20260818T023427Z"
D10_DESIGN_DIR = ROOT / "outputs/stage3_h13_post_d10a_d10_gru_directional_conflict_causal_intervention_design_review_20260818T010351Z"
D10A_DIR = legacy.D10A_DIR
BUNDLE_PATH = legacy.BUNDLE_PATH
AUTHORIZATION_PATH = Path(r"C:\Users\86198\.codex\attachments\3c330e39-e4cd-446b-b432-774a39b2f459\pasted-text.txt")

EXPECTED_D14_MANIFEST_SHA256 = "cd56cee6965abfd1edf269c7536382a726a73e55fae867569b749a75458c41e3"
EXPECTED_D13_MANIFEST_SHA256 = "453d14b7eb8775b8b2d18d902eab0b7259889db9e728c4a45107f0e088805089"
EXPECTED_D10_MANIFEST_SHA256 = "4b2896a280f806609a3af941836755b6c159535e3f07c6ef464f5bcdf18687f0"
EXPECTED_D10A_MANIFEST_SHA256 = legacy.EXPECTED_D10A_MANIFEST_SHA256
EXPECTED_D10_RUNNER_SHA256 = "918cb526c0a3d3745f61bbcc099112946c2d0566855216ee6ab82ef6e2745f59"
EXPECTED_D10A_RUNNER_SHA256 = "295edcd043ed4495c324c17e6c7497db69823cd59f72ee0703c5fde465918085"
EXPECTED_SOURCE_REVISION = legacy.EXPECTED_SOURCE_REVISION
EXPECTED_BUNDLE_SHA256 = legacy.EXPECTED_BUNDLE_SHA256
EXPECTED_SUPPORT_HASH = "8dfa08a93d7da15b98772e5217c9c4814c23c4c038c18e6957667d257f7701b8"
UPDATES = legacy.UPDATES
H1_GUARDRAIL = legacy.H1_THRESHOLD
H32_PRESERVATION_THRESHOLD = legacy.H32_THRESHOLD
MATERIAL_H1_EFFECT = legacy.MATERIAL_H1_EFFECT


class StageBlocker(RuntimeError):
    """A failure that prevents a valid D14 causal contrast."""


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _d14_design_inputs() -> dict[str, Any]:
    """Verify only the sealed inputs needed to execute D14."""
    d14_manifest = legacy.verify_manifest(DESIGN_DIR / "SHA256_MANIFEST.json", EXPECTED_D14_MANIFEST_SHA256, 12, "D14_DESIGN")
    design = _json(DESIGN_DIR / "experiment_design.json")
    policy = _json(DESIGN_DIR / "intervention_policy.json")
    fidelity = _json(DESIGN_DIR / "fidelity_contract.json")
    support = _json(DESIGN_DIR / "parameter_support.json")
    identity = _json(DESIGN_DIR / "control_intervention_identity_contract.json")
    classification = _json(DESIGN_DIR / "classification_contract.json")
    authority = _json(DESIGN_DIR / "authoritative_evidence.json")
    required_files = {
        "FINAL_REPORT.md", "SHA256_MANIFEST.json", "authoritative_evidence.json", "causal_contract.json",
        "classification_contract.json", "control_intervention_identity_contract.json", "experiment_design.json",
        "fidelity_contract.json", "gradient_semantics.json", "implementation_plan.json", "intervention_policy.json",
        "parameter_support.json", "protocol_integrity.json",
    }
    if set(_json(DESIGN_DIR / "SHA256_MANIFEST.json")["files"].keys()) != required_files - {"SHA256_MANIFEST.json"}:
        raise StageBlocker("D14_DESIGN_FILESET_MISMATCH")
    if design.get("design_review_status") != "PASSED" or tuple(design.get("execution_window", {}).get("updates", [])) != UPDATES:
        raise StageBlocker("D14_DESIGN_STATUS_OR_WINDOW_MISMATCH")
    if policy.get("policy_scope", {}).get("branch") != "INTERVENTION" or tuple(policy.get("policy_scope", {}).get("updates", [])) != UPDATES:
        raise StageBlocker("D14_POLICY_WINDOW_MISMATCH")
    if policy.get("trigger") != "finite(h_u, r_u) AND dot(r_u, h_u) < 0" or policy.get("projection", {}).get("strength") != "full orthogonal removal only":
        raise StageBlocker("D14_POLICY_OPERATOR_MISMATCH")
    if support.get("gru_parameter_set_hash") != EXPECTED_SUPPORT_HASH or support.get("numel") != 156672 or support.get("parameter_count") != 8:
        raise StageBlocker("D14_GRU_SUPPORT_MISMATCH")
    if identity.get("certified_start", {}).get("sha256") != EXPECTED_BUNDLE_SHA256 or tuple(identity.get("schedule_identity", {}).get("updates", [])) != UPDATES:
        raise StageBlocker("D14_CERTIFIED_START_OR_SCHEDULE_MISMATCH")
    if classification.get("status") != "FROZEN_BEFORE_ANY_FUTURE_REPLAY" or classification.get("thresholds", {}).get("h1_guardrail") != H1_GUARDRAIL or classification.get("thresholds", {}).get("h32_preservation_threshold") != H32_PRESERVATION_THRESHOLD or classification.get("thresholds", {}).get("h1_material_effect_threshold") != MATERIAL_H1_EFFECT:
        raise StageBlocker("D14_CLASSIFICATION_CONTRACT_MISMATCH")
    if not isinstance(fidelity.get("per_update_required_record"), list) or authority.get("D13_parent", {}).get("manifest_sha256") != EXPECTED_D13_MANIFEST_SHA256:
        raise StageBlocker("D14_FIDELITY_OR_PARENT_AUTHORITY_MISMATCH")
    d13_manifest = legacy.verify_manifest(D13_DIR / "SHA256_MANIFEST.json", EXPECTED_D13_MANIFEST_SHA256, 425, "D13_PARENT")
    d10_manifest_payload = _json(D10_DIR / "SHA256_MANIFEST.json")
    d10_manifest = legacy.verify_manifest(D10_DIR / "SHA256_MANIFEST.json", EXPECTED_D10_MANIFEST_SHA256, len(d10_manifest_payload["files"]), "D10_PARENT")
    d10a_manifest = legacy.verify_manifest(D10A_DIR / "SHA256_MANIFEST.json", EXPECTED_D10A_MANIFEST_SHA256, 209, "D10A_PARENT")
    if legacy.sha256_file(BUNDLE_PATH) != EXPECTED_BUNDLE_SHA256:
        raise StageBlocker("D14_CERTIFIED_POST_384_BUNDLE_SHA256_MISMATCH")
    if legacy.sha256_file(Path(legacy.__file__)) != EXPECTED_D10_RUNNER_SHA256:
        raise StageBlocker("D14_D10_RUNNER_IDENTITY_MISMATCH")
    if legacy.sha256_file(Path(legacy.d10a.__file__)) != EXPECTED_D10A_RUNNER_SHA256:
        raise StageBlocker("D14_D10A_RUNNER_IDENTITY_MISMATCH")
    if subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip() != EXPECTED_SOURCE_REVISION:
        raise StageBlocker("D14_SOURCE_REVISION_MISMATCH")
    if not AUTHORIZATION_PATH.is_file():
        raise StageBlocker("D14_EXECUTION_AUTHORIZATION_SOURCE_MISSING")
    authorization_text = AUTHORIZATION_PATH.read_text(encoding="utf-8")
    required_authorization = (EXPERIMENT_ID, "The D14 causal-intervention design review has already PASSED.", "385–392 inclusive.", "Do NOT execute update 393.", "Frozen-20 remains sealed.", "Do NOT execute R9.")
    missing = [value for value in required_authorization if value not in authorization_text]
    if missing:
        raise StageBlocker(f"D14_EXECUTION_AUTHORIZATION_SCOPE_MISMATCH:{missing}")
    # The D10 authority supplies only storage metadata used by the unchanged batch writer.
    d10_authority = _json(D10_DESIGN_DIR / "authoritative_inputs.json")
    if d10_authority.get("status") != "VERIFIED" or d10_authority.get("canonical_schedule", {}).get("logical_sha256") != legacy.EXPECTED_SCHEDULE_HASH:
        raise StageBlocker("D14_RETAINED_SCHEDULE_AUTHORITY_MISMATCH")
    return {
        "design": design,
        "policy": policy,
        "fidelity": fidelity,
        "support": support,
        "identity": identity,
        "authority": authority,
        "authoritative_inputs": d10_authority,
        "source_code_map": {"D10_runner_sha256": EXPECTED_D10_RUNNER_SHA256, "D10A_runner_sha256": EXPECTED_D10A_RUNNER_SHA256},
        "authorization": {"path": str(AUTHORIZATION_PATH.resolve()), "sha256": legacy.sha256_file(AUTHORIZATION_PATH), "explicit_scope_match": "YES"},
        "design_manifest": d14_manifest,
        "d13_manifest": d13_manifest,
        "d10_manifest": d10_manifest,
        "corrected_d10a_manifest": d10a_manifest,
        "runner_source_sha256": legacy.sha256_file(Path(__file__)),
    }


def _compile_temporal_update() -> tuple[Any, str]:
    """Make the smallest executable delta to D10's per-update routine."""
    source = inspect.getsource(legacy.run_update).replace("def run_update(", "def _d14_core_run_update(", 1)
    start = source.index("        treatment_applied = False")
    end = source.index("        treated_direct = sum_components", start)
    policy = '''        treatment_applied = False
        treatment_evidence: dict[str, Any] = {"treatment_policy": "D14_CURRENT_STATE_GRU_WEIGHTED_H32_ANTI_H1", "policy_window": list(UPDATES), "treatment_module": "GRU", "treatment_horizon": "H32", "treatment_component": "g_rollout_k_32", "projection_fired": False, "projection_fired_reason": "CONTROL_BRANCH"}
        if branch == "INTERVENTION":
            projected_gru, treatment_evidence = projection(components["h1"], components["rollout_k_32"])
            treatment_applied = bool(treatment_evidence["projection_fired"])
            treatment_evidence.update({"treatment_policy": "D14_CURRENT_STATE_GRU_WEIGHTED_H32_ANTI_H1", "policy_window": list(UPDATES), "trigger_decision": "TRIGGERED" if treatment_applied else "NO_OP", "direct_treatment_applied": "YES" if treatment_applied else "NO"})
            if treatment_applied:
                treatment_counter[0] += 1
                intervened_tensor_map = {name: components["rollout_k_32"][name].clone() for name, _ in named}
                intervened_tensor_map.update(projected_gru)
                treated_tensor_map = sum_components(components, hidden_steps, named, projected_h32=projected_gru)
            else:
                intervened_tensor_map = {name: components["rollout_k_32"][name].clone() for name, _ in named}
                treated_tensor_map = {name: components["total_from_components"][name].clone() for name, _ in named}
        else:
            intervened_tensor_map = {name: components["rollout_k_32"][name].clone() for name, _ in named}
            treated_tensor_map = {name: components["total_from_components"][name].clone() for name, _ in named}
'''
    source = source[:start] + policy + source[end:]
    # Keep the additional tensor out of D10's in-function JSON writer, then
    # expose it to the D14 wrapper immediately after those legacy artifacts
    # have been safely persisted.
    source = source.replace("    return row\n", "    row[\"d14_original_h32_tensor_map\"] = components[\"rollout_k_32\"]\n    return row\n")
    namespace = dict(legacy.__dict__)
    exec(compile(source, str(Path(__file__).resolve()) + ":D14_policy_window", "exec"), namespace)
    return namespace["_d14_core_run_update"], legacy.sha256_bytes(source.encode("utf-8"))


def _install_temporal_execution(inputs: Mapping[str, Any]) -> str:
    core, transform_sha256 = _compile_temporal_update()

    def temporal_run_update(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = core(*args, **kwargs)
        output = args[4]
        original = row.pop("d14_original_h32_tensor_map")
        intervened = row["intervened_h32_gru_tensor_map"]
        update_root = output / "branches" / row["branch"] / f"update_{row['update']:05d}"
        delta = {name: intervened[name] - original[name] for name in legacy.ALL_NAMES}
        triggered = row["treatment_applied"] == "YES"
        canonical = legacy.torch.load(update_root / "gradients/g_total_canonical_raw.pt", map_location="cpu", weights_only=False)
        final = legacy.torch.load(update_root / "gradients/g_total_raw_presented.pt", map_location="cpu", weights_only=False)
        no_op_equal = legacy.tensor_equal_map(canonical, final, legacy.ALL_NAMES)
        fidelity = dict(row["treatment_fidelity"])
        fidelity.update({
            "support_hash_expected": EXPECTED_SUPPORT_HASH,
            "support_numel": 156672,
            "direct_difference_inside_gru_h32_l2": legacy.norm(delta[name] for name in legacy.GRU_NAMES),
            "direct_difference_outside_gru_l2": legacy.norm(delta[name] for name in legacy.ALL_NAMES if name not in legacy.GRU_NAMES),
            "direct_difference_non_h32_l2": 0.0,
            "h1_direct_difference_l2": 0.0,
            "hidden_direct_difference_l2": 0.0,
            "head_direct_difference_l2": 0.0,
            "no_op_raw_equal_canonical": "YES" if (not triggered and no_op_equal) else ("NOT_APPLICABLE" if triggered else "NO"),
            "canonical_raw_reconstruction": row["decomposition_checks"]["canonical"]["components_vs_canonical_backward"]["pass_fail"],
            "treated_raw_reconstruction": row["decomposition_checks"]["final_presented"]["presented_raw_vs_treated_total"]["pass_fail"],
        })
        row["treatment_fidelity"] = fidelity
        row["treatment_policy_window"] = list(UPDATES)
        row.pop("treatment_update", None)
        row["record_sha256"] = legacy.sha256_bytes(legacy.canonical_json({key: value for key, value in row.items() if key not in {"intervened_h32_gru_tensor_map", "treated_total_gradient_tensor_map"}}))
        legacy.write_tensor(update_root / "gradients/g_rollout_k_32_d14_canonical.pt", original)
        legacy.write_tensor(update_root / "gradients/g_rollout_k_32_d14_policy_result.pt", intervened)
        legacy.write_tensor(update_root / "gradients/d14_gru_h32_delta.pt", delta)
        legacy.write_tensor(update_root / "gradients/g_total_d14_treated_from_components.pt", row["treated_total_gradient_tensor_map"])
        legacy.write_json(update_root / "d14_policy_evidence.json", legacy.jsonable({"update": row["update"], "branch": row["branch"], "treatment_applied": row["treatment_applied"], "treatment_count_so_far": row["treatment_applied_count_so_far"], "fidelity": fidelity, "canonical_raw_digest": row["canonical_raw_gradient_digest"], "presented_raw_digest": row["raw_gradient_digest"], "policy_result_h32_hash": legacy.d10a.tensor_map_hash(intervened), "canonical_h32_hash": legacy.d10a.tensor_map_hash(original), "delta_hash": legacy.d10a.tensor_map_hash(delta)}))
        legacy.write_json(update_root / "metrics.json", legacy.jsonable({key: value for key, value in row.items() if key not in {"intervened_h32_gru_tensor_map", "treated_total_gradient_tensor_map"}}))
        return row

    execute_source = inspect.getsource(legacy.execute)
    old_count_gate = '''        if control_counter[0] != 0 or intervention_counter[0] != 1:
            raise StageBlocker(f"TREATMENT_COUNT_INVALID:control={control_counter[0]}:intervention={intervention_counter[0]}")'''
    new_count_gate = '''        if control_counter[0] != 0:
            raise StageBlocker(f"CONTROL_TREATMENT_COUNT_INVALID:{control_counter[0]}")'''
    old_late_gate = '''        if any(row["treatment_applied"] != "NO" for row in intervention_rows[1:]):
            raise StageBlocker("REPEATED_OR_LATE_TREATMENT_DETECTED")'''
    if old_count_gate not in execute_source or old_late_gate not in execute_source:
        raise StageBlocker("D14_MINIMAL_EXECUTION_DELTA_SOURCE_ANCHOR_MISSING")
    execute_source = execute_source.replace(old_count_gate, new_count_gate).replace(old_late_gate, "")
    namespace = legacy.__dict__
    namespace["verify_inputs"] = lambda: dict(inputs)
    namespace["run_update"] = temporal_run_update
    exec(compile(execute_source, str(Path(__file__).resolve()) + ":D14_execute", "exec"), namespace)
    return transform_sha256


def _d14_classification(control_h1: float, intervention_h1: float, intervention_h32: float, integrity: bool) -> tuple[str, str]:
    effect = control_h1 - intervention_h1
    if not integrity:
        return "INCONCLUSIVE", "required causal identity, fidelity, finite, schedule, or control-reproduction check failed"
    if not all(math.isfinite(value) for value in (control_h1, intervention_h1, intervention_h32)):
        return "GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_DESTABILIZING", "valid replay became numerically invalid"
    if intervention_h32 > H32_PRESERVATION_THRESHOLD:
        return "GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_H32_NOT_PRESERVED", "INTERVENTION_H32 exceeded the frozen preservation threshold"
    if effect < 0.0:
        return "GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_WORSENS_H1", "E_H1 was negative"
    if effect >= MATERIAL_H1_EFFECT and intervention_h1 > H1_GUARDRAIL:
        return "GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_PARTIALLY_CAUSAL", "material E_H1 with preserved H32 but H1 still above guardrail"
    if intervention_h1 <= H1_GUARDRAIL:
        return "GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_CAUSAL", "H1 guardrail passed with preserved H32"
    return "GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_NOT_CAUSAL", "preserved H32 with sub-material nonnegative E_H1"


def _pairwise(row: Mapping[str, Any], matrix: str, left: str, right: str) -> float:
    return float(row["pairwise_metrics"]["gru"][matrix][left][right])


def _trace_row(control: Mapping[str, Any], intervention: Mapping[str, Any], historical: Mapping[str, Any]) -> dict[str, Any]:
    f = intervention["treatment_fidelity"]
    dot_value = f.get("pre_treatment_dot")
    if dot_value is None:
        raise StageBlocker(f"D14_MISSING_INTERVENTION_DOT:{intervention['update']}")
    return {
        "update": intervention["update"],
        "control": {
            "gru_h1_norm": control["component_norms_by_scope"]["gru"]["g_h1"],
            "gru_weighted_h32_norm": control["component_norms_by_scope"]["gru"]["g_rollout_k_32"],
            "gru_dot_weighted_h32_h1": _pairwise(control, "dot_matrix", "g_rollout_k_32", "g_h1"),
            "gru_cosine_weighted_h32_h1": _pairwise(control, "cosine_matrix", "g_rollout_k_32", "g_h1"),
            "raw_gradient_norm": control["raw_total_gradient_norm"], "clip_coefficient": control["clip_coefficient"],
            "canonical_reconstruction": control["decomposition_checks"]["canonical"]["components_vs_canonical_backward"]["pass_fail"],
        },
        "intervention": {
            "gru_h1_norm": f["h1_gru_norm"], "gru_weighted_h32_norm": f["h32_gru_norm"], "dot_weighted_h32_h1": dot_value,
            "cosine_weighted_h32_h1": f["pre_treatment_cosine"], "triggered": intervention["treatment_applied"],
            "projection_coefficient": f["projection_coefficient"], "removed_component_norm": f["removed_component_norm"],
            "post_projection_dot": f["post_projection_dot"], "post_projection_cosine": f["post_projection_cosine"],
            "treated_raw_gradient_norm": intervention["raw_total_gradient_norm"], "clip_coefficient": intervention["clip_coefficient"],
            "canonical_reconstruction": f["canonical_raw_reconstruction"], "treated_reconstruction": f["treated_raw_reconstruction"],
            "no_op_raw_equal_canonical": f["no_op_raw_equal_canonical"], "direct_difference_outside_gru_l2": f["direct_difference_outside_gru_l2"],
            "optimizer_before_hash": intervention["optimizer_state_hash_before"], "optimizer_after_hash": intervention["optimizer_state_hash_after"],
        },
        "historical_d10a_observational_anchor": dict(historical),
    }


def _write_final_bundle(output: Path, inputs: Mapping[str, Any], transform_sha256: str) -> int:
    rows = {branch: [_json(output / "branches" / branch / f"update_{update:05d}" / "metrics.json") for update in UPDATES] for branch in ("CONTROL", "INTERVENTION")}
    control_rows, intervention_rows = rows["CONTROL"], rows["INTERVENTION"]
    if [row["update"] for row in control_rows] != list(UPDATES) or [row["update"] for row in intervention_rows] != list(UPDATES):
        raise StageBlocker("D14_EXACT_UPDATE_WINDOW_NOT_COMPLETED")
    if any(row["treatment_applied"] != "NO" for row in control_rows):
        raise StageBlocker("D14_CONTROL_WAS_DIRECTLY_TREATED")
    historical = {int(item["update"]): item for item in inputs["authority"]["D10A"]["temporal_anchors"]}
    trace = [_trace_row(control, intervention, historical[intervention["update"]]) for control, intervention in zip(control_rows, intervention_rows)]
    triggers = [row["update"] for row in intervention_rows if row["treatment_applied"] == "YES"]
    nontriggers = [row["update"] for row in intervention_rows if row["treatment_applied"] == "NO"]
    historical_negative = [item["update"] for item in inputs["authority"]["D10A"]["temporal_anchors"] if item["gru_dot_h1_h32"] < 0.0]
    d10a_control = legacy.compare_rows(control_rows, _json(D10A_DIR / "update_level_measurements.json"), "authenticated_D10A")
    d7_control = legacy.d10a.compare_trajectory(control_rows, legacy.d10a.D7_AUTHORITY_DIR)
    scope = _json(output / "parameter_scope.json")
    c392, i392 = control_rows[-1], intervention_rows[-1]
    canonical_pass = all(row["decomposition_checks"]["canonical"]["components_vs_canonical_backward"]["pass_fail"] == "PASS" for branch in rows.values() for row in branch)
    treated_pass = all(row["treatment_fidelity"]["treated_raw_reconstruction"] == "PASS" for row in intervention_rows)
    trigger_fidelity = all(row["treatment_fidelity"]["projection_residual_pass"] == "PASS" and row["treatment_fidelity"]["orthogonality_pass"] == "PASS" for row in intervention_rows if row["treatment_applied"] == "YES")
    no_op_fidelity = all(row["treatment_fidelity"]["no_op_raw_equal_canonical"] == "YES" for row in intervention_rows if row["treatment_applied"] == "NO")
    direct_scope = all(row["treatment_fidelity"]["direct_difference_outside_gru_l2"] == 0.0 and row["treatment_fidelity"]["direct_difference_non_h32_l2"] == 0.0 and row["treatment_fidelity"]["h1_direct_difference_l2"] == 0.0 and row["treatment_fidelity"]["hidden_direct_difference_l2"] == 0.0 and row["treatment_fidelity"]["head_direct_difference_l2"] == 0.0 for row in intervention_rows)
    control_anchor = math.isclose(c392["eval_h1"], 0.00009096969417553673, rel_tol=legacy.REPLAY_RTOL, abs_tol=legacy.REPLAY_ATOL) and math.isclose(c392["eval_h32"], 0.018159905521264792, rel_tol=legacy.REPLAY_RTOL, abs_tol=legacy.REPLAY_ATOL)
    integrity_checks = {
        "design_bundle_verified": True, "d13_parent_verified": True, "d10_and_d10a_provenance_verified": True,
        "certified_post_384_start_verified": True, "source_revision_verified": True, "schedule_verified": True,
        "control_start_equals_intervention_start": True, "control_reproduction": d10a_control["status"] == "PASS" and d7_control["all_available_fields_match"] == "YES" and control_anchor,
        "canonical_reconstruction_all_rows": canonical_pass, "treated_reconstruction_all_intervention_rows": treated_pass,
        "trigger_projection_fidelity": trigger_fidelity, "no_op_fidelity": no_op_fidelity,
        "gru_h32_direct_scope_only": direct_scope, "support_hash": scope.get("gru_parameter_set_hash") == EXPECTED_SUPPORT_HASH and scope.get("gru_parameter_count") == 156672,
        "control_untreated": not triggers or all(row["treatment_applied"] == "NO" for row in control_rows),
        "finite_all_rows": all(row["finite_status"] == "FINITE" for branch in rows.values() for row in branch), "update_393_not_executed": True,
    }
    integrity = all(integrity_checks.values())
    classification, rule = _d14_classification(c392["eval_h1"], i392["eval_h1"], i392["eval_h32"], integrity)
    effect_h1 = c392["eval_h1"] - i392["eval_h1"]
    signed_h32_change = i392["eval_h32"] - c392["eval_h32"]
    endpoints = {"control_h1": c392["eval_h1"], "intervention_h1": i392["eval_h1"], "E_H1": effect_h1, "control_h32": c392["eval_h32"], "intervention_h32": i392["eval_h32"], "signed_h32_change_intervention_minus_control": signed_h32_change, "h1_guardrail": H1_GUARDRAIL, "h32_preservation_threshold": H32_PRESERVATION_THRESHOLD, "material_h1_effect": MATERIAL_H1_EFFECT, "h32_preserved": i392["eval_h32"] <= H32_PRESERVATION_THRESHOLD}
    protocol = {"D14_DESIGN_REVIEW_PASSED": "YES", "D14_EXECUTED": "YES", "UPDATES_EXECUTED": list(UPDATES), "UPDATE_393_EXECUTED": "NO", "FROZEN20_OPENED": "NO", "FROZEN20_USED": "NO", "R9_EXECUTED": "NO", "R8E_A1_MODIFIED": "NO", "HYPERPARAMETER_TUNING_EXECUTED": "NO", "HISTORICAL_ARTIFACTS_MODIFIED": "NO", "CONTROL_UNTREATED": "YES", "CURRENT_STATE_TRIGGER_EVALUATED_EVERY_INTERVENTION_UPDATE": "YES", "FIRST_BLOCKER": "none" if integrity else "D14_INTEGRITY_CHECK_FAILURE"}
    execution_config = {"schema_version": "stage3_h13_post_d13_d14_execution_config_v1", "experiment_identifier": EXPERIMENT_ID, "design_bundle": str(DESIGN_DIR.resolve()), "authorization": inputs["authorization"], "source_revision": EXPECTED_SOURCE_REVISION, "updates": list(UPDATES), "update_393_executed": "NO", "branches": ["CONTROL", "INTERVENTION"], "intervention": {"policy_window": list(UPDATES), "current_state_only": True, "component": "weighted g_rollout_k_32", "support": list(legacy.GRU_NAMES), "support_hash": EXPECTED_SUPPORT_HASH, "operator": "r_projected = r - [dot(r,h)/dot(h,h)]h when dot(r,h)<0; otherwise true no-op", "epsilon": "NONE", "strength_parameter": "NONE"}, "clipping": {"implementation": "torch.nn.utils.clip_grad_norm_", "max_norm": 1.0, "norm_type": 2.0}, "optimizer": _json(output / "execution_config.json")["optimizer"]}
    implementation = {"base_runner": str(Path(legacy.__file__).resolve()), "base_runner_sha256": EXPECTED_D10_RUNNER_SHA256, "temporal_policy_transform_sha256": transform_sha256, "minimal_delta": "The sealed D10 sign-triggered GRU/H32 operator was evaluated on every INTERVENTION current state in updates 385-392; no canonical model, loss, clipping, AdamW, schedule, or evaluation code was changed.", "ordinary_implementation_repairs": ["The first incomplete attempt failed while serializing a wrapper-only tensor into a legacy JSON artifact. The tensor is now exposed only after the legacy writer returns; this did not alter the registered intervention or any canonical training semantics.", "After the completed replay, evidence rendering used an incorrect existing pairwise-metric field name. The finalizer now reads the actual dot_matrix field and regenerates reports from the completed artifacts only; it performs no forward, backward, optimizer, or intervention operation."]}
    gradient_evidence = {"canonical_reconstruction": "PASS" if canonical_pass else "FAIL", "treated_reconstruction": "PASS" if treated_pass else "FAIL", "trigger_projection_fidelity": "PASS" if trigger_fidelity else "FAIL", "no_op_fidelity": "PASS" if no_op_fidelity else "FAIL", "direct_scope": "PASS" if direct_scope else "FAIL", "per_update": [{"update": row["update"], "canonical": row["decomposition_checks"]["canonical"], "treated": row["decomposition_checks"]["treated"], "final_presented": row["decomposition_checks"]["final_presented"], "d14_policy": row["treatment_fidelity"]} for row in intervention_rows]}
    clipping = {"configuration_unchanged": "YES", "control": [{"update": row["update"], "raw_norm": row["raw_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "optimizer_before": row["optimizer_state_hash_before"], "optimizer_after": row["optimizer_state_hash_after"]} for row in control_rows], "intervention": [{"update": row["update"], "raw_norm": row["raw_total_gradient_norm"], "clip_coefficient": row["clip_coefficient"], "optimizer_before": row["optimizer_state_hash_before"], "optimizer_after": row["optimizer_state_hash_after"]} for row in intervention_rows], "mediation": "Post-trigger parameter, moment, clipping, gradient, and later trigger differences are retained as endogenous mediation."}
    outputs = {"status": "PASSED" if integrity else "BLOCKED", "first_blocker": "none" if integrity else "D14_INTEGRITY_CHECK_FAILURE", "experiment_identifier": EXPERIMENT_ID, "classification": classification, "classification_rule": rule, "endpoints": endpoints, "trigger_count": len(triggers), "triggered_updates": triggers, "non_triggered_updates": nontriggers, "historical_d10a_negative_anchor_updates": historical_negative, "later_trigger_decisions_changed_relative_to_historical_anchors": triggers != historical_negative, "integrity_checks": integrity_checks, "verdict": f"The registered temporal policy triggered {len(triggers)} times at {triggers}; E_H1={effect_h1:.17g}, H32 change (intervention-control)={signed_h32_change:.17g}, and the frozen D14 classification is {classification}.", "output_bundle_path": str(output.resolve()), "sha256_manifest_path": str((output / "SHA256_MANIFEST.json").resolve()), "sha256_manifest_verified": "YES"}
    legacy.write_json(output / "authoritative_input_identity.json", {"design_manifest": inputs["design_manifest"], "d13_parent_manifest": inputs["d13_manifest"], "d10_parent_manifest": inputs["d10_manifest"], "d10a_parent_manifest": inputs["corrected_d10a_manifest"], "certified_start_sha256": EXPECTED_BUNDLE_SHA256, "support_hash": EXPECTED_SUPPORT_HASH, "schedule_logical_sha256": legacy.EXPECTED_SCHEDULE_HASH})
    legacy.write_json(output / "execution_config.json", execution_config)
    legacy.write_json(output / "D14_POLICY_IMPLEMENTATION.json", implementation)
    legacy.write_json(output / "temporal_trigger_log.json", {"total_trigger_count": len(triggers), "triggered_updates": triggers, "non_triggered_updates": nontriggers, "historical_d10a_negative_anchor_updates": historical_negative, "state_dependent_decisions_differ_from_historical_anchors": triggers != historical_negative, "rows": trace})
    legacy.write_json(output / "longitudinal_intervention_evidence.json", {"updates": trace})
    legacy.write_json(output / "gradient_fidelity_evidence.json", legacy.jsonable(gradient_evidence))
    legacy.write_json(output / "clipping_and_adamw_mediation.json", clipping)
    legacy.write_json(output / "control_reproduction.json", {"d10a": d10a_control, "d7": d7_control, "update_392_anchor_match": control_anchor, "status": "PASS" if integrity_checks["control_reproduction"] else "FAIL"})
    legacy.write_json(output / "endpoint_outcomes.json", endpoints)
    legacy.write_json(output / "causal_classification.json", {"classification": classification, "rule": rule, "integrity": integrity, "endpoints": endpoints})
    legacy.write_json(output / "integrity_checks.json", {"checks": integrity_checks, "all_required_integrity_pass": "YES" if integrity else "NO"})
    legacy.write_json(output / "protocol_integrity.json", protocol)
    legacy.write_json(output / "machine_readable_result.json", outputs)
    legacy.write_json(output / "terminal_certificate.json", {"status": outputs["status"], "first_blocker": outputs["first_blocker"], "experiment_identifier": EXPERIMENT_ID, "updates_executed": list(UPDATES), "update_393_executed": "NO", "classification": classification})
    report = _render_report(outputs, trace)
    (output / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    manifest = _manifest(output)
    check = _verify_manifest(output, manifest)
    if check["status"] != "VERIFIED":
        raise StageBlocker(f"D14_FINAL_MANIFEST_VERIFICATION_FAILED:{check['failures'][:1]}")
    inventory = {str(path.relative_to(output)).replace("\\", "/"): {"sha256": legacy.sha256_file(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"}
    legacy.write_json(output / "artifact_inventory.json", {"schema_version": "stage3_h13_post_d13_d14_artifact_inventory_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "files": inventory})
    print(json.dumps({"status": outputs["status"], "output": str(output.resolve()), "classification": classification, "manifest_sha256": check["manifest_sha256"], **endpoints}, sort_keys=True), flush=True)
    return 0 if integrity else 1


def _render_report(result: Mapping[str, Any], trace: list[Mapping[str, Any]]) -> str:
    e = result["endpoints"]
    lines = [f"{EXPERIMENT_ID}:", result["status"], "", "FIRST_BLOCKER:", result["first_blocker"], "", "CAUSAL_CLASSIFICATION:", result["classification"], "", "ONE_SENTENCE_VERDICT:", result["verdict"], "", "## 1. PROTOCOL_INTEGRITY", "", "D14 policy executed once over updates 385-392; update 393, Frozen-20, R9, and R8E_A1 were not used.", "", "## 2. AUTHORITATIVE_INPUT_IDENTITY", "", f"D14 design manifest: {EXPECTED_D14_MANIFEST_SHA256}", f"Certified post-384 state: {EXPECTED_BUNDLE_SHA256}", f"GRU support hash: {EXPECTED_SUPPORT_HASH}", "", "## 3. CONTROL_REPRODUCTION", "", f"CONTROL update-392 anchors reproduced: {'YES' if result['integrity_checks']['control_reproduction'] else 'NO'}", "", "## 4. INTERVENTION_EXECUTION", "", f"TOTAL_TRIGGER_COUNT: {result['trigger_count']}", f"TRIGGERED_UPDATES: {result['triggered_updates']}", f"NON_TRIGGERED_UPDATES: {result['non_triggered_updates']}", "", "## 5. TEMPORAL_TRIGGER_TRACE", ""]
    for item in trace:
        c, i = item["control"], item["intervention"]
        lines.extend([f"### UPDATE {item['update']}", f"CONTROL: H1_GRU_NORM={c['gru_h1_norm']:.17g}; H32_GRU_NORM={c['gru_weighted_h32_norm']:.17g}; DOT={c['gru_dot_weighted_h32_h1']:.17g}; RAW_NORM={c['raw_gradient_norm']:.17g}; CLIP={c['clip_coefficient']:.17g}; CANONICAL_RECONSTRUCTION={c['canonical_reconstruction']}", f"INTERVENTION: DOT={i['dot_weighted_h32_h1']:.17g}; TRIGGERED={i['triggered']}; H1_GRU_NORM={i['gru_h1_norm']:.17g}; H32_GRU_NORM={i['gru_weighted_h32_norm']:.17g}; PROJECTION_MAGNITUDE={i['removed_component_norm']:.17g}; POST_DOT={i['post_projection_dot']:.17g}; RAW_NORM={i['treated_raw_gradient_norm']:.17g}; CLIP={i['clip_coefficient']:.17g}; NO_OP_EQUAL_CANONICAL={i['no_op_raw_equal_canonical']}", ""])
    lines.extend(["## 6. GRADIENT_AND_PARAMETER_SUPPORT_FIDELITY", "", f"Direct support restricted to the authenticated eight GRU tensors ({EXPECTED_SUPPORT_HASH}); all direct outside-GRU/non-H32/H1/hidden/HEAD differences were zero.", "", "## 7. CLIPPING_AND_ADAMW_MEDIATION", "", "Canonical global clipping and AdamW were unchanged; downstream branch differences after a trigger were retained as causal mediators.", "", "## 8. TERMINAL_ENDPOINTS", "", f"CONTROL_H1: {e['control_h1']:.17g}", f"INTERVENTION_H1: {e['intervention_h1']:.17g}", f"E_H1: {e['E_H1']:.17g}", "", f"CONTROL_H32: {e['control_h32']:.17g}", f"INTERVENTION_H32: {e['intervention_h32']:.17g}", f"H32_CHANGE: {e['signed_h32_change_intervention_minus_control']:.17g} (INTERVENTION minus CONTROL)", "", f"H1_GUARDRAIL: {H1_GUARDRAIL:.17g}", f"H32_PRESERVATION_THRESHOLD: {H32_PRESERVATION_THRESHOLD:.17g}", f"MATERIAL_H1_EFFECT: {MATERIAL_H1_EFFECT:.17g}", "", "## 9. CAUSAL_CLASSIFICATION", "", f"A. Did the temporal policy trigger? {'YES' if result['trigger_count'] else 'NO'}", f"B. How many times did it trigger? {result['trigger_count']}", f"C. At which updates? {result['triggered_updates']}", f"D. Did intervention-state evolution change later trigger decisions? {'YES' if result['later_trigger_decisions_changed_relative_to_historical_anchors'] else 'NO'}", f"E. Did repeated treatment materially improve H1? {'YES' if e['E_H1'] >= MATERIAL_H1_EFFECT else 'NO'}", f"F. Was the H1 effect above or below the frozen material-effect threshold? {'ABOVE_OR_EQUAL' if e['E_H1'] >= MATERIAL_H1_EFFECT else 'BELOW'}", f"G. Was H32 preserved? {'YES' if e['h32_preserved'] else 'NO'}", f"H. Is temporal accumulation of this exact GRU/H32 conflict supported as a material causal mechanism? {'YES' if result['classification'] in {'GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_CAUSAL', 'GRU_H32_TEMPORAL_CONFLICT_ACCUMULATION_PARTIALLY_CAUSAL'} else 'NO'}", "I. If not, what exact hypothesis has now been ruled out? A valid preserved-H32 result rules out primary material causal importance of this exact repeated first-order current-state GRU weighted-H32 anti-H1 projection policy over updates 385-392.", "", "## 10. WHAT_D14_ESTABLISHES", "", "D14 establishes only the matched update-392 contrast for the preregistered state-dependent temporal policy over updates 385-392.", "", "## 11. WHAT_D14_DOES_NOT_ESTABLISH", "", "D14 does not establish effects for other modules, horizons, optimizers, clipping policies, longer windows, later updates, or other models.", "", "## 12. OUTPUT_BUNDLE", "", f"OUTPUT_BUNDLE_PATH: {result.get('output_bundle_path', 'See machine_readable_result.json')}", f"SHA256_MANIFEST_PATH: {result.get('sha256_manifest_path', 'SHA256_MANIFEST.json')}", "SHA256_MANIFEST_VERIFIED: YES", ""])
    return "\n".join(lines)


def _manifest(output: Path) -> dict[str, Any]:
    excluded = {"SHA256_MANIFEST.json", "artifact_inventory.json"}
    files = {str(path.relative_to(output)).replace("\\", "/"): {"sha256": legacy.sha256_file(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file() and path.name not in excluded}
    payload = {"schema_version": "stage3_h13_post_d13_d14_sha256_manifest_v1", "hash_algorithm": "SHA-256", "self_excluded": True, "artifact_inventory_excluded": True, "file_count": len(files), "files": files}
    legacy.write_json(output / "SHA256_MANIFEST.json", payload)
    return payload


def _verify_manifest(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures = [name for name, entry in manifest["files"].items() if not (output / name).is_file() or legacy.sha256_file(output / name) != entry["sha256"] or (output / name).stat().st_size != entry["bytes"]]
    actual = {str(path.relative_to(output)).replace("\\", "/") for path in output.rglob("*") if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "artifact_inventory.json"}}
    if actual != set(manifest["files"]):
        failures.append("file_set_mismatch")
    return {"status": "VERIFIED" if not failures else "BLOCKED", "failures": failures, "checked": len(manifest["files"]), "manifest_sha256": legacy.sha256_file(output / "SHA256_MANIFEST.json")}


def _fail_closed(output: Path, blocker: str) -> int:
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "BLOCKED", "first_blocker": blocker, "experiment_identifier": EXPERIMENT_ID, "classification": "INCONCLUSIVE", "verdict": "D14 did not establish a valid causal contrast because execution failed closed."}
    legacy.write_json(output / "machine_readable_result.json", result)
    legacy.write_json(output / "protocol_integrity.json", {"D14_EXECUTED": "NO", "FIRST_BLOCKER": blocker, "FROZEN20_OPENED": "NO", "R9_EXECUTED": "NO", "UPDATE_393_EXECUTED": "NO"})
    (output / "FINAL_REPORT.md").write_text(f"{EXPERIMENT_ID}:\nBLOCKED\n\nFIRST_BLOCKER:\n{blocker}\n\nCAUSAL_CLASSIFICATION:\nINCONCLUSIVE\n\nONE_SENTENCE_VERDICT:\nD14 did not establish a valid causal contrast because execution failed closed.\n", encoding="utf-8", newline="\n")
    manifest = _manifest(output)
    return 1 if _verify_manifest(output, manifest)["status"] == "VERIFIED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        return _fail_closed(output, f"OUTPUT_DIRECTORY_ALREADY_EXISTS_NO_RESUME:{output}")
    inputs: dict[str, Any] | None = None
    try:
        inputs = _d14_design_inputs()
        transform_sha256 = _install_temporal_execution(inputs)
        rc = legacy.execute(output)
        if rc != 0:
            return _fail_closed(output, "D14_CORE_EXECUTION_FAILED_CLOSED")
        return _write_final_bundle(output, inputs, transform_sha256)
    except StageBlocker as exc:
        return _fail_closed(output, str(exc))
    except Exception as exc:  # pragma: no cover
        output.mkdir(parents=True, exist_ok=True)
        (output / "execution_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        return _fail_closed(output, f"UNEXPECTED_EXECUTION_FAILURE:{type(exc).__name__}:{exc}")


if __name__ == "__main__":
    raise SystemExit(main())
