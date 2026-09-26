"""Generate an uncertainty-aware, fail-closed D58 clearance report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "D58_STAGE4_FULL_SYSTEM_CLOSURE"
REPEATABILITY_M = 0.00002
REPEATABILITY_SOURCE = "https://www.fairino.com/industry_/2.html"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for short in ("auto0", "auto1"):
        path = OUT / f"full_stateful_metrics_{short}" / "metrics.json"
        obj = load(path)
        env = obj["geometry"]["minimum_environment_clearance_m"]["min"]
        self_clear = obj["geometry"]["minimum_self_clearance_m"]["min"]
        parent_env = obj["comparison_tolerances"]["parent_min_environment_clearance_m"]
        parent_self = obj["comparison_tolerances"]["parent_min_self_clearance_m"]
        rows.append(
            {
                "candidate": short,
                "model_clearance_m": {
                    "environment": env,
                    "self": self_clear,
                },
                "protected_parent_model_clearance_m": {
                    "environment": parent_env,
                    "self": parent_self,
                },
                "delta_vs_protected_parent_m": {
                    "environment": env - parent_env,
                    "self": self_clear - parent_self,
                },
                "repeatability_only_sensitivity_margin_m": {
                    "environment": env - REPEATABILITY_M,
                    "self": self_clear - REPEATABILITY_M,
                },
                "repeatability_only_sensitivity_margin_positive": {
                    "environment": env - REPEATABILITY_M > 0,
                    "self": self_clear - REPEATABILITY_M > 0,
                },
            }
        )
    report = {
        "schema_version": "d58-robust-clearance-uncertainty-v1",
        "status": "UNRESOLVED_NO_TOTAL_UNCERTAINTY_BOUND",
        "scope": "D58 C4 stateful shadow trajectories; model-space clearance only",
        "formula": "sensitivity_margin = model_clearance - specified_repeatability_only",
        "interpretation": "The sensitivity margins are not physical safety certification because repeatability is not total geometric uncertainty or absolute accuracy.",
        "uncertainty_budget": [
            {"contributor": "FR5 pose repeatability", "classification": "manufacturer_specified", "value_m": REPEATABILITY_M, "source": REPEATABILITY_SOURCE, "used_in_sensitivity": True},
            {"contributor": "joint-state / encoder uncertainty", "classification": "unknown_without_hardware_measurement", "value_m": None, "used_in_sensitivity": False},
            {"contributor": "base and TCP calibration", "classification": "unknown_without_calibration_measurement", "value_m": None, "used_in_sensitivity": False},
            {"contributor": "URDF dimensions and mesh approximation", "classification": "unknown_without_geometry_validation", "value_m": None, "used_in_sensitivity": False},
            {"contributor": "backlash, compliance, payload deflection, tracking error", "classification": "unknown_without_hardware_measurement", "value_m": None, "used_in_sensitivity": False},
            {"contributor": "numerical / sampling uncertainty", "classification": "bounded only by the reported software measurement procedure", "value_m": None, "used_in_sensitivity": False},
        ],
        "candidates": rows,
        "collision_semantics": "adaptive_discrete_interpolation",
        "continuous_self_collision": "NOT_AVAILABLE_AS_EXACT_ARTICULATED_FK_QT_CERTIFICATE",
        "hardware_clearance": "NOT_AVAILABLE",
        "acceptance_threshold": "UNRESOLVED_THRESHOLD",
        "promotion_allowed": False,
    }
    json_path = OUT / "D58_ROBUST_CLEARANCE.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# D58 Robust Clearance and Uncertainty",
        "",
        f"- Status: `{report['status']}`",
        "- Formula: `sensitivity_margin = model_clearance - specified_repeatability_only`",
        "- The manufacturer-specified FR5 repeatability is used only as a one-contributor sensitivity subtraction; it is not treated as total uncertainty or absolute accuracy.",
        "- Unknown contributors (joint state, calibration, mesh/URDF, compliance, payload, tracking, numerical certification) keep physical acceptance unresolved.",
        "",
        "| Candidate | Model env (m) | Model self (m) | Env delta vs parent (m) | Self delta vs parent (m) | Repeatability-only env margin (m) | Repeatability-only self margin (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['model_clearance_m']['environment']:.12g} | {row['model_clearance_m']['self']:.12g} | "
            f"{row['delta_vs_protected_parent_m']['environment']:.12g} | {row['delta_vs_protected_parent_m']['self']:.12g} | "
            f"{row['repeatability_only_sensitivity_margin_m']['environment']:.12g} | {row['repeatability_only_sensitivity_margin_m']['self']:.12g} |"
        )
    lines.extend(
        [
            "",
            "The positive sensitivity margins are model-derived diagnostics only. They do not authorize hardware safety or promotion.",
            "",
        ]
    )
    (OUT / "D58_ROBUST_CLEARANCE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(json_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
