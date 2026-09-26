from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "ros2_overlay/src/deterministic_kdl_kinematics_plugin/src/deterministic_kdl_kinematics_plugin.cpp"
MANIFEST = ROOT / "outputs/ik_graph_stage192c/policy_manifest.yaml"


def test_stage192c_policy_names_and_explicit_generator_are_present():
    source = PLUGIN.read_text(encoding="utf-8")
    for policy in [
        "P0_disabled",
        "P1_exact_replay",
        "P2_fixed_prng",
        "P3_axis_basis",
        "P4_multi_joint_templates",
        "P5_jacobian_svd",
        "P6_hybrid",
    ]:
        assert policy in source
    assert "splitmix64_next" in source
    assert "uint64_to_unit" in source
    assert "ChainJntToJacSolver" in source
    assert "Eigen::JacobiSVD" in source
    assert "P5_vmin_vsecond_combo_" in source


def test_stage192c_output_status_is_evidence_bound():
    status = (ROOT / "outputs/ik_graph_stage192c/stage192c_status.yaml").read_text(encoding="utf-8")
    assert "status: failed_coverage" in status
    assert "status: blocked" in status
    assert "full_pipeline_status: not_run_gate_not_met" in status
