from pathlib import Path


PLUGIN_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros2_overlay"
    / "src"
    / "deterministic_kdl_kinematics_plugin"
    / "src"
    / "deterministic_kdl_kinematics_plugin.cpp"
)


def test_stage192a_policies_are_explicit_and_nonrandom_escape_policies_do_not_wiggle_randomly():
    source = PLUGIN_SOURCE.read_text(encoding="utf-8")

    assert 'policy_ != "random"' in source
    assert 'policy_ != "disabled"' in source
    assert 'policy_ != "deterministic_sequence"' in source
    assert 'perturb.data.setRandom()' in source

    disabled_block = source.split('if (policy_ == "disabled")', 1)[1].split(
        'if (policy_ == "deterministic_sequence")', 1
    )[0]
    deterministic_block = source.split('if (policy_ == "deterministic_sequence")', 1)[1].split(
        '} else {', 1
    )[0]
    assert "setRandom" not in disabled_block
    assert "setRandom" not in deterministic_block
