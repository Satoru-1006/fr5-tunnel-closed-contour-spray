"""Focused tests for C3 stress/smoothness shadow semantics."""

from pathlib import Path

import numpy as np

from tools.stage4c_c3_shadow import family_for, smoothness


def test_family_classification_preserves_collision_sensitive_label() -> None:
    assert family_for("collision_sensitive_0051") == "COLLISION_SENSITIVE"
    assert family_for("adversarial_0100") == "ADVERSARIAL"


def test_smoothness_uses_persisted_jerk_and_time(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    fields = ["t"] + [f"j{i}_{suffix}" for suffix in ("q", "dq", "ddq", "jerk") for i in range(1, 7)]
    rows = []
    for t, jerk in ((0.0, 1.0), (1.0, 3.0), (2.0, 1.0)):
        rows.append([t] + [0.0] * 18 + [jerk] + [0.0] * 5)
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(",".join(fields) + "\n")
        for row in rows:
            stream.write(",".join(str(value) for value in row) + "\n")

    result = smoothness(path)

    assert result["max_abs_jerk_rad_s3"] == 3.0
    assert np.isclose(result["integrated_abs_jerk_rad_s2"], 4.0)
