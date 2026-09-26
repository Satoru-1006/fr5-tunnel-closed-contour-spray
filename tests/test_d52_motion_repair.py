import csv

import numpy as np

from tools.d52_motion_repair import NativeTrajectory, repair_execution_form, write_native


def _trajectory(tmp_path, with_bridge=True):
    fields = ("t", "j1_q", "j1_dq", "j1_ddq")
    rows = []
    for index in range(6):
        t = index * 0.01
        q = 0.5 * t * t
        rows.append({"t": t, "j1_q": q, "j1_dq": t, "j1_ddq": 1.0})
    if with_bridge:
        # Simulate the exporter: the bridge has an advanced timestamp, and
        # all later native states retain their regular 10 ms spacing after it.
        bridge_time = 0.547636992
        shift = bridge_time - rows[2]["t"] - 0.01
        bridge = dict(rows[3])
        bridge["t"] = bridge_time
        bridge["j1_dq"] = 100.0
        bridge["j1_ddq"] = -100.0
        for row in rows[4:]:
            row["t"] += shift
        rows[3:] = [bridge, *rows[4:]]
    return fields, rows


def _write_six_joint(path, fields, rows):
    full_fields = ("t",) + tuple(f"j{i}_{suffix}" for i in range(1, 7) for suffix in ("q", "dq", "ddq"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=full_fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, 0.0) for field in full_fields}
            writer.writerow(out)


def test_bridge_is_removed_and_following_times_shifted(tmp_path):
    fields, rows = _trajectory(tmp_path)
    source = tmp_path / "source.csv"
    destination = tmp_path / "repaired.csv"
    _write_six_joint(source, fields, rows)
    report = repair_execution_form(source, destination, expected_dt=0.01)
    assert report["status"] == "REPAIRED"
    with destination.open(encoding="utf-8", newline="") as stream:
        output = list(csv.DictReader(stream))
    times = np.asarray([float(row["t"]) for row in output])
    assert len(output) == len(rows)
    assert np.allclose(np.diff(times), 0.01)


def test_regular_trajectory_is_not_changed(tmp_path):
    fields, rows = _trajectory(tmp_path, with_bridge=False)
    source = tmp_path / "source.csv"
    destination = tmp_path / "repaired.csv"
    _write_six_joint(source, fields, rows)
    report = repair_execution_form(source, destination, expected_dt=0.01)
    assert report["status"] == "NO_REPAIR_NEEDED"
