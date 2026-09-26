from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage3_h6_4.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage3_h6_4", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(index: int, name: str, q: list[float]):
    return {"waypoint_index": index, "candidate_id": name, "joint_values": q, "branch_node_id": "b"}


def test_edge_cache_key_is_candidate_and_layer_exact():
    h = load_module()
    a = candidate(1, "a", [0.0] * 6)
    b = candidate(2, "b", [0.01] * 6)
    assert h.edge_key(a, b) == "1:a->2:b"
    assert h.edge_prefilter(a, b)["branch_prefilter_pass"] is True


def test_exact_dp_never_uses_native_invalid_edge():
    h = load_module()
    a = candidate(0, "a", [0.0] * 6)
    b = candidate(1, "b", [0.01] * 6)
    rows = [{"waypoint_index": 0}, {"waypoint_index": 1}]
    cache = {h.edge_key(a, b): {"state": "NATIVE_INVALID"}}
    result = h.exact_dp(rows, {0: [a], 1: [b]}, cache)
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "native_valid_graph_disconnect"


def test_exact_dp_reports_unknown_until_native_certified():
    h = load_module()
    a = candidate(0, "a", [0.0] * 6)
    b = candidate(1, "b", [0.01] * 6)
    rows = [{"waypoint_index": 0}, {"waypoint_index": 1}]
    result = h.exact_dp(rows, {0: [a], 1: [b]}, {})
    assert result["status"] == "PASSED"
    assert len(result["unknown_selected_edges"]) == 1


def test_segment_split_removes_cusp_and_retains_53_46_break():
    h = load_module()
    old = [
        {"segment_id": 0, "component_id": 1, "target_ids": list(range(12)), "waypoint_start": 0, "waypoint_end": 260},
        {"segment_id": 1, "component_id": 2, "target_ids": [51, 30], "waypoint_start": 261, "waypoint_end": 264},
        {"segment_id": 2, "component_id": 0, "target_ids": list(range(16)), "waypoint_start": 265, "waypoint_end": 587},
    ]
    segments = h.split_authoritative_segments(old)
    assert len(segments) == 5
    assert [(item["waypoint_start"], item["waypoint_end"]) for item in segments] == [(0, 220), (221, 260), (261, 264), (265, 562), (563, 587)]


def test_forbidden_stages_are_not_invoked_by_runner_source():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "send_goal_async(" not in text
    assert "stage3_h7" not in text.lower()
    assert "time_parameterization" not in text.lower()
