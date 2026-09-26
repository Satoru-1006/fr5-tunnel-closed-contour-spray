"""D60 deterministic semantic probes for continuous articulated collision.

This is deliberately a shadow-only experiment.  The synthetic probes have
known answers and distinguish endpoint checks, a rigid endpoint-chord proxy,
and the actual nonlinear path induced by q(t).  They do not claim to certify
the FR5 mesh.  The installed FCL known-answer executable is run separately to
anchor the backend result: FCL can certify the motion model supplied to its
API, while its endpoint API receives rigid begin/end transforms rather than a
joint-space q(t) function.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "D60_SYSTEM_LEVEL_CLOSURE" / "semantic_probes"
DEFAULT_TRAJECTORY = (
    ROOT
    / "outputs"
    / "D59_OFFLINE_ALGORITHM_SYSTEM_CLOSURE"
    / "timing_scale025_auto0"
    / "trajectories"
    / "adversarial_0100.csv"
)
DEFAULT_FCL = (
    ROOT
    / "tmp"
    / "stage4d_install"
    / "lib"
    / "stage4d_continuous_self_collision"
    / "stage4d_fcl_known_answer"
)

Point = tuple[float, float]
PathFn = Callable[[float], Point]


def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    direction = sub(end, start)
    length_sq = direction[0] * direction[0] + direction[1] * direction[1]
    if length_sq == 0.0:
        return norm(sub(point, start))
    u = ((point[0] - start[0]) * direction[0] + (point[1] - start[1]) * direction[1]) / length_sq
    u = min(1.0, max(0.0, u))
    closest = (start[0] + u * direction[0], start[1] + u * direction[1])
    return norm(sub(point, closest))


def sample_min(values: Iterable[float]) -> tuple[float, int]:
    best = math.inf
    best_index = -1
    for index, value in enumerate(values):
        if value < best:
            best = value
            best_index = index
    return best, best_index


def two_link_fk(q1: float, q2: float, second_length: float = 0.6) -> Point:
    return (
        math.cos(q1) + second_length * math.cos(q1 + q2),
        math.sin(q1) + second_length * math.sin(q1 + q2),
    )


def probe_case(
    case_id: str,
    description: str,
    path: PathFn,
    obstacle: Point,
    radius: float,
    *,
    expected_chord_collision: bool = False,
    proxy_contact_time: float | None = None,
    extra: dict[str, object] | None = None,
    dense_count: int = 200_001,
) -> dict[str, object]:
    start = path(0.0)
    end = path(1.0)
    signed_distance = lambda t: norm(sub(path(t), obstacle)) - radius
    dense_values = (signed_distance(index / (dense_count - 1)) for index in range(dense_count))
    true_min, true_index = sample_min(dense_values)
    coarse_count = 17
    coarse_values = [signed_distance(index / (coarse_count - 1)) for index in range(coarse_count)]
    endpoint_values = [signed_distance(0.0), signed_distance(1.0)]
    chord_clearance = point_segment_distance(obstacle, start, end) - radius
    payload: dict[str, object] = {
        "case_id": case_id,
        "description": description,
        "radius_m": radius,
        "start_point": list(start),
        "end_point": list(end),
        "endpoint_min_signed_distance_m": min(endpoint_values),
        "endpoint_discrete_collision": min(endpoint_values) <= 0.0,
        "endpoint_chord_proxy_min_signed_distance_m": chord_clearance,
        "endpoint_chord_proxy_collision": chord_clearance <= 0.0,
        "coarse_sample_count": coarse_count,
        "coarse_sample_min_signed_distance_m": min(coarse_values),
        "coarse_sample_collision": min(coarse_values) <= 0.0,
        "dense_oracle_sample_count": dense_count,
        "dense_oracle_min_signed_distance_m": true_min,
        "dense_oracle_collision": true_min <= 0.0,
        "dense_oracle_first_min_index": true_index,
        "dense_oracle_min_time": true_index / (dense_count - 1),
        "proxy_contact_time": proxy_contact_time,
        "expected": {
            "endpoint_discrete_collision": False,
            "endpoint_chord_proxy_collision": expected_chord_collision,
            "dense_oracle_collision": True,
        },
        "status": (
            "PASS"
            if not min(endpoint_values) <= 0.0
            and (chord_clearance <= 0.0) == expected_chord_collision
            and true_min <= 0.0
            else "FAIL"
        ),
    }
    if extra:
        payload["extra"] = extra
    return payload


def make_probes() -> list[dict[str, object]]:
    # Case 1: the canonical endpoint-only miss.
    case1 = probe_case(
        "case_01_endpoint_free_middle_collision",
        "A point crosses a static disk while both endpoints remain collision-free.",
        lambda t: (-1.0 + 2.0 * t, 0.0),
        (0.0, 0.0),
        0.1,
        expected_chord_collision=True,
    )

    # Case 2: two-link FK path.  The endpoint chord stays on y=0, while the
    # true q(t)->FK(q(t)) path rises through the obstacle at t=0.5.
    midpoint = two_link_fk(math.pi / 2.0, math.pi / 2.0)
    case2 = probe_case(
        "case_02_nonlinear_two_link_fk",
        "A simultaneous two-joint interpolation produces a nonlinear Cartesian link path not represented by the endpoint chord.",
        lambda t: two_link_fk(math.pi * t, math.pi * t),
        midpoint,
        0.05,
        extra={"q_start": [0.0, 0.0], "q_end": [math.pi, math.pi], "obstacle_at_true_midpoint": True},
    )

    # Case 3: a long link rotates a full turn around its joint.  The endpoint
    # transform is identical at t=0 and t=1, so a rigid endpoint proxy sees no
    # motion even though the swept link visits every angle.
    case3 = probe_case(
        "case_03_rapid_long_link_rotation",
        "A long rotating link returns to its endpoint pose but sweeps across a static obstacle during the turn.",
        lambda t: (math.cos(2.0 * math.pi * t), math.sin(2.0 * math.pi * t)),
        (0.0, 1.0),
        0.02,
        extra={"link_length_m": 1.0, "angular_travel_rad": 2.0 * math.pi},
    )

    # Case 4: a collision window narrower than the coarse temporal grid.
    case4 = probe_case(
        "case_04_narrow_grazing_window",
        "A narrow collision interval is deliberately positioned between a 17-sample grid's time points.",
        lambda t: (-1.0 + 2.0 * t, 0.0006),
        (0.013, 0.0),
        0.001,
        expected_chord_collision=True,
        extra={"coarse_grid_is_expected_to_miss": True, "collision_window_approx_width_s": 0.001},
    )

    # Case 5: q(t) is nonuniform.  The geometry is the same arc, but the
    # normalized rigid endpoint interpolation predicts a different contact
    # time than the actual q(t)=pi*t^2 trajectory.
    case5 = probe_case(
        "case_05_nonuniform_q_time_evolution",
        "The joint follows q(t)=pi*t^2; endpoint pose interpolation finds the same geometry but the wrong contact time.",
        lambda t: (math.cos(math.pi * t * t), math.sin(math.pi * t * t)),
        (math.sqrt(0.5), math.sqrt(0.5)),
        0.02,
        proxy_contact_time=0.25,
        extra={"q_of_t": "pi*t^2", "true_contact_time_expected": 0.5, "linear_endpoint_pose_contact_time": 0.25},
    )

    # Case 6: both links move and their relative motion is nonlinear.  Both
    # objects return to their initial endpoints, so a relative endpoint chord
    # is stationary and misses the interior contact.
    def moving_a(t: float) -> Point:
        theta = 2.0 * math.pi * t
        return math.cos(theta), math.sin(theta)

    def moving_b(t: float) -> Point:
        theta = 2.0 * math.pi * t + math.pi * (2.0 * t - 1.0) ** 2
        return math.cos(theta), math.sin(theta)

    case6 = probe_case(
        "case_06_simultaneous_relative_motion",
        "Two moving links have safe identical endpoint pairs but collide when their relative phase closes at mid-interval.",
        lambda t: sub(moving_a(t), moving_b(t)),
        (0.0, 0.0),
        0.05,
        extra={"moving_link_a": "unit_circle_angle=2*pi*t", "moving_link_b": "unit_circle_angle=2*pi*t+pi*(2*t-1)^2"},
    )

    # Case 7: endpoint transform translation is a misleading proxy for direct
    # FK.  q2 bows out and returns, creating a large mid-interval deviation.
    midpoint7 = two_link_fk(math.pi / 2.0, 0.8)
    case7 = probe_case(
        "case_07_endpoint_transform_vs_direct_fk",
        "Endpoint TCP transforms are safe under linear pose translation, while direct FK of the nonlinear joint path enters the obstacle.",
        lambda t: two_link_fk(math.pi * t, 0.8 * math.sin(math.pi * t)),
        midpoint7,
        0.03,
        extra={"q_of_t": ["pi*t", "0.8*sin(pi*t)"], "endpoint_pose_interpolation": "linear_translation_proxy"},
    )
    return [case1, case2, case3, case4, case5, case6, case7]


def run_fcl_known_answer(binary: Path) -> dict[str, object]:
    if not binary.exists():
        return {"status": "NOT_AVAILABLE", "binary": str(binary), "reason": "known_answer_binary_missing"}
    try:
        completed = subprocess.run(
            [
                "wsl.exe",
                "-d",
                "Ubuntu-24.04-D",
                "--",
                str(binary.resolve()).replace("\\", "/").replace("D:/", "/mnt/d/", 1),
            ],
            check=False,
            capture_output=True,
            text=False,
            timeout=120,
        )
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        lines = [line.replace("\x00", "").strip() for line in stdout.splitlines() if line.strip()]
        records = [json.loads(line) for line in lines if line.startswith("{")]
        passed = completed.returncode == 0 and records and all(item.get("status") == "PASS" for item in records)
        return {
            "status": "PASS" if passed else "FAIL",
            "binary": str(binary),
            "returncode": completed.returncode,
            "cases": records,
            "stderr_tail": stderr.replace("\x00", "")[-1000:],
            "semantics": "FCL continuousCollide over supplied rigid begin/end transforms; not articulated FK(q(t))",
        }
    except Exception as error:  # a backend probe failure is evidence, not a false pass
        return {"status": "FAIL", "binary": str(binary), "error": f"{type(error).__name__}:{error}"}


def read_timebase(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = [float(row["t"]) for row in rows]
    deltas = [b - a for a, b in zip(times, times[1:])]
    return {
        "trajectory": str(path),
        "state_count": len(rows),
        "interval_count": len(deltas),
        "duration_s": times[-1] - times[0],
        "dt_min_s": min(deltas),
        "dt_max_s": max(deltas),
        "dt_mean_s": sum(deltas) / len(deltas),
        "nonuniform_timebase": max(deltas) - min(deltas) > 1.0e-12,
        "finite_and_strictly_increasing": all(math.isfinite(value) for value in times) and all(value > 0.0 for value in deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--fcl-binary", type=Path, default=DEFAULT_FCL)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = make_probes()
    fcl = run_fcl_known_answer(args.fcl_binary.resolve())
    timebase = read_timebase(args.trajectory.resolve())
    payload = {
        "schema_version": "d60-adversarial-semantic-probes-v1",
        "scope": "synthetic semantic validation plus installed FCL known-answer; no protected scientific mutation",
        "status": "PASS" if all(item["status"] == "PASS" for item in probes) and fcl["status"] == "PASS" and timebase["finite_and_strictly_increasing"] else "BLOCKED",
        "synthetic_probe_count": len(probes),
        "synthetic_probes": probes,
        "fcl_known_answer": fcl,
        "d59_native_timebase": timebase,
        "interpretation": {
            "endpoint_discrete": "necessary but insufficient",
            "fcl_endpoint_api": "validates the rigid motion model passed to the API",
            "articulated_fk_q_t": "requires a trajectory-aware enclosure or a backend that natively accepts the articulated q(t) semantics",
            "sampling": "used only as a known-answer oracle in this synthetic harness, never promoted as a certificate",
        },
    }
    (output_dir / "D60_SEMANTIC_PROBES.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "D60_SEMANTIC_PROBES.md").write_text(
        "# D60 semantic probes\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "The seven deterministic probes all construct endpoint-safe paths whose dense FK/path oracle collides. "
        "This establishes why endpoint checks, coarse sampling, and rigid endpoint transforms cannot be relabelled as exact articulated `FK(q(t))` certification. "
        "The installed FCL known-answer test is recorded separately and only validates the rigid motion model supplied to `continuousCollide`.\n\n"
        "See `D60_SEMANTIC_PROBES.json` for complete values and the D59 native timebase evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(output_dir / "D60_SEMANTIC_PROBES.json")}, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
