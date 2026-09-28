"""Unconstrained open-path correspondence and local timing diagnostics for P2-B3-C1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class PathProjection:
    segment_index: int
    station_m: float
    distance_m: float


@dataclass(frozen=True)
class LocalSegmentTiming:
    start_index: int
    end_index: int
    distance_m: float
    delta_t_s: float
    speed_m_s: float
    speed_band_status: str
    dwell: bool


def _finite_vector(value: Sequence[float], *, size: int, name: str) -> tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"{name}_must_have_{size}_values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name}_must_be_finite")
    return result


def project_open_polyline(
    point: Sequence[float], path: Sequence[Sequence[float]],
) -> PathProjection:
    """Project onto every segment independently; never force monotone progress."""
    p = _finite_vector(point, size=3, name="point")
    vertices = [_finite_vector(vertex, size=3, name="path_vertex") for vertex in path]
    if len(vertices) < 2:
        raise ValueError("open_path_requires_at_least_two_vertices")

    best: tuple[float, int, float] | None = None
    cumulative = 0.0
    for index, (start, end) in enumerate(zip(vertices, vertices[1:])):
        delta = tuple(end[axis] - start[axis] for axis in range(3))
        length_sq = sum(value * value for value in delta)
        length = math.sqrt(length_sq)
        if length <= 0.0:
            raise ValueError(f"open_path_has_degenerate_segment:{index}")
        ratio = sum((p[axis] - start[axis]) * delta[axis] for axis in range(3)) / length_sq
        ratio = min(1.0, max(0.0, ratio))
        closest = tuple(start[axis] + ratio * delta[axis] for axis in range(3))
        distance = math.sqrt(sum((p[axis] - closest[axis]) ** 2 for axis in range(3)))
        candidate = (distance, index, cumulative + ratio * length)
        if best is None or candidate < best:
            best = candidate
        cumulative += length
    assert best is not None
    return PathProjection(segment_index=best[1], station_m=best[2], distance_m=best[0])


def station_deltas(projections: Sequence[PathProjection]) -> list[float]:
    return [right.station_m - left.station_m for left, right in zip(projections, projections[1:])]


def local_segment_timing(
    positions: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
    *,
    target_speed_m_s: float = 0.003,
    relative_tolerance: float = 0.05,
) -> list[LocalSegmentTiming]:
    """Report every adjacent TCP displacement, time delta, and local speed."""
    if len(positions) != len(timestamps_s):
        raise ValueError("position_timestamp_count_mismatch")
    if len(positions) < 2:
        raise ValueError("local_timing_requires_at_least_two_samples")
    if not math.isfinite(target_speed_m_s) or target_speed_m_s <= 0.0:
        raise ValueError("target_speed_must_be_positive_and_finite")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance_must_be_nonnegative_and_finite")

    points = [_finite_vector(point, size=3, name="tcp_position") for point in positions]
    times = [float(value) for value in timestamps_s]
    if not all(math.isfinite(value) for value in times):
        raise ValueError("timestamps_must_be_finite")
    lower = target_speed_m_s * (1.0 - relative_tolerance)
    upper = target_speed_m_s * (1.0 + relative_tolerance)
    result: list[LocalSegmentTiming] = []
    for index, (start, end) in enumerate(zip(points, points[1:])):
        distance = math.sqrt(sum((end[axis] - start[axis]) ** 2 for axis in range(3)))
        dt = times[index + 1] - times[index]
        if dt <= 0.0:
            raise ValueError(f"timestamps_must_increase:{index}")
        speed = distance / dt
        if speed < lower:
            status = "BELOW_CONFIGURED_BAND"
        elif speed > upper:
            status = "ABOVE_CONFIGURED_BAND"
        else:
            status = "WITHIN_CONFIGURED_BAND"
        result.append(LocalSegmentTiming(
            start_index=index,
            end_index=index + 1,
            distance_m=distance,
            delta_t_s=dt,
            speed_m_s=speed,
            speed_band_status=status,
            dwell=distance <= 1.0e-12,
        ))
    return result


def waypoint_semantics(index: int, *, corrected_candidate: bool) -> dict[str, object]:
    if not 0 <= index < 181:
        raise ValueError("waypoint_index_out_of_range")
    if corrected_candidate:
        return {
            "solver_role": "DLS_TARGET_SOLVED",
            "expected_target_index": index,
            "target_solved": True,
            "retained_seed": False,
            "upstream_source_row": 0 if index == 0 else None,
            "semantic_classification": "DLS_TARGET_STATE_SEEDED_FROM_D39_ROW0" if index == 0 else "DLS_TARGET_STATE",
        }
    if index < 16:
        return {
            "solver_role": "RETAINED_OBSERVED_WARM_START",
            "expected_target_index": index,
            "target_solved": False,
            "retained_seed": True,
            "upstream_source_row": index,
            "semantic_classification": "OBSERVED_D39_ROW_EMITTED_AS_UNSOLVED_PREFIX",
        }
    return {
        "solver_role": "DLS_TARGET_SOLVED",
        "expected_target_index": index,
        "target_solved": True,
        "retained_seed": False,
        "upstream_source_row": None,
        "semantic_classification": "DLS_TARGET_STATE",
    }
