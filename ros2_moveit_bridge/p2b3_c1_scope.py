"""Pure policy helper for the P2-B3-C1 ordered R0 process scope."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DlsTrajectoryScope:
    emit_warm_start_prefix: bool
    initial_seed_row: int
    first_target_index: int


def parse_initialization_only_flag(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if normalized not in {"", "false", "0", "true", "1"}:
        raise ValueError("P2B3_C1_INITIALIZATION_ONLY_must_be_boolean")
    return normalized in {"true", "1"}


def resolve_dls_trajectory_scope(
    *,
    initialization_only: bool,
    p2b2_variant: str,
    solver_variant: str,
    warm_start_rows: int,
    secondary_objective: str,
) -> DlsTrajectoryScope:
    """Keep frozen defaults; C1 uses D39 observations only to seed target 0."""

    if not initialization_only:
        return DlsTrajectoryScope(
            emit_warm_start_prefix=True,
            initial_seed_row=warm_start_rows - 1,
            first_target_index=warm_start_rows,
        )
    if not (
        p2b2_variant.upper() == "R0"
        and solver_variant.upper() == "B0"
        and warm_start_rows == 16
        and secondary_objective == "none"
    ):
        raise ValueError("P2B3_C1_initialization_only_requires_frozen_R0_B0_policy")
    return DlsTrajectoryScope(
        emit_warm_start_prefix=False,
        initial_seed_row=0,
        first_target_index=0,
    )
