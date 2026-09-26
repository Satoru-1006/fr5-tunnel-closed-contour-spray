"""CLI for the one-purpose Stage 2.8S-R2 canonical identity migration.

This is an administrative ZERO-GOAL operation.  It never starts ROS, creates
an ActionClient, consumes the Formal R2 budget, or calls a transport send.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage28sr2_formal_ledger import CANONICAL_LEDGER_PATH  # noqa: E402
from src.stage28sr2_ledger_identity_migration import (  # noqa: E402
    AlreadyMigratedError,
    MigrationError,
    migrate_runner_identity,
)


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "outputs" / f"stage28sr2_h4_1_identity_migration_{stamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2.8S-R2 H4.1 canonical runner identity migration")
    parser.add_argument("--ledger", type=Path, default=CANONICAL_LEDGER_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--runner", type=Path, default=ROOT / "scripts" / "stage28sr2_production_runner.py")
    parser.add_argument("--expected-old-runner-sha256", required=True)
    parser.add_argument("--expected-ledger-sha256", default=None)
    args = parser.parse_args(argv)
    output = (args.output or _default_output()).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    try:
        certificate = migrate_runner_identity(
            ledger_path=args.ledger,
            output_dir=output,
            expected_old_runner_sha256=args.expected_old_runner_sha256,
            expected_ledger_sha256=args.expected_ledger_sha256,
            runner_path=args.runner,
            repo_root=ROOT,
            migration_tool_path=Path(__file__),
        )
    except AlreadyMigratedError as error:
        print(json.dumps({"migration_status": "already_migrated", "first_blocker": str(error)}, indent=2))
        return 2
    except MigrationError as error:
        print(json.dumps({"migration_status": "blocked", "first_blocker": str(error)}, indent=2))
        return 2
    print(json.dumps(certificate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
