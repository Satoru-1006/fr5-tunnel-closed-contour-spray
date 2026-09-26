"""Authoritative Stage 3 final-release entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stage3_d45_finalize_release import main


if __name__ == "__main__":
    raise SystemExit(main())
