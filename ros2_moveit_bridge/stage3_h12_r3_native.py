#!/usr/bin/env python3
"""H12-R3 native entry point.

The implementation is shared with the patched primitive worker so the
MoveIt2/TOTG/PlanningScene code path stays identical.  The worker now
classifies native Ruckig results explicitly and refuses post-certification for
``Working`` or any negative result.
"""

from stage3_h12_r2_native import main


if __name__ == "__main__":
    raise SystemExit(main())
