#!/usr/bin/env python3
"""Resume the existing H12-R native certification without rebuilding the model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage3_h12_r_certification import discover_h11, load_json, run_native_shards, sha256_file


OUTPUT = ROOT / "outputs/stage3_h12_r_constraint_aware_residual_repair_20260811T175146Z"


def main() -> int:
    h11 = discover_h11()
    contract = load_json(h11 / "h11_dataset_contract.json")
    h10 = Path(str(contract["source_h10"]))
    if not h10.is_absolute():
        h10 = ROOT / h10
    selected = load_json(OUTPUT / "selected_model_manifest.json")
    repair_sha = sha256_file(ROOT / "src/stage3_h12_r_residual.py")
    aggregate = run_native_shards(
        OUTPUT,
        OUTPUT / "stage3_h12_r_native_input.npz",
        h11 / "h11_window_manifest.json",
        h10.resolve(),
        h11,
        str(selected["checkpoint_sha256"]),
        repair_sha,
        4096,
    )
    print(aggregate, flush=True)
    return 0 if aggregate.get("status") == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
