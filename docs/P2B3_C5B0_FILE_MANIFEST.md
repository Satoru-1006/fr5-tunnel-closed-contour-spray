# P2-B3-C5B0 — Minimal Reproducible GitHub Surface

This branch intentionally contains only the core implementation and the smallest
reproducible checks for the registration-specification closure. Generated result
JSON, build trees, install trees, logs, scratch probes, and drafts are not part of
the GitHub checkout; they are preserved in the Drive `P2-B3-C5B0/PROCESS_ARCHIVE`
folder.

## Core files

| Path | Purpose |
|---|---|
| `src/p2b3_c5b0_registration_spec.py` | Pure-Python registration transforms, semantic checks, failure taxonomy, and bounded comparison helpers. |
| `tests/test_p2b3_c5b0_registration_spec.py` | Known-answer and invariant tests for identity, SE(3), object preservation, q/timestamp preservation, and fail-closed semantics. |
| `cpp/p2b3_c5b0_ccd/CMakeLists.txt` | Minimal native Bullet fixture build definition. |
| `cpp/p2b3_c5b0_ccd/p2b3_c5b0_bullet_fixture.cpp` | Synthetic static-world Bullet sweep capability check; it is not an FR5 boundary replay. |
| `docs/P2B3_C5B0_README.md` | Frozen authority, scope, evidence boundaries, and reproduction commands. |
| `docs/P2B3_C5B0_FILE_MANIFEST.md` | This inventory and the explicit exclusion policy. |

## Reproduction

From the repository root:

```powershell
python -m pytest -q tests/test_p2b3_c5b0_registration_spec.py tests/test_p2b3_c5a_metrics.py tests/test_p2b3_c3_physical_uncertainty.py tests/test_p2b3_c4_scene_contract.py
```

Expected focused result at closure: `69 passed, 0 failed`.

The Bullet fixture is built outside the repository with the installed Bullet
development package. Its command is documented in `P2B3_C5B0_README.md`.

## Deliberate exclusions

- `p2b3_c5b0_result.json` is the authoritative generated result and is Drive-only.
- No build, install, cache, log, convergence shadow, or duplicated JSONL output is
  committed for C5B0.
- The process archive is evidence storage, not a new scientific authority and does
  not change the frozen Stage 3/C4/C5A inputs.
