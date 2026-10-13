# FAIRINO FR5 C5B1 R3-R2 — minimal, self-contained offline replication

Contents: exact frozen CRLF C1 FK and target CSVs; original C5B1 native JSON; GitHub-authoritative R2 module and R3 baseline; new `src/p2b3_c5b1_r3r2_robust_raycast.py`; tests; `R3R2_RESULT.json`; research report.

No external Python dependencies beyond Python 3.10+ standard library. From extracted ZIP root:

```bash
python -m unittest discover -s tests -p 'test_p2b3_c5b1_*.py'
python -m src.p2b3_c5b1_r3r2_robust_raycast --c1-fk inputs/p2b3_c1_r0_fk_trace.csv --c1-target inputs/open_arch_tcp_poses_base_link.csv --c5b1-scan inputs/P2B3_C5B1_NATIVE_RESULT.json --output-json R3R2_REPLAY_FRESH.json
```

Expected new replay SHA-256: `b6078b5c62617fbbe7e989d2507ab03a968f39e8f3d808b6f1de967a12747def`.

The release is a **reference-geometry evaluation only**. Do not treat as actual tunnel mesh raycasting, coating validation, robot physical margin or CCD. Output refuses overwrite. The full immutable input CSVs preserve Windows CRLF intentionally; hashing LF GitHub rendered text yields a different byte SHA-256 although semantic numeric values match.
