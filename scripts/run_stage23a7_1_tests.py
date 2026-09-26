import json, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "outputs/ik_graph_stage23a7_1_runtime_audit/fr5_scaled_horseshoe_demo_v44"
cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_stage23a7_1_outputs.py", "tests/test_stage23a7_outputs.py"]
r = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
(out / "test_results.txt").write_text("COMMAND: " + " ".join(cmd) + "\nEXIT_CODE: " + str(r.returncode) + "\n\nSTDOUT\n" + r.stdout + "\nSTDERR\n" + r.stderr, encoding="utf-8")
(out / "test_command_manifest.json").write_text(json.dumps({"command": cmd, "cwd": str(root), "stdout": r.stdout, "stderr": r.stderr, "exit_code": r.returncode}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(r.stdout, end="")
print(r.stderr, end="", file=sys.stderr)
sys.exit(r.returncode)
