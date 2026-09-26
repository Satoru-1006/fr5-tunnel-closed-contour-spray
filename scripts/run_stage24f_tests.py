"""Run and persist the Stage 2.4F and frozen Stage 2.3B/2.4A-2.4E tests."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/test_stage23b_outputs.py",
    "tests/test_stage24_outputs.py",
    "tests/test_stage24a_outputs.py",
    "tests/test_stage24b_outputs.py",
    "tests/test_stage24c_outputs.py",
    "tests/test_stage24d_outputs.py",
    "tests/test_stage24e_outputs.py",
    "tests/test_stage24f_outputs.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wsl", action="store_true", help="run the regression suite in the native WSL ROS/Python environment")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.wsl:
        linux_tests = " ".join(TESTS)
        command = ["wsl.exe", "bash", "-lc", f"cd /mnt/c/Users/86198/Desktop/robotfucker && PYTHONPATH=/mnt/c/Users/86198/Desktop/robotfucker python3 -m pytest -q {linux_tests}"]
    else:
        command = ["pytest", "-q", *TESTS]
    proc = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    report = [
        f"started_utc: {datetime.now(timezone.utc).isoformat()}",
        f"command: {' '.join(command)}",
        f"exit_code: {proc.returncode}",
        "",
        "--- stdout ---",
        proc.stdout,
        "--- stderr ---",
        proc.stderr,
    ]
    (out / "stage24f_test_report.txt").write_text("\n".join(report), encoding="utf-8")
    (out / "stage24f_test_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out / "stage24f_test_stderr.log").write_text(proc.stderr, encoding="utf-8")
    (out / "stage24f_test_report.json").write_text(
        json.dumps(
            {
                "status": "passed" if proc.returncode == 0 else "failed",
                "exit_code": proc.returncode,
                "command": " ".join(command),
                "tests": TESTS,
                "native_environment": bool(args.wsl),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    commands_log = out / "stage24f_commands.log"
    with commands_log.open("a", encoding="utf-8") as handle:
        handle.write("\n" + " ".join(command) + "\n")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
