"""Run the D50 FCL continuous-self-collision gate one case per process."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CASES = (
    "regression_0000", "regression_0001", "normal_0000", "normal_0100",
    "boundary_0000", "boundary_0100", "collision_sensitive_0000", "collision_sensitive_0051",
    "adversarial_0100", "adversarial_0101", "perturbation_0000", "perturbation_0100",
)


def run_case(case_id: str, native_root: str, binary: str, urdf: str, srdf: str) -> dict[str, object]:
    output = f"{native_root}/c2_cases/{case_id}"
    shell = "set -e; mkdir -p {out}; source /opt/ros/jazzy/setup.bash; source /home/robot/d50_d41_native_v1/install/setup.bash; export LD_LIBRARY_PATH={binary_parent}:/mnt/d/robotfucker/install/lib:/mnt/d/robotfucker/tmp/d41_install/lib:/opt/ros/jazzy/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/sdformat_vendor/lib:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_tools_vendor/lib:/usr/lib/x86_64-linux-gnu; exec {binary} --manifest {manifest} --urdf {urdf} --srdf {srdf} --output {out} --case {case}".format(
        out=shlex.quote(output), binary_parent=shlex.quote(str(Path(binary).parent.parent)), binary=shlex.quote(binary), manifest=shlex.quote(f"{native_root}/manifest.csv"), urdf=shlex.quote(urdf), srdf=shlex.quote(srdf), case=shlex.quote(case_id)
    )
    log = Path(f"outputs/D50_STAGE4_INTEGRATED_SHADOW/phase_c_global_0075/c2_process_logs/{case_id}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04-D", "--", "bash", "-lc", shell], stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, timeout=60 * 60 * 4)
    return {"case_id": case_id, "return_code": completed.returncode, "native_output": output, "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", default="/home/robot/d50_global_c2")
    parser.add_argument("--binary", default="/mnt/d/robotfucker/tmp/stage4d_install/lib/stage4d_continuous_self_collision/stage4d_continuous_self_collision")
    parser.add_argument("--urdf", default="/home/robot/d50_d41_native_v2/derived_robot_model.urdf")
    parser.add_argument("--srdf", default="/home/robot/d50_d41_native_v2/fairino5_v6_spray_tcp.srdf")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_case, case, args.native_root, args.binary, args.urdf, args.srdf) for case in CASES]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    results.sort(key=lambda item: str(item["case_id"]))
    Path("outputs/D50_STAGE4_INTEGRATED_SHADOW/phase_c_global_0075/c2_process_results.json").write_text(json.dumps({"schema_version": "d50-c2-process-isolation-v1", "workers": args.workers, "case_count": len(results), "cases": results}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(int(item["return_code"]) == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
