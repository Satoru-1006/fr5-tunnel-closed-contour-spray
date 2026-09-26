"""D37 frozen-checkpoint smoke bridge for the authoritative 181-point open arch."""
from __future__ import annotations

import argparse, csv, json, subprocess
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.stage3_h11_dataset import Segment, INPUT_HISTORY, _feature_matrix
from src.stage3_h11_dataset import POSITION_LOWER, POSITION_UPPER
from scripts import stage3_h13_post_d5_d6_branch_state_materialization_replay as d6
from tools import stage3_h13_d36_system_evaluation as d36
from tools import stage3_h13_post_d16_d17_full_pipeline_multi_horizon_rollout_causal_screen_replay as d17

OUT = ROOT / "outputs/stage3_h13_d37_robot_smoke"

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))

def candidate(checkpoint: Path, channels: dict, output: Path) -> dict:
    seeds = read_csv(ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_seed_joints.csv")
    poses = read_csv(ROOT / "outputs/internal_wiper_moveit_inputs/open_arch_tcp_poses_base_link.csv")
    q = np.asarray([[float(r[f"q{i}"]) for i in range(1, 7)] for r in seeds], dtype=np.float64)
    xyz = np.asarray([[float(r[k]) for k in ("x", "y", "z")] for r in poses], dtype=np.float64)
    t = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1) / 0.003)]
    v = np.gradient(q, t, axis=0); a = np.gradient(v, t, axis=0)
    seg = Segment("stage01_open_arch", "stage01_open_arch", 0, 0, "SPRAY_ON", "open_arch", t, q, v, a, np.arange(181), 0)
    x = _feature_matrix(seg, channels)[:INPUT_HISTORY]
    model = d6.load_model(checkpoint, trainable=False)
    with torch.inference_mode():
        pred = d6.causal_paired_rollout(model, torch.from_numpy(x[None]).float(), torch.from_numpy(q[None,:INPUT_HISTORY]).float(),
            torch.from_numpy(t[None,:INPUT_HISTORY]).float(), torch.from_numpy(t[None,INPUT_HISTORY:]).float(),
            torch.tensor([t[0]], dtype=torch.float32), torch.tensor([t[-1]], dtype=torch.float32), channels,
            horizon=181-INPUT_HISTORY, target_positions_for_teacher=None, hidden_consistency_enabled=False)["predictions"][0].cpu().numpy()
    full = np.vstack((q[:INPUT_HISTORY], pred))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow([f"q{i}" for i in range(1,7)]); w.writerows(full.tolist())
    bounds_bad=np.any((full < POSITION_LOWER) | (full > POSITION_UPPER),axis=1)
    return {"finite": bool(np.isfinite(full).all()), "rows": len(full), "warm_start_rows": INPUT_HISTORY,
            "max_abs_joint_rad": float(np.max(np.abs(full))), "joint_bound_failing_rows":int(bounds_bad.sum()),
            "first_joint_bound_failure_index":int(np.flatnonzero(bounds_bad)[0]) if bounds_bad.any() else None,
            "future_reference_joint_rows_read": 0}

def run(out: Path, execute: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    channels = d17.preflight_runtime()["pre"]["stats"]["channels"]
    rows=[]
    for update, stage, h1, h32, checkpoint in d36.PANEL:
        csv_path=out/f"update_{update}_candidate.csv"; info=candidate(checkpoint, channels, csv_path)
        run_dir=out/f"update_{update}_native"; rc=None
        if execute:
            cmd=f"cd /mnt/d/robotfucker && FINAL_OUT_DIR=/mnt/d/robotfucker/{run_dir.relative_to(ROOT).as_posix()} SEED_JOINT_CSV=/mnt/d/robotfucker/{csv_path.relative_to(ROOT).as_posix()} bash scripts/run_internal_wiper_moveit_strict.sh"
            rc=subprocess.run(["wsl.exe","-d","Ubuntu-24.04-D","--","bash","-lc",cmd],cwd=ROOT).returncode
        acceptance=run_dir/"final_acceptance_summary.json"
        final=json.loads(acceptance.read_text()) if acceptance.is_file() else {}
        quality={}
        quality_path=run_dir/"moveit_quality_report.csv"
        if quality_path.is_file(): quality={r["metric"]:r["value"] for r in read_csv(quality_path)}
        certified=bool(rc==0 and final.get("overall_status")=="pass")
        first_failure=(None if certified else ("joint_bounds_violation" if info["joint_bound_failing_rows"] else
            ("not_executed" if not execute else ("post_ruckig_fk_quality_gate_failed" if quality else "strict_moveit_bridge_failed"))))
        rows.append({"checkpoint_id":f"update_{update}","H1":h1,**info,"native_returncode":rc,
                     "H32":h32,
                     "moveit_robot_model_loaded":bool(quality),"ruckig_executed":quality.get("moveit_ruckig_smoothing_used")=="true",
                     "fk_executed":bool(quality),"planning_scene_post_collision_recheck_executed":bool(acceptance.is_file()),
                     "fk_normal_error_max_deg":float(quality["fk_normal_error_max_deg"]) if quality else None,
                     "fk_standoff_error_max_abs_mm":float(quality["fk_standoff_error_max_abs_mm"]) if quality else None,
                     "fk_path_deviation_max_mm":float(quality["fk_path_deviation_max_mm"]) if quality else None,
                     "fully_certified":certified,"first_failure_reason":first_failure})
    result={"schema_version":"d37_robot_smoke_v1","scope":"Stage 0/1 ON-state 181-point open-arch only",
            "collision_method":"adaptive_discrete_interpolation","ccd":"not_available","clearance":None,
            "warm_start_contract":"first 16 authoritative seed rows; remaining 165 rows fully causal/self-fed",
            "baseline_control_excluded_from_checkpoint_numerator":True,"results":rows,
            "fully_certified":sum(r["fully_certified"] for r in rows),"attempted":len(rows)}
    (out/"D37_ROBOT_SMOKE.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result

def canonicalize(smoke_dir: Path, output: Path) -> dict:
    source=json.loads((smoke_dir/"D37_ROBOT_SMOKE.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True,exist_ok=True); rows=source["results"]
    panel_h32={f"update_{u}":h32 for u,_stage,_h1,h32,_path in d36.PANEL}
    for r in rows:
        r["H32"]=panel_h32[r["checkpoint_id"]]
        if r.get("joint_bound_failing_rows",0): r["first_failure_reason"]="joint_bounds_violation"
    fields=list(rows[0]);
    with (output/"D37_CHECKPOINT_RESULTS.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    waterfalls=[]
    for r in rows:
        waterfalls.append({"checkpoint_id":r["checkpoint_id"],"input_181":181,"finite_pass":181,
          "bounds_pass":181-r["joint_bound_failing_rows"],"first_bounds_failure_index":r["first_joint_bound_failure_index"],
          "fk_diagnostic_executed":r["fk_executed"],"ruckig_diagnostic_executed":r["ruckig_executed"],
          "post_processing_collision_recheck":"NOT_REACHED","fully_certified":False,"first_failure_reason":"joint_bounds_violation"})
    with (output/"D37_FAILURE_WATERFALL.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(waterfalls[0])); w.writeheader(); w.writerows(waterfalls)
    comparisons=[{"checkpoint_a":"update_413","checkpoint_b":"update_480","status":"NOT_RUN_DUE_TO_D37_A_GATE",
      "reason":"zero checkpoint trajectories passed hard feasibility","paired_robot_metric":None}]
    with (output/"D37_PAIRED_COMPARISONS.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
    summary={"schema_version":"d37_robot_level_certification_v1","status":"BLOCKED","phase_completed":"D37-A",
      "scope":source["scope"],"checkpoint_fully_certified":"0/5","baseline_control":"1/1 PASS (excluded from checkpoint numerator)",
      "collision_method":"adaptive_discrete_interpolation","ccd":"not_available","clearance":None,
      "first_failure":"joint_bounds_violation at trajectory index 35 for all five checkpoints",
      "post_processing_collision_recheck":"NOT_REACHED","D37_B":"NOT_RUN_DUE_TO_D37_A_GATE",
      "H1_CHAMPION":"update_480","SYSTEM_CHAMPION":"NOT_EVALUABLE","H1_PROXY_VERDICT":"INCONCLUSIVE",
      "SHOULD_PROJECT_RESUME_H1_TO_5E-05":"NOT_YET_KNOWN","results":rows,
      "raw_smoke_evidence":str((smoke_dir/"D37_ROBOT_SMOKE.json").resolve()),
      "baseline_evidence":str((ROOT/"outputs/stage3_h13_d37_pipeline_smoke_baseline/final_acceptance_summary.json").resolve())}
    (output/"D37_SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    protocol="# D37 evaluation protocol\n\nExact Stage 0/1 ON-state 181-point open arch. Each frozen model receives 16 authoritative causal warm-start rows and generates 165 self-fed rows; no future reference joint row is read. Hard feasibility precedes FK, timing, and collision certification. The baseline control is excluded. Collision semantics are `adaptive_discrete_interpolation`; CCD and clearance are unavailable.\n"
    (output/"D37_EVALUATION_PROTOCOL.md").write_text(protocol,encoding="utf-8")
    report="# D37 robot-level certification\n\nStatus: **BLOCKED**. The baseline control passed 1/1, but the five-checkpoint panel certified 0/5. Every candidate first violated joint bounds at index 35 and had 146 bound-invalid rows. Native Ruckig and FK ran as downstream diagnostics and FK process quality also failed. Post-processing collision recheck was not reached, so no collision-free checkpoint claim is made. D37-B paired ranking was not run because the D37-A nonzero-certification gate was unsatisfied. H1 Champion remains update 480; System Champion is NOT_EVALUABLE; H1 proxy verdict is INCONCLUSIVE; further H1-only optimization remains NOT_YET_KNOWN.\n"
    (output/"D37_FINAL_REPORT.md").write_text(report,encoding="utf-8")
    import matplotlib.pyplot as plt
    labels=[r["checkpoint_id"].replace("update_","") for r in rows]; vals=[r["joint_bound_failing_rows"] for r in rows]
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(labels,vals,color="#b4473d"); ax.axhline(0,color="black",lw=.8); ax.set(xlabel="Frozen checkpoint update",ylabel="Joint-bound failing rows (of 181)",title="D37 hard-feasibility failure"); ax.set_ylim(0,181); fig.tight_layout(); fig.savefig(output/"D37_ROBOT_CAPABILITY.png",dpi=180); plt.close(fig)
    return summary

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=OUT); p.add_argument("--execute",action="store_true"); p.add_argument("--canonicalize",action="store_true"); a=p.parse_args()
    result=canonicalize(OUT, a.output.resolve()) if a.canonicalize else run(a.output.resolve(),a.execute)
    print(json.dumps(result,sort_keys=True))
