from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


CONTACT_RE = re.compile(
    r"Found a contact between '(?P<object>[^']+)' \(type '(?P<object_type>[^']+)'\) "
    r"and '(?P<link>[^']+)' \(type '(?P<link_type>[^']+)'\)"
)
RUNTIME_RE = re.compile(r"RuntimeError: MoveIt2 IK failed the closed contour gate: (?P<body>.*)")
INDEX_RE = re.compile(r"index=(?P<index>\d+), (?P<detail>.*?)(?=; index=\d+,|$)")
JOINT_STEP_RE = re.compile(r"joint=(?P<joint>j\d+),joint_step=(?P<step>[0-9.]+)deg")
ALL_COLLIDING_RE = re.compile(r"all IK candidates are colliding(?:, best=(?P<best>[^;]+))?")


def _parse_failure_detail(detail: str) -> dict[str, object]:
    joint_steps = [
        {"joint": match.group("joint"), "step_deg": float(match.group("step"))}
        for match in JOINT_STEP_RE.finditer(detail)
    ]
    all_colliding = ALL_COLLIDING_RE.search(detail)
    return {
        "detail": detail,
        "all_candidates_colliding": bool(all_colliding),
        "collision_best": all_colliding.group("best") if all_colliding and all_colliding.group("best") else "",
        "joint_steps": joint_steps,
        "max_reported_joint_step_deg": max((row["step_deg"] for row in joint_steps), default=0.0),
    }


def summarize_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    contacts: list[dict[str, str]] = []
    for match in CONTACT_RE.finditer(text):
        contacts.append(
            {
                "object": match.group("object"),
                "object_type": match.group("object_type"),
                "link": match.group("link"),
                "link_type": match.group("link_type"),
                "pair": f"{match.group('object')} <-> {match.group('link')}",
            }
        )

    failures: list[dict[str, object]] = []
    runtime_match = RUNTIME_RE.search(text)
    if runtime_match:
        body = runtime_match.group("body")
        for match in INDEX_RE.finditer(body):
            parsed = _parse_failure_detail(match.group("detail"))
            failures.append({"index": int(match.group("index")), **parsed})

    contact_pair_counts = Counter(contact["pair"] for contact in contacts)
    joint_counter: Counter[str] = Counter()
    max_joint_step = 0.0
    for failure in failures:
        for row in failure["joint_steps"]:
            joint_counter[str(row["joint"])] += 1
            max_joint_step = max(max_joint_step, float(row["step_deg"]))

    first_failure = failures[0] if failures else None
    return {
        "source": str(log_path),
        "status": "fail" if failures or contacts else "unknown",
        "runtime_error_found": runtime_match is not None,
        "first_failure_index": first_failure["index"] if first_failure else -1,
        "failure_preview_count": len(failures),
        "all_colliding_failure_count": sum(1 for failure in failures if failure["all_candidates_colliding"]),
        "max_reported_joint_step_deg": max_joint_step,
        "reported_joints": " ".join(sorted(joint_counter)),
        "contact_count": len(contacts),
        "contact_pairs": [
            {"pair": pair, "count": count}
            for pair, count in contact_pair_counts.most_common()
        ],
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MoveIt2 closed-contour IK failure logs.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_log(args.log_path)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote IK failure summary to {args.out_json}")


if __name__ == "__main__":
    main()
