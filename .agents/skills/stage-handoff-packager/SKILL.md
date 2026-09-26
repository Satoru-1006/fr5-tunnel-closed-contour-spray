---
name: stage-handoff-packager
description: Use when a robot stage is complete or needs a Codex/ChatGPT/human handoff package with evidence manifests, reproducibility commands, protected-state notes, and a verified archive at or below 50 MB.
---

# Stage Handoff Packager

Build a concise, evidence-backed handoff for the named stage. Read existing
`HANDOFF.md`, `AGENT_HANDOFF.md`, `.agent-handoff/`, `AGENTS.md`, reports, and
repository facts before editing. Preserve existing handoff protocol and merge
incrementally; never replace a repository's handoff state with a template.

## Package contract

Include applicable items such as `HANDOFF_README.md`, `STAGE_REPORT.md`,
`RESULT.json`, `metrics.csv`, `commands.txt`, `environment.txt`,
`git_status.txt`, `git_diff.patch`, `manifest.json`, `SHA256SUMS`, configs,
key scripts, important logs, and plots. Include only files relevant to the
stage and keep the default archive at `<= 50 MB`. For excluded large or
disposable files, record path, size, hash when identity matters, and a clear
`excluded_reason` in the manifest.

Validate the package by listing members, extracting to a fresh directory,
comparing expected members and byte sizes, and checking hashes for the
authoritative files. Hash identity-bearing artifacts once; do not turn
temporary-file hashing into the experiment. Redact credentials, unrelated
working-tree data, build/install/log trees, and chat dumps.

State exact `PASS`, `FAIL`, `UNRESOLVED`, `NOT_RUN`, and `UNAVAILABLE` evidence
boundaries. A completed replay is not proof that all points passed. Preserve
canonical input, protected baseline, official limits, robot model, and
historical result semantics; packaging must never mutate them. Pair the
technical skill with `robot-experiment-auditor` and include the top remaining
bottlenecks, coverage gaps, and Stage 4B next actions.
