# Repository execution rules

- The authoritative Stage 0/1 input is the 181-point open-arch pair under `outputs/internal_wiper_moveit_inputs/`.
- Do not mix the legacy 720-point outputs or the legacy spray-off/reorientation reports into the Stage 0/1 graph.
- Stage 0/1 is ON-state open-arch planning only. Do not add OFF states, reorientation edges, GNN, PPO, LSTM, Transformer, retreat, approach, or closed-contour transitions.
- Collision results must be labelled `adaptive_discrete_interpolation`; they are not strict continuous collision detection.
- Bullet CCD and clearance are unavailable unless a real backend reports them. Use `not_available` and JSON nulls; never infer clearance from collision-free states.
- A result is valid only when MoveIt2 PlanningScene, FK, dynamics, and post-Ruckig checks have actually run. Do not create placeholder pass outputs.
- Preserve unrelated user changes. Never run `git reset`, `git clean`, `git checkout .`, or `git restore .`.

# D46 — AGGRESSIVE MEASUREMENT, CONTROLLED EXPOSURE & NON-REGRESSION REPAIR DOCTRINE

Add the following doctrine as an authoritative execution rule for D46.

---

## 1. CORE EXECUTION PERSONALITY

D46 must not become a defensive, hash-heavy, provenance-heavy ceremonial audit.

The required execution personality is:

> **Aggressive measurement, controlled exposure, conservative acceptance.**

Or more simply:

> **大胆测，大胆压，大胆把问题逼出来；问题关在可控盒子里，正确成果绝不被污染。**

The purpose of D46 is to discover the truth of the frozen Stage 3 robot system as efficiently and thoroughly as possible.

Therefore prefer:

* real execution;
* large-scale measurement;
* adversarial cases;
* boundary cases;
* stress cases;
* perturbation tests;
* direct reproduction of suspicious behavior;
* fast root-cause isolation;
* targeted instrumentation;
* independent cross-checks;
* aggressive debugging of measurement infrastructure;
* fast regression after repair;
* continued forward progress.

Avoid spending major execution time on:

* repetitive SHA-256 calculations;
* recursively hashing disposable files;
* redundant manifest regeneration;
* excessive provenance probes;
* duplicating the same evidence many times;
* defensive wrappers added without a demonstrated need;
* stopping execution after every small uncertainty;
* long read-only audits before attempting actual execution;
* creating paperwork merely to prove that paperwork exists.

Hashing is allowed where identity matters.

It must remain:

> **a seal, not the steering wheel.**

---

# 2. THE CONTROLLED-BOX PRINCIPLE

Treat D46 execution as if all uncertainty, failures, experiments and diagnostic work occur inside a controlled box.

Outside the box is:

# `STAGE3_FINAL_RELEASE`

It is frozen.

It is protected.

It is read-only scientific state.

Inside the box are:

* measurement tooling;
* benchmark harnesses;
* instrumentation;
* diagnostic scripts;
* synthetic tests;
* stress tests;
* adversarial cases;
* temporary patches;
* measurement repair branches;
* alternative implementations used for cross-checking;
* failed debugging attempts;
* experimental measurement code.

Inside this box, Codex may be aggressive.

It may:

* break experimental tooling;
* rewrite a suspicious measurement function;
* add instrumentation;
* replace an unreliable evaluator;
* try multiple algorithms;
* compare independent implementations;
* add synthetic cases;
* construct adversarial inputs;
* create temporary diagnostic branches;
* discard failed patches;
* combine several debugging techniques.

Failure inside the box is acceptable.

Pollution of the frozen Stage 3 scientific baseline is not.

The governing architecture is:

`FROZEN_STAGE3_RELEASE`

↓

`CONTROLLED_MEASUREMENT_SANDBOX`

↓

`EXPOSE`

↓

`DIAGNOSE`

↓

`REPAIR MEASUREMENT DEFECT IF NEEDED`

↓

`REGRESSION`

↓

`ACCEPT OR DISCARD`

↓

`CONTINUE MEASURING THE SAME FROZEN SYSTEM`

---

# 3. TWO FUNDAMENTALLY DIFFERENT TYPES OF PROBLEMS

Every discovered problem must first be classified.

This distinction is mandatory.

---

## TYPE A — MEASUREMENT / INFRASTRUCTURE DEFECT

Examples include:

* benchmark generator bug;
* wrong metric implementation;
* incorrect unit conversion;
* wrong joint ordering;
* incorrect frame interpretation;
* incorrect timestamp handling;
* numerical differentiation defect;
* incorrect jerk calculation;
* incorrect collision API usage;
* incorrect self-collision interpretation;
* broken clearance computation;
* incorrect singularity computation;
* Ruckig API misuse;
* dynamics evaluator bug;
* wrong torque normalization;
* stale benchmark state;
* nondeterministic harness bug;
* incorrect seed handling;
* corrupted benchmark metadata;
* incorrectly loaded limits;
* malformed test configuration;
* measurement pipeline crash;
* false positive;
* false negative;
* measurement result that cannot be trusted because the measurement implementation itself is wrong.

These defects belong to D46 itself.

### THEY MUST BE REPAIRED.

Do not simply report:

`MEASUREMENT DEFECT FOUND`

and stop if the defect is technically actionable.

Instead:

> reproduce → isolate → understand → repair → regression-test → rerun affected measurements → continue.

The objective is to finish D46 with a trustworthy measurement system.

---

## TYPE B — REAL FROZEN ROBOT-SYSTEM WEAKNESS

Examples include:

* actual collision;
* actual self-collision;
* low environment clearance;
* low self-clearance;
* jerk violation;
* acceleration violation;
* velocity violation;
* joint-limit violation;
* continuity problem produced by the frozen system;
* poor terminal accuracy;
* high TCP tracking error;
* singularity-risk configuration;
* excessive model-based required torque;
* poor robustness;
* Ruckig-invalid trajectory genuinely produced by the frozen system;
* post-Ruckig degradation;
* difficult-case failure;
* reproducible bad behavior of the frozen Stage 3 system.

These are not D46 tooling defects.

They are **scientific baseline findings**.

### DO NOT REPAIR THEM IN D46.

Expose them.

Reproduce them.

Measure them accurately.

Quantify their severity.

Classify them.

Determine affected benchmark families.

Record representative worst cases.

Include them in:

`STAGE4_FAILURE_TAXONOMY_V1`

and:

`STAGE4_RISK_RANKED_BOTTLENECKS_V1`

Then carry them into Stage 4B.

The governing rule is:

> **Measurement defects are repaired in Stage 4A.
> Robot-system weaknesses are measured in Stage 4A and repaired in Stage 4B.**

Do not confuse the two.

---

# 4. DO NOT OPTIMIZE AWAY BAD BASELINE RESULTS

This is an absolute rule.

If the frozen Stage 3 system produces:

* a collision;
* poor clearance;
* bad jerk;
* excessive acceleration;
* poor TCP accuracy;
* singularity risk;
* high torque;
* robustness failure;

do not respond by tuning the robot until the case passes.

Do not:

* change Stage 3 weights;
* modify canonical parameters;
* change the checkpoint;
* tune Ruckig limits or settings to improve the score;
* add smoothing to hide the failure;
* change trajectory parameters;
* alter collision thresholds;
* remove the case;
* reduce perturbation magnitude;
* relax the acceptance rule;
* modify difficult scenarios;
* silently replace the trajectory.

That would destroy the meaning of:

`STAGE4_SYSTEM_BENCHMARK_V1`.

Stage 4A must answer:

> **What does the frozen system actually do?**

not:

> **What can we quickly change so that it looks better?**

---

# 5. AGGRESSIVELY REPAIR MEASUREMENT PROBLEMS

Although robot-system weaknesses must remain untouched in D46, measurement-tool defects must not be treated passively.

If the benchmark itself is wrong, attack the problem.

Do not stop because:

* the first patch failed;
* one API is inconvenient;
* one implementation is numerically unstable;
* a historical script no longer works;
* a metric is ambiguous;
* an external library behaves differently than expected.

Use multiple approaches where useful.

Possible methods include:

* independent implementation;
* differential testing;
* synthetic known-answer cases;
* property-based testing;
* metamorphic tests;
* numerical sensitivity analysis;
* A/B implementation comparison;
* local library-source inspection;
* authoritative documentation comparison;
* instrumentation;
* state tracing;
* minimal reproducer extraction;
* targeted fuzzing;
* cross-library verification;
* analytical sanity checks.

The goal is not merely:

`ISSUE IDENTIFIED`.

The goal is:

`ISSUE RESOLVED AND MEASUREMENT TRUST RESTORED`.

---

# 6. REPAIR UNTIL THE MEASUREMENT CHAIN WORKS

Do not abandon D46 at the first actionable measurement blocker.

If an actionable measurement problem remains, keep working.

For example:

Attempt A fails.

Try B.

If B reveals a deeper root cause, attack that.

If necessary combine A+B+C.

Use bounded parallel investigations where useful.

Do not confuse:

> “the first method failed”

with:

> “the problem cannot be solved.”

D46 should make a serious best effort to complete the measurement system rather than simply catalogue implementation difficulties.

A material unresolved measurement defect is a legitimate blocker.

A failed experimental repair branch is not.

---

# 7. REPAIR LOOP

For every Type-A measurement defect:

## STEP 1 — Reproduce

Create the smallest reliable reproducer.

Do not repair an unconfirmed suspicion.

---

## STEP 2 — Root cause

Identify the underlying cause.

Avoid downstream cosmetic patches when the upstream defect can be identified.

---

## STEP 3 — Blast radius

Determine which measurement results might have been affected.

Do not assume only the visible metric was affected.

---

## STEP 4 — Experimental repair

Inside the controlled box, try the strongest practical repair.

Aggressive experimentation is allowed.

The frozen Stage 3 canonical remains untouched.

---

## STEP 5 — Known-answer verification

Where possible, prove the repaired implementation using a known-answer or deliberately constructed case.

Examples:

* a forced collision must be detected;
* a known non-collision must not become a false positive;
* an intentionally excessive velocity must trigger the auditor;
* an intentionally excessive jerk must trigger the auditor;
* a known finite valid state must remain finite;
* known invalid Ruckig input must fail;
* a synthetic clearance change should move the reported distance in the expected direction.

---

## STEP 6 — Targeted regression

Re-test:

* the original failure;
* the modified component;
* direct dependencies;
* nearby critical behavior.

---

## STEP 7 — Protected measurement regression

Confirm that previously correct measurement capabilities did not regress.

---

## STEP 8 — Rerun affected benchmark measurements

Any result produced by the defective implementation must be considered suspect until recomputed.

---

## STEP 9 — Accept only if non-regressive

Promote the measurement repair only if:

`DEFECT_FIXED = YES`

and:

`PREVIOUSLY_CORRECT_MEASUREMENT_REGRESSED = NO`

Otherwise discard the repair and try another method.

---

# 8. ROLLING-SNOWBALL RULE FOR MEASUREMENT TOOLING

D46 measurement infrastructure should improve monotonically.

Conceptually:

`MEASUREMENT_BASELINE_0`

↓

`FIX_1`

↓

`MEASUREMENT_BASELINE_1`

↓

`FIX_2`

↓

`MEASUREMENT_BASELINE_2`

↓

...

↓

`FINAL_STAGE4A_MEASUREMENT_PIPELINE`

Once a measurement capability has been demonstrated correct, later accepted repairs must preserve it.

If the measurement pipeline is at an effective quality of:

`80`

and a repair raises it to:

`81`

then 81 becomes the new floor.

Future accepted repairs must not return it to 80.

Experimental branches may regress arbitrarily.

Accepted tooling may not.

This is:

> **Aggressive exploration, monotonic acceptance.**

---

# 9. PROTECT SCIENTIFIC STATE, NOT EVERY TEMPORARY FILE

Do not confuse scientific protection with filesystem paranoia.

The things that require strong protection are:

* Stage 3 Final Release;
* canonical checkpoint;
* canonical configuration;
* benchmark definition once frozen;
* benchmark acceptance set once frozen;
* authoritative final measurement results.

Temporary measurement scripts do not need ceremonial protection.

Temporary logs do not need repeated hashing.

Temporary failed repair branches do not need permanent preservation unless they contain important diagnostic evidence.

Protect what matters.

Experiment freely with everything else.

---

# 10. MINIMAL HASH / PROVENANCE POLICY

D46 must not waste substantial execution time on cryptographic bookkeeping.

Use existing Stage 3 release identity evidence.

Perform only the identity checks genuinely necessary to confirm:

* correct Stage 3 release;
* correct canonical checkpoint;
* correct benchmark version;
* final authoritative D46 release artifacts where needed.

Do not:

* SHA-256 every generated result;
* hash every benchmark case;
* hash every temporary repair script;
* repeatedly recompute identical checkpoint hashes;
* build giant recursive hash inventories;
* stop execution to perform redundant integrity probes.

A useful principle is:

> **One trustworthy identity check is better than twenty redundant integrity rituals.**

Spend compute budget on:

* measurement;
* adversarial testing;
* statistics;
* difficult cases;
* bug discovery;
* root-cause analysis;
* reproducibility.

Not ceremony.

---

# 11. BE AGGRESSIVE IN EXPOSING THE FROZEN SYSTEM

Do not protect Stage 3 from bad news.

Protect it from mutation.

Those are different concepts.

D46 should deliberately try to expose weaknesses through:

* near-limit states;
* near-singular states;
* obstacle-near states;
* narrow-clearance transitions;
* high-curvature trajectories;
* long trajectories;
* short trajectories;
* difficult orientation changes;
* combined translation and orientation;
* perturbations;
* clean restarts;
* repeated deterministic runs;
* post-Ruckig auditing;
* model-based dynamics checks;
* collision-sensitive transitions.

If a real weakness exists, finding it is a D46 success.

Do not design the benchmark merely to prove Stage 3 is good.

---

# 12. BAD RESULT ≠ BAD D46

This principle is mandatory.

D46 may return:

`TASK_STATUS = PASS`

while also finding:

`COLLISION_FAILURES > 0`

or:

`JERK_VIOLATIONS > 0`

or:

`LOW_CLEARANCE_CASES > 0`

or:

`ROBUSTNESS_FAILURES > 0`.

This means the benchmark successfully revealed a real weakness.

The distinction must remain:

`MEASUREMENT_PIPELINE_STATUS`

versus:

`FROZEN_ROBOT_BASELINE_PERFORMANCE_STATUS`.

Never artificially improve the second merely to make the first look successful.

---

# 13. THE BOX MUST CONTAIN FAILURE — NOT HIDE IT

“Controlled box” does not mean suppressing errors.

It means:

> failure is allowed to appear without being allowed to corrupt authoritative scientific state.

Inside the box:

* reproduce the issue;
* isolate it;
* instrument it;
* stress it;
* understand it.

Then:

### If it is a tooling defect:

repair it.

### If it is a real robot weakness:

freeze the evidence and carry it forward.

This is controlled exposure.

Not defensive suppression.

---

# 14. NO SILENT WORKAROUNDS

Do not “fix” a problem by avoiding the code path that exposes it.

Examples of unacceptable behavior:

* removing a difficult benchmark family;
* skipping failed cases;
* silently reducing trajectory density;
* disabling CCD because it finds failures;
* disabling self-collision checks;
* loosening jerk thresholds;
* switching off strict Ruckig validation;
* suppressing singularity metrics;
* treating missing torque limits as automatically safe;
* replacing NaN with zero;
* clipping bad metrics to look valid;
* dropping outliers because they hurt statistics.

A detected problem must remain visible until correctly classified and handled.

---

# 15. DISTINGUISH UNCERTAINTY FROM FAILURE

If a capability cannot be scientifically evaluated because required information is absent, report:

`UNAVAILABLE`

or:

`UNVERIFIED`

or:

`UNRESOLVED_THRESHOLD`

rather than:

`PASS`.

Examples:

* unknown jerk-limit provenance;
* missing torque limit;
* unavailable continuous self-collision checker;
* unavailable calibrated TCP uncertainty;
* insufficient dynamic model fidelity.

Do not turn missing knowledge into a false success.

But also do not stop the entire benchmark unnecessarily.

Measure everything that can be measured correctly.

---

# 16. OPTIMAL ORDER OF ATTACK

When multiple measurement problems appear, solve them in dependency order.

Use a straight highway, not a winding route.

Recommended priority:

## FIRST — foundational correctness

* robot identity;
* checkpoint identity;
* units;
* joint ordering;
* frames;
* timebase;
* benchmark case identity.

↓

## SECOND — raw trajectory/state correctness

* q;
* v;
* a;
* timestamps;
* interpolation;
* finite state.

↓

## THIRD — kinematic measurements

* velocity;
* acceleration;
* jerk;
* limits;
* continuity.

↓

## FOURTH — geometric measurements

* FK;
* collision;
* self-collision;
* CCD;
* clearance.

↓

## FIFTH — advanced system measurements

* singularity;
* dynamics;
* torque;
* Ruckig post-processing;
* robustness.

↓

## SIXTH — statistics / scorecard / ranking

Do not debug percentile tables before confirming that the underlying physical measurement is correct.

Do not debug torque normalization while joint ordering is still uncertain.

Solve the highest upstream uncertainty first.

Then roll forward.

---

# 17. EXPOSURE-FIRST EXECUTION

After minimum baseline authentication, execute the benchmark pipeline early.

Do not spend a long period trying to predict every possible problem before running.

Prefer:

> **run → expose → isolate → repair tooling if needed → rerun → continue**

rather than:

> inspect → inspect → hash → inspect → write manifest → inspect → finally run.

The fastest way to discover the real failure frontier is often actual execution.

---

# 18. EVIDENCE SHOULD SUPPORT DECISIONS

Preserve evidence that answers:

* What failed?
* Is the failure real?
* Is it a measurement defect or robot weakness?
* What caused it?
* Was a repair accepted?
* Did the repair regress another correct measurement?
* Which benchmark result changed after recomputation?
* What remains for Stage 4B?

Do not preserve evidence merely because it exists.

Prefer a concise defect ledger and final trustworthy results over huge raw logs.

---

# 19. FINAL TYPE-A DEFECT REQUIREMENT

D46 should not finish with an unresolved material Type-A measurement defect if the defect remains technically actionable inside the available workspace.

Continue working until either:

### A.

the measurement defect is fixed and affected measurements are recomputed;

or:

### B.

strong evidence establishes that resolution is impossible with the available software/environment, in which case report the exact limitation and mark the affected measurement domain:

`UNAVAILABLE / BLOCKED`

Do not prematurely stop after the first failed repair attempt.

---

# 20. FINAL TYPE-B WEAKNESS REQUIREMENT

D46 must **not** eliminate Type-B robot weaknesses.

Instead, every significant Type-B finding must be transformed into a Stage 4B-ready problem definition containing:

* problem category;
* affected benchmark family;
* affected case IDs;
* frequency;
* worst magnitude;
* severity;
* reproducibility;
* safety impact;
* current evidence;
* likely subsystem;
* measurement confidence.

This turns:

> “something seems wrong”

into:

> “Stage 4B has a precise target.”

---

# 21. STAGE 4A → STAGE 4B CONTRACT

The best D46 result is not:

> “everything passed.”

The best D46 result is:

> **“we now know exactly what is correct, exactly what is weak, exactly what we can measure reliably, and exactly what Stage 4B must attack first.”**

Stage 4B should receive:

`TOP_1_BOTTLENECK`

`TOP_2_BOTTLENECK`

`TOP_3_BOTTLENECK`

plus:

* exact baseline metrics;
* worst cases;
* regression barrier;
* acceptance set;
* measurement pipeline;
* coverage gaps;
* unresolved thresholds.

D46 must stop before optimizing these weaknesses.

---

# 22. FINAL OPERATING DOCTRINE

The final D46 doctrine is:

> **Do not be afraid of problems. Hunt them.**

> **Do not let discovered problems mutate the frozen Stage 3 system. Contain them.**

> **If the measuring instrument is wrong, repair the instrument aggressively.**

> **If the robot itself is weak, measure the weakness honestly and hand it to Stage 4B.**

> **Do not hide failure.**

> **Do not optimize the baseline.**

> **Do not let temporary experiments contaminate authoritative results.**

> **Do not waste the campaign on hashes, redundant manifests or defensive ceremony.**

> **Spend effort on execution, measurement, adversarial testing, debugging and scientific truth.**

The shortest summary is:

# **大胆把问题逼出来，把问题关在盒子里。**

# **测量工具有错，当场狠狠干净。**

# **机器人本身有缺陷，当场测清楚，但绝不在 D46 偷偷优化掉。**

# **好的 Stage 3 成果永远留在盒子外面，不被污染、不被破坏。**

# **Measure first. Repair later.**

D46 ends only when the measurement system itself is trustworthy enough to tell Stage 4B exactly what must be improved next.

# Codex Skills Infrastructure — Project Constitution and Routing

This repository uses the Codex repo-local Skill discovery location
`.agents/skills`. Global upstream Skills are installed under
`C:\Users\86198\.codex\skills` when the active Codex installation resolves
the default user home there. Skill discovery is an aid to routing, not evidence
that a robot experiment ran.

## Scientific protection rules

- `NO_FAKE_PASS`: never promote a proxy, replay completion, valid BRep, solver
  row, or partial result into hardware, safety, compatibility, or task proof.
- `SHADOW_FIRST`: test new tooling and candidate methods in an isolated
  diagnostic/shadow area before any promotion decision.
- `UNRESOLVED != PASS` and `NOT_RUN != PASS`; unavailable CCD, clearance,
  torque limits, calibrated TCP, hardware, or other evidence stays explicitly
  unavailable or unverified.
- `NO_SILENT_REGRESSION`: compare the protected valid set, metrics, and
  authoritative artifacts after any tooling change; discard non-regressive
  status only when the evidence supports it.
- `PROTECTED_BASELINE` and `CANONICAL_PROTECTION`: do not mutate frozen
  canonical inputs, protected floors, formal robot models, waypoint sets,
  official results, or experiment metrics merely to improve a score.
- `OFFICIAL_JOINT_LIMITS_IMMUTABLE_WITHOUT_AUTHORIZATION`: joint limits are
  read-only unless the user separately authorizes a formal model change.
- `REPRODUCIBLE_EXPERIMENTS`: record stage, source, seed/config, input,
  candidate, baseline, backend, and output identity before comparison.
- `NO_LARGE_C_DRIVE_WRITES`: avoid build, dataset, model, and cache growth on
  C:. Use bounded project or scratch locations and report any large file.
- Do not execute an external Skill's install, hook, server, benchmark, or
  network script unless the task explicitly requires it and the source has
  been audited. Installing a Skill does not authorize installing its runtime
  dependencies or running its examples.
- `robot-collision-safety` must preserve the distinction between
  `adaptive_discrete_interpolation` and strict continuous collision detection.
  Collision-free samples do not prove continuous safety or clearance.

## Skill routing

- IK, FK, waypoint feasibility, exact roots, roll, base/layout probes,
  240-point validity, joint margins, or old-valid retention ->
  `robot-state-feasibility`.
- PlanningScene, self/environment/tool/ground collision, ACM, FCL, CCD,
  clearance, collision meshes, or fail-closed safety ->
  `robot-collision-safety`.
- IK branch, joint jump, discontinuity, multi-solution path, graph selection,
  or `WP66 -> WP67` continuity -> `trajectory-continuity-optimizer`.
- Velocity, acceleration, jerk, time parameterization, Ruckig, synchronization,
  or post-Ruckig validation -> `trajectory-dynamics-ruckig`.
- Baseline, candidate, champion, ablation, reproducibility, metric routing, or
  honest PASS/FAIL/UNRESOLVED -> `robot-experiment-auditor`.
- Stage completion, handoff, manifest, redaction, archive, or `<=50 MB` package
  -> `stage-handoff-packager`.
- ROS2 graph, nodes, topics, services, actions, TF, QoS, launch, or colcon ->
  upstream `ros2`.
- Modern Gazebo/gz, SDF, ros_gz bridge, sensors, worlds, or headless simulation
  -> upstream `gazebo`; pair with `ros2` and `testing` as needed.
- ROS/node/launch/simulation smoke and deterministic test evidence -> upstream
  `testing`.
- RViz2 RobotModel, TF, Marker, Path, or topic visualization -> upstream `rviz2`.
- URDF links/joints/limits/inertials/mesh/frame validation -> upstream `urdf`.
- MoveIt2 semantic groups, end effectors, virtual joints, group states, or
  disabled-collision review -> upstream `srdf`.
- Durable Codex/Claude handoff protocol -> upstream `agent-handoff`; preserve
  this repository's existing `HANDOFF.md` and handoff state.
- Skill/plugin static analysis, trigger analysis, token budget, or starter
  benchmark -> upstream `plugin-eval` when its runtime is available.

## Cross-Skill closures

- Feasibility closure: `robot-state-feasibility` + `robot-collision-safety` +
  `robot-experiment-auditor`.
- Continuity closure: `robot-state-feasibility` +
  `trajectory-continuity-optimizer` + `robot-collision-safety` +
  `robot-experiment-auditor`.
- Dynamic closure: `trajectory-continuity-optimizer` +
  `trajectory-dynamics-ruckig` + `robot-collision-safety` +
  `robot-experiment-auditor`.
- Simulation later: `ros2` + `gazebo` + `rviz2` + `testing` plus the relevant
  project validation Skill. This route does not authorize a full simulation in
  a documentation/discovery task.
- Stage closure: the relevant technical Skill + `robot-experiment-auditor` +
  `stage-handoff-packager`.

## External Skill boundary

Upstream Skills provide reusable ROS2, Gazebo, RViz2, testing, MoveIt/IK,
URDF/SRDF, handoff, and evaluation knowledge. They do not override this file,
the frozen Stage 3/D46 scientific boundaries, or the project's canonical
inputs. The project Skills contain only project-specific SOPs, metrics,
protection rules, and routing. Future `paper-evidence-builder`,
`research-ablation-benchmark`, and `robot-literature-scout` are reference-only
until separately reviewed and authorized; do not install them implicitly.
