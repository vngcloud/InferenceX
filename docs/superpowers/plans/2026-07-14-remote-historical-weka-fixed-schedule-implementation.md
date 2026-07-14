# Remote Historical Weka Fixed-Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a scenario-free AIPerf fixed-schedule path for the stable historical Weka corpus, while preserving existing agentic-replay and CCU behavior apart from the approved removal of their hard-coded failed-request threshold.

**Architecture:** Add two opt-in matrix fields, fixed-schedule and max-context-length, and branch only the replay command builder when fixed scheduling is selected. Fix AIPerf fixed scheduling so only root conversations are scheduled directly; child conversations continue to be spawned by the Weka branch orchestrator. Existing scenario-driven agentic replay remains the default path.

**Tech Stack:** Bash, Python/Pydantic, GitHub Actions YAML, pytest, local AIPerf source, Git submodule.

**Safety boundaries:**

- Work on dev-remote-bmk; do not open a PR.
- Do not run a local or remote benchmark.
- Do not dispatch GitHub Actions or change perf-changelog.yaml.
- Preserve unrelated dirty files and existing dataset/spec edits.
- Commit implementation locally; do not push the InferenceX branch unless requested.

---

### Task 1: Make AIPerf fixed scheduling root-only

**Files:**

- Modify: /Users/lap15120/greennode-code/aiperf/src/aiperf/timing/strategies/fixed_schedule.py
- Test: /Users/lap15120/greennode-code/aiperf/tests/unit/timing/strategies/test_fixed_schedule.py
- Update later: utils/aiperf-mooncake submodule pointer

**Step 1: Confirm impact and conventions**

Run GitNexus upstream impact for FixedScheduleStrategy.setup_phase before editing. The known risk is MEDIUM and its direct callers are timing tests; stop and warn if a fresh result is HIGH or CRITICAL.

Inspect the checked-out AIPerf tests and Weka metadata types. Local code is authoritative; do not use Context7.

**Step 2: Write the failing regression test**

Add one root conversation with turns and one explicit child with is_root=False. Set up FixedScheduleStrategy and assert that the initial schedule contains only the root.

Run:

    cd /Users/lap15120/greennode-code/aiperf
    uv run pytest tests/unit/timing/strategies/test_fixed_schedule.py -q

Expected: the new root-only assertion fails because both entries are currently scheduled.

**Step 3: Implement the minimum fix**

In FixedScheduleStrategy.setup_phase, skip non-root metadata entries before checking turns:

    for conversation in self._conversation_source.dataset_metadata.conversations:
        if not conversation.is_root:
            continue
        if not conversation.turns:
            continue

Do not change timing, grace period, branch spawning, or request pacing.

**Step 4: Verify AIPerf**

Run:

    uv run pytest tests/unit/timing/strategies/test_fixed_schedule.py -q
    uv run pytest tests/unit/timing -q

Expected: all tests pass.

**Step 5: Commit and publish the dependency**

Commit in the standalone AIPerf repository:

    git add src/aiperf/timing/strategies/fixed_schedule.py tests/unit/timing/strategies/test_fixed_schedule.py
    git commit -m "fix: schedule only root conversations"
    git push origin benchtool/agentx-weka

The commit must be remotely reachable before InferenceX references it. This publishes a dependency; it is not an InferenceX PR or benchmark.

---

### Task 2: Add fixed-schedule matrix fields end to end

**Files:**

- Modify: utils/matrix_logic/validation.py
- Modify: utils/matrix_logic/generate_sweep_configs.py
- Test: utils/matrix_logic/test_validation.py
- Test: utils/matrix_logic/test_generate_sweep_configs.py
- Modify: .github/workflows/e2e-tests.yml
- Modify: .github/workflows/run-sweep.yml
- Modify: .github/workflows/templates/test-replay-client.yml

**Step 1: Confirm symbol impact**

Run GitNexus upstream impact for AgenticReplayConfig and any generator function being edited. Report direct callers and affected flows; stop if risk is HIGH or CRITICAL.

**Step 2: Add failing schema tests**

Test that fixed-schedule defaults false, max-context-length defaults None, a positive value serializes with its kebab-case alias, zero and negatives fail, and unknown fields remain rejected.

Run:

    uv run pytest utils/matrix_logic/test_validation.py -q

Expected: new fields are rejected before implementation.

**Step 3: Add failing generator tests**

Add fixed-schedule: true and max-context-length: 100000 to an agentic replay fixture. Assert both reach the generated entry. Also prove an existing AgentX fixture is unchanged when the fields are absent.

Run:

    uv run pytest utils/matrix_logic/test_generate_sweep_configs.py -q

Expected: generated output lacks the new values.

**Step 4: Implement validation and generation**

Add enum values FIXED_SCHEDULE and MAX_CONTEXT_LENGTH. Add these fields to AgenticReplayConfig and SingleNodeAgenticReplayMatrixEntry:

    fixed_schedule: bool = Field(default=False, alias=Fields.FIXED_SCHEDULE.value)
    max_context_length: Optional[int] = Field(
        default=None,
        ge=1,
        alias=Fields.MAX_CONTEXT_LENGTH.value,
    )

Read and emit both fields in both existing agentic-replay generation paths. Do not refactor their duplication.

**Step 5: Wire workflow inputs**

Add an optional boolean fixed-schedule input defaulting false and an optional string max-context-length input defaulting empty. Export:

    FIXED_SCHEDULE: ${{ inputs.fixed-schedule && 'true' || 'false' }}
    MAX_CONTEXT_LENGTH: ${{ inputs.max-context-length }}

Forward both fields from e2e-tests.yml and run-sweep.yml. While touching run-sweep.yml, forward its already-supported remote replay fields to match the e2e path: remote URL, remote API-key secret, tokenizer, tokenizer trust, and dataset-entry limit. Do not change triggers or job selection.

**Step 6: Verify matrix logic**

Run:

    uv run pytest utils/matrix_logic/ -q

Expected: all matrix tests pass.

**Step 7: Commit**

Run GitNexus staged change detection first, then commit only these files:

    git add utils/matrix_logic/validation.py utils/matrix_logic/generate_sweep_configs.py
    git add utils/matrix_logic/test_validation.py utils/matrix_logic/test_generate_sweep_configs.py
    git add .github/workflows/e2e-tests.yml .github/workflows/run-sweep.yml
    git add .github/workflows/templates/test-replay-client.yml
    git commit -m "feat: add fixed schedule replay config"

---

### Task 3: Build the scenario-free fixed-schedule command

**Files:**

- Modify: benchmarks/benchmark_lib.sh
- Modify: benchmarks/single_node/agentic/_remote_replay.sh
- Test: use the existing benchmark-library shell test location discovered during implementation

**Step 1: Confirm impact and test harness**

Use GitNexus where indexed, then rg for Bash call sites. Confirm build_replay_cmd is shared and the new selector remains default-off. Inspect the current shell test convention before choosing a test path.

**Step 2: Add failing command-construction tests**

For FIXED_SCHEDULE=true, require aiperf profile, remote chat endpoint, local Weka input, weka_trace type, fixed schedule, duration, ignore_eos:true, seed, slice duration, tokenizer flags, output directory, and conditional max context.

Require absence of scenario, think time, grace override, concurrency, server token count, trajectory flags, and failed-request threshold.

For the default path, prove scenario-driven agentic replay remains selected, except that its hard-coded failed-request threshold is absent.

Run the focused test and confirm it fails before implementation.

**Step 3: Add the fixed builder branch**

At the top of build_replay_cmd, branch on FIXED_SCHEDULE=true and construct only the approved scenario-free command. Add MAX_CONTEXT_LENGTH only when non-empty. Omit the grace flag so AIPerf uses its finite default.

Do not pass CONC; it stays only as a matrix identity value for current InferenceX plumbing.

**Step 4: Remove hard-coded thresholds**

Remove the hard-coded failed-request threshold from the scenario replay command and local Weka replay command. Keep the generic optional threshold support in run_aiperf_benchmark.

Delete report_failed_request_abort and its sole call from _remote_replay.sh.

**Step 5: Mount local input**

When INPUT_FILE is set, add a read-only bind mount from that path to the identical nested-container path. Keep result and Hugging Face cache mounts unchanged.

**Step 6: Verify without traffic**

Run:

    bash -n benchmarks/benchmark_lib.sh
    bash -n benchmarks/single_node/agentic/_remote_replay.sh

Run only the focused command-construction test. Source the library and inspect the produced argument array if no harness exists; never execute it.

**Step 7: Commit**

Run staged GitNexus change detection, then stage only the two scripts and actual test file:

    git commit -m "feat: build fixed schedule replay command"

---

### Task 4: Register main and smoke configurations

**Files:**

- Modify: .github/configs/nvidia-master.yaml
- Modify: benchmarks/single_node/agentic/datasets/README.md
- Modify: benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/README.md
- Modify: utils/aiperf-mooncake submodule pointer

**Step 1: Update the submodule**

Fetch the Task 1 commit in utils/aiperf-mooncake and check out that exact commit. Verify only the intended pointer changes.

**Step 2: Add two additive entries**

Add glm5-2-greennode-historical-fixed-remote and glm5-2-greennode-historical-fixed-remote-smoke.

Both use the existing GreenNode remote URL/secret, model z-ai/glm-5.2, tokenizer zai-org/GLM-5.2 with trust enabled, the checked-in Weka sessions directory, weka_trace, fixed-schedule true, max model length 131072, tp1, ep1, and concurrency-list [13] for matrix compatibility only.

Main: duration 3000 and no max-context cap.

Smoke: duration 60 and max-context-length 100000.

Do not alter glm5-2-greennode-bench-client-remote; it remains the legacy scenario/CCU entry.

**Step 3: Document the canonical command**

Document the exact scenario-free command shape and state that only roots are initially scheduled, Weka children spawn through branch metadata, no think-time/grace override/server token count is used, ignore_eos:true is supplied, and main/smoke settings differ as above.

Retain the lineage counts and hashes. Mark the command as not yet executed.

**Step 4: Validate configuration locally**

Generate only the two new entries with --no-evals and inspect output. Config generation must stop before any launcher or workflow. Verify both names, field propagation, and legacy AgentX validation.

Run the matrix suite again.

**Step 5: Commit**

Run staged GitNexus detection, then:

    git add .github/configs/nvidia-master.yaml
    git add benchmarks/single_node/agentic/datasets/README.md
    git add benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/README.md
    git add utils/aiperf-mooncake
    git commit -m "feat: register historical fixed replay configs"

Do not stage perf-changelog.yaml.

---

### Task 5: Final non-benchmark verification and handoff

**Step 1: Run all permitted checks**

Run:

    uv run pytest utils/matrix_logic/ -q
    bash -n benchmarks/benchmark_lib.sh
    bash -n benchmarks/single_node/agentic/_remote_replay.sh

Also run the focused builder tests and AIPerf timing tests from earlier tasks.

Never run aiperf profile, the replay launcher, Docker replay, workflow dispatch, curl against the endpoint, or any command sending inference traffic.

**Step 2: Audit scope**

Run GitNexus staged detection before a final commit and inspect:

    git status --short
    git diff --stat
    git log --oneline --decorate -8

Confirm unrelated dirty files remain untouched, perf-changelog.yaml is unchanged, the legacy CCU config remains, agentic replay still defaults to its prior strategy, only the approved hard-coded threshold behavior changed, and no benchmark artifacts exist.

**Step 3: Report and stop**

Report commit hashes on dev-remote-bmk, the AIPerf dependency commit/push, verification results, confirmation that no benchmark ran and no PR opened, and the future main/smoke config names marked not executed.

Do not push the InferenceX branch or start a benchmark without a new instruction.
