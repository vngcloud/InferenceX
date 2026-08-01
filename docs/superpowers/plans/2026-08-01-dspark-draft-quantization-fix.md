# DSpark Draft Quantization Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the GLM-5.2 W4AFP8 agentic launcher loads its BF16 DSpark checkpoint without target quantization inheritance and prove non-zero acceptance locally.

**Architecture:** Keep SGLang unchanged and fix only the affected InferenceX launcher. Exercise the real launcher through a temporary repository layout and shell stubs, then validate the production DP4/EP8 CUDA-graph path on H200.

**Tech Stack:** Bash, pytest, Docker, SGLang metrics.

## Global Constraints

- The target keeps `--quantization w4afp8`.
- The draft receives `--speculative-draft-model-quantization unquant`.
- The automated test asserts the generated command, not source text.
- The local acceptance test uses DP4/EP8 and CUDA graph.

---

### Task 1: Protect the generated launcher command

**Files:**
- Create: `utils/test_glm52_dspark_recipe.py`
- Modify: `benchmarks/single_node/agentic/glm5.2dspark_fp4_h200_sglang.sh:27-35`

**Interfaces:**
- Consumes: launcher environment variables and `benchmark_lib.sh` shell functions.
- Produces: a generated `sglang_command.txt` containing separate target and draft quantization arguments.

- [ ] **Step 1: Write the failing test**

Create a temporary `benchmarks/single_node/agentic` layout, copy the real launcher, provide a no-op `benchmark_lib.sh`, a fake `nvidia-smi`, and a temporary draft directory. Run the launcher with `DRAFT_MODEL_PATH` overridden and assert the generated command contains the literal sequence `--speculative-draft-model-quantization unquant` while retaining `--quantization w4afp8`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest utils/test_glm52_dspark_recipe.py -v`

Expected: FAIL because the current launcher overwrites `DRAFT_MODEL_PATH` and does not emit the draft quantization override.

- [ ] **Step 3: Implement the minimal launcher fix**

Use an overridable draft path and extend `SPEC_ARGS`:

```bash
export DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/RedHatAI/GLM-5.2-speculator.dspark}"
SPEC_ARGS=(
  --speculative-algorithm DSPARK
  --speculative-draft-model-path "$DRAFT_MODEL_PATH"
  --speculative-draft-model-quantization unquant
  --speculative-dspark-block-size 4
)
```

- [ ] **Step 4: Run focused and neighboring tests**

Run: `python -m pytest utils/test_glm52_dspark_recipe.py utils/test_benchmark_lib.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit implementation**

Commit the test and launcher with a bilingual conventional commit message.

### Task 2: Verify local acceptance

**Files:**
- No repository changes.

**Interfaces:**
- Consumes: `thangquang0909/sglang:v0.5.16-dspark-v2-g4`, local target and draft checkpoints on `h200`.
- Produces: direct server-log and Prometheus acceptance evidence.

- [ ] **Step 1: Launch production-path server**

Run the W4AFP8 target with DP4/EP8, DP Attention, CUDA graph, gamma 4, and `--speculative-draft-model-quantization unquant`.

- [ ] **Step 2: Send controlled requests**

Send deterministic chat requests at small local concurrency with at least 128 generated tokens.

- [ ] **Step 3: Validate acceptance**

Read `Decode batch` logs and `sglang:spec_accept_rate` / `sglang:spec_accept_length` metrics. Require at least one active DP rank to have accept rate greater than zero and accept length greater than 1.

- [ ] **Step 4: Clean up and verify**

Stop the exact debug container and verify no `dspark-debug-*` container or GPU process remains.

