# InferenceX Glossary

Terms specific to this project. Implementation details belong in CLAUDE.md, not here.

---

## benchmark_client

The tool that generates load against the serving endpoint during a benchmark run. Distinct from the **serving framework** (vLLM, SGLang, etc.), which is the process being measured.

Valid values:
- `inferencex_native` — InferenceX's built-in `benchmark_serving.py` client (default for all existing entries)
- `aiperf` — NVIDIA AIPerf (ai-dynamo/aiperf), invoked via the **AIPerf adapter**

A result tagged `framework=vllm, benchmark_client=aiperf` means: vLLM served the model; AIPerf drove the load.

---

## serving framework

The inference engine that hosts the model and handles requests. Expressed as the `framework` field in config and results. Examples: `vllm`, `sglang`, `trt`, `atom`, `dynamo-sglang`. Never set to the benchmark client name.

---

## AIPerf adapter

The translation layer at `utils/bench_serving/aiperf_adapter.py`. Reads AIPerf's output artifact directory and writes an InferenceX-compatible result JSON that `process_result.py` can consume. Handles two artifact shapes: fixed-concurrency runs and search-recipe runs (extracting best concurrency from `search_history.json`).

---

## fixed-concurrency run

The standard InferenceX benchmark model: one matrix entry = one fixed concurrency point. The matrix loop (driven by `generate_sweep_configs.py`) calls the benchmark script once per concurrency value. Both `inferencex_native` and `aiperf` (v0.9.0) support this model.

---

## search recipe

An AIPerf-internal mode (`--search-recipe max-concurrency-under-sla`) where AIPerf runs its own sweep to find the maximum concurrency that passes a given SLO (e.g. TTFT p95 < 400 ms). The result is a variable best-concurrency, not a fixed point. This mode **does not fit InferenceX's fixed-concurrency matrix model** and is tracked as a follow-up feature — not part of the initial `aiperf` benchmark_client integration.

---

## workload archetypes (MEP-0001)

Three capacity classes used to structure benchmark suites:

- **Ferrari** — interactive latency-sensitive (coding assistant, RAG online QA). SLO: TTFT p95 and ITL p95 tight.
- **Fast Food** — relaxed-latency support/chatbot workload. SLO: TTFT p95 looser (e.g. 2 s). Read throughput and cost more than latency rank.
- **Deep Thinker** — batch or long-generation workload. No TTFT SLO; maximize throughput and tokens/Watt.

---

## agentic replay

A multi-turn coding-agent trace replay driven by AIPerf. Simulates concurrent agent sessions with realistic multi-turn timing and prefix-reuse patterns. Distinct from synthetic single-turn benchmarks.

Two first-class invocation paths, kept for different purposes:

**Mooncake path (the main prod-trace flow):** a `mooncake_trace` custom dataset replayed with multi-turn session support, where the server's own response is threaded back into the next turn's context (FORK-mode DAG replay via `build_assistant_turn`, capturing both text and `tool_calls`). This is what gives realistic KV/prefix-cache reuse across turns. Drives our **own prod-derived traces** (minimax claude-code, gemma chat, gemma RAG), committed in-repo and pointed at via `--input-file`. The team's dataset generators live in `InferenceOptiAIPlan/workload-gen/`. aiperf for this path installs from the **`utils/aiperf-mooncake` submodule** — a GreenNode-owned **clean fork of `v0.9.0`** (no agentx commits), wired via `AIPERF_SOURCE_DIR` into the isolated `ensure_aiperf` venv, so we can patch aiperf internals without touching the weka fork (ADR-0003). Distinct from the `utils/aiperf` submodule below.

**Weka path (retained for SemiAnalysis comparability):** the `inferencex-agentx-mvp` scenario + `semianalysis_cc_traces_weka` **public** corpus (fetched at runtime), installed via `install_agentic_deps` from the `utils/aiperf` submodule (`vngcloud/aiperf` @ `cjq/weka-live-assistant-responses`). A preset bundle (locked params + SemiAnalysis corpus). **Not deprecated** — kept precisely for direct numeric comparability with SemiAnalysis's published agentic leaderboard. The mooncake path does not replace it; the two answer different questions.

---

## inferencex_native

The default value of `benchmark_client`. Refers to InferenceX's own `utils/bench_serving/benchmark_serving.py`. All existing config entries implicitly use this client even though the field was not present before the AIPerf integration.
