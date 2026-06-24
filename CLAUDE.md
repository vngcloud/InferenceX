# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Read [`AGENTS.md`](AGENTS.md) first** — it is the canonical reference for architecture, conventions, common tasks, and CI/CD patterns. [`CONTEXT.md`](CONTEXT.md) defines project-specific terminology (STP/MTP, benchmark_client, search recipe, workload archetypes). [`KLAUD_DEBUG.md`](KLAUD_DEBUG.md) catalogs recurring failure modes before you debug a failing PR.

## Commands

```bash
# Run tests
python -m pytest utils/matrix_logic/ -v
python -m pytest utils/matrix_logic/ -v -m slow
python -m pytest utils/matrix_logic/ -v -m integration

# Validate / generate benchmark matrix from configs
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --model-prefix dsr1 --framework sglang --runner-type h200 --seq-lens 1k1k

# Process results
python utils/process_result.py
python utils/summarize.py ./results

# Dispatch a one-off benchmark (see AGENTS.md for full syntax)
gh api -X POST /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=main' \
  -f 'inputs[test-name]=<name>' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files .github/configs/nvidia-master.yaml ...'
```

## Architecture

InferenceX benchmarks LLM inference across hardware (NVIDIA/AMD) and serving frameworks (vLLM, SGLang, TRT-LLM, ATOM). The pipeline is:

1. **Config** — `.github/configs/nvidia-master.yaml` / `amd-master.yaml` define all benchmark configurations. `runners.yaml` maps hardware labels to runner nodes.
2. **Trigger** — `perf-changelog.yaml` is append-only; committing changes to it fires `run-sweep.yml`.
3. **Matrix generation** — `utils/matrix_logic/generate_sweep_configs.py` parses master configs, expands concurrency ranges, and emits a JSON matrix. Pydantic schemas in `validation.py` validate both input configs and output matrix entries.
4. **Execution** — CI fans out one job per matrix entry. Each job launches a Docker-containerized server, runs the benchmark client (`benchmark_serving.py` or AIPerf), and emits per-config JSON.
5. **Aggregation** — `collect-results.yml` downloads artifacts and builds `agg_bmk.json` for leaderboard ingestion.

### Key invariants

- `perf-changelog.yaml` is **append-only** with whitespace-sensitive formatting — new entries always go at the **END**, never mid-file. Never 3-way-merge it; see `KLAUD_DEBUG.md §1.1` for the rebase recipe.
- MTP (`*_mtp.sh`) scripts **must** pass `--use-chat-template` to `run_benchmark_serving` — EAGLE-style speculative decoding regresses silently without it.
- Multi-node `srt-slurm-recipes/` changes must update both the recipe YAML and `nvidia-master.yaml` atomically.
- Never create new directories under `/workspace` during a benchmark run.
