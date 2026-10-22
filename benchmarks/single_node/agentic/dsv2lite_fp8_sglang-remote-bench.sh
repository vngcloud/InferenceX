#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

# Remote-bench recipe: benchmarks an already-running, externally-managed
# SGLang endpoint instead of launching one on this box. No local server, no
# router, no nvidia-smi — this runner only needs network access to the
# target and a Python venv for aiperf. Unlike glm5.2_fp4_*_sglang.sh (one
# per hardware target, since local server tuning is hw-specific), this file
# is hw-agnostic: it never launches a server, so there's one per
# model+precision+framework combo, not one per hardware.
#
# Required and optional inputs, all self-reported by whoever owns the remote
# endpoint, are documented on remote_bench_preflight() in benchmark_lib.sh —
# that function is the authoritative contract. In short: REMOTE_BASE_URL,
# REMOTE_ENGINE_METRICS_URL and REMOTE_RUNNER_TYPE are required;
# REMOTE_GPU_TELEMETRY_URL, REMOTE_MAX_CONTEXT_LENGTH, REMOTE_API_KEY,
# REMOTE_TOKENIZER and REMOTE_RESET_URL are optional. Behavior changes belong
# in remote_bench_preflight(), not here — a per-recipe fix silently misses
# every other remote target.
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE

mkdir -p "$RESULT_DIR"

remote_bench_preflight

resolve_trace_source
install_agentic_deps

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
