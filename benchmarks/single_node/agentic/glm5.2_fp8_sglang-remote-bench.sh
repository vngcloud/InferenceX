#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

# Remote-bench recipe: benchmarks an already-running, externally-managed
# SGLang endpoint instead of launching one on this box. No local server, no
# router, no nvidia-smi — this runner only needs network access to the target
# and a Python venv for aiperf. Hardware-agnostic by design: nothing here
# launches a server, so there is one file per model+precision+framework
# combo, not one per hardware target.
#
# Built for GreenNode's MaaS gateway, which is authenticated and exposes
# neither /health nor a DCGM exporter. All of that is handled by
# remote_bench_preflight in benchmark_lib.sh; see that function's header for
# the full env-var contract, and
# docs/superpowers/specs/2026-07-29-remote-bench-authenticated-gateway-design.md
# for why each requirement is what it is.
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE

mkdir -p "$RESULT_DIR"

remote_bench_preflight

resolve_trace_source
install_agentic_deps

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
