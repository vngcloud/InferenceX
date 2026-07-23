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
# Required inputs, all self-reported by whoever owns the remote endpoint:
#   REMOTE_BASE_URL            e.g. http://10.0.4.12:30000
#   REMOTE_GPU_TELEMETRY_URL   DCGM /metrics endpoint on the remote box
#   REMOTE_ENGINE_METRICS_URL  SGLang /metrics endpoint on the remote box
#   REMOTE_RUNNER_TYPE         real, GPU_KEYS-resolvable hw string (e.g. h200-nv) —
#                              NOT the cluster:remote-bench runner label
# Optional:
#   REMOTE_RESET_URL           called before each concurrency point to clear
#                              KV/prefix cache + router affinity on the
#                              long-lived remote engine (see #2 in issue #26:
#                              local recipes get a fresh process per conc job,
#                              a remote target does not)
check_env_vars MODEL CONC RESULT_DIR DURATION \
    REMOTE_BASE_URL REMOTE_GPU_TELEMETRY_URL REMOTE_ENGINE_METRICS_URL REMOTE_RUNNER_TYPE

mkdir -p "$RESULT_DIR"

# Required-endpoint pre-flight. Unlike aiperf's own telemetry probing (which
# soft-fails and disables telemetry on an unreachable URL — the right default
# for a general-purpose client), a remote-bench result without GPU + engine
# telemetry isn't usable for our purposes, so we fail the job here rather
# than let the run finish without them. Scoped entirely to this recipe;
# aiperf's own graceful-degradation behavior is untouched.
for pair in "REMOTE_GPU_TELEMETRY_URL:$REMOTE_GPU_TELEMETRY_URL" "REMOTE_ENGINE_METRICS_URL:$REMOTE_ENGINE_METRICS_URL"; do
  name="${pair%%:*}"
  url="${pair#*:}"
  if ! curl --output /dev/null --silent --fail --max-time 10 "$url"; then
    echo "ERROR: $name ($url) is not reachable. Required for remote-bench." >&2
    exit 1
  fi
done

# Health check against the remote endpoint itself. No --server-pid: there's
# no local process to monitor, the remote operator owns the engine's lifecycle.
if ! curl --output /dev/null --silent --fail --max-time 10 "$REMOTE_BASE_URL/health"; then
  echo "ERROR: REMOTE_BASE_URL ($REMOTE_BASE_URL) is not reachable at /health." >&2
  exit 1
fi

if [ -n "${REMOTE_RESET_URL:-}" ]; then
  echo "Resetting remote engine state via REMOTE_RESET_URL before this concurrency point ..."
  curl --output /dev/null --silent --fail --max-time 30 -X POST "$REMOTE_RESET_URL"
fi

# Self-report the real hardware key for downstream ingest. RUNNER_TYPE is
# otherwise set by the workflow to inputs.runner (the GH Actions runs-on
# label, e.g. cluster:remote-bench), which is not a GPU_KEYS-resolvable
# value. Overriding it here, in-process before aiperf runs, is what
# process_agentic_result.py (os.environ.get("RUNNER_TYPE")) actually sees.
export RUNNER_TYPE="$REMOTE_RUNNER_TYPE"
export AIPERF_GPU_TELEMETRY_URL="$REMOTE_GPU_TELEMETRY_URL"
export AIPERF_SERVER_METRICS_URLS="$REMOTE_ENGINE_METRICS_URL"
export REMOTE_BASE_URL

resolve_trace_source
install_agentic_deps

build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
