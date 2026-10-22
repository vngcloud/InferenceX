#!/usr/bin/env bash
set -uo pipefail

# Kill aiperf replay trees left running by a benchmark shell that was
# SIGKILLed. Invoked from benchmark-tmpl.yml's shared resource-cleanup step,
# both pre-run (a previous cancelled job's leftovers double-load the endpoint
# and corrupt this run's numbers) and post-run (`if: always()` steps do run
# after a cancellation -- verified in run 30508401430, where every upload step
# executed at 02:32:46-02:32:54 -- which makes this the one layer that survives
# SIGKILL of the benchmark shell itself).
#
# Best-effort by design: cleanup must never fail a job, hence no `set -e`.

# benchmark_lib.sh's source-time agentic gate validates KV_OFFLOADING and
# `exit 1`s when the pair is inconsistent -- and a sourced exit would take this
# script with it. The workflow exports SCENARIO_TYPE=agentic-coding and
# IS_AGENTIC=1 job-wide, so pin both off here: this script only borrows
# stop_background_process_tree, it never benchmarks.
export IS_AGENTIC=0
export SCENARIO_TYPE=""

# shellcheck source=../benchmarks/benchmark_lib.sh
source "$(dirname "$0")/../benchmarks/benchmark_lib.sh"

reap_orphan_agentic_replays
