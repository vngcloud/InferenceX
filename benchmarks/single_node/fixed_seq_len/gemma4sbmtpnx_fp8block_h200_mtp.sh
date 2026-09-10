#!/usr/bin/env bash

# Gemma-4 31B FP8-block -- STAGE 5 cell: native Gemma-4 MTP, google/gemma-4-31B-it-assistant
# on the SPEED-Bench throughput_8k mixed split, ignore-eos OFF -- artifact control.
#
# One of sixteen cells that differ only in the three variables set below; the
# engine configuration, dataset preparation, client invocation and acceptance
# accounting all live in the shared body sourced at the bottom, so the sweep
# provably cannot drift between arms.
#
# The separate file exists only because runners/launch_h200-greennode.sh derives
# the recipe path from the matrix model-prefix
# (benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200_mtp.sh)
# and neither the SPEED-Bench category nor the eos mode is a matrix field.
# Matrix key: gemma4sbmtpnx-fp8block-h200-vllm, model-prefix gemma4sbmtpnx.
#
# See gemma4sb_body.sh for the rationale: why this is a separate stage, why it
# bypasses run_benchmark_serving, why prefix caching is off, and why acceptance
# is read from Prometheus instead of the server log.
SB_ARM=mtp
SB_CATEGORY=mixed
SB_IGNORE_EOS=0
# NUM_SPEC_TOKENS is intentionally unset: pin the Stage-3 winning depth here
# before dispatching. The body aborts if it is missing.
# NUM_SPEC_TOKENS=<stage-3 winner>
source "$(dirname "$0")/gemma4sb_body.sh"
