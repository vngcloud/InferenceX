#!/usr/bin/env bash

# Gemma-4 31B FP8-block -- STAGE 5 cell: no speculator (Stage-5 gate denominator)
# on the SPEED-Bench throughput_8k high-entropy split, ignore-eos on.
#
# One of sixteen cells that differ only in the three variables set below; the
# engine configuration, dataset preparation, client invocation and acceptance
# accounting all live in the shared body sourced at the bottom, so the sweep
# provably cannot drift between arms.
#
# The separate file exists only because runners/launch_h200-greennode.sh derives
# the recipe path from the matrix model-prefix
# (benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200.sh)
# and neither the SPEED-Bench category nor the eos mode is a matrix field.
# Matrix key: gemma4sbbasehi-fp8block-h200-vllm, model-prefix gemma4sbbasehi.
#
# See gemma4sb_body.sh for the rationale: why this is a separate stage, why it
# bypasses run_benchmark_serving, why prefix caching is off, and why acceptance
# is read from Prometheus instead of the server log.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
source "$(dirname "$0")/gemma4sb_body.sh"
