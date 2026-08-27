#!/usr/bin/env bash

# Gemma-4 31B FP8-block 8k1k -- STAGE 3, native Gemma-4 MTP at speculative
# depth 8. One of three arms (depths 2/4/8) that differ in this integer and
# nothing else: the entire engine configuration, dataset, ladder and telemetry
# live in the shared body sourced below, so the sweep provably cannot drift.
#
# The separate file exists only because runners/launch_h200-greennode.sh
# derives the recipe path from the matrix model-prefix
# (benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200_mtp.sh),
# so each depth needs its own model-prefix and therefore its own filename.
# Matrix key: gemma4v28mtp8-fp8block-h200-vllm, model-prefix gemma4v28m8.
#
# See gemma4v28mtp_body.sh for the drafter, the "method":"mtp" rationale, the
# v0.28.0 floor, and the caveats.
NUM_SPEC_TOKENS=8
source "$(dirname "$0")/gemma4v28mtp_body.sh"
