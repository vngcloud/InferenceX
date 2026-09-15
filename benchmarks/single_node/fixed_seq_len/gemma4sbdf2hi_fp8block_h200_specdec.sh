#!/usr/bin/env bash

# Gemma-4 31B FP8-block -- STAGE 5 cell: DFlash2, in-house checkpoint
# (/mnt/models/gemma4-31b-it-dflash2 on runner h200-greennode_06) on the
# SPEED-Bench throughput_8k high-entropy split, ignore-eos on.
#
# PREFLIGHT ONLY -- one cell, conc 1. This is a custom checkpoint trained
# outside vLLM's published DFlash lineage; confirm it loads and produces a
# sane acceptance length before extending to more categories/concurrencies,
# same precedent as the RedHatAI dflash arm (see gemma4sb_body.sh dflash
# case, and matrix comment above gemma4sbdfhi-fp8block-h200-vllm).
#
# DISPATCH REQUIRES --runner-node-filter h200-greennode_06: the checkpoint is
# local to that host only, mounted in-container via the existing
# MODEL_STORE_MOUNT (see gemma4sb_body.sh dflash2 case for the full mount
# rationale). It will not resolve on any other node in the h200-greennode pool.
#
# One of the gemma4sb cells that differ only in the three variables set below;
# the engine configuration, dataset preparation, client invocation and
# acceptance accounting all live in the shared body sourced at the bottom, so
# the sweep provably cannot drift between arms.
#
# The separate file exists only because runners/launch_h200-greennode.sh derives
# the recipe path from the matrix model-prefix
# (benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200_specdec.sh)
# and neither the SPEED-Bench category nor the eos mode is a matrix field.
# Matrix key: gemma4sbdf2hi-fp8block-h200-vllm, model-prefix gemma4sbdf2hi.
#
# See gemma4sb_body.sh for the rationale: why this is a separate stage, why it
# bypasses run_benchmark_serving, why prefix caching is off, and why acceptance
# is read from Prometheus instead of the server log.
SB_ARM=dflash2
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=7
source "$(dirname "$0")/gemma4sb_body.sh"
