#!/usr/bin/env bash

# SGLang SpeedBench cell, tau2 serving config with 64 running slots and mem-fraction-static 0.97 for MTP (Gemma-4 31B FP8-block, mtp, high_entropy).
# Matrix key gemma4sbat2s64m97mtphi-fp8block-gnslurm-sglang. Identical to gemma4sbat2s64mtphi (run 37949639715)
# except SB_MEM_FRACTION=0.97 (was 0.92): s64 left ~20 GiB free after the pool and ran only 39 of 64 slots, retracting.
# Checks whether a larger pool lets more of the 64 slots run without retracts. Use with --conc 64.
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.97
SB_SWA_RATIO=0.2
SB_CHUNKED_PREFILL=8192
SB_SWA_EVICTION=32
SB_MAX_RUNNING=64
SB_EXTRA_ARGS="--schedule-policy hrrn --enable-cache-report"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
