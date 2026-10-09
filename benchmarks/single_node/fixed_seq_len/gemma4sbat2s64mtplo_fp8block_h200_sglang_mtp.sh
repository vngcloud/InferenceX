#!/usr/bin/env bash

# SGLang SpeedBench cell, tau2 serving config with 64 running slots for MTP (Gemma-4 31B FP8-block, mtp, low_entropy).
# Matrix key gemma4sbat2s64mtplo-fp8block-gnslurm-sglang. Identical to gemma4sbat2mtplo (run 37884264773)
# except SB_MAX_RUNNING=64: the recipe caps MTP at 32 slots above conc 32 because the old pool sizing
# (radix off) needed ~107 GiB at 64. With radix on the pool is full x ratio, so this checks whether
# 64 slots now start and how they perform. Use with --conc 64.
SB_ARM=mtp
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.92
SB_SWA_RATIO=0.2
SB_CHUNKED_PREFILL=8192
SB_SWA_EVICTION=32
SB_MAX_RUNNING=64
SB_EXTRA_ARGS="--schedule-policy hrrn --enable-cache-report"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
