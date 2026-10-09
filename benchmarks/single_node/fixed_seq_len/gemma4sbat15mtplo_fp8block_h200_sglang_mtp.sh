#!/usr/bin/env bash

# SGLang SpeedBench cell, tau2 serving config with swa-full-tokens-ratio 0.15 (Gemma-4 31B FP8-block, mtp, low_entropy).
# Matrix key gemma4sbat15mtplo-fp8block-gnslurm-sglang. Identical to gemma4sbat2mtplo (ratio 0.2,
# run 37884264773) except SB_SWA_RATIO: mem 0.92, chunked-prefill 8192, schedule-policy hrrn,
# enable-cache-report, SWA eviction 32, context 262144, radix on.
SB_ARM=mtp
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.92
SB_SWA_RATIO=0.15
SB_CHUNKED_PREFILL=8192
SB_SWA_EVICTION=32
SB_EXTRA_ARGS="--schedule-policy hrrn --enable-cache-report"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
