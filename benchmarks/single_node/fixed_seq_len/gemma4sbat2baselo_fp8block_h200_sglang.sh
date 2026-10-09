#!/usr/bin/env bash

# SGLang SpeedBench cell with the tau2 agentic serving config (Gemma-4 31B FP8-block, base, low_entropy).
# Matrix key gemma4sbat2baselo-fp8block-gnslurm-sglang. Server flags copied from run 37879258874
# (gemma4tau2 sglang MTP4, commit cd19d215): mem 0.92, swa-full-tokens-ratio 0.2,
# chunked-prefill 8192, schedule-policy hrrn, enable-cache-report, SWA eviction 32,
# context 262144, radix on.
SB_ARM=base
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_SWA_RATIO=0.2
SB_CHUNKED_PREFILL=8192
SB_SWA_EVICTION=32
SB_EXTRA_ARGS="--schedule-policy hrrn --enable-cache-report"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
