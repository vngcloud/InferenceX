#!/usr/bin/env bash

# SGLang probe B: schedule-conservativeness 1.3 + RETRACT_DECODE_STEPS 40 (base lo), Gemma-4 31B FP8-block, low_entropy, mem 0.92 (chunk 16384, default evict).
# Matrix key gemma4sbqb-fp8block-gnslurm-sglang. Probe only; compare to the mem-only sweep cell.
SB_ARM=base
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_EXTRA_ARGS="--schedule-conservativeness 1.3"
SB_EXTRA_ENV="SGLANG_RETRACT_DECODE_STEPS=40"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
