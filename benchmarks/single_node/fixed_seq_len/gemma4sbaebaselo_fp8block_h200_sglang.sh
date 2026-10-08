#!/usr/bin/env bash

# SGLang mem+evict sweep cell (Gemma-4 31B FP8-block, base, low_entropy): mem 0.92 + SWA eviction 32, chunk 16384 (default).
# Matrix key gemma4sbaebaselo-fp8block-gnslurm-sglang. Differs from the 0.88 sweep by mem 0.92 and SGLANG_SWA_EVICTION_INTERVAL=32 only.
SB_ARM=base
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_SWA_EVICTION=32
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
