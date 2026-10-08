#!/usr/bin/env bash

# SGLang mem-only sweep cell (Gemma-4 31B FP8-block, base, high_entropy): mem 0.92 only (engine-default eviction, chunk 16384).
# Matrix key gemma4sbambasehi-fp8block-gnslurm-sglang. Only --mem-fraction-static differs from the 0.88 sweep.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
