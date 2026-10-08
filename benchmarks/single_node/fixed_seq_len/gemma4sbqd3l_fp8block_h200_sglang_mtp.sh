#!/usr/bin/env bash

# SGLang probe D: MTP depth 3 (lo), Gemma-4 31B FP8-block, low_entropy, mem 0.92 (chunk 16384, default evict).
# Matrix key gemma4sbqd3l-fp8block-gnslurm-sglang. Probe only; compare to the mem-only sweep cell.
SB_ARM=mtp
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=3
SB_MEM_FRACTION=0.92
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
