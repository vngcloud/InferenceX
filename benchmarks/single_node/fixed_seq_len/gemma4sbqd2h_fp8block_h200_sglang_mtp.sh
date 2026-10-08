#!/usr/bin/env bash

# SGLang probe D: MTP depth 2 (hi), Gemma-4 31B FP8-block, high_entropy, mem 0.92 (chunk 16384, default evict).
# Matrix key gemma4sbqd2h-fp8block-gnslurm-sglang. Probe only; compare to the mem-only sweep cell.
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=2
SB_MEM_FRACTION=0.92
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
