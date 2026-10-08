#!/usr/bin/env bash

# SGLang probe A: FP8 KV (base lo), Gemma-4 31B FP8-block, low_entropy, mem 0.92 (chunk 16384, default evict).
# Matrix key gemma4sbqakb-fp8block-gnslurm-sglang. Probe only; compare to the mem-only sweep cell.
SB_ARM=base
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_EXTRA_ARGS="--kv-cache-dtype fp8_e4m3"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
