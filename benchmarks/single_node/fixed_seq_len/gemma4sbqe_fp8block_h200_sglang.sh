#!/usr/bin/env bash

# SGLang probe E: cuda-graph-max-bs-decode 128 (base hi c64), Gemma-4 31B FP8-block, high_entropy, mem 0.92 (chunk 16384, default evict).
# Matrix key gemma4sbqe-fp8block-gnslurm-sglang. Probe only; compare to the mem-only sweep cell.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_EXTRA_ARGS="--cuda-graph-max-bs-decode 128"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
