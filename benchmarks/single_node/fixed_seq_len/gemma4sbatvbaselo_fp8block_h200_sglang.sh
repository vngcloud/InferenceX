#!/usr/bin/env bash

# SGLang tuned variant v (Gemma-4 31B FP8-block, base, low entropy): mem 0.92 + SWA eviction 32 +
# chunked-prefill 16384 (default chunk keeps a larger SWA pool: pool = slots x per_request + 2 x chunk).
# Matrix key gemma4sbatvbaselo-fp8block-gnslurm-sglang. Investigates the base lo c8/c32 retract storm of the 8192-chunk tuned sweep.
SB_ARM=base
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_SWA_EVICTION=32
SB_CHUNKED_PREFILL=16384
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
