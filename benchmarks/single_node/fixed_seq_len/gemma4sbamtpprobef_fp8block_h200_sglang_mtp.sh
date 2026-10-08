#!/usr/bin/env bash

# SGLang MTP (depth 4) tuned smoke, Gemma-4 31B FP8-block, high entropy, conc 32:
# probe-c settings (mem 0.92 + eviction 32 + chunked-prefill 8192), 32 slots.
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.92
SB_MAX_RUNNING=32
SB_SWA_EVICTION=32
SB_CHUNKED_PREFILL=8192
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
