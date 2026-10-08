#!/usr/bin/env bash

# SGLang tuned smoke, Gemma-4 31B FP8-block, base, high entropy, conc 8:
# probe-c settings (mem 0.92 + eviction 32 + chunked-prefill 8192), checks that
# the tuning does not hurt low concurrency.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
SB_SWA_EVICTION=32
SB_CHUNKED_PREFILL=8192
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
