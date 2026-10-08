#!/usr/bin/env bash

# SGLang MTP (depth 4) KV-pool smoke, Gemma-4 31B FP8-block, high entropy, conc 64:
# --mem-fraction-static 0.92, --max-running-requests 64. SWA eviction interval 32 (the most favourable setting).
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.92
SB_MAX_RUNNING=64
SB_SWA_EVICTION=32
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
