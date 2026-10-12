#!/usr/bin/env bash

# SGLang DFLASH smoke (Gemma-4 31B FP8-block, z-lab/gemma-4-31B-it-DFlash, high_entropy),
# target --attention-backend fa4, DRAFT attention backend: fa4, --speculative-num-draft-tokens 8.
# Matrix key gemma4sbdfa-fp8block-gnslurm-sglang. Serving knobs = the tau2 "B" config
# (mem 0.92, swa ratio 0.2, chunk 8192, hrrn, evict 32, radix on); only the spec
# arm changes (NEXTN MTP4 -> DFLASH block 16).
NUM_SPEC_TOKENS=8
SB_ARM=dflash
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
SB_DRAFT_ATTN=fa4
SB_MEM_FRACTION=0.92
SB_SWA_RATIO=0.2
SB_CHUNKED_PREFILL=8192
SB_SWA_EVICTION=32
SB_EXTRA_ARGS="--schedule-policy hrrn --enable-cache-report"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
