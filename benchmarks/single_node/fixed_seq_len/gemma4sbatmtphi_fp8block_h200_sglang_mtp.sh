#!/usr/bin/env bash

# SGLang tuned sweep cell (Gemma-4 31B FP8-block, mtp, high_entropy): mem 0.92 + SWA eviction 32 + chunked-prefill 8192.
# Matrix key gemma4sbatmtphi-fp8block-gnslurm-sglang. 'SGLang tuned': chunk 8192 has no vLLM equivalent.
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.92
SB_SWA_EVICTION=32
SB_CHUNKED_PREFILL=8192
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
