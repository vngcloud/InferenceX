#!/usr/bin/env bash

# SGLang KV-pool probe a (Gemma-4 31B FP8-block, base, high entropy): mem 0.92.
# Matrix key gemma4sbaprobea-fp8block-hardware-hcm-sglang. Probe only; the
# sweep recipe is gemma4sbabasehi_fp8block_h200_sglang.sh.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
SB_MEM_FRACTION=0.92
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
