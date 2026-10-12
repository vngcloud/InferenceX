#!/usr/bin/env bash
# Static k sweep arm of the glm5.3w4adpa campaign: same recipe, num_steps=5.
# DeepEP low-latency buffer: graphs capture up to 32 reqs/rank (max-running 256 / dp 8),
# 32 x (5+1) tokens exceeds the default 128-token cap, so raise it to 256.
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
SPEC_NUM_STEPS=5 exec "$(dirname "$0")/glm5.3w4adpa_fp4_h200_sglang_mtp.sh"
