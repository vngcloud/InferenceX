#!/usr/bin/env bash

# MNBT sweep wrapper: gemma4eagle DP2+EAGLE3 recipe with --max-num-batched-tokens
# pinned to 8192 (the current baseline), on GPUs 0,1 of h200-greennode_03 (GPUs
# 4-7 host another tenant). This is the sweep's control point -- same MNBT as the
# production _04 config, isolated on _03 so all three arms share one clean box.
# All logic lives in the shared gemma4eagle recipe. See gemma4eagle_fp8block_h200.sh.
export MAX_NUM_BATCHED_TOKENS=8192
export GPU_IDS="${GPU_IDS:-0,1}"
exec bash "$(dirname "$0")/gemma4eagle_fp8block_h200.sh"
