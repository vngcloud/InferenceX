#!/usr/bin/env bash

# MNBT sweep wrapper: gemma4eagle DP2+EAGLE3 recipe with --max-num-batched-tokens
# pinned to 2048, on GPUs 0,1 of h200-greennode_03 (GPUs 4-7 host another
# tenant). All logic lives in the shared gemma4eagle recipe; this file only fixes
# the swept value + GPU pin, then delegates. See gemma4eagle_fp8block_h200.sh.
export MAX_NUM_BATCHED_TOKENS=2048
export GPU_IDS="${GPU_IDS:-0,1}"
exec bash "$(dirname "$0")/gemma4eagle_fp8block_h200.sh"
