#!/usr/bin/env bash

# Gemma-4 31B FP8-block SPEED-Bench cell (aiperf client): SB_ARM=base on the
# throughput_8k low_entropy split, ignore-eos on. Matrix key
# gemma4sbabaselo-fp8block-h200-vllm. Everything except these variables lives in
# gemma4sba_body.sh; see its header for the arm/category contract.
SB_ARM=base
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
source "$(dirname "$0")/gemma4sba_body.sh"
