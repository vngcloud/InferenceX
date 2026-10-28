#!/usr/bin/env bash

# Gemma-4 31B FP8-block GreenSpec cell (aiperf client): GS_ARM=base on the
# agentic_tools lite subset. Matrix key gemma4gsatlbase-fp8block-hardware-hcm-vllm.
# Everything except these variables lives in gemma4gs_body.sh.
GS_ARM=base
GS_SUBSET=agentic_tools
GS_LITE=1
source "$(dirname "$0")/gemma4gs_body.sh"
