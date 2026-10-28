#!/usr/bin/env bash

# Gemma-4 31B FP8-block GreenSpec cell (aiperf client): GS_ARM=mtp (depth 4) on
# the agentic_tools lite subset. Matrix key gemma4gsatlmtp-fp8block-hardware-hcm-vllm.
# Everything except these variables lives in gemma4gs_body.sh.
GS_ARM=mtp
GS_SUBSET=agentic_tools
GS_LITE=1
NUM_SPEC_TOKENS=4
source "$(dirname "$0")/gemma4gs_body.sh"
