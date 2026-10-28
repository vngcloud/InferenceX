#!/usr/bin/env bash

# Gemma-4 31B FP8-block GreenSpec cell (aiperf client): GS_ARM=mtp (depth 6) on
# the full diverse subset. Matrix key gemma4gsdvmtp6-fp8block-hardware-hcm-vllm.
# Everything except these variables lives in gemma4gs_body.sh.
GS_ARM=mtp
GS_SUBSET=diverse
GS_LITE=0
NUM_SPEC_TOKENS=6
source "$(dirname "$0")/gemma4gs_body.sh"
