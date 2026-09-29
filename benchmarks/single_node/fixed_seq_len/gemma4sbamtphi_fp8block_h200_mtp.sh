#!/usr/bin/env bash

# Gemma-4 31B FP8-block SPEED-Bench cell (aiperf client): SB_ARM=mtp on the
# throughput_8k high_entropy split, ignore-eos on. Matrix key
# gemma4sbamtphi-fp8block-h200-vllm. Everything except these variables lives in
# gemma4sba_body.sh; see its header for the arm/category contract.
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
source "$(dirname "$0")/gemma4sba_body.sh"
