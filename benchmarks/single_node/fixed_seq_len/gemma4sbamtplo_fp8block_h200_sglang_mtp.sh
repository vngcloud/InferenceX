#!/usr/bin/env bash

# Gemma-4 31B FP8-block SPEED-Bench cell on SGLang (aiperf client): SB_ARM=mtp
# (depth 4) on the throughput_8k low_entropy split, ignore-eos on. Matrix key
# gemma4sbamtplo-fp8block-hardware-hcm-sglang. Everything except these
# variables lives in gemma4sba_sglang_body.sh.
SB_ARM=mtp
SB_CATEGORY=low_entropy
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
