#!/usr/bin/env bash

# Gemma-4 31B FP8-block SPEED-Bench cell on SGLang (aiperf client): SB_ARM=base
# on the throughput_8k high_entropy split, ignore-eos on. Matrix key
# gemma4sbabasehi-fp8block-hardware-hcm-sglang. Everything except these
# variables lives in gemma4sba_sglang_body.sh.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
