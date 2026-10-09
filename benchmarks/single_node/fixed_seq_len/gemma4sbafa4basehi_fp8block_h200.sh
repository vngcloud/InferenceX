#!/usr/bin/env bash

# Smoke: gemma4sbabasehi with vLLM FLASH_ATTN version 4 instead of FLASHINFER.
SB_ARM=base
SB_CATEGORY=high_entropy
SB_IGNORE_EOS=1
SB_VLLM_ATTN=FLASH_ATTN
SB_VLLM_FA_VERSION=4
source "$(dirname "$0")/gemma4sba_body.sh"
