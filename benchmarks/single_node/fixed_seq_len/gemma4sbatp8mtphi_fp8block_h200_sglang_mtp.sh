#!/usr/bin/env bash

# SGLang SpeedBench TP8 smoke (Gemma-4 31B FP8-block, mtp, high_entropy, throughput_16k).
# Matrix key gemma4sbatp8mtphi-fp8block-hardware-hcm-sglang. Same serving config as
# gemma4sbat2mtphi (config B: mem 0.92, swa ratio 0.2, chunk 8192, hrrn, evict 32,
# context 262144, radix on, MTP4); only TP (1 -> 8, from the matrix) and the dataset
# (throughput_8k -> throughput_16k) differ. max-running-requests stays min(CONC, 32).
SB_ARM=mtp
SB_CATEGORY=high_entropy
SB_CONFIG=throughput_16k
SB_IGNORE_EOS=1
NUM_SPEC_TOKENS=4
SB_MEM_FRACTION=0.92
SB_SWA_RATIO=0.2
SB_CHUNKED_PREFILL=8192
SB_SWA_EVICTION=32
SB_EXTRA_ARGS="--schedule-policy hrrn --enable-cache-report"
source "$(dirname "$0")/gemma4sba_sglang_body.sh"
