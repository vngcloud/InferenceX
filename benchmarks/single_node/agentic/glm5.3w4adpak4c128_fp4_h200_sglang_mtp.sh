#!/usr/bin/env bash
# Static k=4 with the default DeepEP cap (128), to isolate the cap cost from the k4 arm (cap 256).
# Graphs stop at bs 24/rank so capture fits: 24 x (4+1) = 120 tokens. Eager bs >= 26/rank would
# exceed the cap; CCU 16 keeps per-rank bs far below that.
SPEC_NUM_STEPS=4 SPEC_EXTRA_ARGS="--cuda-graph-max-bs-decode 24" exec "$(dirname "$0")/glm5.3w4adpa_fp4_h200_sglang_mtp.sh"
