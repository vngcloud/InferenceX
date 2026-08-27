#!/usr/bin/env bash

# Gemma-4 31B FP8-block, single-GPU vLLM, fixed-seq-len 8k1k, conc 8.
#
# PURPOSE: one-arm A/B against gemma4cp_fp8block_h200.sh (Actions run
# 32946714039, job 98108978549) to test whether pinning FlashAttention to v3
# lets the FP8 KV cache keep a FlashAttention kernel instead of falling back to
# Triton.
#
# WHAT WE OBSERVED. With --kv-cache-dtype fp8_e4m3 the engine logs
#   [cuda.py:476] Using TRITON_ATTN attention backend out of potential
#                 backends: ['TRITON_ATTN'].
# while the otherwise-identical BF16-KV twin (gemma4fi, run 32946716838) logs
#   [cuda.py:476] Using FLASH_ATTN attention backend out of potential
#                 backends: ['FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'].
# The diff between those two `vllm serve` lines is exactly --kv-cache-dtype
# plus --max-model-len, so fp8 KV is what removes FLASH_ATTN from the candidate
# set. Cost measured at conc 8: TTFT mean 798 -> 1376 ms (+72%), TPOT 17.9 ->
# 22.5 ms (+25%), 0.45 -> 0.36 req/s (-20%).
#
# WHY v3 IS THE KNOB. vLLM v0.25.0 FlashAttentionBackend.supports_kv_cache_dtype:
#   if kv_cache_dtype in ("fp8", "fp8_e4m3"):
#       return (get_flash_attn_version() == 3
#               and current_platform.is_device_capability_family(90))
# H200 IS capability family 90, so the only failing term is the version -- and
# the BF16 run logged "[flash_attn.py:718] Using FlashAttention version 4".
# FlashAttention is not fp16-only; this build just resolves to v4, which does
# not take fp8 KV. --attention-config.flash_attn_version=3 forces the branch
# that does. head_dim is 256 (< the >256 threshold that would upgrade 3 -> 4
# again inside get_flash_attn_version), so the pin should stick.
#
# FAILURE MODE IS BENIGN. If FA3 is not built into this image,
# get_flash_attn_version() hits `assert is_fa_version_supported(fa_version)`
# and the enclosing `except (ImportError, AssertionError): return None`, so it
# returns None -- None != 3 -> FLASH_ATTN stays excluded -> TRITON_ATTN, i.e.
# we simply reproduce the baseline, with an explanatory
# "Cannot use FA version 3 is not supported due to ..." line in the log. No
# crash either way. Read the verdict off [cuda.py:476] and [flash_attn.py:718].
#
# CONTROL FIDELITY: this file is gemma4cp_fp8block_h200.sh with exactly one
# added server flag. Same checkpoint, same fp8_e4m3 KV, same chunked-prefill
# tuning, same MAX_BATCHED_TOKENS cap, same GPU pin, same 80-prompt workload,
# so any delta is attributable to the FA version.
#
# Exploration-only: dispatched via e2e-tests.yml against a branch, never merged.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

# Same card gemma4cp pinned for the baseline, so the two runs sit on identical
# silicon. h200-greennode_03 carries a non-InferenceX tenant on the high GPU
# indices and the launcher runs --gpus all with no pin of its own, so without
# this the engine would land on GPU 0. GPU_IDS is not in the launcher's env
# allow-list, so this is the pin -- edit here if 3 is occupied at dispatch.
BENCH_GPU="${GPU_IDS:-3}"

# Host port selection. The launcher uses --network host, so $PORT is a
# box-global port; the harness default 8888 has been squatted by the other
# tenant's inference server before (run 32934050806 died on
# "Address already in use" while wait_for_server_ready was satisfied by the
# SQUATTER). Bind-test candidates rather than HTTP-probe them: a squatter that
# answers nothing on / would still pass a curl.
PORT_CANDIDATES="${PORT_CANDIDATES:-8888 8901 8902 8903 8904 8905}"
PORT="$(python3 - $PORT_CANDIDATES <<'PY'
import socket, sys
for port in (int(a) for a in sys.argv[1:]):
    sock = socket.socket()
    try:
        # No SO_REUSEADDR/SO_REUSEPORT on purpose: this must fail whenever
        # anything at all -- including a TIME_WAIT remnant -- owns the port.
        sock.bind(("0.0.0.0", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    sys.exit(0)
sys.exit(1)
PY
)" || { echo "ERROR: no free host port among: $PORT_CANDIDATES" >&2; exit 1; }
export PORT
echo "Serving on host port $PORT"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# Unfiltered on purpose: records the whole box, including what the other tenant
# holds. nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the GPU monitor's extra
# cards are cosmetic telemetry; throughput is TP-derived.
nvidia-smi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
fi

SERVER_LOG=/workspace/server.log

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_P2P_LEVEL=NVL

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

# max_num_batched_tokens must not exceed max_num_seqs * max_model_len, else the
# warmup dummy batch overruns a single sequence's max length and CUDA graph
# capture dies with an illegal memory access (hit at conc 1: 16384 > 1*9472).
# Cap it to CONC*MAX_MODEL_LEN so low-conc points stay valid; at conc 8 with
# max_model_len 9472 the cap is 75776, so this keeps the full 16384 -- exactly
# what the gemma4cp baseline ran.
MAX_BATCHED_TOKENS=16384
CAP=$(( CONC * MAX_MODEL_LEN ))
if [ "$MAX_BATCHED_TOKENS" -gt "$CAP" ]; then MAX_BATCHED_TOKENS="$CAP"; fi

set -x
CUDA_VISIBLE_DEVICES="$BENCH_GPU" vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-config.flash_attn_version=3 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$CONC" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 8192 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# The whole point of the run: surface which backend and which FA version the
# engine actually resolved, in the job log, without digging through server.log.
echo "--- attention backend selection ---"
grep -E "attention backend|FlashAttention version|FA version|flash_attn" "$SERVER_LOG" || \
    echo "(no attention-backend lines found in $SERVER_LOG)"
echo "-----------------------------------"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
