#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay for Kimi-K3 MXFP4 on B300 (b300-netperf) using vLLM.
#
# Mirrors the single_node_tp + blackwell profile of
# https://recipes.vllm.ai/moonshotai/Kimi-K3 (source:
# vllm-project/recipes models/moonshotai/Kimi-K3.yaml).
#
# Model shape (from the checkpoint's config.json):
#   2.8T params, 896 routed experts + 2 shared, 16 experts/token, 93 layers
#   split into 24 full-attention MLA layers (kv_lora_rank 512, qk_rope 64) and
#   69 Kimi Delta Attention (KDA) linear layers, 1M max_position_embeddings.
#   MXFP4 weights; self-attn / shared-experts / dense MLP / lm_head stay bf16.
#
# Weights live on the runner at $MODEL_PATH; launch_b300-netperf.sh mounts
# /mnt/models read-only into the container, so no download happens here.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, RESULT_DIR, DURATION, EP_SIZE,
#   DP_ATTENTION, SPEC_DECODING

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING RESULT_DIR DURATION EP_SIZE DP_ATTENTION SPEC_DECODING

# The 2.8T MXFP4 checkpoint fills ~1.42 TiB of the node's 2149 GiB of HBM, so
# every arm here is GPU-resident; there is no CPU-offload ladder to select.
require_agentic_kv_offload_none

export MODEL_PATH="${MODEL_PATH:-/mnt/models/moonshotai/Kimi-K3}"
if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
    echo "Error: MODEL_PATH='$MODEL_PATH' is missing or empty on this runner." >&2
    exit 1
fi

# K3 is a 1M-context family and serves the full window, so it replays the
# unfiltered corpus like the other 1M recipes (dsv4, glm5.2, minimaxm3). The
# 256k-capped variant exists for families whose max_model_len cannot reach 1M;
# that is not this model. Pinned explicitly to match the sibling b300-netperf
# glm5.2 recipes rather than leaning on the MODEL_PREFIX default.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0

# recipe: hardware_overrides.blackwell.extra_env
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1

resolve_trace_source
install_agentic_deps
nvidia-smi

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    # vLLM has no --enable-dp-attention; attention-DP is implicit in
    # --data-parallel-size N with TP1 (what dsv4_fp4_b300_vllm.sh does). K3
    # cannot use it on one node:
    #   - the recipe's only single-node strategy is single_node_tp; unlike
    #     sibling MoE recipes it omits single_node_dep/tep, and multi_node_dep
    #     carries a 16-GPU floor. DP replicates the dense/attention/KDA weights
    #     per rank, and ~1.68 TB of weights will not replicate into 2149 GiB.
    #   - DP+EP over hybrid GatedDeltaNet/Mamba layers deadlocks during CUDA
    #     graph capture (vllm-project/vllm#41862), which K3's 69 KDA layers hit.
    echo "Error: DP_ATTENTION is not supported for Kimi-K3 on a single 8xB300 node." >&2
    exit 1
fi

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# The matrix's spec-decoding knob is Literal["mtp","draft_model","none"]; K3's
# speculator is the DSpark draft model, so it rides the draft_model value.
SPEC_ARGS=()
case "$SPEC_DECODING" in
    none) ;;
    draft_model)
        SPEC_ARGS=(
            --speculative-config
            '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7,"method":"dspark","attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
        )
        ;;
    *)
        echo "Error: unsupported SPEC_DECODING='$SPEC_DECODING' for Kimi-K3 (want none|draft_model)." >&2
        exit 1
        ;;
esac

# Serve the model's full window, as the recipe's blackwell profile does, so the
# unfiltered corpus replays without truncation.
#
# Measured on b300-netperf_00 at TP8 / gpu-memory-utilization 0.95: the MXFP4
# weights leave 56.17 GiB of KV per GPU = 4,322,142 tokens of pool at ~13.6
# KiB/token. vLLM prints "Maximum concurrency for 1,048,576 tokens per request:
# 4.12x" -- that is the worst case where every request holds a full 1M window,
# not a cap on CCU; real concurrency tracks the traces' live context (~132x at
# 32k, ~66x at 64k, ~33x at 128k). Expect the knee where the working set
# outgrows the pool: the sibling glm5.2 TP8 arm hit it past conc 8 with a larger
# pool, which is why the ladder starts low.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS=$((2 * CONC))

echo "Starting vllm server..."
{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --gpu-memory-utilization 0.95
    --load-format fastsafetensors
    --no-enable-flashinfer-autotune
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --kv-cache-dtype fp8
    --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TRTLLM_RAGGED"}'
    # The KDA layers force the attention block size up to 1536 tokens so the
    # attention page is >= the mamba page, so prefix reuse is coarse-grained
    # here in a way it is not for the GQA/MLA-only recipes. Also watch
    # vllm-project/vllm#50147 (K3 TP8 + prefix caching -> illegal memory access
    # under concurrency); drop this flag and re-baseline if the sweep hits it.
    --enable-prefix-caching
    # Claude Code traces are text-only; skipping the vision tower frees its
    # weights and its share of the multimodal preprocessing path.
    --language-model-only
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --disable-uvicorn-access-log
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
set -x

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Defaulted, not bare: this script runs under `set -u`, and the launcher only
# forwards EVAL_ONLY when the workflow set it.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
