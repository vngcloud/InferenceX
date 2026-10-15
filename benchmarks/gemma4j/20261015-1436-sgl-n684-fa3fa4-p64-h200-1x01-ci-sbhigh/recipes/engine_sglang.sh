# SGLang engine launch. Sourced by run_cell.sh after launch.env is loaded.
# Needs: MODEL SERVED_MODEL_NAME PORT TP DP ROUTER CONTEXT_LENGTH MAX_RUNNING ARGS_FILE SPEC_KIND SPEC_DRAFT SPEC_DEPTH
engine_spec_args() {
    case "$SPEC_KIND" in
        none) ;;
        mtp)    printf '%s\n' --speculative-algorithm NEXTN --speculative-draft-model-path "$SPEC_DRAFT" \
                    --speculative-num-steps "$SPEC_DEPTH" --speculative-eagle-topk 1 \
                    --speculative-num-draft-tokens "$((SPEC_DEPTH + 1))" ;;
        dflash) printf '%s\n' --speculative-algorithm DFLASH --speculative-draft-model-path "$SPEC_DRAFT" \
                    --speculative-num-draft-tokens "$SPEC_DEPTH" --speculative-draft-attention-backend fa4 ;;
        eagle3) printf '%s\n' --speculative-algorithm EAGLE3 --speculative-draft-model-path "$SPEC_DRAFT" \
                    --speculative-num-steps "$SPEC_DEPTH" --speculative-eagle-topk 1 \
                    --speculative-num-draft-tokens "$((SPEC_DEPTH + 1))" ;;
        *) echo "unknown SPEC_KIND=$SPEC_KIND" >&2; return 1 ;;
    esac
}
engine_expected_backend() { grep -E '^--attention-backend=' "$ARGS_FILE" | cut -d= -f2; }
engine_expected_spec_regex() {
    case "$SPEC_KIND" in none) echo none ;; mtp) echo "to FROZEN_KV_MTP" ;; dflash) echo "DFLASH draft runner" ;; eagle3) echo "EAGLE3" ;; esac
}
engine_start() {  # <launch_dir>; sets SERVER_PID, SERVER_LOG
    local dir="$1"
    # DP: SGLang divides --max-running-requests (and --chunked-prefill-size) by --dp internally, so the cell
    # total is passed as-is. router=sgl-router fronts the replicas with sglang_router (cache_aware);
    # router=builtin uses launch_server's own --dp dispatch.
    local -a launcher=(python3 -m sglang.launch_server) dp_args=() extra=() spec=()
    local max_running="$MAX_RUNNING"
    if (( ${DP:-1} > 1 )); then
        dp_args=(--dp "$DP")
        if [[ "${ROUTER:-}" == "sgl-router" ]]; then
            # sglang_router starts one full server per worker: each gets the per-replica share
            launcher=(python3 -m sglang_router.launch_server --router-policy cache_aware)
            max_running=$(( MAX_RUNNING / DP ))
        fi
    fi
    local -a cmd=("${launcher[@]}" --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
                  --host 0.0.0.0 --port "$PORT" --trust-remote-code --tp "$TP" "${dp_args[@]}"
                  --context-length "$CONTEXT_LENGTH" --max-running-requests "$max_running" --enable-metrics)
    local line  # bash 3.2-compatible (no mapfile): one native arg per line
    while IFS= read -r line; do [[ -n "$line" ]] && extra+=("$line"); done < "$ARGS_FILE"
    while IFS= read -r line; do [[ -n "$line" ]] && spec+=("$line"); done < <(engine_spec_args)
    cmd+=("${extra[@]}" "${spec[@]}" --tool-call-parser gemma4 --reasoning-parser gemma4)
    SERVER_LOG="$dir/server.log"
    printf '%q ' "${cmd[@]}" > "$dir/launch_command.txt"; echo >> "$dir/launch_command.txt"
    [[ -n "${BENCH_DRY_START:-}" ]] && return 0
    if [[ -n "${BENCH_FAKE_SERVER:-}" ]]; then $BENCH_FAKE_SERVER > "$SERVER_LOG" 2>&1 & else "${cmd[@]}" > "$SERVER_LOG" 2>&1 & fi
    SERVER_PID=$!
}
