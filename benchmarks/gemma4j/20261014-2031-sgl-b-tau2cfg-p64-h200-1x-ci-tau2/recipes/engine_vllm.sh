# vLLM engine launch. Sourced by run_cell.sh after launch.env is loaded.
engine_spec_args() {
    case "$SPEC_KIND" in
        none) ;;
        mtp|dflash|eagle3)
            printf '%s\n' --speculative-config \
                "{\"method\": \"$SPEC_KIND\", \"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": $SPEC_DEPTH}" ;;
        *) echo "unknown SPEC_KIND=$SPEC_KIND" >&2; return 1 ;;
    esac
}
engine_expected_backend() { echo FLASH_ATTN; }
engine_expected_spec_regex() {
    case "$SPEC_KIND" in none) echo none ;; *) echo "SpeculativeConfig\\(method='${SPEC_KIND}'" ;; esac
}
engine_start() {  # <launch_dir>; sets SERVER_PID, SERVER_LOG
    local dir="$1"
    # DP: vLLM keeps --max-num-seqs per replica, so the cell total is divided by DP here; vLLM's
    # --data-parallel-size has its own internal load balancer (router=builtin only).
    local -a dp_args=() extra=() spec=()
    if (( ${DP:-1} > 1 )); then
        [[ "${ROUTER:-}" == "sgl-router" ]] && { echo "vLLM cells support router: builtin only" >&2; return 1; }
        dp_args=(--data-parallel-size "$DP")
    fi
    local -a cmd=(vllm serve "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$PORT"
                  --trust-remote-code --tensor-parallel-size "$TP" "${dp_args[@]}" --max-model-len "$CONTEXT_LENGTH"
                  --max-num-seqs "$(( MAX_RUNNING / ${DP:-1} ))")
    local line  # bash 3.2-compatible (no mapfile): one native arg per line
    while IFS= read -r line; do [[ -n "$line" ]] && extra+=("$line"); done < "$ARGS_FILE"
    while IFS= read -r line; do [[ -n "$line" ]] && spec+=("$line"); done < <(engine_spec_args)
    cmd+=("${extra[@]}" "${spec[@]}" --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4)
    SERVER_LOG="$dir/server.log"
    printf '%q ' "${cmd[@]}" > "$dir/launch_command.txt"; echo >> "$dir/launch_command.txt"
    [[ -n "${BENCH_DRY_START:-}" ]] && return 0
    export VLLM_DISABLE_COMPILE_CACHE=1 VLLM_SERVER_DEV_MODE=1   # dev mode exposes /reset_prefix_cache
    if [[ -n "${BENCH_FAKE_SERVER:-}" ]]; then $BENCH_FAKE_SERVER > "$SERVER_LOG" 2>&1 & else "${cmd[@]}" > "$SERVER_LOG" 2>&1 & fi
    SERVER_PID=$!
}
