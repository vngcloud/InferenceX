# One SPEED-Bench point through InferenceX's run_speedbench_aiperf.
point_run() {  # <conc> <repeat> <point_dir>
    local conc="$1" repeat="$2" pdir="$3"
    local num_prompts=$(( conc * 10 )); (( num_prompts < 64 )) && num_prompts=64
    export CONC="$conc" OSL=1024 ISL=8192 MAX_MODEL_LEN="$CONTEXT_LENGTH" PORT
    export SPEEDBENCH_CONFIG=throughput_8k SPEEDBENCH_CATEGORY="$CATEGORY" SPEEDBENCH_IGNORE_EOS=1
    export SPEEDBENCH_EXTRA_INPUTS='{"temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
    export SPEEDBENCH_NUM_PROMPTS="$num_prompts" BENCH_PROMPT_OFFSET=$(( repeat * num_prompts ))
    export SPEEDBENCH_SERVER_PID="$SERVER_PID" SERVED_MODEL_NAME
    export SPEEDBENCH_META="engine=$ENGINE"
    export RESULT_DIR="$pdir/raw" RESULT_FILENAME="bench_${POINT_KEY}"
    mkdir -p "$RESULT_DIR"
    # aiperf reads the category pool sequentially from the top; a repeat reads a disjoint slice so
    # prompts never repeat across repeats (cache warmth would otherwise leak between them).
    export SPEEDBENCH_DIR="$(dirname "$SPEEDBENCH_FILE")"
    if (( repeat > 0 )); then
        mkdir -p "$pdir/raw/slice"
        python3 - "$SPEEDBENCH_FILE" "$CATEGORY" "$BENCH_PROMPT_OFFSET" "$pdir/raw/slice/throughput_8k.jsonl" <<'PY' || return 1
import json, sys
src, cat, offset, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = [line for line in open(src) if line.strip() and (not cat or json.loads(line).get("category") == cat)]
if len(rows) <= offset:
    sys.exit(f"category pool has {len(rows)} rows, repeat offset {offset} leaves none")
open(dst, "w").writelines(rows[offset:])
PY
        export SPEEDBENCH_DIR="$pdir/raw/slice"
    fi
    install_agentic_deps || return 1
    run_speedbench_aiperf || return $?
    cp "$INFMAX_CONTAINER_WORKSPACE/$RESULT_FILENAME.json" "$pdir/raw/result.json"
    if [[ -d "$RESULT_DIR/speedbench_aiperf" ]]; then rm -rf "$pdir/aiperf"; mv "$RESULT_DIR/speedbench_aiperf" "$pdir/aiperf"; fi
    return 0
}
