# One tau2 point through InferenceX's agentic replay (mooncake_trace).
point_run() {  # <conc> <repeat> <point_dir>
    local conc="$1" repeat="$2" pdir="$3" entries turns
    export CONC="$conc" DURATION="$DURATION_S" KV_OFFLOADING=none KV_OFFLOAD_BACKEND="" PORT
    export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
    export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="${ENGINE}:"
    # priming: aiperf replays 2 warmup turns per lane (= 2 x conc) unscored, after the cache reset
    export AIPERF_WARMUP_REQUESTS_PER_LANE=2 BENCH_PRIME_N=$(( 2 * conc ))
    export TRACE_SOURCE_FLAG="--custom-dataset-type mooncake_trace --input-file $TAU2_FILE"
    export AIPERF_UNSAFE_OVERRIDE=true ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 IS_MULTINODE=false
    # build_replay_cmd requires it; CI's workflow env sets it from the agentx-fast input (default off -> 0)
    export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}"
    export RESULT_DIR="$pdir/raw" RESULT_FILENAME="bench_${POINT_KEY}" AGENTIC_OUTPUT_DIR="$pdir/raw"
    # same pinned sampling as SPEED-Bench; build_replay_cmd word-splits this value, so no spaces
    export AIPERF_EXTRA_INPUTS='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
    export MODEL="$SERVED_MODEL_NAME"   # aiperf tokenizer + wire name, as the team's tau2 recipes do
    mkdir -p "$RESULT_DIR"
    # dual stop as in the team recipe: every session once (entries) or DURATION, whichever first
    entries=$(python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; print(len({r.get('session_id', i) for i, r in enumerate(rows)}))" "$TAU2_FILE") || return 1
    turns=$(grep -c . "$TAU2_FILE")
    install_agentic_deps || return 1
    build_replay_cmd "$RESULT_DIR" || return 1
    [[ "$REPLAY_CMD" == *"--num-dataset-entries 393"* ]] || { echo "build_replay_cmd no longer emits --num-dataset-entries 393" >&2; return 1; }
    REPLAY_CMD="${REPLAY_CMD/--num-dataset-entries 393/--num-dataset-entries $entries} --request-count $turns"
    run_agentic_replay_and_write_outputs "$RESULT_DIR" || return $?
    cp "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json" "$pdir/raw/agg.json"
    if [[ -d "$RESULT_DIR/aiperf_artifacts" ]]; then rm -rf "$pdir/aiperf"; mv "$RESULT_DIR/aiperf_artifacts" "$pdir/aiperf"; fi
    return 0
}
