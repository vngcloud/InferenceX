# One mooncake-trace point through InferenceX's agentic replay (scenario inferencex-agentx-mvp). Sourced by
# point_tau2.sh and point_livekit.sh, which set the trace-specific knobs below.
#   TRACE_FILE               trace jsonl (one request per line, grouped by session_id)
#   TRACE_WARMUP_PER_LANE    aiperf's unscored warmup requests per lane, after the t* snapshot primers
#   TRACE_RECYCLE            0: every session once or DURATION, whichever first (request cap = trace length)
#                            1: no request cap; lanes draw sessions from a seeded shuffle and recycle them with a
#                               fresh cache-bust marker per play, for the whole DURATION
trace_replay_run() {  # <conc> <point_dir>
    local conc="$1" pdir="$2" entries turns
    export CONC="$conc" DURATION="$DURATION_S" KV_OFFLOADING=none KV_OFFLOAD_BACKEND="" PORT
    export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
    export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="${ENGINE}:"
    export AIPERF_WARMUP_REQUESTS_PER_LANE="$TRACE_WARMUP_PER_LANE"
    export TRACE_SOURCE_FLAG="--custom-dataset-type mooncake_trace --input-file $TRACE_FILE"
    # mooncake_trace defaults to sequential: lanes would replay sessions in file (chronological) order
    (( TRACE_RECYCLE )) && TRACE_SOURCE_FLAG+=" --dataset-sampling-strategy shuffle"
    export AIPERF_UNSAFE_OVERRIDE=true ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 IS_MULTINODE=false
    # build_replay_cmd requires it; CI's workflow env sets it from the agentx-fast input (default off -> 0)
    export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-0}"
    export RESULT_DIR="$pdir/raw" RESULT_FILENAME="bench_${POINT_KEY}" AGENTIC_OUTPUT_DIR="$pdir/raw"
    # same pinned sampling as SPEED-Bench; build_replay_cmd word-splits this value, so no spaces
    export AIPERF_EXTRA_INPUTS='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
    export MODEL="$SERVED_MODEL_NAME"   # aiperf tokenizer + wire name, as the team's tau2 recipes do
    mkdir -p "$RESULT_DIR"
    # streamed: the LiveKit traces are 1.4 GB, one ~100 KB line per request
    entries=$(python3 -c "
import json, sys
print(len({json.loads(l).get('session_id', i) for i, l in enumerate(open(sys.argv[1])) if l.strip()}))" "$TRACE_FILE") || return 1
    turns=$(grep -c . "$TRACE_FILE")
    install_agentic_deps || return 1
    build_replay_cmd "$RESULT_DIR" || return 1
    [[ "$REPLAY_CMD" == *"--num-dataset-entries 393"* ]] || { echo "build_replay_cmd no longer emits --num-dataset-entries 393" >&2; return 1; }
    REPLAY_CMD="${REPLAY_CMD/--num-dataset-entries 393/--num-dataset-entries $entries}"
    (( TRACE_RECYCLE )) || REPLAY_CMD+=" --request-count $turns"
    run_agentic_replay_and_write_outputs "$RESULT_DIR" || return $?
    cp "$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json" "$pdir/raw/agg.json"
    if [[ -d "$RESULT_DIR/aiperf_artifacts" ]]; then rm -rf "$pdir/aiperf"; mv "$RESULT_DIR/aiperf_artifacts" "$pdir/aiperf"; fi
    return 0
}
