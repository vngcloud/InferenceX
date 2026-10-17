# One tau2 point: the customer-service trace replayed once (every session, or DURATION), 2 warmup turns per lane.
source "$BENCH_DIR/recipes/trace_replay.sh"
point_run() {  # <conc> <repeat> <point_dir>
    TRACE_FILE="$TAU2_FILE" TRACE_WARMUP_PER_LANE=2 TRACE_RECYCLE=0 trace_replay_run "$1" "$3"
}
