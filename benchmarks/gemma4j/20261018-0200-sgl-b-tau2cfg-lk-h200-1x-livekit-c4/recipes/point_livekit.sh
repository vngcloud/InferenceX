# One LiveKit 111868 point (CATEGORY = trace variant v1 | v2): CONC lanes = concurrent rooms (CCU), each replaying
# whole rooms with their logged think time, no request cap, 10 warmup requests per lane (as the team's
# custom-agentic-livekit111868 scenarios).
source "$BENCH_DIR/recipes/trace_replay.sh"
point_run() {  # <conc> <repeat> <point_dir>
    local file
    case "$CATEGORY" in v1) file="$LIVEKIT_V1_FILE" ;; v2) file="$LIVEKIT_V2_FILE" ;; *) echo "livekit: unknown variant '$CATEGORY'" >&2; return 1 ;; esac
    TRACE_FILE="$file" TRACE_WARMUP_PER_LANE=10 TRACE_RECYCLE=1 trace_replay_run "$1" "$3"
}
