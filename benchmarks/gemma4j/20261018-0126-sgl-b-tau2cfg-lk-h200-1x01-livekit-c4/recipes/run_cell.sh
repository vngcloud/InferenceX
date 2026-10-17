#!/usr/bin/env bash
# In-container driver (spec §5/§6): one server per launch, every conc point of the launch back to back.
# Layout: $BENCH_DIR/{run.env, recipes/, launches/<key>/{launch.env,args.txt,env.txt}} -> points/<key>/raw/*, status.json
set -o pipefail
BENCH_DIR="${BENCH_DIR:-/bench}"
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/workspace}"
export INFERENCEX_REPO_ROOT="$INFMAX_CONTAINER_WORKSPACE"
BENCH_LIB="${BENCH_LIB:-$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh}"
cd "$INFMAX_CONTAINER_WORKSPACE" || exit 1
set -a; source "$BENCH_DIR/run.env"; set +a
# torch (vLLM >= 0.25) calls getpass.getuser() at import: a container uid without a passwd entry needs USER/LOGNAME
export USER="${USER:-bench}" LOGNAME="${LOGNAME:-bench}"
# hgx-h200-01 Slurm counts GPUs but not indices (every 1-GPU job gets CUDA_VISIBLE_DEVICES=0): pin the lane's GPU
[[ -n "${BENCH_GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$BENCH_GPU"
# CI sources these defaults before every recipe; the agentic helpers require them
for rs in "$INFMAX_CONTAINER_WORKSPACE/benchmarks/runtime_settings.sh"; do [[ -f "$rs" ]] && source "$rs"; done
export AIPERF_PYTHON_VERSION="${AIPERF_PYTHON_VERSION:-3.11}"
source "$BENCH_LIB"
# benchmark_lib.sh points the uv cache at a per-job tmp dir; keep the persistent mount when present
[[ -d /mnt/uv-cache ]] && export AIPERF_UV_CACHE_DIR=/mnt/uv-cache
source "$BENCH_DIR/recipes/common.sh"

STATUS="$BENCH_DIR/status.json"
status_set() {  # <point_key> <status> [reason]
    python3 - "$STATUS" "$RUN_ID" "$1" "$2" "${3:-}" <<'PY'
import json, os, sys
path, run_id, key, st, reason = sys.argv[1:6]
d = json.load(open(path)) if os.path.exists(path) else {"run_id": run_id, "status": "running", "points": {}}
d["points"][key] = {"status": st, "fail_reason": reason or None}
vals = [p["status"] for p in d["points"].values()]
d["status"] = "running" if any(v in ("pending", "running") for v in vals) else (
    "done" if all(v == "done" for v in vals) else ("partial" if any(v == "done" for v in vals) else "failed"))
json.dump(d, open(path, "w"), indent=2)
PY
}
log_tail() { tail -n 40 "$1" 2>/dev/null | tr '\n' ' ' | cut -c1-2000; }
point_key() { local k="${LAUNCH_KEY}-c$1"; (( $2 > 0 )) && k="$k-r$2"; echo "$k"; }

# Register every point as pending first, so a crash leaves an honest status.json.
for lenv in "$BENCH_DIR"/launches/*/launch.env; do
    ( set -a; source "$lenv"; set +a
      for pr in $POINTS; do status_set "$(point_key "${pr%%:*}" "${pr##*:}")" pending; done )
done

GPU_DESC=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sort | uniq -c | tr -s ' ' | paste -sd ';' -)
GPU_DESC="${GPU_DESC:-unknown}${BENCH_GPU:+;index=$BENCH_GPU}"

RC=0
for lenv in "$BENCH_DIR"/launches/*/launch.env; do
    set -a; source "$lenv"; set +a
    ldir="$(dirname "$lenv")"
    [[ -f "$ENV_FILE" ]] && { set -a; source "$ENV_FILE"; set +a; }
    source "$BENCH_DIR/recipes/engine_${ENGINE}.sh"
    source "$BENCH_DIR/recipes/point_${WORKLOAD}.sh"
    WCAT=""   # warmup row filter: SPEED-Bench rows carry "category"; a LiveKit category is a trace variant
    case "$WORKLOAD" in
        speedbench) WCAT="${CATEGORY:-}"; IGN=1; WARM="$SPEEDBENCH_FILE"; DSREPO=Noridom1/speed-bench-prepared; DSREV="$SPEEDBENCH_REV"; DSSHA="$SPEEDBENCH_SHA" ;;
        tau2) IGN=0; WARM="$TAU2_FILE"; DSREPO=thangquang09/customer-service-agent-traces; DSREV="$TAU2_REV"; DSSHA="$TAU2_SHA" ;;
        livekit) v="LIVEKIT_$(printf '%s' "$CATEGORY" | tr '[:lower:]' '[:upper:]')"; f="${v}_FILE" r="${v}_REV" h="${v}_SHA"
                 IGN=0; WARM="${!f}"; DSREPO=Green-Lab/2026-10-16-111868-agentic-traces; DSREV="${!r}"; DSSHA="${!h}" ;;
    esac
    DSFILE="$(basename "$WARM")"
    export BENCH_ENGINE="$ENGINE" BENCH_PORT="$PORT" BENCH_MODEL="$SERVED_MODEL_NAME" BENCH_OSL=1024
    export BENCH_IGNORE_EOS="$IGN" BENCH_WARMUP_FILE="$WARM" BENCH_CATEGORY="$WCAT" BENCH_PRIME_N=0
    LAUNCH_MODEL="$MODEL"

    echo "===== launch $LAUNCH_KEY ($ENGINE tp=$TP dp=${DP:-1} router=${ROUTER:-none} max_running=$MAX_RUNNING) ====="
    engine_start "$ldir"
    launch_ok=1
    # wait_for_ready `exit`s when the server dies: probe in a subshell (with a deadline) so a dead launch fails
    # its points instead of killing the driver; then register the healthy server for run_server_client.
    ( wait_for_ready --endpoint "http://localhost:${PORT}/health" --log "$SERVER_LOG" --pid "$SERVER_PID" \
        --timeout "${BENCH_READY_TIMEOUT_S:-3600}" ) || launch_ok=0
    (( launch_ok )) && { wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" || launch_ok=0; }
    if (( launch_ok )) && [[ -z "${BENCH_FAKE_SERVER:-}" ]]; then
        bench_qualify_log "$SERVER_LOG" "$(engine_expected_backend)" "$(engine_expected_spec_regex)" "$CONTEXT_LENGTH" || launch_ok=0
    fi
    if (( launch_ok )); then
        bench_metrics_snapshot "$ldir/metrics_prewarm.txt" || launch_ok=0
        (( launch_ok )) && { bench_warmup || launch_ok=0; }
        # proof that /metrics covers every replica: all warmup requests show up in the counters
        (( launch_ok )) && { bench_check_request_count "$ldir/metrics_prewarm.txt" "${BENCH_WARMUP_N:-8}" || launch_ok=0; }
    fi

    for pr in $POINTS; do
        conc="${pr%%:*}"; repeat="${pr##*:}"
        POINT_KEY="$(point_key "$conc" "$repeat")"
        pdir="$BENCH_DIR/points/$POINT_KEY"; mkdir -p "$pdir/raw"
        export BENCH_POINT_DIR="$pdir" POINT_KEY
        if (( ! launch_ok )); then status_set "$POINT_KEY" failed "launch failed: $(log_tail "$SERVER_LOG")"; RC=1; continue; fi
        status_set "$POINT_KEY" running
        export BENCH_PROMPT_OFFSET=0
        [[ "$WORKLOAD" == "speedbench" ]] && BENCH_PROMPT_OFFSET=$(( repeat * (conc * 10 < 64 ? 64 : conc * 10) ))
        [[ "$WORKLOAD" == "tau2" ]] && export BENCH_PRIME_N=$(( 2 * conc ))
        [[ "$WORKLOAD" == "livekit" ]] && export BENCH_PRIME_N=$(( 11 * conc ))   # 1 snapshot primer + 10 per lane
        export BENCH_META_JSON
        BENCH_META_JSON=$(python3 - "$RUN_ID" "$CELL" "$POINT_KEY" "$ENGINE" "$IMAGE" "$HARDWARE" "$EXECUTOR" \
            "$WORKLOAD" "${CATEGORY:-}" "$conc" "$repeat" "$MAX_RUNNING" "$RECIPE_SHA" "$(hostname -s)" \
            "$ldir/launch_command.txt" "$DSREPO" "$DSREV" "$DSSHA" "$DSFILE" "$TP" "${DP:-1}" "${ROUTER:-}" \
            "${INFERENCEX_SHA:-unknown}" "$GPU_DESC" <<'PY'
import json, sys
a = sys.argv
print(json.dumps(dict(inferencex_sha=a[23], gpu=a[24],
    run_id=a[1], cell=a[2], cell_key=a[3], engine=a[4], image=a[5], hardware=a[6], executor=a[7], workload=a[8],
    category=a[9] or None, conc=int(a[10]), repeat=int(a[11]), max_running=int(a[12]), recipe_sha=a[13], node=a[14],
    launch_command=open(a[15]).read().strip(), dataset=dict(repo=a[16], revision=a[17], file=a[19], sha256=a[18]),
    topology=dict(tp=int(a[20]), dp=int(a[21]), router=a[22] or None, max_running_per_replica=int(a[12]) // int(a[21])))))
PY
)
        t0=$(date +%s)
        if ! bench_cache_reset; then status_set "$POINT_KEY" failed "cache reset unverified"; RC=1; continue; fi
        bench_write_meta '{}'
        bench_metrics_snapshot "$pdir/raw/metrics_before.txt"
        if point_run "$conc" "$repeat" "$pdir"; then
            bench_metrics_snapshot "$pdir/raw/metrics_after.txt"
            python3 - "$pdir/raw/point_meta.json" "$(( $(date +%s) - t0 ))" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d["duration_s"] = int(sys.argv[2]); json.dump(d, open(p, "w"), indent=2)
PY
            status_set "$POINT_KEY" done
        else
            bench_metrics_snapshot "$pdir/raw/metrics_after.txt" || true
            status_set "$POINT_KEY" failed "point_run failed: $(log_tail "$SERVER_LOG")"; RC=1
        fi
        MODEL="$LAUNCH_MODEL"
    done
    bench_stop_server "$SERVER_PID" || RC=1
    sleep "${BENCH_LAUNCH_GAP_S:-5}"
done
exit $RC
