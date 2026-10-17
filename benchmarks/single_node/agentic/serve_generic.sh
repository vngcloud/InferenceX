#!/usr/bin/env bash
set -eo pipefail
set -x

# /data on this node has no ACL support: keep the shared HF cache writable for every runner uid.
umask 0000

# Generic engine recipe for the serving-YAML path
# (docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md).
# The launcher runs this instead of a per-arm recipe when SERVE_CMD is set.
# Everything arm-specific comes from infx.serving.to_matrix via env:
#   SERVE_CMD          shell-quoted engine argv, without --host/--port
#   SERVE_ENV          KEY=VALUE lines exported before the engine starts
#   SERVED_MODEL_NAME  --served-model-name from the command, if any
#   SERVING_SCENARIO   scenarios/<name>.yaml (aiperf workload, incl. duration)
#   SERVING_LOADER     dataset of that scenario: public loader alias or scenarios/datasets/lock.yaml key
#   SERVING_YAML       the user's serving YAML, saved next to the results
#   ROUTER_CMD         set on the router path: the router runs in a sidecar (serve_router.sh); with ROUTER_PROFILE=dynamo the engine has no HTTP port, otherwise it listens on ENGINE_PORT
#   ROUTER_PROFILE     dynamo | generic (launcher)
#   ROUTER_METRICS_PORT generic router: its /metrics port, empty = the router port
#   ENGINE_PORT        generic router: the engine's HTTP port (launcher)
#   DYN_SYSTEM_PORT    Dynamo engine metrics/health port (launcher)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL CONC PORT RESULT_DIR FRAMEWORK SERVE_CMD SERVING_SCENARIO SERVING_LOADER

mkdir -p "$RESULT_DIR"
# xtrace off here: SERVING_YAML and SERVE_ENV may hold tokens (env: values);
# log only the exported names.
{ set +x; } 2>/dev/null
printf '%s' "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"
if [[ -n "${SERVE_ENV:-}" ]]; then
    while IFS= read -r assignment; do
        if [[ -n "$assignment" ]]; then
            # Storage sidecar placeholders ({{storage_master_port}}, {{storage_meta_port}})
            # expand here so the engine env can reference ports the launcher only allocates
            # at runtime (e.g. MOONCAKE_MASTER=127.0.0.1:{{storage_master_port}}).
            assignment="$(expand_storage_placeholders "$assignment")"
            export "$assignment"
            echo "serve env: ${assignment%%=*}"
        fi
    done <<< "$SERVE_ENV"
fi
set -x

nvidia-smi

install_agentic_deps
# Scenario dataset: a lock-pinned file (custom-agentic-*: download, sha256 check,
# IX_CUSTOM_DATASET_FILE for the YAML) or a public Weka loader (resolve_trace_source
# validates and pre-downloads it; the --public-dataset flag it builds is unused,
# the scenario YAML names the dataset).
custom_rc=0
is_custom_dataset "$SERVING_LOADER" || custom_rc=$?
case "$custom_rc" in
    # the YAML env's offline flags are for the engine; the dataset fetch must reach the Hub
    # (a prefix assignment on a function call is temporary in bash non-posix mode)
    0) HF_HUB_OFFLINE= TRANSFORMERS_OFFLINE= HF_DATASETS_OFFLINE= fetch_custom_dataset "$SERVING_LOADER" ;;
    1) export WEKA_LOADER_OVERRIDE="$SERVING_LOADER"
       resolve_trace_source ;;
    *) echo "ERROR: cannot tell whether $SERVING_LOADER is a lock-pinned dataset (exit $custom_rc)" >&2; exit 1 ;;
esac
# the user's env may set HF_HUB_OFFLINE=1 for the engine; the pre-download must reach the Hub
HF_HUB_OFFLINE= TRANSFORMERS_OFFLINE= HF_DATASETS_OFFLINE= "$AIPERF_HF_CLI" download "$MODEL"

if [[ -n "${ROUTER_CMD:-}" ]]; then
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        check_env_vars DYN_SYSTEM_PORT
        # Frontend (dynamo_frontend_*) + engine (sglang:* via the Dynamo system port).
        export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics,http://localhost:${DYN_SYSTEM_PORT}/metrics"
    else
        check_env_vars ENGINE_PORT
        # Router (its own metrics port, else the router port) + every engine's own /metrics.
        metrics_urls="http://localhost:${ROUTER_METRICS_PORT:-$PORT}/metrics"
        for engine_port in ${ENGINE_PORTS:-$ENGINE_PORT}; do
            metrics_urls+=",http://localhost:${engine_port}/metrics"
        done
        export AIPERF_SERVER_METRICS_URLS="$metrics_urls"
    fi
else
    export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
fi
case "$FRAMEWORK" in
    vllm) export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:" ;;
    sglang) export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:" ;;
    *) echo "ERROR: serve_generic.sh supports vllm and sglang, got FRAMEWORK=$FRAMEWORK" >&2; exit 1 ;;
esac

SERVER_LOG="$RESULT_DIR/server.log"
eval "set -- $SERVE_CMD"
if [[ "${REPLICAS:-1}" -gt 1 ]]; then
    # Serving-YAML replicas (spec 2026-10-16): N copies of SERVE_CMD, each on its own GPU slice and
    # ENGINE_PORTS entry. Replica 0 keeps server.log / server_command.txt: the vLLM server-metrics
    # adapter reads "GPU KV cache size" from server.log (replicas are identical).
    read -r -a replica_ports <<< "${ENGINE_PORTS:-}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=, read -r -a replica_gpus <<< "$CUDA_VISIBLE_DEVICES"
    else
        mapfile -t replica_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
    if (( ${#replica_ports[@]} != REPLICAS || ${#replica_gpus[@]} == 0 || ${#replica_gpus[@]} % REPLICAS )); then
        echo "ERROR: replicas $REPLICAS needs $REPLICAS ports in ENGINE_PORTS (got '${ENGINE_PORTS:-}') and a GPU count divisible by $REPLICAS (got ${#replica_gpus[@]})" >&2
        exit 1
    fi
    gpus_per_replica=$(( ${#replica_gpus[@]} / REPLICAS ))
    replica_pids=()
    for ((replica = 0; replica < REPLICAS; replica++)); do
        suffix=""
        (( replica == 0 )) || suffix=".$replica"
        replica_cmd=("$@" --host 0.0.0.0 --port "${replica_ports[replica]}")
        write_command "$RESULT_DIR/server_command${suffix}.txt" "${replica_cmd[@]}"
        CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${replica_gpus[*]:replica*gpus_per_replica:gpus_per_replica}")" \
            "${replica_cmd[@]}" > "$RESULT_DIR/server${suffix}.log" 2>&1 &
        replica_pids+=("$!")
    done
    for ((replica = 0; replica < REPLICAS; replica++)); do
        suffix=""
        (( replica == 0 )) || suffix=".$replica"
        wait_for_server_ready --port "${replica_ports[replica]}" --server-log "$RESULT_DIR/server${suffix}.log" \
            --server-pid "${replica_pids[replica]}"
    done
    # ponytail: the server watch follows replica 0 only; another replica dying shows up as failed
    # requests (failed-request threshold 0.1 stops the run). Watch every PID if that proves too slow.
    SERVER_PID="${replica_pids[0]}"
else
    if [[ -n "${ROUTER_CMD:-}" && "${ROUTER_PROFILE:-}" == dynamo ]]; then
        SERVER_CMD=("$@")
    elif [[ -n "${ROUTER_CMD:-}" ]]; then
        SERVER_CMD=("$@" --host 0.0.0.0 --port "$ENGINE_PORT")
    else
        SERVER_CMD=("$@" --host 0.0.0.0 --port "$PORT")
    fi
    write_command "$RESULT_DIR/server_command.txt" "${SERVER_CMD[@]}"
    "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
fi

if [[ -n "${ROUTER_CMD:-}" ]]; then
    wait_for_frontend_model --port "$PORT" --model "${SERVED_MODEL_NAME:-$MODEL}" \
        --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --router-exited "$RESULT_DIR/.router_exited"
else
    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
fi

# offline mode is for the engine (it already inherited it); the aiperf client must reach the Hub to load the trace dataset
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
build_replay_cmd_yaml "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
