#!/usr/bin/env bash
# Engine step of one serving-YAML point (started by job.sh in the engine container, cwd /workspace):
# prepare the client, start the engine replica(s), wait until they (and the router) serve, run the client.
set -eo pipefail
: "${RESULT_DIR:?}" "${SERVE_CMD:?}" "${PORT:?}" "${MODEL:?}"
export PYTHONPATH=/workspace AGENTIC_OUTPUT_DIR=/workspace
mkdir -p "$RESULT_DIR"
printf '%s' "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"

expand_storage() {
    local text="$1"
    [[ -n "${STORAGE_MASTER_PORT:-}" ]] && text="${text//\{\{storage_master_port\}\}/$STORAGE_MASTER_PORT}"
    [[ -n "${STORAGE_META_PORT:-}" ]] && text="${text//\{\{storage_meta_port\}\}/$STORAGE_META_PORT}"
    printf '%s' "$text"
}
# SERVE_ENV values may hold tokens: export them without echoing the values.
while IFS= read -r assignment; do
    [[ -z "$assignment" ]] && continue
    assignment="$(expand_storage "$assignment")"
    export "$assignment"
    echo "serve env: ${assignment%%=*}"
done <<< "${SERVE_ENV:-}"

nvidia-smi || true
python3 -m infx.serving.client prepare

dynamo=""
[[ -n "${ROUTER_CMD:-}" && "${ROUTER_PROFILE:-}" == dynamo ]] && dynamo=1
if [[ -n "$dynamo" ]]; then
    # Dynamo engines have no HTTP port: each replica serves /health and /metrics on its system
    # port, and the frontend serves $PORT.
    read -r -a ports <<< "$DYN_SYSTEM_PORTS"
    read -r -a kv_ports <<< "$DYN_KV_EVENT_PORTS"
    urls="http://localhost:$PORT/metrics"
    for port in "${ports[@]}"; do urls+=",http://localhost:$port/metrics"; done
    export AIPERF_SERVER_METRICS_URLS="$urls"
elif [[ -n "${ROUTER_CMD:-}" ]]; then
    read -r -a ports <<< "$ENGINE_PORTS"
    urls="http://localhost:${ROUTER_METRICS_PORT:-$PORT}/metrics"
    for port in "${ports[@]}"; do urls+=",http://localhost:$port/metrics"; done
    export AIPERF_SERVER_METRICS_URLS="$urls"
else
    ports=("$PORT")
    export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
replicas=${#ports[@]}
if (( ${#gpus[@]} == 0 || ${#gpus[@]} % replicas )); then
    echo "ERROR: ${#gpus[@]} GPUs do not split into $replicas replicas" >&2
    exit 1
fi
per=$(( ${#gpus[@]} / replicas ))

eval "set -- $SERVE_CMD"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for ((replica = 0; replica < replicas; replica++)); do
    suffix=""
    (( replica == 0 )) || suffix=".$replica"
    launch_env=(CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpus[*]:replica*per:per}")")
    if [[ -n "$dynamo" ]]; then
        launch_env+=(DYN_SYSTEM_PORT="${ports[replica]}")
        command=("$@")
        # The engine's KV event publisher binds a fixed port unless told otherwise: one per replica.
        # dynamo.sglang publishes no KV events at all without --kv-events-config.
        if [[ " $* " != *" --kv-events-config"[\ =]* ]]; then
            if [[ "$3" == dynamo.vllm ]]; then
                command+=(--kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${kv_ports[replica]}\"}")
            elif [[ "$3" == dynamo.sglang ]]; then
                command+=(--kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${kv_ports[replica]}\"}")
            fi
        fi
    else
        command=("$@" --host 0.0.0.0 --port "${ports[replica]}")
    fi
    printf '%q ' "${command[@]}" > "$RESULT_DIR/server_command$suffix.txt"
    printf '\n' >> "$RESULT_DIR/server_command$suffix.txt"
    env "${launch_env[@]}" "${command[@]}" > "$RESULT_DIR/server$suffix.log" 2>&1 &
    pids+=("$!")
done

# A Dynamo frontend lists the model once the first engine registered: with several Dynamo
# replicas, wait for each one's system port /health too.
if [[ -z "$dynamo" ]] || (( replicas > 1 )); then
    for ((replica = 0; replica < replicas; replica++)); do
        suffix=""
        (( replica == 0 )) || suffix=".$replica"
        python3 -m infx.bench wait --url "http://localhost:${ports[replica]}/health" \
            --pid "${pids[replica]}" --log "$RESULT_DIR/server$suffix.log"
    done
fi
if [[ -n "${ROUTER_CMD:-}" ]]; then
    # The router is ready once it lists the model (a Dynamo frontend only does after the engine registered).
    model="${SERVED_MODEL_NAME:-$MODEL}"
    deadline=$(( SECONDS + 3600 ))  # engines are already healthy here; only the router can be this slow
    until python3 - "$PORT" "$model" <<'PY' 2>/dev/null
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as response:
    ids = [entry.get("id") for entry in json.load(response).get("data", [])]
sys.exit(0 if model in ids else 1)
PY
    do
        for ((replica = 0; replica < replicas; replica++)); do
            if ! kill -0 "${pids[replica]}" 2>/dev/null; then
                suffix=""
                (( replica == 0 )) || suffix=".$replica"
                echo "ERROR: engine replica $replica exited before the router listed $model; last lines of server$suffix.log:" >&2
                tail -n 50 "$RESULT_DIR/server$suffix.log" >&2
                exit 1
            fi
        done
        if [[ -f "$RESULT_DIR/.router_exited" ]]; then
            echo "ERROR: router step exited ($(cat "$RESULT_DIR/.router_exited")) before it listed $model; last lines of router.log:" >&2
            tail -n 50 "$RESULT_DIR/router.log" >&2
            exit 1
        fi
        if (( SECONDS > deadline )); then
            echo "ERROR: router did not list $model within 3600 s; last lines of router.log:" >&2
            tail -n 50 "$RESULT_DIR/router.log" >&2
            exit 1
        fi
        sleep 5
    done
fi

# the engine keeps its offline switches; the client must reach the Hub for datasets and the tokenizer
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
python3 -m infx.serving.client run
