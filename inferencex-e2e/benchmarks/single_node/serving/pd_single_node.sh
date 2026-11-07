#!/usr/bin/env bash
# Single-node prefill/decode disaggregation for a serving YAML whose router is a Dynamo frontend: bench.sh runs
# the serve: command once with every allocated GPU, and this script splits them in two halves, the first for one
# dynamo.sglang prefill worker and the second for one decode worker. The frontend finds both through the file
# discovery that job.sh sets up, and routes each request to the prefill worker, then to the decode worker.
#
#   serve: bash benchmarks/single_node/serving/pd_single_node.sh <args for both> --prefill <args> --decode <args>
#
# Per worker the script adds --disaggregation-mode and a free --dist-init-addr; the prefill worker also gets a
# free --disaggregation-bootstrap-port and, if KV_EVENTS_PREFILL=1, a free kv-events ZMQ endpoint. The decode
# worker gets its own DYN_SYSTEM_PORT (bench.sh scrapes the prefill worker's). Each worker logs to
# $RESULT_DIR/server_{prefill,decode}.log; when either exits, the other is stopped and the script exits with the
# first exit code, so bench.sh sees the engine die.
set -uo pipefail
: "${RESULT_DIR:?}"

common=()
prefill=()
decode=()
role=common
for arg in "$@"; do
    case "$arg" in
        --prefill) role=prefill; continue ;;
        --decode) role=decode; continue ;;
    esac
    case "$role" in
        common) common+=("$arg") ;;
        prefill) prefill+=("$arg") ;;
        decode) decode+=("$arg") ;;
    esac
done

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
if (( ${#gpus[@]} < 2 || ${#gpus[@]} % 2 )); then
    echo "ERROR: pd_single_node.sh needs an even number of GPUs, got ${#gpus[@]}" >&2
    exit 1
fi
half=$(( ${#gpus[@]} / 2 ))
prefill_gpus="$(IFS=,; echo "${gpus[*]:0:half}")"
decode_gpus="$(IFS=,; echo "${gpus[*]:half:half}")"

free_ports() {  # free_ports <n>: n distinct free TCP ports in 20000-32767, one per line
    python3 - "$1" <<'PY'
import random, socket, sys
want, found = int(sys.argv[1]), []
for port in random.sample(range(20000, 32768), 4000):
    if len(found) == want:
        break
    with socket.socket() as probe:
        try:
            probe.bind(("", port))
        except OSError:
            continue
    found.append(port)
print("\n".join(map(str, found)))
PY
}
mapfile -t ports < <(free_ports 6)
(( ${#ports[@]} == 6 )) || { echo "ERROR: no free ports for the PD workers" >&2; exit 1; }
prefill+=(--disaggregation-mode=prefill --dist-init-addr="127.0.0.1:${ports[0]}"
          --disaggregation-bootstrap-port="${ports[1]}")
decode+=(--disaggregation-mode=decode --dist-init-addr="127.0.0.1:${ports[2]}")
if [[ "${KV_EVENTS_PREFILL:-0}" == 1 ]]; then
    prefill+=(--kv-events-config="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${ports[3]}\"}")
fi

launch() {  # launch <role> <gpus> <system port> <args...>
    local name=$1 visible=$2 system_port=$3
    shift 3
    printf '%q ' "$@" > "$RESULT_DIR/server_command_$name.txt"
    printf '\n' >> "$RESULT_DIR/server_command_$name.txt"
    echo "pd_single_node: $name on GPUs $visible, DYN_SYSTEM_PORT=$system_port, log server_$name.log"
    CUDA_VISIBLE_DEVICES="$visible" DYN_SYSTEM_PORT="$system_port" "$@" > "$RESULT_DIR/server_$name.log" 2>&1 &
}

launch prefill "$prefill_gpus" "${DYN_SYSTEM_PORT:-${ports[5]}}" "${common[@]}" "${prefill[@]}"
prefill_pid=$!
launch decode "$decode_gpus" "${ports[4]}" "${common[@]}" "${decode[@]}"
decode_pid=$!
trap 'kill "$prefill_pid" "$decode_pid" 2>/dev/null || true' EXIT TERM INT

wait -n "$prefill_pid" "$decode_pid"
rc=$?
for name in prefill decode; do
    pid_var="${name}_pid"
    if ! kill -0 "${!pid_var}" 2>/dev/null; then
        echo "pd_single_node: $name worker exited ($rc); last lines of server_$name.log:" >&2
        tail -n 60 "$RESULT_DIR/server_$name.log" >&2
    fi
done
exit "$rc"
