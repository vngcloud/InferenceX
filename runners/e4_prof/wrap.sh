#!/usr/bin/env bash
# e4 profiling wrapper around an InferenceX serve command (e4A/e4B frontend A/B):
#   serve: |
#     bash /workspace/runners/e4_prof/wrap.sh <mode> python3 -m sglang.launch_server ...
# Modes: sample (sampler only), torch, pyspy, nsys, gdbpmp (sampler + one profile window, see
# agent.py). Output goes to $RESULT_DIR/e4prof/, which the workflow uploads with results/**.
# Tunables (YAML env): E4_WINDOW_DELAY_S (30), E4_WINDOW_S (60), E4_TORCH_STEPS (40).
set -uo pipefail

mode="$1"
shift
case "$mode" in
    sample|torch|pyspy|nsys|gdbpmp) ;;
    *) echo "[e4prof] unknown mode '$mode'" >&2; exit 2 ;;
esac
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$RESULT_DIR/e4prof"
rm -rf "$out"
mkdir -p "$out"
# First on the path: py/sitecustomize.py chains the one in runners/sglang_gemma4_bcg.
export PYTHONPATH="$here/py${PYTHONPATH:+:$PYTHONPATH}"

launch=("$@")
case "$mode" in
    torch)
        export E4_TORCH_TRIGGER="$out/torch.trigger" E4_TORCH_DIR="$out/torch"
        export E4_TORCH_STEPS="${E4_TORCH_STEPS:-40}" E4_TORCH_ID="e4prof-$(date +%s)"
        export SGLANG_PROFILE_RECORD_SHAPES=false
        mkdir -p "$E4_TORCH_DIR"
        ;;
    nsys)
        export SGLANG_ENABLE_NVTX_SCHEDULER=1
        # perf_event_paranoid=4 on these nodes: no CPU sampling or context-switch tracing.
        launch=(nsys launch --session-new=e4prof --trace=cuda,nvtx,osrt --sample=none
                --cpuctxsw=none --cuda-graph-trace=node "$@")
        ;;
esac

echo "[e4prof] mode=$mode out=$out launch: ${launch[*]}"
"${launch[@]}" &
server_pid=$!
trap 'kill -TERM "$server_pid" 2>/dev/null' TERM INT
python3 "$here/agent.py" --mode "$mode" --server-pid "$server_pid" --out "$out" \
    --benchmark-log "$RESULT_DIR/benchmark.log" >> "$out/agent.log" 2>&1 &
wait "$server_pid"
rc=$?
# wait returns early when a trapped signal arrives; wait again for the real exit status.
if kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    rc=$?
fi
exit "$rc"
