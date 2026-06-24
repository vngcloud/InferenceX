#!/usr/bin/env bash
# Verify LMCache behaviour on a HYBRID-ATTENTION model (linear/Mamba + full attention).
#
# Hybrid models (Qwen3.5 / Qwen3-Next class) interleave linear-attention (Gated
# DeltaNet) layers — which keep a recurrent STATE cache of a different tensor
# shape — with periodic full-attention layers. The lmcache 0.4.5 / LMCacheConnectorV1
# stack baked into vllm v0.21.0 + sglang v0.5.12 assumes a single unified KV shape,
# so it cannot offload these models. This script reproduces the two distinct
# failure modes documented in REPORT.md §9.
#
#   ENGINE=vllm   ./verify_hybrid.sh   # expect: engine-core crash, server never starts
#   ENGINE=sglang ./verify_hybrid.sh   # expect: serves, but LMCache silently inert
set -euo pipefail

ENGINE="${ENGINE:-vllm}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"   # hybrid: layer_types = 3x linear_attention + 1x full_attention
GPU="${GPU:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo " HYBRID-ATTENTION LMCache verification"
echo "   ENGINE=$ENGINE  MODEL=$MODEL  GPU=$GPU"
echo "==================================================================="

if [ "$ENGINE" = "vllm" ]; then
  MODEL="$MODEL" GPU="$GPU" "$HERE/run_vllm.sh" >/dev/null
  NAME="vllm-lmcache"
  echo "Waiting for startup or crash..."
  until docker logs "$NAME" 2>&1 | grep -E -q \
    "Application startup complete|Engine core initialization failed|failed to convert the KV cache specs"; do
    docker ps -a --filter "name=$NAME" --format '{{.Status}}' | grep -q "Exited" && break
    sleep 5
  done
  echo "--- result ---"
  if docker logs "$NAME" 2>&1 | grep -q "failed to convert the KV cache specs to one unified type"; then
    echo "❌ CRASH (expected): vLLM cannot serve the hybrid model with LMCache."
    docker logs "$NAME" 2>&1 | grep -E "Turning off hybrid kv cache manager|ValueError: Hybrid KV cache|Engine core initialization failed" | sed 's/^/    /'
  elif docker logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
    echo "⚠️ Server started — hybrid support may have changed; re-check LMCache version."
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true

else  # sglang
  MODEL="$MODEL" GPU="$GPU" "$HERE/run_sglang.sh" >/dev/null
  NAME="sglang-lmcache"
  echo "Waiting for startup..."
  until docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; do
    docker ps -a --filter "name=$NAME" --format '{{.Status}}' | grep -q "Exited" && { echo "Server exited unexpectedly"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
    sleep 5
  done
  echo "Server up. Sending cold -> flush radix -> warm (warm can only hit LMCache)..."
  REQ=$(mktemp)
  python3 - "$REQ" "$MODEL" <<'PY'
import json, sys
out, model = sys.argv[1], sys.argv[2]
para=("LMCache stores the key-value tensors produced during prefill so repeated prompt "
      "prefixes are not recomputed; it splits state into fixed-size chunks. ")
body="".join(f"Background document section {i}:\n{para}\n" for i in range(120))
prompt=f"Hybrid verify.\n\n{body}\nQuestion: Summarize in one sentence."
json.dump({"model":model,"messages":[{"role":"user","content":prompt}],
           "max_tokens":8,"temperature":0,"stream":False}, open(out,"w"))
PY
  send(){ curl -s http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d @"$REQ" -o /dev/null -w '%{time_total}'; }
  c=$(send); echo "  cold=${c}s"
  curl -s -X POST http://localhost:30000/flush_cache >/dev/null; sleep 2
  w=$(send); echo "  warm(after flush)=${w}s"; sleep 2
  rm -f "$REQ"
  echo "--- result ---"
  CACHED=$(docker logs "$NAME" 2>&1 | grep "Prefill batch" | tail -4 | grep -oE "#cached-token: [0-9]+" | tail -1)
  HASLMC=$(docker logs "$NAME" 2>&1 | grep -icE "lmcache (stored|retrieve)|retrieve [0-9]+ new tokens" || true)
  echo "    post-flush warm prefill -> $CACHED   (0 == LMCache did NOT restore the flushed prefix)"
  echo "    LMCache store/retrieve log lines: $HASLMC   (0 == LMCache never engaged)"
  echo "    => SGLang serves the hybrid model via its native radix, but LMCache is INERT."
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi
echo "Done. GPU $GPU freed."
