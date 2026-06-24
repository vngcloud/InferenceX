#!/usr/bin/env bash
# Engine-agnostic LMCache cache-hit test + metrics scrape.
#
# Sends a long shared-prefix prompt as a COLD run (KV gets stored into LMCache),
# then a WARM run of the same prompt. For SGLang we flush the GPU radix between
# the two runs so the warm prefix can ONLY come from LMCache. We scrape the
# right metrics endpoint for each engine and report the cache-hit signal.
#
#   ENGINE=vllm   ./test_and_scrape.sh     # vLLM on :8100,  LMCache native metrics on :7001
#   ENGINE=sglang ./test_and_scrape.sh     # SGLang on :30000 (metrics merged into same port)
set -euo pipefail

ENGINE="${ENGINE:-vllm}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
REPEAT="${REPEAT:-120}"          # ~5k-token prompt; comfortably exceeds chunk_size
MAX_TOKENS="${MAX_TOKENS:-8}"    # tiny: keep prefill (what LMCache accelerates) dominant
NONCE="${NONCE:-$(date +%s)-$RANDOM}"

if [ "$ENGINE" = "sglang" ]; then
  HOST="${HOST:-http://localhost:30000}"
  METRICS="${METRICS:-http://localhost:30000/metrics}"   # lmcache + sglang metrics share this port
  FLUSH_URL="$HOST/flush_cache"
else
  HOST="${HOST:-http://localhost:8100}"
  METRICS="${METRICS:-http://localhost:7001/metrics}"    # LMCache internal API server
  FLUSH_URL=""                                           # vLLM prefix cache disabled in run_vllm.sh
fi

REQ=$(mktemp)
python3 - "$REQ" "$MODEL" "$REPEAT" "$MAX_TOKENS" "$NONCE" <<'PY'
import json, sys
out, model, repeat, mt, nonce = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
para = ("LMCache stores the key-value tensors produced during prefill so repeated prompt "
        "prefixes are not recomputed; it splits state into fixed-size chunks and loads the "
        "longest matching prefix straight into memory instead of recomputing it. ")
body = "".join(f"Background document section {i}:\n{para}\n" for i in range(repeat))
prompt = f"Document set {nonce}.\n\n{body}\nQuestion: Summarize in one sentence."
json.dump({"model": model, "messages": [{"role": "user", "content": prompt}],
           "max_tokens": mt, "temperature": 0, "stream": False}, open(out, "w"))
print(f"prompt ~{len(prompt)//4} tokens", file=sys.stderr)
PY

send() { curl -s "$HOST/v1/chat/completions" -H "Content-Type: application/json" \
           -d @"$REQ" -o /dev/null -w "%{time_total}"; }

scrape() {
  if [ "$ENGINE" = "sglang" ]; then
    echo "  [sglang native]"; curl -s "$METRICS" | grep -E "^sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total)" | grep -v 'le="' | sed 's/^/    /' || true
    echo "  [lmcache native]"; curl -s "$METRICS" | grep -E "^lmcache:(num_hit_tokens_total|num_stored_tokens_total|retrieve_hit_rate)" | grep -v 'le="' | sed 's/^/    /' || true
  else
    echo "  [vllm external KV-connector view]"; curl -s "$HOST/metrics" | grep -E "^vllm:external_prefix_cache_(hits|queries)_total" | sed 's/^/    /' || true
    echo "  [lmcache native :7001]"; curl -s "$METRICS" | grep -E "^lmcache:(num_hit_tokens_total|num_stored_tokens_total|retrieve_hit_rate|num_requested_tokens_total)" | grep -v 'le="' | sed 's/^/    /' || true
  fi
}

echo "== ENGINE=$ENGINE MODEL=$MODEL HOST=$HOST =="
echo "== metrics BEFORE =="; scrape
echo "== COLD run (stores KV into LMCache) =="; c=$(send); echo "  total=${c}s"
if [ -n "$FLUSH_URL" ]; then echo "== flush GPU radix (LMCache keeps its copy) =="; curl -s -X POST "$FLUSH_URL" >/dev/null; sleep 2; fi
echo "== WARM run (prefix served from LMCache) =="; w=$(send); echo "  total=${w}s"
sleep 3
echo "== metrics AFTER =="; scrape
python3 - "$c" "$w" <<'PY'
import sys; c, w = float(sys.argv[1]), float(sys.argv[2])
print(f"\n== latency: cold={c:.3f}s warm={w:.3f}s => {(1-w/c)*100:.1f}% faster ({c/w:.2f}x)"
      if w < c else f"\n== latency: cold={c:.3f}s warm={w:.3f}s => no speedup")
PY
rm -f "$REQ"
