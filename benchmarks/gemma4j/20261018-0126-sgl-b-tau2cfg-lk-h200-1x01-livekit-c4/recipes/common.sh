#!/usr/bin/env bash
# Shared measurement protocol for every recipe (spec §6). Sourced, not executed.
# Uses: BENCH_ENGINE BENCH_PORT BENCH_MODEL BENCH_POINT_DIR BENCH_WARMUP_FILE BENCH_WARMUP_N BENCH_OSL
#       BENCH_IGNORE_EOS BENCH_CATEGORY BENCH_META_JSON BENCH_PRIME_N BENCH_PROMPT_OFFSET

_common_fail() { echo "COMMON: $*" >&2; return 1; }

bench_metrics_snapshot() {  # <out-file>
    curl -fsS --max-time 30 "http://localhost:${BENCH_PORT}/metrics" > "$1" || _common_fail "metrics snapshot failed"
}

# Summed value of a metric across label sets; accepts the declared name or its _total counter form.
# Prints nothing when the metric is absent (absent is not zero).
_metric_value() {  # <name> [file]
    local src
    if [[ -n "${2:-}" ]]; then src=$(cat "$2"); else src=$(curl -fsS --max-time 30 "http://localhost:${BENCH_PORT}/metrics") || return 1; fi
    printf '%s\n' "$src" | awk -v m="$1" '
        $0 !~ /^#/ { split($1, a, "{"); if (a[1] == m || a[1] == m "_total") { s += $NF; found = 1 } }
        END { if (found) printf "%d\n", s }'
}

# Unscored warmup: exactly BENCH_WARMUP_N chat requests (rows cycled), concurrency <= 8, scored sampling.
# Rows come from the end of the (category-filtered) pool so they never overlap scored prompts; rows without
# "messages" (mooncake trace rows) get a synthetic user message of ~input_length words.
bench_warmup() {
    python3 - "$BENCH_WARMUP_FILE" "${BENCH_WARMUP_N:-8}" "$BENCH_PORT" "$BENCH_MODEL" "$BENCH_OSL" \
        "${BENCH_IGNORE_EOS:-1}" "${BENCH_CATEGORY:-}" <<'PY' || _common_fail "warmup failed"
import concurrent.futures as cf
import json
import sys
import urllib.request

path, n, port, model, osl, ignore_eos, category = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6] == "1", sys.argv[7]
rows = [json.loads(line) for line in open(path) if line.strip()]
if category:
    rows = [r for r in rows if r.get("category") == category]
if not rows:
    sys.exit("no warmup rows")
rows = rows[-n:]
rows = (rows * n)[:n]


def messages(row):
    msgs = row.get("messages")
    if msgs:
        first_user = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), len(msgs) - 1)
        return msgs[: first_user + 1]
    words = max(1, min(int(row.get("input_length") or 512), 16384) * 3 // 4)
    return [{"role": "user", "content": " ".join(["warmup"] * words)}]


def one(row):
    body = {"model": model, "messages": messages(row), "temperature": 0, "max_tokens": osl, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    done = int((out.get("usage") or {}).get("completion_tokens") or 0)
    if ignore_eos and done < osl:
        raise RuntimeError(f"warmup request returned {done} < {osl} tokens with ignore_eos")
    return done


with cf.ThreadPoolExecutor(max_workers=min(8, n)) as ex:
    toks = list(ex.map(one, rows))
print(f"warmup: {len(toks)} requests, {sum(toks)} tokens")
PY
}

# A probe several KV blocks long (vLLM counts prefix hits per full block; a short probe can never hit).
_probe_request() {
    local words
    words=$(python3 -c "print(' '.join('probe%d' % i for i in range(400)))")
    curl -fsS --max-time 600 -H 'Content-Type: application/json' "http://localhost:${BENCH_PORT}/v1/chat/completions" \
        -d "{\"model\":\"${BENCH_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${words}\"}],\"max_tokens\":1,\"temperature\":0}" >/dev/null
}

_post_ok() {  # <path>: POST must answer 2xx
    local code
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 120 -X POST "http://localhost:${BENCH_PORT}$1") || return 1
    [[ "$code" == 2* ]] || { echo "COMMON: POST $1 -> HTTP $code" >&2; return 1; }
}

# Verified cache reset. Exports BENCH_CACHE_RESET=verified|unverified; fails on unverified.
#   SGLang: /flush_cache answers 2xx and sglang:num_used_tokens reads 0 afterwards.
#   vLLM (needs VLLM_SERVER_DEV_MODE=1 for /reset_prefix_cache): a multi-block probe sent twice must hit the
#   cache (so the check can detect anything), and after a reset the same probe must not; then a final reset.
bench_cache_reset() {
    local attempt h0 h1 h2 h3 used
    export BENCH_CACHE_RESET=unverified
    for attempt in 1 2 3; do
        if [[ "$BENCH_ENGINE" == "sglang" ]]; then
            if _post_ok /flush_cache; then
                sleep 1
                used=$(_metric_value sglang:num_used_tokens)
                if [[ "$used" == "0" ]]; then export BENCH_CACHE_RESET=verified; break; fi
            fi
        else
            if _post_ok /reset_prefix_cache; then
                h0=$(_metric_value vllm:prefix_cache_hits); _probe_request; _probe_request
                h1=$(_metric_value vllm:prefix_cache_hits)
                if _post_ok /reset_prefix_cache; then
                    h2=$(_metric_value vllm:prefix_cache_hits); _probe_request
                    h3=$(_metric_value vllm:prefix_cache_hits)
                    if [[ -n "$h0" && "$h1" -gt "$h0" && "$h3" == "$h2" ]] && _post_ok /reset_prefix_cache; then
                        export BENCH_CACHE_RESET=verified; break
                    fi
                fi
            fi
        fi
        sleep "${BENCH_RESET_RETRY_S:-5}"
    done
    [[ "$BENCH_CACHE_RESET" == "verified" ]] || _common_fail "cache reset could not be verified after 3 attempts"
}

# Every warmup request counted: delta of the request counter (summed over label sets, i.e. every DP replica)
# since <before-file> must equal <n>.
bench_check_request_count() {  # <before-file> <n>
    local name b a seen
    if [[ "$BENCH_ENGINE" == "sglang" ]]; then name="sglang:num_requests"; else name="vllm:request_success"; fi
    b=$(_metric_value "$name" "$1"); a=$(_metric_value "$name")
    seen=$(( ${a:-0} - ${b:-0} ))
    echo "warmup requests seen by /metrics: $seen"
    (( seen == $2 )) || _common_fail "metrics counted $seen of $2 warmup requests (a DP replica not scraped?)"
}

# Merge BENCH_META_JSON + extra JSON + sampling block into raw/point_meta.json.
bench_write_meta() {  # [extra-json]
    local extra="${1:-}"
    [[ -n "$extra" ]] || extra='{}'
    mkdir -p "$BENCH_POINT_DIR/raw"
    BENCH_EXTRA_JSON="$extra" python3 - "$BENCH_POINT_DIR/raw/point_meta.json" <<'PY' || _common_fail "point_meta.json write failed"
import datetime
import json
import os
import sys

meta = json.loads(os.environ.get("BENCH_META_JSON") or "{}")
meta.update(json.loads(os.environ.get("BENCH_EXTRA_JSON") or "{}"))
meta["sampling"] = {"temperature": 0, "enable_thinking": False,
                    "ignore_eos": os.environ.get("BENCH_IGNORE_EOS", "1") == "1",
                    "warmup_requests": int(os.environ.get("BENCH_WARMUP_N", "8")),
                    "prime_requests": int(os.environ.get("BENCH_PRIME_N", "0")),
                    "prompt_offset": int(os.environ.get("BENCH_PROMPT_OFFSET", "0")), "seed": 42,
                    "cache_reset": os.environ.get("BENCH_CACHE_RESET", "unverified")}
meta["started_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
with open(sys.argv[1], "w") as f:
    json.dump(meta, f, indent=2)
PY
}

# Qualification gate: <server.log> <backend> <spec-regex|none> <context-length>
bench_qualify_log() {
    local log="$1" backend="$2" spec="$3" ctx="$4" rc=0
    if [[ "$BENCH_ENGINE" == "sglang" ]]; then
        grep -q "'attention_backend': '${backend}'" "$log" || { echo "COMMON: attention backend != ${backend}"; rc=1; }
        grep -q "'context_length': ${ctx}" "$log" || { echo "COMMON: context_length != ${ctx}"; rc=1; }
        grep -qE "KV Cache is allocated" "$log" || { echo "COMMON: no KV pool line"; rc=1; }
    else
        grep -q "Using ${backend} attention backend" "$log" || { echo "COMMON: attention backend != ${backend}"; rc=1; }
        grep -q "'max_model_len': ${ctx}" "$log" || { echo "COMMON: max_model_len != ${ctx}"; rc=1; }
        grep -qE "GPU KV cache size" "$log" || { echo "COMMON: no KV pool line"; rc=1; }
    fi
    if [[ "$spec" != "none" ]]; then
        grep -qE "$spec" "$log" || { echo "COMMON: spec promotion '${spec}' not found"; rc=1; }
    fi
    return $rc
}

# Stop the server and every descendant, then wait until the port is closed and (when nvidia-smi exists)
# no compute process holds GPU memory, so the next launch sizes its KV pool against a free GPU.
bench_stop_server() {  # <pid>
    local pid="$1" all=() frontier=("$1") next=() p i
    while (( ${#frontier[@]} )); do
        next=()
        for p in "${frontier[@]}"; do
            all+=("$p")
            while read -r c; do [[ -n "$c" ]] && next+=("$c"); done < <(pgrep -P "$p" 2>/dev/null)
        done
        frontier=("${next[@]}")
    done
    kill -TERM "${all[@]}" 2>/dev/null
    for i in $(seq 1 60); do kill -0 "${all[@]}" 2>/dev/null || break; sleep 1; done
    kill -KILL "${all[@]}" 2>/dev/null
    wait "$pid" 2>/dev/null
    for i in $(seq 1 120); do
        curl -fsS --max-time 2 "http://localhost:${BENCH_PORT}/health" >/dev/null 2>&1 || break
        sleep 1
    done
    if command -v nvidia-smi >/dev/null 2>&1; then
        for i in $(seq 1 120); do
            [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && return 0
            sleep 1
        done
        _common_fail "GPU still has compute processes 120 s after teardown"
    fi
    return 0
}
