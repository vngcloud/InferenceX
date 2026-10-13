#!/usr/bin/env python3
"""Check whether a category-classification request can reuse the entropy request's KV cache.

For each sample, the entropy request is the token-id prompt from ``build_scoring_request``
(sent with ``prompt_logprobs``). The classify request is the same token ids plus a
suffix (new user turn with the instruction + generation prompt), answered with one token.
If vLLM's prefix cache serves the shared prefix, classification is nearly free.

Phases, all sequential so per-request numbers are clean:
  1. entropy(S1) -> classify(S1)   key question: does classify hit what entropy computed?
  2. classify(S1) again            control: plain request after plain request
  3. entropy(S1) again             does a prompt_logprobs request *read* the cache?
  4. classify(S2), fresh samples   cold baseline (latency without cache)

Cached tokens come from ``usage.prompt_tokens_details.cached_tokens``
(needs ``--enable-prompt-tokens-details``) and from the server's prefix-cache counters.

Usage:
  python scripts/probe_prefix_cache.py --model Qwen/Qwen3-4B \
      --data-path data/gateway-smoke/raw/train.jsonl --num-samples 40 \
      --vllm-executable /shared/phucnlt2/venv-vllm/bin/vllm --output probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import urllib.request
from typing import Any

from spec_data.data.quality.category import (
    CLASSIFY_INSTRUCTION,
    PROMPT_VERSION,
    build_classify_suffix,
    parse_verdict,
)
from spec_data.data.quality.entropy_model import VLLMEntropyBackend, build_scoring_request
from spec_data.data.quality.loader import load_dataset
from spec_data.data.quality.vllm_server import VLLMServer

# Token counters, e.g. vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total.
_METRIC_RE = re.compile(
    r"^(vllm:(?:gpu_)?prefix_cache_(?:queries|hits))(?:_total)?(?:\{.*\})?\s+(\S+)$"
)


def read_prefix_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        text = resp.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC_RE.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def send(backend: VLLMEntropyBackend, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    data = backend._post({"model": backend.model, "temperature": 0.0, "max_tokens": 1, **payload})
    return data, time.perf_counter() - t0


def run_phase(
    name: str,
    kind: str,
    reqs: list[dict[str, Any]],
    backend: VLLMEntropyBackend,
    base_url: str,
    top_k: int,
) -> dict[str, Any]:
    before = read_prefix_metrics(base_url)
    rows = []
    for r in reqs:
        if kind == "entropy":
            payload = {"prompt": r["entropy_ids"], "prompt_logprobs": top_k}
        else:
            payload = {"prompt": r["classify_ids"], "logprobs": top_k}
        data, secs = send(backend, payload)
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        row: dict[str, Any] = {
            "id": r["id"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "prefix_len": len(r["entropy_ids"]),
            "seconds": round(secs, 4),
        }
        if kind == "classify":
            lp = data["choices"][0].get("logprobs") or {}
            top = (lp.get("top_logprobs") or [{}])[0] or {}
            verdict = parse_verdict(top, usage)
            row["label"] = verdict.label
            row["probs"] = {k: round(v, 3) for k, v in verdict.probs.items()}
            row["label_mass"] = round(verdict.label_mass, 3)
            row["has_tool_calls"] = r["has_tool_calls"]
        rows.append(row)
    after = read_prefix_metrics(base_url)
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}

    prompt = sum(x["prompt_tokens"] or 0 for x in rows)
    cached = [x["cached_tokens"] for x in rows if x["cached_tokens"] is not None]
    summary = {
        "phase": name,
        "requests": len(rows),
        "prompt_tokens": prompt,
        "cached_tokens": sum(cached) if cached else None,
        "cached_fraction": round(sum(cached) / prompt, 4) if cached and prompt else None,
        "median_seconds": round(statistics.median(x["seconds"] for x in rows), 4) if rows else None,
        "server_metrics_delta": delta,
    }
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument(
        "--num-samples", type=int, default=40, help="per set; S1 and S2 each get this many"
    )
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--vllm-executable", default="vllm")
    ap.add_argument("--server-log", default="")
    ap.add_argument("--base-url", default="", help="use a running server instead of launching one")
    ap.add_argument("--server-extra-arg", action="append", default=[], help="repeatable")
    ap.add_argument("--server-env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--seed", type=int, default=0, help="sample order shuffle; -1 keeps file order")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    samples = load_dataset(args.data_path).samples
    if args.seed >= 0:
        random.Random(args.seed).shuffle(samples)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    suffix_ids = build_classify_suffix(tokenizer)
    print(
        f"[probe] prompt {PROMPT_VERSION}, suffix: {len(suffix_ids)} tokens\n"
        f"{tokenizer.decode(suffix_ids)!r}"
    )

    # Leave room for the suffix and the one generated token.
    budget = args.max_model_len - 1 - len(suffix_ids)
    reqs = []
    for s in samples:
        sr = build_scoring_request(s, tokenizer, budget)
        if sr is None or not sr.templated:
            continue
        reqs.append(
            {
                "id": s.id,
                "entropy_ids": sr.token_ids,
                "classify_ids": sr.token_ids + suffix_ids,
                "has_tool_calls": any(
                    m.get("tool_calls") or m.get("role") == "tool" for m in s.conversations
                ),
            }
        )
        if len(reqs) >= 2 * args.num_samples:
            break
    half = len(reqs) // 2
    s1, s2 = reqs[:half], reqs[half:]
    print(
        f"[probe] {len(samples)} samples loaded, {len(reqs)} templated -> S1={len(s1)} S2={len(s2)}"
    )

    def probe(base_url: str) -> dict[str, Any]:
        backend = VLLMEntropyBackend(
            base_url=base_url,
            model=args.model,
            top_k=args.top_k,
            tokenizer=tokenizer,
            timeout=1800,  # prompt_logprobs over a 100k+ token prompt is a large response
        )
        send(backend, {"prompt": suffix_ids})  # warm-up (compile/cuda graphs), not measured
        phases = [
            run_phase("1_entropy_S1", "entropy", s1, backend, base_url, args.top_k),
            run_phase("1_classify_after_entropy_S1", "classify", s1, backend, base_url, args.top_k),
            run_phase("2_classify_again_S1", "classify", s1, backend, base_url, args.top_k),
            run_phase("3_entropy_again_S1", "entropy", s1, backend, base_url, args.top_k),
            run_phase("4_classify_cold_S2", "classify", s2, backend, base_url, args.top_k),
        ]
        return {
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "instruction": CLASSIFY_INSTRUCTION,
            "suffix_tokens": len(suffix_ids),
            "phases": phases,
        }

    if args.base_url:
        result = probe(args.base_url)
    else:
        with VLLMServer(
            model=args.model,
            max_model_len=args.max_model_len,
            max_logprobs=args.top_k,
            ready_timeout=3600,  # large models: weight download + compile on first start
            gpu_memory_utilization=args.gpu_memory_utilization,
            log_path=args.server_log or None,
            extra_args=[
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
                *args.server_extra_arg,
            ],
            executable=args.vllm_executable,
            env=dict(kv.split("=", 1) for kv in args.server_env),
        ) as server:
            result = probe(server.base_url)

    print("\n[probe] summary")
    for p in result["phases"]:
        s = p["summary"]
        print(
            f"  {s['phase']:<30} cached {s['cached_fraction']}  "
            f"median {s['median_seconds']}s  metrics {s['server_metrics_delta']}"
        )
    # One classify pass per sample: S1 from phase 2, S2 from phase 4.
    once = ("2_classify_again_S1", "4_classify_cold_S2")
    labels = [r for p in result["phases"] if p["summary"]["phase"] in once for r in p["rows"]]
    tool = [r["label"] for r in labels if r["has_tool_calls"]]
    print(f"  tool-call samples labelled agentic: {tool.count('agentic')}/{len(tool)}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[probe] wrote {args.output}")


if __name__ == "__main__":
    main()
