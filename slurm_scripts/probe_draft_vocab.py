#!/usr/bin/env python3
"""Draft-vocab coverage of a reduced-vocab speculator on the gateway bench outputs.

A draft with a reduced vocab (DFlash / EAGLE-3 ``t2d`` mask) can only propose tokens
inside its vocab: a target token outside it is always rejected. This probe tokenizes
the prod reference responses (``rows.jsonl``, joined to ``bench.jsonl`` by id for the
category and sampling group) and reports, per ``<category>__<group slug>`` (as in the
bench summary):

- ``coverage``: share of output tokens inside the draft vocab;
- ``prefix_bound[i]``: share of positions whose next ``i`` tokens are all in vocab —
  the ceiling on the per-position acceptance rate ``i`` measured by vLLM;
- ``al_bound``: ``1 + sum(prefix_bound)``, the ceiling on acceptance length with
  ``block`` draft tokens (a perfect draft within its vocab);
- the most frequent out-of-vocab tokens overall.

Plan: ``__plan__/[30-09-26] draft-token-sweep-and-dflash-bench.md`` §4.1.

Usage (cluster, repo root, uses the job's HF cache):
  HF_HOME=/shared/phucnlt2/hf-cache /shared/phucnlt2/venv-vllm/bin/python \
      scripts/probe_draft_vocab.py --output probe_draft_vocab.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from spec_data.eval.gateway_bench_run import group_slug

BENCH_DIR = "data/gemma-4-31b-it/bench/2026-09-18_2026-09-25"


def load_t2d(draft: str) -> np.ndarray:
    """Boolean mask over the target vocab from a local checkpoint dir or an HF repo id."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(draft)
    if not path.is_dir():
        path = Path(snapshot_download(draft, allow_patterns=["*.json", "*.safetensors"]))
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(str(file), "np") as f:
            if "t2d" in f.keys():
                return f.get_tensor("t2d").astype(bool)  # type: ignore[no-any-return]
    raise SystemExit(f"✗ no t2d tensor in {path}")


def load_outputs(bench_dir: Path) -> list[dict[str, Any]]:
    """Reference response text per bench row, with its category and sampling group."""
    meta = {}
    with open(bench_dir / "bench.jsonl") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                meta[row["id"]] = (row.get("category") or "unknown", row["sampling_group"], row)
    out = []
    with open(bench_dir / "rows.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in meta:
                continue
            category, group, bench_row = meta[row["id"]]
            resp = row["response"]
            text = (resp.get("reasoning_content") or "") + (resp.get("content") or "")
            if resp.get("tool_calls"):
                text += json.dumps(resp["tool_calls"], ensure_ascii=False)
            out.append(
                {
                    "category": category,
                    "group": group_slug(group),
                    "text": text,
                    "output_tokens": bench_row.get("output_tokens"),
                }
            )
    return out


def prefix_bound(in_vocab: np.ndarray, block: int) -> np.ndarray:
    """Counts per i=1..block of start positions t whose tokens t..t+i-1 are all in vocab.

    Start positions run over the whole sequence (the first output token comes from the
    prefill, so this slightly over-counts; negligible for multi-hundred-token outputs).
    Windows past the end count as failed, matching a draft that is cut off by EOS.
    """
    n = len(in_vocab)
    counts = np.zeros(block)
    # run[t] = length of the in-vocab run starting at t
    run = np.zeros(n + 1, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        run[t] = run[t + 1] + 1 if in_vocab[t] else 0
    for i in range(1, block + 1):
        counts[i - 1] = int((run[:n] >= i).sum())
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draft", default="RedHatAI/gemma-4-31B-it-speculator.dflash")
    ap.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    ap.add_argument("--bench-dir", default=BENCH_DIR)
    ap.add_argument("--block", type=int, default=8, help="draft tokens per step")
    ap.add_argument("--top-oov", type=int, default=40)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    t2d = load_t2d(args.draft)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_outputs(Path(args.bench_dir))
    print(f"draft vocab {int(t2d.sum())}/{len(t2d)}, {len(rows)} rows")

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "tokens": 0, "in_vocab": 0, "prefix": np.zeros(args.block)}
    )
    oov: Counter[int] = Counter()
    len_err = []
    for r in rows:
        ids = np.asarray(tok(r["text"], add_special_tokens=False)["input_ids"], dtype=np.int64)
        if r["output_tokens"]:
            len_err.append(abs(len(ids) - r["output_tokens"]) / r["output_tokens"])
        in_vocab = t2d[ids] if len(ids) else np.zeros(0, dtype=bool)
        oov.update(ids[~in_vocab].tolist())
        pb = prefix_bound(in_vocab, args.block)
        for key in (f"{r['category']}__{r['group']}", f"{r['category']}__all", "all"):
            s = stats[key]
            s["rows"] += 1
            s["tokens"] += len(ids)
            s["in_vocab"] += int(in_vocab.sum())
            s["prefix"] += pb

    groups = {}
    for key, s in sorted(stats.items()):
        prefix = (s["prefix"] / s["tokens"]).tolist() if s["tokens"] else []
        groups[key] = {
            "rows": s["rows"],
            "tokens": s["tokens"],
            "coverage": s["in_vocab"] / s["tokens"] if s["tokens"] else None,
            "prefix_bound": prefix,
            "al_bound": 1 + sum(prefix),
        }
    total = stats["all"]["tokens"]
    top = [
        {"id": i, "token": tok.convert_ids_to_tokens(i), "count": c, "share": c / total}
        for i, c in oov.most_common(args.top_oov)
    ]
    result = {
        "draft": args.draft,
        "draft_vocab": int(t2d.sum()),
        "block": args.block,
        "token_len_rel_err_p50": float(np.median(len_err)) if len_err else None,
        "groups": groups,
        "top_oov": top,
    }

    header = f"{'group':34} {'rows':>5} {'tokens':>9} {'cover':>6} {'AL≤':>5}"
    print(f"{header}  prefix bound 1..{args.block}")
    for key, g in groups.items():
        print(
            f"{key:34} {g['rows']:5d} {g['tokens']:9d} {g['coverage']:6.3f} {g['al_bound']:5.2f}  "
            + " ".join(f"{p:.2f}" for p in g["prefix_bound"])
        )
    print("\ntop out-of-vocab tokens:")
    for t in top[:20]:
        print(f"  {t['token']!r:24} {t['count']:7d}  {t['share']:.3%}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n→ {args.output}")


if __name__ == "__main__":
    main()
