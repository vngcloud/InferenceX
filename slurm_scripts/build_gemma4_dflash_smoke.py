#!/usr/bin/env python3
"""Build the Gemma-4-31B DFlash smoke datasets from gateway traffic.

Each row is rendered exactly as vLLM serves it: the prompt goes through the
Gemma 4 chat template with ``add_generation_prompt=True`` (which appends the
empty thought channel), then the logged reply and ``<turn|>\\n`` follow. The
target therefore sees the same prefix during capture as in production.

Outputs (``--out-dir``):
  overfit.jsonl   short rows, trained on for many epochs (gate: loss collapses)
  heldout.jsonl   disjoint short rows, offline eval set (gate: eval path works)
  long.jsonl      rows at the real agentic length (gate: memory / step time)
  expect.jsonl    id -> exact supervised text, read by check_gemma4_dflash.py
  manifest.json   selection rules, token stats, feature-storage estimate

Usage:
  .venv/bin/python scripts/build_gemma4_dflash_smoke.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

END_OF_TURN = "<turn|>\n"
# 5 captured layers x hidden 5376 x bf16.
FEATURE_BYTES_PER_TOKEN = 5 * 5376 * 2


def _render(tokenizer, conversation: list[dict]) -> tuple[str, str]:
    prompt = tokenizer.apply_chat_template(
        conversation[:-1], tokenize=False, add_generation_prompt=True
    )
    reply = conversation[-1]["content"]
    return prompt + reply + END_OF_TURN, reply + END_OF_TURN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="data/gemma-4-31b-it/train/2026-09-18_2026-09-25/train.jsonl")
    parser.add_argument("--out-dir", default="data/gemma4-dflash-smoke")
    parser.add_argument("--tokenizer", default="google/gemma-4-31B-it")
    parser.add_argument("--short-max-tokens", type=int, default=2048)
    parser.add_argument("--long-min-tokens", type=int, default=8192)
    parser.add_argument("--long-max-tokens", type=int, default=9984)
    parser.add_argument("--min-reply-tokens", type=int, default=64)
    parser.add_argument("--num-overfit", type=int, default=128)
    parser.add_argument("--num-heldout", type=int, default=32)
    parser.add_argument("--num-long", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(line) for line in open(args.source)]

    short, long_ = [], []
    seen_text: set[str] = set()
    for index, row in enumerate(rows):
        conversation = row["conversations"]
        if conversation[-1]["role"] != "assistant":
            continue
        text, supervised = _render(tokenizer, conversation)
        if text in seen_text:
            continue
        seen_text.add(text)
        num_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        reply_tokens = len(tokenizer(supervised, add_special_tokens=False)["input_ids"])
        if reply_tokens < args.min_reply_tokens:
            continue
        record = {
            # Gateway ids are request timestamps and can repeat; the source
            # line index makes them unique.
            "id": f"{row['id']}#{index}",
            "text": text,
            "supervised": supervised,
            "num_tokens": num_tokens,
            "reply_tokens": reply_tokens,
        }
        if num_tokens <= args.short_max_tokens:
            short.append(record)
        elif args.long_min_tokens < num_tokens <= args.long_max_tokens:
            long_.append(record)

    rng = random.Random(args.seed)
    rng.shuffle(short)
    rng.shuffle(long_)
    needed = args.num_overfit + args.num_heldout
    if len(short) < needed or len(long_) < args.num_long:
        raise SystemExit(
            f"not enough rows: short={len(short)} (need {needed}), "
            f"long={len(long_)} (need {args.num_long})"
        )
    splits = {
        "overfit": short[: args.num_overfit],
        "heldout": short[args.num_overfit : needed],
        "long": long_[: args.num_long],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": args.source,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "rendering": "chat_template(prompt, add_generation_prompt=True) + reply + '<turn|>\\n'",
        "filters": {
            "min_reply_tokens": args.min_reply_tokens,
            "short_max_tokens": args.short_max_tokens,
            "long_tokens": [args.long_min_tokens, args.long_max_tokens],
        },
        "eligible": {"short": len(short), "long": len(long_)},
        "splits": {},
    }
    with open(out_dir / "expect.jsonl", "w") as expect:
        for name, records in splits.items():
            with open(out_dir / f"{name}.jsonl", "w") as f:
                for record in records:
                    f.write(json.dumps({"id": record["id"], "text": record["text"]}) + "\n")
                    expect.write(
                        json.dumps({"id": record["id"], "supervised": record["supervised"]}) + "\n"
                    )
            tokens = sum(r["num_tokens"] for r in records)
            manifest["splits"][name] = {
                "rows": len(records),
                "tokens": tokens,
                "reply_tokens": sum(r["reply_tokens"] for r in records),
                "max_tokens": max(r["num_tokens"] for r in records),
                "feature_gb": round(tokens * FEATURE_BYTES_PER_TOKEN / 1e9, 2),
            }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))
    print(f"eligible short={len(short)} long={len(long_)} -> {out_dir}")


if __name__ == "__main__":
    main()
