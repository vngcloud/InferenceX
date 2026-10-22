"""Generate the E3 hidden-prefix trace (mooncake_trace JSONL, `messages` + `tools` mode).

    python utils/serving/gen_hidden_prefix_trace.py out.jsonl
    python utils/serving/gen_hidden_prefix_trace.py out.jsonl --groups 320 --sessions 2400 --tokenizer tokenizer.json

What it isolates: a large prefix shared ACROSS DP groups that the router cannot see.
- sgl-model-gateway (smg) cache_aware routes on extract_text_for_routing(), which reads only
  the `messages` contents (openai-protocol chat.rs); `tools` is not part of it.
- The GLM-5.3 chat template renders `tools` first, right after the fixed reasoning-effort
  header, so the tool catalog is the leading token prefix of every request.
Every request = one group's tool catalog (H tokens, hidden from the router) + a short unique
user message (V tokens, all the router sees). The router's match rate is ~0, so requests
land on DP ranks independently of which rank holds that catalog. A shared Mooncake L3 serves
the catalog to whichever rank gets the request; per-rank L2 (baseline) only hits when that
rank happens to hold it, and G x H is sized well above one rank's L1+L2.

Each session has --turns independent requests (aiperf skips 1-turn sessions; the first turn
of the first wave is warmup). `messages` mode replays each turn verbatim
(MESSAGE_ARRAY_WITH_RESPONSES), so later turns carry no history and are not pinned to a rank.
The group of each turn is drawn uniformly at random. Text is random hex (~1.5 bytes/token on
the GLM tokenizer) to keep the file small; KV cost per token does not depend on the text.
Deterministic for a given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

HEX = "0123456789abcdef"
TOKENS_PER_HEX_CHAR = 0.666  # GLM-5.3 tokenizer, measured on random hex


def hex_text(rng: random.Random, tokens: int) -> str:
    return "".join(rng.choice(HEX) for _ in range(int(tokens / TOKENS_PER_HEX_CHAR)))


def catalog(rng: random.Random, group: int, tools: int, tokens_per_tool: int) -> list[dict]:
    return [{"type": "function", "function": {
        "name": f"g{group:04d}_tool{t:02d}",
        "description": hex_text(rng, tokens_per_tool),
        "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"]},
    }} for t in range(tools)]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--groups", type=int, default=320, help="distinct tool catalogs")
    p.add_argument("--tools", type=int, default=24, help="tools per catalog")
    p.add_argument("--tokens-per-tool", type=int, default=1000, help="description tokens per tool")
    p.add_argument("--sessions", type=int, default=2400)
    p.add_argument("--turns", type=int, default=4, help="independent requests per session (>= 2)")
    p.add_argument("--visible-tokens", type=int, default=1024, help="unique user message per request")
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--delay-min-ms", type=int, default=500)
    p.add_argument("--delay-max-ms", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", help="tokenizer.json to report exact catalog/request token counts")
    a = p.parse_args(argv)
    if a.turns < 2:
        p.error("--turns must be >= 2: aiperf AgentX replay skips 1-turn sessions")

    rng = random.Random(a.seed)
    catalogs = [catalog(rng, g, a.tools, a.tokens_per_tool) for g in range(a.groups)]
    n = 0
    with open(a.out, "w") as f:
        for s in range(a.sessions):
            for t in range(a.turns):
                g = rng.randrange(a.groups)
                rec = {"session_id": f"s{s:05d}", "output_length": a.output_len,
                       "tools": catalogs[g],
                       "messages": [{"role": "user",
                                     "content": f"task {s}-{t}: " + hex_text(rng, a.visible_tokens)}]}
                if t:
                    rec["delay"] = rng.randint(a.delay_min_ms, a.delay_max_ms)
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
    print(f"{a.out}: {a.sessions} sessions, {n} requests, {a.groups} catalogs", file=sys.stderr)

    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        h = len(tok.encode(json.dumps(catalogs[0]), add_special_tokens=False).ids)
        v = len(tok.encode(hex_text(random.Random(0), a.visible_tokens), add_special_tokens=False).ids)
        print(f"catalog ~{h} tokens (JSON as rendered, approx), visible ~{v} tokens, "
              f"hidden share {h / (h + v):.1%}, unique catalogs {a.groups * h / 1e6:.2f}M tokens",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
