"""Generate the E2 shared-prefix trace (mooncake_trace JSONL) for the HiCache L2-vs-Mooncake-L3 A/B.

    python utils/serving/gen_shared_prefix_trace.py out.jsonl            # defaults below
    python utils/serving/gen_shared_prefix_trace.py out.jsonl --groups 300 --siblings 8

What it isolates: KV reuse ACROSS DP groups. G groups (think: repos) each own a shared
first-turn prefix of P tokens; S sibling sessions per group start with that prefix plus a
private part. With the router's --cache-threshold=0.3, a prefix below 30% of the request
does not pin the sibling to the DP rank that already holds it, so siblings spread over the
ranks by load. Baseline (per-rank L2 cache) must re-prefill P on every rank a group reaches
for the first time; a shared Mooncake L3 serves it from the rank that computed it.

Shape per session (aiperf DELTAS_WITHOUT_RESPONSES, the default for synthesized prompts):
turn 1 = shared group blocks + private blocks; turns 2..T = small private deltas after a
delay; aiperf threads the live responses into the history, so later turns reuse turn 1.

Order: sessions are written in windows of W groups, siblings round-robin inside a window
(s1 of every group, then s2, ...), so a group's next sibling starts ~W sessions after the
previous one: late enough for write_through_selective to have pushed the prefix to L3
(after its second hit, i.e. the first sibling's turn 2), early enough that it is not
evicted. aiperf's file datasets sample sequentially in first-appearance order.

hash_ids are block ids (block_size 512, the mooncake_trace default): equal ids decode to
equal text, so equal leading ids give an identical token prefix. Ids never repeat across
groups or sessions except the group prefix. Deterministic for a given --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

BLOCK = 512


def build(groups: int, siblings: int, window: int, shared_blocks: int, private_blocks: int,
          turns: int, delta_blocks: int, output_len: int, delay_ms: tuple[int, int],
          seed: int) -> list[dict]:
    rng = random.Random(seed)
    next_id = 1
    group_prefix: list[list[int]] = []
    for _ in range(groups):
        group_prefix.append(list(range(next_id, next_id + shared_blocks)))
        next_id += shared_blocks

    records: list[dict] = []
    for start in range(0, groups, window):
        window_groups = range(start, min(start + window, groups))
        for s in range(siblings):
            for g in window_groups:
                sid = f"g{g:04d}-s{s}"
                private = list(range(next_id, next_id + private_blocks))
                next_id += private_blocks
                ids = group_prefix[g] + private
                records.append({"session_id": sid, "input_length": len(ids) * BLOCK,
                                "output_length": output_len, "hash_ids": ids})
                for _ in range(turns - 1):
                    delta = list(range(next_id, next_id + delta_blocks))
                    next_id += delta_blocks
                    records.append({"session_id": sid, "delay": rng.randint(*delay_ms),
                                    "input_length": len(delta) * BLOCK,
                                    "output_length": output_len, "hash_ids": delta})
    return records


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--groups", type=int, default=300)
    p.add_argument("--siblings", type=int, default=8, help="sessions per group (8 = one per DP rank at dp8)")
    p.add_argument("--window", type=int, default=32, help="groups interleaved at a time")
    p.add_argument("--shared-blocks", type=int, default=32, help="group prefix, x512 tokens (16384)")
    p.add_argument("--private-blocks", type=int, default=88, help="private turn-1 part, x512 tokens (45056)")
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--delta-blocks", type=int, default=4, help="turns 2..T input, x512 tokens (2048)")
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--delay-min-ms", type=int, default=5000)
    p.add_argument("--delay-max-ms", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    share = a.shared_blocks / (a.shared_blocks + a.private_blocks)
    if share >= 0.3:
        p.error(f"shared prefix is {share:.1%} of turn 1; keep it below the router's --cache-threshold=0.3")

    records = build(a.groups, a.siblings, a.window, a.shared_blocks, a.private_blocks, a.turns,
                    a.delta_blocks, a.output_len, (a.delay_min_ms, a.delay_max_ms), a.seed)
    with open(a.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    sessions = a.groups * a.siblings
    print(f"{a.out}: {sessions} sessions, {len(records)} turns, turn-1 "
          f"{(a.shared_blocks + a.private_blocks) * BLOCK} tokens "
          f"({a.shared_blocks * BLOCK} shared = {share:.1%})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
