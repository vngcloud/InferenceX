#!/usr/bin/env python3
"""Verify a mooncake trace against the measured target distribution.

Reads the generated JSONL back off disk -- deliberately NOT the in-memory
arrays -- so the check covers serialization, rounding, and field naming, not
just the sampler. Percentiles use numpy linear interpolation (k = (n-1)*p),
the same definition aiperf `analyze-trace` uses, so this table and that
command are directly comparable.

Cache hit rates are recomputed here from scratch rather than imported from the
generator, so a bug in the generator's own accounting cannot hide behind a
shared helper. The replay reimplements aiperf's algorithm (prefix_analyzer
`_compute_per_request_hit_rates`); `aiperf analyze-trace` on the same file is
the third, fully independent check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The measured table from production. This is the ground truth being checked
# against; `mean` is included because it is an independent constraint the
# sampler was fitted to, not merely another percentile.
ISL_TARGET = {
    "n": 910,
    "min": 14,
    "mean": 27755,
    "p50": 34919,
    "p90": 42244,
    "p95": 42570,
    "p99": 43324,
    "max": 44220,
}

STATS = ["n", "min", "mean", "p50", "p90", "p95", "p99", "max"]

BLOCK_SIZE = 512
UNIQUE_ID_BASE = 900_000_000


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def load(path: Path) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    isl, osl, hashes = [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            isl.append(rec["input_length"])
            osl.append(rec.get("output_length", 0))
            hashes.append(rec.get("hash_ids", []))
    return np.array(isl), np.array(osl), hashes


def replay(isl: np.ndarray, hashes: list[list[int]]) -> dict[str, float]:
    """Infinite-cache replay. See module docstring for why this is duplicated."""
    seen: set[int] = set()
    hit_tokens = 0
    aiperf_rates: list[float] = []
    all_rates: list[float] = []
    for length, ids in zip(isl, hashes):
        if not ids:
            all_rates.append(0.0)
            continue
        first_unseen = len(ids)
        for idx, hid in enumerate(ids):
            if hid not in seen:
                first_unseen = idx
                break
        seen.update(ids)
        rate = first_unseen / len(ids)
        aiperf_rates.append(rate)
        all_rates.append(rate)
        hit_tokens += min(first_unseen * BLOCK_SIZE, int(length))
    total = int(isl.sum())
    return {
        "token_weighted": hit_tokens / total if total else 0.0,
        "aiperf": float(np.mean(aiperf_rates)) if aiperf_rates else 0.0,
        "per_request": float(np.mean(all_rates)) if all_rates else 0.0,
        "hit_tokens": float(hit_tokens),
    }


def check_tiling(isl: np.ndarray, hashes: list[list[int]]) -> list[str]:
    """Structural checks that aiperf's loader would otherwise fail on at runtime.

    1. Block count must be exactly ceil(L / BLOCK_SIZE). Too few and aiperf
       takes the prefix-only path (silently changing the layout); too many and
       `_build_token_sequence` raises ConfigurationError on the implied
       negative partial block.
    2. Shared (reused) blocks must never be the final, partial block -- a
       partial shared block would be content-identical only by luck.
    3. Shared block ids must form a contiguous run at the FRONT. A cache hit
       requires a shared PREFIX; shared blocks in the middle never hit.
    """
    errors: list[str] = []
    for i, (length, ids) in enumerate(zip(isl, hashes)):
        if not ids:
            continue
        expected = max(1, -(-int(length) // BLOCK_SIZE))
        if len(ids) != expected:
            errors.append(
                f"line {i}: {len(ids)} blocks for isl {length}, expected {expected}"
            )
        shared_flags = [h < UNIQUE_ID_BASE for h in ids]
        n_shared = sum(shared_flags)
        if shared_flags[:n_shared] != [True] * n_shared:
            errors.append(f"line {i}: shared blocks are not a contiguous prefix")
        if n_shared == len(ids) and int(length) % BLOCK_SIZE != 0:
            errors.append(f"line {i}: final partial block is shared")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument(
        "--expect-cache-level",
        type=float,
        default=0.0,
        help="Target token-weighted hit rate this trace was built for.",
    )
    ap.add_argument("--tolerance", type=float, default=0.005)
    args = ap.parse_args()

    isl, osl, hashes = load(args.trace)
    got = describe(isl)

    print(f"Trace: {args.trace}")
    print()
    print("ISL -- target vs realized")
    print(f"  {'stat':<6} {'target':>10} {'realized':>10} {'delta':>10} {'rel':>8}")
    for k in STATS:
        tgt = float(ISL_TARGET[k])
        val = got[k]
        delta = val - tgt
        rel = (delta / tgt * 100) if tgt else 0.0
        print(f"  {k:<6} {tgt:>10,.0f} {val:>10,.0f} {delta:>+10,.0f} {rel:>+7.2f}%")

    print()
    print("OSL -- realized")
    o = describe(osl)
    for k in STATS:
        print(f"  {k:<6} {o[k]:>10,.0f}")
    cap_n = int((osl == osl.max()).sum())
    print(f"  at cap ({int(osl.max()):,}): {cap_n:,} / {len(osl):,} = "
          f"{cap_n / len(osl) * 100:.1f}%")

    n_with_hash = sum(1 for h in hashes if h)
    rates = replay(isl, hashes)
    struct_errors = check_tiling(isl, hashes)

    print()
    print("Cache structure")
    print(f"  requests carrying hash_ids : {n_with_hash:,} / {len(hashes):,}")
    if n_with_hash:
        shared = {h for ids in hashes for h in ids if h < UNIQUE_ID_BASE}
        print(f"  distinct shared blocks     : {len(shared):,} "
              f"({len(shared) * BLOCK_SIZE:,} tokens of hot KV)")
        print(f"  distinct prefix roots      : "
              f"{len({ids[0] for ids in hashes if ids and ids[0] < UNIQUE_ID_BASE}):,}")
    print(f"  structural errors          : {len(struct_errors)}")
    for e in struct_errors[:5]:
        print(f"    - {e}")

    print()
    print("Hit rate (infinite cache -- an UPPER BOUND on the server's real rate)")
    print(f"  target (token-weighted)    : {args.expect_cache_level:.2%}")
    print(f"  realized token-weighted    : {rates['token_weighted']:.2%}")
    print(f"  aiperf analyze-trace metric: {rates['aiperf']:.2%}")
    print(f"  per-request mean           : {rates['per_request']:.2%}")
    print(f"  cached prefill tokens      : {int(rates['hit_tokens']):,} "
          f"of {int(isl.sum()):,}")

    print()
    print(f"  prefill:decode token ratio : {isl.sum() / osl.sum():,.1f} : 1")
    print(f"  min required max-model-len : {int(isl.max() + osl.max()):,}")

    mean_ok = abs(got["mean"] - ISL_TARGET["mean"]) / ISL_TARGET["mean"] < 0.01
    rate_ok = abs(rates["token_weighted"] - args.expect_cache_level) <= args.tolerance
    ok = mean_ok and rate_ok and not struct_errors
    print()
    print("PASS" if ok else "FAIL")
    if not ok:
        if not mean_ok:
            print("  - ISL mean drifted from target")
        if not rate_ok:
            print("  - realized hit rate outside tolerance")
        if struct_errors:
            print("  - structural errors in hash_ids tiling")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
