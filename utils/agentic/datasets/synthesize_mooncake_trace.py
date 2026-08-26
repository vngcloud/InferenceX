#!/usr/bin/env python3
"""Synthesize a mooncake-format trace from a measured ISL/OSL distribution.

Two variants, selected by --cache-level:

  --cache-level 0    COLD FLOOR. No `hash_ids`, so aiperf synthesizes a fresh
                     prompt per request and the server prefix cache never hits.
                     The pessimistic bound, and the direct comparable to the
                     existing "no cache" fixed-seq-len runs.

  --cache-level F    WARM. Each request is fully tiled into `hash_ids`; the
                     leading fraction of blocks is drawn from a shared prefix
                     chain, the rest are unique. F is the TOKEN-WEIGHTED share
                     of prefill tokens that are served from cache -- see
                     "Which hit rate" below, because there is more than one.

Three fields are deliberately omitted from every line:

  timestamp  -- omitting it keeps the run CLOSED-loop (concurrency-driven, like
                the current `--request-rate inf --max-concurrency CONC` path).
                Verified against `_should_use_fixed_schedule_for_trace_dataset`
                in aiperf's user_config.py: auto-fixed-schedule engages only if
                a `timestamp` field is actually present in the file.
  session_id -- absent, so the aiperf loader assigns each line its own session
                (base_trace_loader `_group_traces`). Cache reuse here is
                expressed through hash_ids, not through session grouping.
  delay      -- closed-loop runs have no inter-arrival model to express.

`output_length` becomes `max_tokens`, NOT a guarantee -- the model may stop
early at EOS. To force exact output lengths the way the fixed-seq-len path does
with `--ignore-eos`, run aiperf with `--extra-inputs ignore_eos:true`.

--------------------------------------------------------------------------
Which hit rate
--------------------------------------------------------------------------
Three different numbers can all be called "cache hit rate", and they do not
agree on a heavy-tailed ISL distribution:

  token-weighted   sum(cached tokens) / sum(all prefill tokens).
                   This is what the SERVER reports and what actually predicts
                   prefill work saved. THIS is what --cache-level targets.

  aiperf           mean over requests of (leading cached blocks / total blocks),
                   unweighted, skipping requests that carry no hash_ids.
                   This is what `aiperf analyze-trace` prints as
                   `cache_hit_rate`. Reported below so the two can be compared
                   without confusion.

  per-request mean same as aiperf but counting hash-less requests as 0.0.

Full block tiling is used (every request's tokens are covered by hash_ids)
specifically so these three converge. The alternative "prefix-only" layout --
hash_ids for the cached prefix, un-hashed tail padded with fresh tokens -- is
equally valid to the server, but makes `aiperf analyze-trace` report ~100%
regardless of the real rate, because the un-hashed tail is invisible to its
denominator. Tiling keeps the external tool usable as a check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Step 1: the measured ISL distribution, as a quantile -> value table.
# ---------------------------------------------------------------------------
# All values are MEASURED except p25, which is DERIVED: it is the value that
# makes the piecewise-linear reconstruction integrate to the measured mean of
# 27,755. The measured percentiles alone fix only the top half (they average
# 39,417), so the bottom half must average 16,093 for the overall mean to land
# on 27,755 -- and p25 = 14,719 is what produces that.
#
# Sanity anchor: a uniform draw over [14, 34919] would average 17,467, only 8%
# above the required 16,093. So the lower half is close to uniform, weighted
# slightly low. If you can pull a real p10/p25 from the source logs, replace the
# derived row -- it is the single softest number in this file.
ISL_QUANTILES: list[tuple[float, float]] = [
    (0.00,    14.0),   # measured (min)
    (0.25, 14719.0),   # DERIVED to satisfy the measured mean
    (0.50, 34919.0),   # measured
    (0.90, 42244.0),   # measured
    (0.95, 42570.0),   # measured
    (0.99, 43324.0),   # measured
    (1.00, 44220.0),   # measured (max)
]
ISL_TARGET_MEAN = 27755.0

# ---------------------------------------------------------------------------
# Step 1b: the OSL model.
# ---------------------------------------------------------------------------
# 30% of requests hit the max_tokens cap exactly. That is a truncation spike,
# not a bump, so it is modelled as a point mass rather than smoothed into the
# body. The remaining 70% is a placeholder: real OSL bodies are usually
# right-skewed toward short, so a uniform body likely overstates the mean.
# It barely moves total token count (~0.6%) but it moves DECODE STEPS by ~30%,
# which is exactly the axis EAGLE3 acts on. Revisit before drawing spec-decode
# conclusions from this trace.
OSL_CAP = 1024
OSL_CAP_FRACTION = 0.30
OSL_BODY_MIN = 100
OSL_BODY_MAX = 1000

N_REQUESTS = 910
SEED = 20260826

# ---------------------------------------------------------------------------
# Step 6 (warm variants only): hash_ids block layout.
# ---------------------------------------------------------------------------
# BLOCK_SIZE must match what aiperf uses to expand hash_ids into tokens
# (InputTokensDefaults.BLOCK_SIZE) and what `analyze-trace --block-size` assumes.
# It is unrelated to vLLM's own --block-size (default 16); 512 is a multiple of
# 16, so shared-prefix boundaries stay aligned to vLLM block boundaries and no
# hit is lost to misalignment.
BLOCK_SIZE = 512

# Shared chain block for (root r, depth j). Disjoint id spaces so a shared block
# can never collide with a unique one.
SHARED_ID_BASE = 1_000_000
ROOT_STRIDE = 100_000
UNIQUE_ID_BASE = 900_000_000

# How many distinct shared prefix chains ("documents", "system prompts",
# "conversation roots") the population reuses. This is a REALISM knob, not a
# rate knob: the rate is hit by solving for chain depth, whatever this is set
# to. Fewer roots = a smaller hot working set that is easier for the server to
# keep resident; more roots = the same nominal hit rate spread over more KV,
# which a capacity-bound server will fail to deliver. 32 roots over 910
# requests is ~28 requests per root.
DEFAULT_NUM_ROOTS = 32


def stratified_uniforms(n: int) -> np.ndarray:
    """Midpoint rule: u_i = (i - 0.5) / n for i = 1..n.

    Step 2 rationale. Drawing n random uniforms would make the realized
    percentiles wobble around the targets by a few hundred tokens at n=910 --
    sampling noise that has nothing to do with the workload. The midpoint rule
    covers [0,1] evenly, so the realized distribution reproduces the target
    almost exactly and the trace *is* the distribution rather than one noisy
    draw from it. It is also the best quadrature rule for recovering the mean.
    """
    return (np.arange(1, n + 1) - 0.5) / n


def invert_cdf(u: np.ndarray, anchors: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear inverse CDF through the quantile anchors."""
    qs = np.array([q for q, _ in anchors])
    vs = np.array([v for _, v in anchors])
    return np.interp(u, qs, vs)


def build_isl(n: int) -> np.ndarray:
    """Step 2: sample ISL by inverting the quantile table."""
    return np.rint(invert_cdf(stratified_uniforms(n), ISL_QUANTILES)).astype(int)


def build_osl(n: int) -> np.ndarray:
    """Step 3: the two-component OSL mixture, both components stratified."""
    n_cap = int(round(n * OSL_CAP_FRACTION))
    n_body = n - n_cap
    cap = np.full(n_cap, OSL_CAP, dtype=int)
    body = np.rint(
        OSL_BODY_MIN + stratified_uniforms(n_body) * (OSL_BODY_MAX - OSL_BODY_MIN)
    ).astype(int)
    return np.concatenate([cap, body])


# ---------------------------------------------------------------------------
# Warm-cache construction
# ---------------------------------------------------------------------------


def build_hash_ids(
    isl: np.ndarray, roots: np.ndarray, depth_fraction: float
) -> list[list[int]]:
    """Tile each request into blocks; the leading `depth_fraction` are shared.

    For request i of length L: M = ceil(L / BLOCK_SIZE) blocks total, of which
    the first k = round(depth_fraction * M) are taken from its root's shared
    chain at depths 0..k-1, and the remaining M - k are freshly minted ids that
    are never reused by anyone.

    Because every request on a root takes chain depths 0,1,2,... in order, any
    two requests on the same root share a genuine PREFIX of min(k_i, k_j)
    blocks -- which is the only kind of sharing a prefix cache can exploit.
    Blocks shared in the middle of a sequence would not hit.

    The trailing block is partial whenever L is not a multiple of BLOCK_SIZE.
    aiperf handles this: `_build_token_sequence` detects
    `len(hash_ids) * block_size > input_length` and sizes the final block to
    `L - (M-1)*BLOCK_SIZE`. k is clamped below so that partial block is always
    a UNIQUE id -- see the comment in the loop for why sharing it would corrupt
    the prompt length.
    """
    out: list[list[int]] = []
    next_unique = UNIQUE_ID_BASE
    for length, root in zip(isl, roots):
        n_blocks = max(1, math.ceil(int(length) / BLOCK_SIZE))
        k = int(round(depth_fraction * n_blocks))
        # A shared block must never be the trailing PARTIAL one. aiperf caches
        # block tokens by hash_id on first synthesis, so if two requests with
        # different remainders shared a partial block id, the second would be
        # handed the first one's token list and its prompt would not match its
        # declared input_length. Full trailing blocks are safe to share.
        max_shared = n_blocks if int(length) % BLOCK_SIZE == 0 else n_blocks - 1
        k = max(0, min(k, max_shared))
        shared = [SHARED_ID_BASE + int(root) * ROOT_STRIDE + j for j in range(k)]
        unique = list(range(next_unique, next_unique + (n_blocks - k)))
        next_unique += n_blocks - k
        out.append(shared + unique)
    return out


def measure_hit_rates(
    isl: np.ndarray, hash_ids: list[list[int]]
) -> dict[str, float]:
    """Replay the trace against an infinite cache and report all three rates.

    Replicates aiperf's algorithm exactly (prefix_analyzer
    `_compute_per_request_hit_rates`): scan for the first block id never seen
    before, everything left of it is a hit, then mark every block of this
    request as seen. Infinite cache, no eviction -- so every number here is an
    UPPER BOUND on what a capacity-limited server will actually deliver.
    """
    seen: set[int] = set()
    hit_tokens = 0
    total_tokens = int(isl.sum())
    aiperf_rates: list[float] = []
    all_rates: list[float] = []

    for length, ids in zip(isl, hash_ids):
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

    return {
        "token_weighted": hit_tokens / total_tokens if total_tokens else 0.0,
        "aiperf": float(np.mean(aiperf_rates)) if aiperf_rates else 0.0,
        "per_request": float(np.mean(all_rates)) if all_rates else 0.0,
        "hit_tokens": hit_tokens,
    }


def solve_depth_fraction(
    isl: np.ndarray, roots: np.ndarray, target: float, tol: float = 1e-4
) -> tuple[float, dict[str, float]]:
    """Find the chain depth fraction whose token-weighted hit rate == target.

    A direct depth_fraction of F does NOT yield a hit rate of F, because the
    first request to touch each root pays a full miss on blocks nobody has seen
    yet. With R roots that cold-start cost is real and has to be compensated,
    so the depth is solved for numerically rather than set equal to the target.
    Hit rate is monotone in depth, so bisection is exact and cheap.
    """
    lo, hi = 0.0, 1.0
    best = build_and_measure(isl, roots, hi)
    if best["token_weighted"] < target - tol:
        raise SystemExit(
            f"Target cache level {target:.0%} is unreachable with "
            f"{len(set(roots.tolist()))} roots: even a fully shared chain only "
            f"reaches {best['token_weighted']:.1%}, because each root's first "
            f"request is always a cold miss. Use fewer roots (--num-roots) or "
            f"more requests (--num-requests)."
        )

    for _ in range(60):
        mid = (lo + hi) / 2
        stats = build_and_measure(isl, roots, mid)
        if stats["token_weighted"] < target:
            lo = mid
        else:
            hi = mid
    depth = (lo + hi) / 2
    return depth, build_and_measure(isl, roots, depth)


def build_and_measure(
    isl: np.ndarray, roots: np.ndarray, depth: float
) -> dict[str, float]:
    return measure_hit_rates(isl, build_hash_ids(isl, roots, depth))


def describe(values: np.ndarray) -> dict[str, float]:
    """Percentiles via numpy linear interpolation (k = (n-1)*p).

    This is the same definition aiperf `analyze-trace` uses, so the table
    printed here is directly comparable to what that command will report on the
    generated file.
    """
    return {
        "n": len(values),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--num-requests", type=int, default=N_REQUESTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--cache-level",
        type=float,
        default=0.0,
        help="Target TOKEN-WEIGHTED prefix cache hit rate, 0.0-1.0. "
        "0 emits the cold floor (no hash_ids).",
    )
    ap.add_argument("--num-roots", type=int, default=DEFAULT_NUM_ROOTS)
    args = ap.parse_args()

    if not 0.0 <= args.cache_level < 1.0:
        raise SystemExit("--cache-level must be in [0.0, 1.0)")

    n = args.num_requests
    rng = np.random.default_rng(args.seed)

    isl = build_isl(n)
    osl = build_osl(n)

    # Step 4: pair independently, then randomize arrival order.
    #
    # Both arrays arrive sorted from stratification, so they are shuffled
    # SEPARATELY before zipping. Shuffling the zipped pairs instead would weld
    # the largest ISL to the largest OSL and manufacture a correlation the data
    # does not support. Independent pairing is the honest default given that no
    # ISL/OSL joint distribution was measured -- it is an assumption, and if the
    # real workload correlates them (long context -> long answer, or the
    # reverse) this trace will not reproduce that.
    rng.shuffle(isl)
    rng.shuffle(osl)

    # Step 6: shared prefix structure. Drawn AFTER the shuffles so that
    # --cache-level 0 consumes no extra randomness and reproduces the original
    # cold-floor file byte for byte.
    hash_ids: list[list[int]] = [[] for _ in range(n)]
    stats: dict[str, float] | None = None
    depth = 0.0
    if args.cache_level > 0:
        roots = rng.integers(0, args.num_roots, size=n)
        depth, stats = solve_depth_fraction(isl, roots, args.cache_level)
        hash_ids = build_hash_ids(isl, roots, depth)

    # Step 5: emit. One JSON object per line, no trailing fields.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        for i_len, o_len, ids in zip(isl, osl, hash_ids):
            rec: dict[str, object] = {
                "input_length": int(i_len),
                "output_length": int(o_len),
            }
            if ids:
                rec["hash_ids"] = ids
            fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {n:,} requests to {args.output}")
    print(f"Seed: {args.seed} (regenerating with this seed reproduces the file exactly)")
    print()
    print(f"Total prefill tokens : {int(isl.sum()):>12,}")
    print(f"Total decode tokens  : {int(osl.sum()):>12,}")
    print(f"prefill:decode ratio : {isl.sum() / osl.sum():>12,.1f} : 1")
    print()

    if stats is None:
        print("Cache: COLD FLOOR (no hash_ids emitted)")
    else:
        shared_blocks = len({h for ids in hash_ids for h in ids if h < UNIQUE_ID_BASE})
        print(f"Cache: WARM, target token-weighted hit rate {args.cache_level:.0%}")
        print(f"  roots (distinct chains)  : {args.num_roots}")
        print(f"  solved chain depth       : {depth:.4f} of each request's blocks")
        print(f"  realized token-weighted  : {stats['token_weighted']:.2%}  <- the target")
        print(f"  realized aiperf metric   : {stats['aiperf']:.2%}  <- analyze-trace prints this")
        print(f"  cached prefill tokens    : {int(stats['hit_tokens']):,}")
        print(f"  recompute prefill tokens : {int(isl.sum()) - int(stats['hit_tokens']):,}")
        print()
        print(f"  hot working set          : {shared_blocks * BLOCK_SIZE:,} tokens of KV")
        print("    ^ must stay resident to realize the rate above. Compare against the")
        print("      'GPU KV cache size: N tokens' line in the vLLM startup log; if the")
        print("      working set does not fit alongside CONC live sequences, the server")
        print("      will evict and the measured hit rate will fall short.")

    print()
    print(
        f"max ISL + max OSL    : {int(isl.max() + osl.max()):>12,}  "
        f"-> max-model-len must exceed this"
    )


if __name__ == "__main__":
    main()
