# HiCache L3 (SSD KV cache) evaluation — GLM-5.2 FP8 on B300

**Date:** 2026-07-25 · **Hardware:** 8× B300 (`b300-netperf_00`) · **Engine:** SGLang v0.5.15.post1
**Workload:** agentic-coding trace, `semianalysis_cc_traces_weka_062126`, 393 sessions

---

## Recommendation

**Do not deploy HiCache L3 (Mooncake + SSD) for single-node serving of this workload.**
It functions correctly and contributes **0.03–0.4%** of served tokens across CCU 16–64,
including under conditions deliberately arranged to favour it.

**Invest in GPU KV capacity instead.** The same model on a DP-attention topology reaches
96–98% cache hit from GPU memory alone, versus 61% here. That is an 8× larger GPU pool, and it
removes the need for any host-side tier.

Keep host DRAM (L2) as configured, and do not enlarge it. It supplies 5–7% of input tokens at
every concurrency tested (11.3% of *cached* tokens at CCU 16), and that share does not grow with
concurrency. Whether removing L2 would actually cost those tokens was **not** isolated — the one
run that shrank it also changed concurrency (§6.6).

---

## 1. Problem

Agentic coding requests carry very large prefixes (median ~110k tokens, p95 ~410k). A prefix
found in cache costs nothing; a prefix not found must be recomputed by prefill, which is GPU
time. Throughput therefore scales as `1/(1 − hit_rate)`.

GPU memory holds only 2.73M tokens (140 GB) on this configuration — about 11 session contexts.
The proposition under test: a 16 TB NVMe tier can hold ~311M tokens for a fraction of the cost
of HBM, so it should absorb the misses and raise effective capacity.

**Question:** does an SSD-backed KV cache tier raise cache hit rate enough to matter?

---

## 2. Method

Four-tier hierarchy, each tier consulted on a miss of the one above:

```
  L1  GPU HBM              2.73M tokens / 140 GB
  L2  host DRAM            configurable via hicache-ratio
  L3a Mooncake DRAM        write staging for L3b
  L3b Mooncake SSD         16 TB on NVMe RAID0
```

Metric of record is `cached_tokens_by_source` from the engine's own counters, which attributes
every served token to the tier that supplied it. Latency was not used as a primary metric, as
it confounds cache effects with queueing.

Sequence of tests, each 3600 s except where noted:

| # | Run | CCU | L2 ratio | L3 store at start | Purpose |
|---|---|---|---|---|---|
| 0 | 30075783267 | 12 | 1.0 | — | Pre-existing baseline. **113 s only**, L3 non-functional |
| 1 | 30141767135 | 32 | 1.0 | empty | 90 s smoke: verify L3 writes and reads at all |
| 2 | 30143803931 | 32 | 1.0 | empty | Warmth point 1 |
| 3 | 30146707202 | 32 | 1.0 | 126 GB | Warmth point 2 |
| 4 | 30150276114 | 32 | 10.0 | n/a | DRAM control — **failed**, see §5 |
| 5 | 30153620526 | 64 | **0.25** | 196 GB | Decisive test: starve L2, force traffic to L3 |
| 6 | 30159497063 | 16 | 1.0 | 250 GB | Full-length low-concurrency reference, replacing test 0 |

Test 5 is the critical one. L2 was reduced to 0.68M tokens so that L1 misses had no destination
except the SSD tier, at high offered concurrency, against a store already warmed by two prior
runs of the identical trace.

---

## 3. Results

| # | CCU | L2 ratio | L1 GPU hit | L2 host hit | **L3 hit** | overall hit |
|---|---|---|---|---|---|---|
| 0† | 12 | 1.0 | 0.622 | 0.340 | — | 0.962 |
| 6 | **16** | 1.0 | 0.553 | **0.071** | 0.00254 | 0.627 |
| 2 | 32 | 1.0 | 0.535 | 0.051 | 0.00298 | 0.589 |
| 3 | 32 | 1.0 | 0.538 | 0.053 | 0.00226 | 0.594 |
| 5 | 64 | 0.25 | 0.609 | 0.00027 | 0.00029 | 0.610 |

† 113 s window, 101 requests. **Contradicted by test 6** — at full duration and adjacent
concurrency, L2 serves 0.071, not 0.340. Treat the 0.340 as a short-window artifact.

Reference, same model, DP-attention topology (run 30099003901): **L1 GPU hit 0.96–0.98**,
host tier 0.0007–0.001.

**Result 1 — more store did not help.** Between tests 2 and 3 the store grew from empty to
126 GB. L3 hit rate did not rise (0.00298 → 0.00226); no other metric moved.

**Result 1b — no concurrency favours the host tiers.** Across every full-length run, L2 serves
5–7% and L3 serves 0.2–0.4%. L3 peaks at CCU 16 (0.00254) and is flat thereafter. There is no
operating band in which either host tier becomes material.

**Result 2 — L3 does not substitute for DRAM.** Starving L2 in test 5 removed its contribution
(0.053 → 0.00027). L3 did not absorb it (0.00226 → 0.00029). Overall hit was unchanged.

Token attribution, test 5:

```
  L1 GPU                  54,470,720   99.91%
  L3 SSD                      25,856    0.05%
  L2 host                     24,576    0.05%
```

**Result 3 — the tier was exercised, and returned misses.** Store-side request counters:

```
  ExistKey (lookups)   21.6 req/s   →  46,932 keys/s
  Get      (fetches)    9.6 req/s   →   1,309 items/s   ≈ 2.8% conversion
```

The engine interrogated the store ~47,000 keys per second; it answered "not present" ~97% of
the time. Store utilisation at run end: 433 GB of 16 TB (2.7%), 491,152 keys. The tier was
neither idle nor full.

---

## 4. Mechanism

The store was interrogated and returned misses, so the constraint is not capacity or wiring.
Two kinds of reuse exist, and L3 reaches neither:

| Reuse | Timescale | Why L3 misses it |
|---|---|---|
| Within-run (turn N → N+1) | seconds | Write path is GPU → DRAM segment → eviction → disk. The reuse moment passes before data lands. |
| Cross-run (same trace later) | hours | Prefix keys form a chain — each page's key depends on all preceding tokens. Generated tokens enter the next context and are not bit-identical between runs, so chains diverge after the first generated token. |

Confidence: §3 is measured. This section is inference consistent with it; key-chain divergence
was not tested in isolation.

## 5. Defects found and fixed

The evaluation began by repairing the feature, which had never worked. All fixed and committed.

| Defect | Effect |
|---|---|
| `offload_on_evict` not set on the Mooncake master | SSD tier mounted but received **0 bytes**; 990 eviction attempts, 0 successes |
| `global_segment_size` hardcoded `1gb` | Saturated in seconds; master spun 74,531 eviction retries in a ~10 ms loop |
| Store path keyed by run ID and concurrency | Every job started against an empty store, measuring warm-up |
| `hicache-ratio` misunderstood as global | It is **per TP rank**. Ratio 10 requested 1226 GB × 8 ranks ≈ 9.8 TB against 3023 GB of RAM; OOM-killed the container and the CI runner (test 4). Ceiling is ~2.4 |
| Dispatch preflight `--allow-unverified-model` | Nulled the value it then required; unusable for any recipe |
| Runner env allowlist | Silently stripped four tuning variables, so documented overrides had no effect |

Test 4 (the equal-DRAM control) was lost to the fourth defect and could not be repeated within
the hardware window.

---

## 6. Limitations

1. **Equal-DRAM control not performed.** `hicache-ratio` is per TP rank, capping L2 at ~2.4×
   (~2 TB) — too small to reach a meaningful fraction of the working set. Test 4 OOM'd attempting
   10×.
2. **Concurrency coverage** is CCU 16, 32 and 64 at full duration. CCU 20–28 untested, but the
   16→64 trend is flat, so a hidden band there is unlikely.
3. **SSD read bandwidth/latency unmeasured** — L3 never served enough reads to sample. The cost
   side of the trade is unquantified.
4. **Single model/precision** (GLM-5.2 FP8, 51.4 KB/token). Do not generalise to different KV
   costs.
5. **Cross-instance reuse untested.** Where prefixes are stable and shared across tenants (e.g. a
   common system prompt), key-chain divergence does not apply. This is L3's remaining plausible
   case.
6. **L2's causal value not isolated.** Its 5–7% hit share is measured, but no run removed L2 with
   concurrency held constant — test 5 changed both. So "keep L2" rests on the share, not on a
   measured penalty for removing it. A single CCU 32 run at `hicache-ratio 0.25` would settle it.

## 7. Reproduction

```
  Config keys   glm5.2-fp8-b300-sglang-eagle-hicache-r10-l3-mooncake-ssd
                glm5.2-fp8-b300-sglang-eagle-hicache-r025-l3-mooncake-ssd-c64
  Recipe        benchmarks/single_node/agentic/glm5.2_fp8_b300-netperf_sglang.sh
  Branch        vng-benchmark
  Method doc    docs/HICACHE_L3_SMOKE_TEST.md   (segment sizing, 40-min pre-check)
  Detail log    docs/HICACHE_L3_MOONCAKE_PLAN.md
```

Before any further L3 work, run the pre-check in `HICACHE_L3_SMOKE_TEST.md` §5. A misconfigured
L3 tier reports success and silently serves nothing — that is how the original defect survived.
