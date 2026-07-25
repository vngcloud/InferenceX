# HiCache L3 (Mooncake + SSD) bring-up plan — GLM-5.2 FP8 B300

Status: **COMPLETE — negative result, see §10**. Target config key
`glm5.2-fp8-b300-sglang-eagle-hicache-r10-l3-mooncake-ssd`, recipe
`benchmarks/single_node/agentic/glm5.2_fp8_b300-netperf_sglang.sh`, node `b300-netperf_00`.

## 1. Why the existing L3 run was a no-op

Reference run: [30075783267](https://github.com/vngcloud/InferenceX/actions/runs/30075783267)
(job 89426121271) — "GLM-5.2 FP8 B300 SGLang EAGLE HiCache L3 Mooncake SSD CCU12 smoke".
The job went green, but Mooncake never served a single read.
oms
Evidence from `mooncake_master.log` (154,724 lines) and the result JSON:

```
enable_offload=1, offload_on_evict=0, offload_force_evict=0
Mem Storage: 967.38 MB / 1.00 GB (94.5%) | SSD Storage: 0 B / 16.00 TB | Keys: 976
Eviction: Success/Attempts=0/990, AllocFail=0, keys=0, size=0 B
[EVICT-TRIGGER] memory_ratio=0.944 high_watermark=0.95 need_mem_eviction=1   x74,531
```

```json
"external_cache_hit_rate": null,
"cached_tokens_by_source": {"device": 7586496, "host": 4141568}
```

Three defects, in order of severity:

1. **`offload_on_evict=0` — fatal.** `--enable_offload=true` mounts the disk segment (the
   master correctly detects 16 TB on `/mnt/test-raid0`) but data only reaches SSD *via
   eviction*, and that path was disabled. All 990 eviction attempts freed 0 keys / 0 bytes,
   so the master spun 74,531 eviction triggers in a ~10 ms loop for the whole run. Reading:
   with a disk tier present but `offload_on_evict=0` the master will not destroy the only
   replica, so eviction can never succeed. `soft-pinned: 0` and `AllocFail=0` rule out
   lease/pin and allocator pressure.
2. **`global_segment_size: "1gb"` on a 2.9 TB node.** Divided by TP
   (`per_tp_global_segment_size = global_segment_size // tp_size`,
   `mooncake_store.py:396`), so each of 8 ranks got 128 MB. Saturated to 94.5% in minutes.
3. **`--hicache-storage-prefetch-policy timeout`** — the conservative policy, and
   **`--hicache-write-policy write_back`**, which only pushes pages down on eviction. Neither
   populates nor reads an L3 store aggressively.

Secondary: the smoke measured only **113 s / 101 requests** — far too short to warm any
deep tier even if the above were fixed.

## 2. Node facts (read from CI logs; no SSH required)

| | |
|---|---|
| `/mnt/test-raid0` | **16.00 TB**, already bind-mounted (`-v /mnt/test-raid0:/mnt/test-raid0`) |
| Host DRAM | 2,964,436 MiB ≈ **2895 GiB** (`configs/runners.yaml`, `cluster:b300-nv`) |
| Harness DRAM budget | `TOTAL_CPU_DRAM_GB=2399` at `dram-utilization: 0.80` |
| GLM-5.2 FP8 KV cost | **51.4 KB/token** (`KV size: 140.35 GB` for 2,728,896 tokens) + 1.80 GB EAGLE draft KV |
| L1 GPU pool | 2,728,896 tokens (140 GB) — saturates at CCU 12 already (`gpu_usage_pct: 1`) |
| L2 host pool today | `hicache_ratio=1.0` → ~142 GB, i.e. **94% of the DRAM budget is unused** |

## 3. Tier plan

| Tier | Setting | Capacity | Tokens |
|---|---|---|---|
| L1 GPU | `mem-fraction-static 0.88` | 140 GB | 2.73M |
| L2 host (hicache) | `--hicache-ratio 1.0` *(unchanged)* | 142 GB | 2.73M |
| L3 Mooncake DRAM | `global_segment_size` derived, [smoke §3](HICACHE_L3_SMOKE_TEST.md) | 140 GB @ 3600 s/CCU 32 | ~2.7M |
| L3 Mooncake SSD | auto-detected on `/mnt/test-raid0` | 16 TB | ~311M |
| Headroom | weights page cache, aiperf, tokenizers | ~1230 GB | |

L2 is deliberately left at `ratio 1.0` so this run is a clean A/B against
run 30075783267 and so the deep tier is actually exercised. Growing L2 instead would just
reproduce the FP4 trap (see §6).

Trace working-set estimate: 393 sessions x ~200–300k final-turn context x 51.4 KB
≈ **4–6 TB**. Past any DRAM configuration, comfortably inside 16 TB of SSD. That is the
case for L3.

## 4. Changes

### 4.1 Mooncake master (`recipe:43`) — the unblock

```bash
mooncake_master --rpc_port=50051 --metrics_port=9003 \
    --enable_offload=true \
    --offload_on_evict=true \
    --eviction_high_watermark_ratio=0.85 \
    --eviction_ratio=0.10 \
    --logtostderr=true
```

`--offload_on_evict=true` is the one-line fix. Watermark 0.85 + ratio 0.10 begins offload
before saturation and frees 2x more per pass, which removes the 10 ms thrash loop.

### 4.2 SGLang client flags (`recipe:79`, `recipe:97-101`)

```
--hicache-write-policy write_through              # was write_back (moves into the L3 block)
--hicache-storage-prefetch-policy wait_complete   # was timeout
--hicache-storage-backend-extra-config '{
  "master_server_address":"127.0.0.1:50051",
  "metadata_server":"P2PHANDSHAKE",
  "local_hostname":"127.0.0.1",
  "protocol":"tcp",
  "global_segment_size":"<derived, see HICACHE_L3_SMOKE_TEST.md §3>",
  "enable_ssd_offload":true,
  "ssd_offload_path":"<MOONCAKE_SSD_DIR>",
  "prefetch_threshold":64
}'
```

- `write_through` is the aggressive write policy: every page goes to L2+L3 as produced,
  rather than only on eviction. Required to populate the store.
- `wait_complete` blocks the batch until a prefetch fully lands
  (`can_terminate = completed`, `hiradix_cache.py:1451`). Validated against the pinned build:
  `choices=["best_effort","wait_complete","timeout"]` (`server_args.py:2044`).
- `prefetch_threshold` 256 -> 64 (= `page_size`) so short prefixes are still fetched.
- `page_first_direct` + `--hicache-io-backend direct` is retained: mooncake rejects
  `layer_first` and normalizes to exactly this pair (`server_args.py:5414`).

### 4.3 CCU ladder (`configs/nvidia-master.yaml:8029`)

`conc-list: [32, 48, 64]`, ramping up. Duration 3600 s, ingest off. Order is preserved into
the matrix (`conc_values = conc_list`, `generate_sweep_configs.py:459`), so 32 runs first and
gives an early read on whether L3 works at all before the two expensive rungs.

Low concurrency cannot demonstrate L3: L1 and L2 are already 100% full at CCU 12
(`gpu_usage_pct: 1`, `cpu_usage_pct: 0.9998`), so anything that fits in those tiers never
reaches the store. All three rungs are past that point.

Oversubscription, stated plainly. The GPU pool holds ~17 full contexts (2.73M tokens / ~157k
mean input):

| CCU | live set needed | vs pool | `max_running_requests` | admitted vs resident |
|---|---|---|---|---|
| 32 | ~5.0M tok | 1.8x | 64 | 3.8x |
| 48 | ~7.5M tok | 2.8x | 96 | 5.6x |
| 64 | ~10.1M tok | 3.7x | 128 | 7.5x |

This is the regime L3 exists for, and also the regime where a *non-working* L3 reproduces
upstream's c32 failure mode (TTFT p50 = 207 s, prefix hit 0.098, run 29741710665). Either
outcome is informative; a run that fits comfortably in L1/L2 is not.

No CCU-matched A/B against run 30075783267 (c12) exists at this ladder. Acceptance criteria
1 and 2 below do not require one — they are absolute, not comparative.

## 5. Known risks

1. **`wait_complete` has no timeout escape**, and the decision is all-reduced across TP
   workers (`_all_reduce_attn_groups`), so one slow rank stalls the batch. Over `protocol:
   tcp` against SSD, a cold multi-hundred-MB fetch can take seconds. Expect TTFT p99 to
   degrade; a stalled fetch can hang a request to the 1800 s watchdog. Fallback — a "soft
   wait_complete" with a bounded blast radius (defaults are `base=2.0`, `per_ki_token=0.1`,
   `max=30.0`):
   `"prefetch_timeout_base":5.0,"prefetch_timeout_per_ki_token":0.5,"prefetch_timeout_max":120.0`
   with `--hicache-storage-prefetch-policy timeout`.
2. **The upper rungs may collapse rather than showcase L3** — see the table in §4.3. A
   working L3 should convert the recompute into an SSD read; that is the experiment. If it
   collapses anyway, that is a result, not a misconfiguration. CCU 32 runs first precisely so
   this is known before 48 and 64 spend node time.
3. **`max_running_requests = 2 * CONC`** stays at the established recipe formula. Capping it
   (e.g. to 32) is the first lever if the run thrashes, and is cheaper to retry than changing
   the ladder.
4. **1024 GB Mooncake DRAM segment** is untested on this node. If the container OOMs, drop
   to `512gb`.

## 6. Do not run this on the FP4 config

`glm5.2-fp4-b300-sglang-eagle-hicache-r15-dpattn-cacheaware-agentic` sits at 96–98%
`gpu_cache_hit_rate` with a 21.9M-token pool (run 30099003901), and its existing DRAM tier
already contributes ~nothing (`cpu_cache_hit_rate` 0.00069–0.00101). L3 cannot help a config
whose working set already fits in GPU. This FP8 recipe runs at `gpu 0.62 / cpu 0.34`, which
is where the headroom is.

## 7. Acceptance criteria

This run answers exactly one question — **does the L3 path function** — and it is judged on
absolute signals only. It cannot answer "is L3 faster", see §7.1.

1. `mooncake_master.log`: **`SSD Storage:` non-zero**, and `Eviction: Success/Attempts` is
   not `0/N`. If SSD stays at 0 B, `offload_on_evict` did not take and nothing else matters.
2. Result JSON: `external_cache_hit_rate` is **no longer `null`**, and
   `cached_tokens_by_source` gains a third source beyond `device`/`host`.
3. Recorded for later use, **not** pass/fail: per-tier hit rates, TTFT p50/p95, total tok/s,
   and whether `wait_complete` moved the TTFT tail.

### 7.1 Do not compare this run to 30075783267

Eight parameters differ between the two runs: `offload_on_evict`, both eviction watermarks,
`global_segment_size` (1gb -> 1024gb), write policy, prefetch policy, `prefetch_threshold`,
CCU (12 -> 32, so `max_running_requests` 24 -> 64), and duration (90 s -> 3600 s). No delta
between them is attributable to any single change.

The tempting number is actively misleading. That run's `overall_cache_hit_rate: 0.96196`
decomposes as **L1 0.62226 + L2 0.33970 — L3 contributed zero.** It is an "L1+L2 were
sufficient" result measured at a concurrency where the working set fit. At CCU 32 it will not
fit, so this run can report a *lower* overall hit rate while L3 works correctly. Treating
0.96196 as a baseline would score a success as a regression.

A real perf comparison needs the matched-CCU, matched-duration arm-B control in §8.2. Until
that exists there is no valid baseline for this config, and none should be implied.

## 8. Follow-up: the real comparison run

**Prepare after CCU 32 of run 30139838426 lands. Do nothing here until acceptance criteria 1
and 2 hold — if L3 is still a no-op, none of this matters.**

Run 30139838426 is still a bring-up: it proves the L3 path works and nothing more. Two
things in it are deliberately *not* tuned for performance, and both must be unlocked before
any number is comparable to a best run.

### 8.1 Every job starts with an empty store — the biggest problem

`ssd_offload_path` is `/mnt/test-raid0/mooncake/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-c${CONC}`,
unique per run **and per CCU**. So c32, c48 and c64 each begin against a cold 16 TB tier and
spend the measured window populating it. That measures warm-up, not steady state, and it
systematically understates L3.

The trace is 393 sessions replayed for 3600 s, so a cold store cannot reach the reuse regime
L3 exists for. Options, cheapest first:

1. Drop `-c${CONC}` from the path so the three CCUs share one store and 48/64 inherit 32's
   fill. Asymmetric — c32 stays cold — but nearly free.
2. Drop `${GITHUB_RUN_ID}` too, giving a persistent store across runs. Fastest path to a warm
   measurement, but results then depend on prior runs, so it must be stated in any comparison
   and the directory wiped deliberately between experiments.
3. Add an explicit pre-measurement warm pass. Correct and self-contained, most work.

Whichever is chosen, the store state at window start has to be reported alongside the
numbers, or the comparison is not interpretable.

### 8.2 L2 is throttled on purpose, and it competes with L3 for the same DRAM

`hicache-ratio 1.0` gives L2 ~142 GB of a 2399 GB budget — chosen in §3 to force traffic
into the store, not because it is good. The trap: **L2 and the Mooncake DRAM segment are the
same physical memory.** You cannot maximise both; a real run has to pick a split.

That makes the decisive experiment a three-way, not a before/after:

| Arm | L2 (hicache) | Mooncake DRAM | Mooncake SSD | Question it answers |
|---|---|---|---|---|
| A (current) | 142 GB / 2.73M tok | 1024 GB | 16 TB | does the L3 path function |
| B — **all-DRAM control** | ~1800 GB / ~34.7M tok | none | none | is L3 better than simply spending the DRAM on L2? |
| C — balanced | ~600 GB / ~11.6M tok | 1024 GB | 16 TB | best achievable with L3 |

**Arm B is the one that can kill the feature**, and it does not exist yet. On a single node
Mooncake's DRAM segment is redundant with L2 — same memory, one extra hop — so L3 only earns
its place via the 16 TB SSD. If B beats C, Mooncake is not worth running single-node and the
FP4 result (§6) is the honest best run.

Hypothesis worth stating so it can be falsified: at 51.4 KB/token, the ~7k tok/s of real
uncached prefill this trace sustains is only ~360 MB/s of KV-equivalent, while a RAID0 NVMe
array reads at multiple GB/s. So an SSD hit should beat recompute by roughly 5–20x, and C
should beat B once the store is warm. If it does not, suspect §8.1 (cold store) before
concluding the tier is useless.

### 8.3 Not comparable to run 30099003901 as-is

The current best run is FP4, TP8/**EP8**/DP-attention, MTP, cache-aware router, 21.9M-token
GPU pool. This L3 config is FP8, TP8/**EP1**, no DP-attention, no router, 2.73M-token pool —
an 8x smaller L1 and a different precision. Any throughput gap between them is dominated by
topology and precision, not by L3.

To compare against the best run rather than to a strawman, the FP8 arm needs `ep: 8`,
`dp-attn: true`, the `sglang-router` with `cache_aware`, and `lpm` (already present). Note
this cuts both ways: DP-attention raises the GPU pool ~8x, which shrinks the headroom L3 has
to work with — the same reason §6 says not to bother on FP4. Expect the honest answer to be
"L3 helps small-L1 configs and does little for large-L1 ones", and design the arms to show
that rather than to hide it.

### 8.4 Carried-over levers

- `max_running_requests = 2 * CONC` (§5.3) — cap it if the upper rungs thrash.
- `MOONCAKE_PREFETCH_POLICY` / `MOONCAKE_GLOBAL_SEGMENT_SIZE` are env-overridable, so retries
  need no commit.
- If `wait_complete` wrecks the TTFT tail, switch to the bounded `timeout` variant in §5.1
  before abandoning the aggressive policy.

## 9. The two factors that decide the result

Everything else is plumbing. These are the variables that determine whether L3 helps.

### 9.1 Store warmth — the gating factor

L3 can only help on a prefix it already holds. Working set is ~4-6 TB; fill rate is
~9.7 GB/min at CCU 32, and one 3600 s job delivers ~88 min of traffic (~28 min warmup +
60 min measured) ~= **850 GB**. So a single run covers ~15-20% of the working set at best,
and its *measured window* starts at only ~6%.

Consequence: one run cannot show L3's value, regardless of configuration. Warmth has to
accumulate across runs on the stable store path, which makes coverage-at-window-start the
independent variable and `external_cache_hit_rate` the dependent one. Report both, always.

### 9.2 `wait_complete` — the cost factor

`wait_complete` maximises hit rate by blocking a batch until a prefetch fully lands, with no
timeout escape and an all-reduce across TP ranks (`hiradix_cache.py:1451`). Smoke run
30141767135 returned **TTFT p50 9.19 s / p95 50.5 s** at CCU 32. That sample is confounded
(cold store, 49 profiled requests, L1 and L2 both saturated) so it is not yet a verdict — but
it is the number that decides whether an aggressive policy is usable.

The expected shape: as coverage rises, SSD hits replace recompute and the TTFT tail should
*improve*. If it does not improve with warmth, `wait_complete` is the wrong trade and the
bounded `timeout` variant in §5.1 is the fallback.

## 10. RESULT — host-tier caching is irrelevant on this config

Status: **experiment complete.** L3 works mechanically and contributes nothing.

### 10.1 All runs

| run | CCU | L2 ratio | store at open | GPU hit | host hit | L3 hit | overall |
|---|---|---|---|---|---|---|---|
| CCU 12 ref | 12 | 1.0 | — | 0.622 | **0.340** | — | **0.962** |
| smoke | 32 | 1.0 | 0 | 0.656 | 0.032 | 0.00025 | 0.688 |
| warmth 1 | 32 | 1.0 | 4 KB | 0.535 | 0.051 | 0.00298 | 0.589 |
| warmth 2 | 32 | 1.0 | 126 GB | 0.538 | 0.053 | 0.00226 | 0.594 |
| final | 64 | **0.25** | 196 GB | 0.609 | 0.00027 | 0.00029 | 0.610 |

### 10.2 The decisive test

The final run starved L2 to 0.68M tokens so that L1 misses had nowhere to go but the warm SSD
tier. L2's contribution duly collapsed — and **L3 did not pick it up**:

```
                 L2 hit      L3 hit      overall
  ratio 1.00     0.053       0.00226     0.594
  ratio 0.25     0.00027     0.00029     0.610     <- L2 removed, L3 flat, overall UNCHANGED
```

Token sources in the final run: `device 99.91%`, `host 0.05%`, `storage_MooncakeStore 0.05%`.
**The GPU tier does 99.9% of all cache serving.** Both host tiers are noise, and removing L2
entirely did not hurt — overall hit actually rose slightly, because the GPU tier gained.

### 10.3 Why — reuse interval vs tier residency

A tier below HBM only helps when a prefix is re-requested **before** that tier evicts it. The
CCU 12 reference shows the content itself is highly reusable (96.2% hit, with L2 serving 0.340
of it). What changes with concurrency is the reuse *interval*:

```
  CCU 12   reuse interval  <  L2 residency        → L2 serves 34% of tokens
  CCU 32   reuse interval  >  L2 residency        → L2 serves 5%
  CCU 64   reuse interval >> L2 and L3 residency  → both serve ~0.03%
```

At high concurrency the request stream churns faster than any host tier can retain a prefix, so
capacity below HBM stops mattering. Adding a 16 TB SSD does not fix a residency problem.

### 10.4 Conclusions

1. **Do not deploy HiCache L3 (Mooncake+SSD) for this config.** It functions correctly and
   contributes 0.03% of tokens even when it is the only tier below HBM with a warm store.
2. **The lever is GPU pool size, not host capacity.** The FP4 DP-attention config with a
   21.9M-token pool reaches 96–98% GPU hit (run 30099003901). This FP8 TP8/EP1 config has 2.73M
   and cannot be rescued by DRAM or SSD.
3. **Host DRAM (L2) is worth keeping at low concurrency only** — 0.340 hit at CCU 12 versus 0.05
   at CCU 32. Its value is a function of offered load, not of its size.
4. **L3's remaining plausible use case is cross-instance or cross-restart reuse**, where the
   reuse interval is hours rather than seconds — not single-node steady-state serving. Untested.

### 10.5 What this experiment did not settle

- **Equal-DRAM L2 vs L3.** Arm B was attempted at `hicache-ratio 10.0` and OOM-killed the node's
  runner; `hicache-ratio` is applied **per TP rank** (~8 x ratio x 122.6 GB), so the ceiling is
  ~2.4 and the intended 10x L2 comparison is impossible on this node. Corrected to 2.0 in
  `configs/nvidia-master.yaml`.
- **The middle concurrency band (CCU 16-24)** where L2 residency may still hold but L1 overflows.
  Never run; the most likely place for a positive result if this is revisited.
- **SSD read bandwidth and latency.** Never measured, because L3 never served enough reads.

## 11. References

- Broken L3 run: [30075783267](https://github.com/vngcloud/InferenceX/actions/runs/30075783267)
- FP4 DP-attn + MTP baseline: [30099003901](https://github.com/vngcloud/InferenceX/actions/runs/30099003901)
- Upstream plain-TP8 collapse at c16/c32: [29741710665](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/29741710665)
- Smoke test / segment sizing math: [HICACHE_L3_SMOKE_TEST.md](HICACHE_L3_SMOKE_TEST.md)
- Dispatch workflow: `.agents/skills/inferencex-agentic-dispatch/SKILL.md`
