# HiCache L3 (Mooncake + SSD) bring-up plan — GLM-5.2 FP8 B300

Status: **plan, not yet validated**. Target config key
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
| L3 Mooncake DRAM | `global_segment_size: 1024gb` | 1024 GB | ~19.9M |
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
  "global_segment_size":"1024gb",
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

Must all hold, or the run is another no-op:

1. `mooncake_master.log`: **`SSD Storage:` non-zero**, and `Eviction: Success/Attempts` is
   not `0/N`. If SSD stays at 0 B, `offload_on_evict` did not take and nothing else matters.
2. Result JSON: `external_cache_hit_rate` is **no longer `null`**, and
   `cached_tokens_by_source` gains a third source beyond `device`/`host`.
3. `overall_cache_hit_rate` at CCU 12 vs the 0.96196 baseline from run 30075783267.
4. TTFT p50/p95 and total tok/s per CCU, with an explicit note on whether `wait_complete`
   moved the TTFT tail.

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

## 9. References

- Broken L3 run: [30075783267](https://github.com/vngcloud/InferenceX/actions/runs/30075783267)
- FP4 DP-attn + MTP baseline: [30099003901](https://github.com/vngcloud/InferenceX/actions/runs/30099003901)
- Upstream plain-TP8 collapse at c16/c32: [29741710665](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/29741710665)
- Dispatch workflow: `.agents/skills/inferencex-agentic-dispatch/SKILL.md`
