# HiCache L3 smoke test — validating the eviction→SSD trigger cheaply

Companion to [HICACHE_L3_MOONCAKE_PLAN.md](HICACHE_L3_MOONCAKE_PLAN.md): that doc is the
experiment, this is the pre-check that the plumbing works.

Validated by run 30141767135 — 272 GB reached SSD, `external_cache_hit_rate` 0.00025,
`cached_tokens_by_source` gained `storage_MooncakeStore`. Run this before any 3600 s run; a
misconfigured L3 is silent, not loud.

## 1. The one constraint everything follows from

SSD is never written directly. It sits downstream of eviction, and eviction only fires above a
watermark:

```
      KV pages produced by prefill/decode
                    │
                    ▼
        ┌───────────────────────┐
        │ L1   GPU        140 GB│   2.73M tokens
        └───────────┬───────────┘
                    │  hicache
                    ▼
        ┌───────────────────────┐
        │ L2   host DRAM  142 GB│   --hicache-ratio 1.0
        └───────────┬───────────┘
                    │  write_through  →  Mooncake put()
                    ▼
        ┌───────────────────────┐
        │ L3a  Mooncake DRAM  S │ ◄── fills at ~9.7 GB/min @ CCU 32
        └───────────┬───────────┘
                    │
                    │   ONLY path to disk: eviction, and
                    │   eviction only fires once fill > 0.85·S
                    │   (and only with offload_on_evict=true)
                    ▼
        ┌───────────────────────┐
        │ L3b  SSD       16 TB  │
        └───────────────────────┘
```

`global_segment_size` (`S`) therefore sets a **deadline**, not a capacity: too large and the
watermark is never crossed inside the run, so nothing offloads.

## 2. The measurement

From run 30141767135 (CCU 32, `write_through`, 6gb segment, offload active):

```
  272 GB written to SSD over ~28 min of traffic

  rate                  =  272 GB / 28 min   ~=  9.7 GB/min
  per unit concurrency  =  9.7 / 32          ~=  0.31 GB/min/CCU
```

Assumes write volume scales with in-flight requests, since `write_through` pushes every page
produced. First-order, adequate.

## 3. The derivation

Let `t_cross` be the moment fill reaches the watermark:

```
        rate · t_cross  =  0.85 · S

                            0.85 · S
        t_cross      =    ────────────
                             rate
```

Then choose *when* that should happen — a fraction `f` of the window, `t_cross = f · DURATION`:

```
        0.85 · S
      ───────────  =  f · DURATION
          rate

                     f · DURATION · rate
        S      =    ─────────────────────
                            0.85
```

Substituting `rate = 0.31 · CONC` GB/min, `DURATION` in seconds (÷60 → minutes), `f = 0.20`:

```
              0.20 · (DURATION/60) · 0.31 · CONC
        S  =  ──────────────────────────────────
                            0.85
```

Clearing decimals — rate ×100 (`0.31 → 31`), watermark ×100 (`0.85 → 85`), and `1/f = 5`:

```
                 31 · CONC · DURATION            31 · CONC · DURATION
        S  =  ─────────────────────────   =    ──────────────────────
                  60 · 85 · 5                          25500
                  ▲    ▲    ▲
                  │    │    └─ 1/f, the 20% target
                  │    └────── watermark ×100
                  └─────────── seconds → minutes
```

The two ×100 factors cancel, which is why it lands on a clean integer form for bash. Implemented
in `benchmarks/single_node/agentic/glm5.2_fp8_b300-netperf_sglang.sh`, floored at 4gb (1gb proved
pathological) and capped at 512gb (host RAM safety).

Resulting values:

| DURATION | CONC 12 | CONC 32 | CONC 64 |
|---|---|---|---|
| 90 s   | 4gb  | 4gb       | 7gb   |
| 3600 s | 52gb | **140gb** | 280gb |

## 4. Why `S` must be small

Fill timeline, CCU 32, 3600 s window. `▓` = filling, no offload. `█` = offload active.

```
 t=0min      12          24          36          48        60min
 ├───────────┼───────────┼───────────┼───────────┼───────────┤

 S=1gb     ▓ 0:05   watermark hit, but offload_on_evict was off → 0 B to SSD
 S=1024gb  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  never reached → 0 B to SSD
 S=140gb   ▓▓▓▓▓▓▓▓▓▓█████████████████████████████████████████  crossed 11:53
 S=4gb     ▓█████ (90 s smoke)                                   crossed 0:20
```

Too small is recoverable; too large silently yields zero bytes on disk. When in doubt, undersize.

## 5. Running the smoke test

`duration-override` is a workflow input, and `S` derives from it automatically, so no commit is
needed to switch between smoke and real:

```bash
gh workflow run e2e-tests.yml --repo vngcloud/InferenceX --ref vng-benchmark \
  -f 'generate-cli-command=test-config --config-files configs/nvidia-master.yaml --config-keys glm5.2-fp8-b300-sglang-eagle-hicache-r10-l3-mooncake-ssd --conc 32 --runner-node-filter b300-netperf_00 --scenario-type agentic-coding --no-evals' \
  -f 'test-name=TRIGGER TEST offload_on_evict CCU 32 90s seg6gb no-ingest' \
  -f ref=vng-benchmark -f duration-override=90 -f skip-agentic-ingest=true
```

Expect **~40 min wall-clock**, not 90 s: `duration-override` sets only the measured window, and
model load (~8 min) plus aiperf warmup (~28 min) dominate. The long warmup is what fills L3.

Overrides now reach the container (`MOONCAKE_GLOBAL_SEGMENT_SIZE`, `MOONCAKE_PREFETCH_POLICY`,
`MAX_RUNNING_REQUESTS_CAP`, `SCHEDULE_CONSERVATIVENESS` were added to `RUN_ENV` in
`runners/launch_b300-netperf.sh`; before that the launcher silently stripped them).

## 6. Reading the result

Pull the per-job artifact and check `mooncake_master.log`. These signals are absolute — no
baseline run is needed or valid (see plan §7.1):

```bash
gh run download <RUN_ID> --repo vngcloud/InferenceX \
  -n "server_logs_glm5.2_tp8_conc32_kvdram-hicache_spec-mtp_fp8_sglang_tp8-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_conc32_b300-netperf_00" -D out

grep -oE "Mem Storage: [^|]*\| NVMe-oF SSD: [^|]*\| SSD Storage: [^|]*\| Keys: [0-9]+" \
  out/results/mooncake_master.log | tail -3
grep "Master Admin Metrics" out/results/mooncake_master.log | tail -1 \
  | grep -oE "Eviction: [^|]*\| Mem Eviction: [^|]*"
```

| Observation | Meaning | Next step |
|---|---|---|
| `SSD Storage:` non-zero and rising; `Mem Storage` recycling below its cap | works | run 3600 s |
| `SSD Storage: 0 B`, `Mem Storage` pinned at cap | offload refuses | `--offload_force_evict=true` |
| `SSD Storage: 0 B`, `Mem Storage` well below cap | watermark not reached, `S` too large | recheck the rate in §2 |
| `Mem Storage: 0 B`, `Keys: 0` | nothing written at all | write path, not offload — check `write_through` and backend creation |

Ignore `Eviction: Success/Attempts` — it stays `0/0` even when offload works, because offload
moves objects to disk rather than dropping them. Judge by `SSD Storage` and the master line
`Offload-on-evict mode enabled: DRAM offload to LOCAL_DISK will occur at eviction time`.

If `--offload_force_evict=true` also fails, stop tuning Mooncake. The repo already has
`glm5.2-fp8-b300-sglang-eagle-hicache-r10-l3-nixl-posix` — a different storage backend with no
master service, no watermarks, and no eviction-triggered offload, so it sidesteps this entire
failure class. It is also where a GDS plugin would eventually attach (currently `POSIX`).

## 7. Limits of the math

- **`0.31 GB/min/CCU` is one data point** — CCU 32, GLM-5.2 FP8, `write_through`. Different model
  or precision changes KV bytes/token and invalidates it.
- **The rate decays** as the store warms and more lookups hit, so real crossing lands later than
  predicted.
- **`f = 0.20` is the margin.** At 4x lower than measured, crossing slips to 80% of the window and
  still works; below that it breaks.

## 8. The cheaper test that does not exist yet

None of this needs a GPU, a model, or a benchmark. `mooncake_master` plus a small client writing
objects until the watermark trips answers §6 on any box with a disk, in ~2 min instead of ~15.

The only reason it is not the default is that we have no SSH access to `b300-netperf_00`, so
every hypothesis costs a CI run. Getting node access (the premise of the `debug-runs` skill) is
the single change that would make the remaining Mooncake tuning cheap.

## Appendix A — config record

Source: `benchmarks/single_node/agentic/glm5.2_fp8_b300-netperf_sglang.sh`, config key
`glm5.2-fp8-b300-sglang-eagle-hicache-r10-l3-mooncake-ssd`. Real-run values; the smoke test
changes only `duration-override`.

### A.1 mooncake_master

```
--rpc_port=50051
--metrics_port=9003
--enable_offload=true
--offload_on_evict=true
--eviction_high_watermark_ratio=0.85
--eviction_ratio=0.10
--logtostderr=true
```

At master defaults: `default_kv_lease_ttl=5000`, `default_kv_soft_pin_ttl=1800000`,
`allow_evict_soft_pinned_objects=1`, `offload_force_evict=0`, `memory_allocator=offset`.

### A.2 SGLang HiCache flags

```
--enable-hierarchical-cache
--hicache-ratio 1.0
--hicache-write-policy write_through
--hicache-io-backend direct
--hicache-mem-layout page_first_direct
--hicache-storage-backend mooncake
--hicache-storage-prefetch-policy wait_complete
```

`--hicache-size` unset.

### A.3 Mooncake client extra_config

```json
{
  "master_server_address": "127.0.0.1:50051",
  "metadata_server":       "P2PHANDSHAKE",
  "local_hostname":        "127.0.0.1",
  "protocol":              "tcp",
  "enable_ssd_offload":    true,
  "ssd_offload_path":      "/mnt/test-raid0/mooncake/${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-c${CONC}",
  "prefetch_threshold":    64
}
```

Prefetch timeout knobs at defaults: `base=2.0`, `per_ki_token=0.1`, `max=30.0`.

### A.4 Capacity

51.4 KB/token (GLM-5.2 FP8, `KV size: 140.35 GB` for 2,728,896 tokens).

| Tier | Backing | Size | Tokens |
|---|---|---|---|
| L1 | GPU HBM, `mem-fraction-static 0.88` | 140 GB | 2.73M |
| L2 | host DRAM, `hicache-ratio 1.0` | 142 GB | 2.73M |
| L3a | host DRAM, Mooncake segment | 140 GB @ 3600 s / CCU 32 | ~2.7M |
| L3b | `/mnt/test-raid0` NVMe RAID0 | 16.00 TB | ~311M |

Host DRAM: 142 + 140 = 282 GB committed of `TOTAL_CPU_DRAM_GB=2399`.

### A.5 Admission

```
--max-running-requests   min(2*CONC, 32)
--schedule-policy        lpm
--schedule-conservativeness 2.0
--chunked-prefill-size   8192
--max-prefill-tokens     8192
--cuda-graph-max-bs      256
--watchdog-timeout       1800
--mem-fraction-static    0.88
```

### A.6 Model / topology

```
model      zai-org/GLM-5.2-FP8  (/mnt/models/zai-org/GLM-5.2-FP8)
image      lmsysorg/sglang:v0.5.15.post1-cu130
tp 8, ep 1, no dp-attention, no router
kv-cache-dtype fp8_e4m3, page_size 64, attention_backend dsa
spec       EAGLE, num-steps 3, eagle-topk 1, num-draft-tokens 4
dataset    semianalysis_cc_traces_weka_062126 (393 entries)
node       b300-netperf_00 (cluster:b300-nv)
```
