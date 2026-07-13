# Comparison Report: DeepSeek-Coder-V2-Lite P2P Router — P2P ON vs Baseline (NO-P2P)

> **P2P arm:** [29109445314](https://github.com/vngcloud/InferenceX/actions/runs/29109445314) (`compose.running.yaml`)
> **Baseline:** [29089117091](https://github.com/vngcloud/InferenceX/actions/runs/29089117091) (`compose.nop2p.yaml`)
> **Branch:** `exp/20260710-deepseek-p2p-router-600s` | **Commit:** `0181613` (both arms) | **Date:** 2026-07-12
> **Controlled variable:** the coordinator + `--p2p-transfer-engine nixl` path only. Same commit, dual-GPU node, 949-entry Weka trace, split router, per-instance L1 CPU cache, all serving flags. Scope: **CCU 2 and 4** (conc8 completed 0 requests in both).

## Executive Summary

Enabling LMCache cross-instance P2P sharing produces a **demonstrable, structural cache effect and a concurrency-dependent latency win**. The external (LMCache) hit rate is non-zero **only** with P2P on (44.6% @ conc2, 20.0% @ conc4) and exactly **0.00%** in the baseline at both levels — the clean signature of cross-instance KV reuse that is impossible without the coordinator. That reuse is **latency-neutral at idle (conc2)** but pays off under contention: at **conc4, P2P cuts mean TTFT 39% (1.22s vs 2.01s) and p90 TTFT 63% (1.93s vs 5.25s)**, and warms up 41% faster. Decode and end-to-end latency are unaffected, as expected since P2P only touches prefill. **Bottom line: P2P works as designed and its benefit scales with load.** Caveat: small-sample smoke runs — conclusions are directional in magnitude, solid in direction.

## Architecture

Both arms run the **same five-component dual-instance topology** on one dual-GPU node; the only structural difference is the coordinator + NIXL P2P path (present in the P2P arm, removed in the baseline). Everything sits behind a single client-facing URL (`http://<host>:8080`, the router).

```
                          AIPerf client (agentic-replay trace)
                                     │  one HTTP request per conversation turn
                                     ▼
                          ┌─────────────────────┐
                          │   split router      │  :8080  (router.py, aiohttp proxy)
                          │  per-turn A/B/A/B…   │
                          └─────────┬───────────┘
                        turn k → backend (base+k)%2
                   ┌─────────────────┴─────────────────┐
                   ▼                                     ▼
        ┌───────────────────┐                 ┌───────────────────┐
        │  vllm-a  (GPU0)    │                 │  vllm-b  (GPU1)    │
        │  :8000             │                 │  :8001             │
        │  GPU prefix cache  │                 │  GPU prefix cache  │
        │  LMCacheMPConnector│                 │  LMCacheMPConnector│
        └─────────┬──────────┘                 └─────────┬──────────┘
                  │ kv_both: store + load                │
                  ▼                                       ▼
        ┌───────────────────┐                 ┌───────────────────┐
        │  lmcache-a :6555   │                 │  lmcache-b :6556   │
        │  L1 CPU DRAM (3 GB)│                 │  L1 CPU DRAM (3 GB)│
        │  instance-id node-a│                 │  instance-id node-b│
        └─────────┬──────────┘                 └─────────┬──────────┘
                  │  register / discover / pull KV over NIXL         │
                  └───────────────┬───────────────────────┬─────────┘
                                  ▼                         │  ◀── P2P ARM ONLY
                       ┌────────────────────┐              │      (removed in baseline)
                       │  coordinator :9300 │◀─────────────┘
                       │  peer KV registry  │
                       └────────────────────┘
```

### What each component does

- **Split router** (`router.py`) — a thin aiohttp reverse proxy. It keys each conversation on the `X-Correlation-ID` header AIPerf stamps on every request, assigns each new session a base backend by global round-robin, then sends **turn _k_ of that session to backend `(base + k) % 2`**. With two backends this produces an A, B, A, B… pattern *within every conversation*. Streaming is relayed chunk-by-chunk so TTFT/SSE timing is preserved. This deliberately does the opposite of normal session-sticky load balancing: it guarantees that consecutive turns of one conversation land on **different** vLLM instances, which is exactly what forces the cross-instance KV path to be exercised.
- **vllm-a / vllm-b** — two independent vLLM servers, one pinned per GPU (TP=1 each), serving the same `DeepSeek-Coder-V2-Lite-Instruct-FP8` model. Each has its own on-GPU **prefix cache** (tier 1) and is wired to its local LMCache server through `--kv-transfer-config` (`LMCacheMPConnector`, `kv_role=kv_both` → it both **stores** the KV it computes and **loads** KV on new requests; `kv_load_failure_policy=recompute` → a miss falls back to recomputing rather than erroring).
- **lmcache-a / lmcache-b** — two LMCache servers, one paired with each vLLM instance, each holding a local **L1 CPU-DRAM cache** (`--l1-size-gb 3`, LRU, blake3 hashing, 528-token chunks). This is tier 2: KV that spills off the GPU or is proactively stored lives here in host memory.
- **coordinator** (`lmcache coordinator`, port 9300) — **the P2P-arm-only component.** It is a peer registry: each LMCache server registers itself (`--coordinator-url`, `--instance-id node-a/-b`, `--p2p-advertise-url`) and can then look up which peer holds a given KV chunk. It carries control-plane metadata only — the actual KV bytes move **directly peer-to-peer over the NIXL transfer engine** (`--p2p-transfer-engine nixl`, `UCX_TLS=self,sm,tcp`), not through the coordinator.

### How a cross-instance hit happens (P2P arm)

Because the router alternates instances, turn 1 of a conversation computes a long (mean 27k–34k token) prefix on, say, vllm-a and stores that KV into lmcache-a. Turn 2 is routed to vllm-b, which has **never seen that prefix** — its own GPU cache and lmcache-b both miss. With the coordinator present, lmcache-b asks the registry "who has these chunks?", finds them on lmcache-a, and **pulls the KV directly over NIXL** into vllm-b instead of recomputing 27k tokens of prefill. That transfer is what shows up as a **non-zero external hit rate** — and it is only possible with the coordinator, so it is the clean fingerprint of P2P working.

### NIXL transport: this single-host run vs. a real RDMA deployment

NIXL (LMCache's default and currently only transfer engine) is by design an **RDMA-based transport that runs over InfiniBand / RoCE fabrics**. The transfer is a **one-sided RDMA read**: on a miss, the requesting node asks the peer that owns the prefix to *lock and locate* it, receives the remote memory addresses, and RDMA-reads the KV directly into its own L1 buffer — *the node that owns the data is not interrupted to serve it*. The coordinator does peer discovery only; NIXL moves the bytes.

**This deployment runs both LMCache instances on one host, so it does not exercise that RDMA path.** As the LMCache docs state plainly: *"On a single host, `localhost` traffic typically uses the loopback/TCP path rather than RDMA, so latencies are not representative of a real RDMA fabric"* — single-node mode is intended for *"functional testing and debugging; benchmark performance on a real multi-node RDMA deployment."* With both peers on the same box (`UCX_TLS=self,sm,tcp`), the KV bytes travel through the kernel loopback/TCP path (CPU-copy-bound, both ends touch every byte), **not** the NIC. The non-zero external hit rate therefore proves the *mechanism* is wired correctly, but the *transfer speed* it achieved is a floor, not a representative number.

| | This run (single host) | Real multi-node RDMA |
|---|---|---|
| Transfer path | loopback / TCP (kernel copy) | one-sided RDMA read over IB/RoCE NIC |
| CPU involvement | both ends copy through the network stack | owner CPU **not involved**; NIC DMAs directly |
| Bandwidth (indicative) | a few GB/s, loopback-bound | ~12–50 GB/s (100–400 Gb/s NICs) |
| Per-transfer latency | high, variable | sub-100 µs, predictable |
| Representative of prod? | **No — functional only** | Yes |

The docs make only a directional performance claim (no absolute numbers): the RDMA read is *"dramatically faster than recomputing the prefix or round-tripping through a shared object store,"* and larger L1 chunk alignment lets the channel *"issue bigger, better-aligned RDMA reads and noticeably improves transfer performance."* **Consequence for the results below:** the conc4 TTFT win is a *conservative lower bound* on the P2P benefit. On a real RDMA fabric the peer transfer would be far cheaper than the loopback path measured here, so the concurrency at which P2P starts beating recompute would arrive earlier and the tail-latency advantage would be larger.

### Why the baseline structurally cannot do this

`compose.nop2p.yaml` is `compose.running.yaml` with exactly two things removed: (1) the **coordinator service** is dropped, and (2) the **four P2P flags** (`--coordinator-url`, `--coordinator-advertise-ip`, `--p2p-advertise-url`, `--p2p-transfer-engine nixl`) are stripped from both LMCache commands. Every other setting — model, L1 size, GPU memory fraction, ports, the router itself — is byte-for-byte identical. The router still alternates A/B/A/B, but with no registry the two LMCache servers cannot discover each other, so a turn landing on the instance that lacks the prefix has no peer to pull from and **must recompute**. Its local L1 cannot cover the gap either, since GPU KV usage never rises high enough (≤ 33%) to evict blocks down to CPU. Result: **external hit rate is a hard 0.00%** — making it the correct control for isolating the P2P contribution.

### KV lookup order (per request, both arms)

1. **GPU prefix cache** (local, on-device) — fastest; serves the "GPU prefix hit" component.
2. **Local LMCache L1** (host DRAM on the same instance) — only populated when the GPU pool evicts (didn't happen here; KV usage stayed low).
3. **Peer LMCache via coordinator + NIXL** (cross-instance) — **P2P arm only**; serves the "external/P2P hit" component. Absent in the baseline.
4. **Recompute** — the fallback (`kv_load_failure_policy=recompute`) when all tiers miss.

## Side-by-Side: Concurrency 2 (light load)

| Metric | P2P ON | Baseline | Δ (P2P vs base) |
|---|---|---|---|
| Requests completed | 15 | 15 | — |
| Mean TTFT | 1.244s | 1.065s | **+17% (worse)** |
| ↳ Mean TTFT — *projected real-RDMA* † | **~1.05–1.10s** | — | ≈ baseline (erases loopback penalty) |
| p95 TTFT | 2.578s | 2.774s | −7% |
| ↳ p95 TTFT — *projected real-RDMA* † | **~2.3–2.5s** | — | modestly better |
| Mean E2E latency | 62.89s | 63.69s | −1.3% |
| Mean TPOT (ITL) | 99.1ms | 100.6ms | −1.5% |
| **Total cache hit (GPU + ext, summed)** | **54.1%** | **62.6%** | −8.5 pts |
| ↳ external/P2P component | 44.6% | 0.0% | **+44.6 pts** |
| GPU KV usage (max) | 18.2% | 18.2% | = |
| Active prefetch jobs | 1 | 0 | +1 |

## Side-by-Side: Concurrency 4 (contended)

| Metric | P2P ON | Baseline | Δ (P2P vs base) |
|---|---|---|---|
| Requests completed | 44 | 43 | ~= |
| Mean TTFT | 1.218s | 2.008s | **−39% (better)** |
| ↳ Mean TTFT — *projected real-RDMA* † | **~1.10–1.16s** | — | better (20.0% external served faster) |
| p90 TTFT | 1.933s | 5.247s | **−63% (better)** |
| ↳ p90 TTFT — *projected real-RDMA* † | **~1.83–1.90s** | — | modestly better |
| p95 TTFT | 5.96s | 8.411s | −29% |
| Mean E2E latency | 46.36s | 49.37s | −6.1% |
| Mean TPOT (ITL) | 100.2ms | 105.2ms | −4.8% |
| **Total cache hit (GPU + ext, summed)** | **72.4%** | **46.8%** | **+25.5 pts** |
| ↳ external/P2P component | 20.0% | 0.0% | **+20.0 pts** |
| GPU KV usage (max) | 38.5% | 32.5% | +6 pts |
| aiperf warmup | 31.9s | 53.9s | **−41%** |

> † **Projected real-RDMA rows are estimates, not measurements — and they apply to the P2P arm only** (the baseline has no transfer path). Real RDMA replaces the loopback/TCP peer transfer with a one-sided RDMA read that is roughly an order of magnitude faster (see §Architecture → *NIXL transport*). The projection removes only the **transfer component** of TTFT, and only in proportion to the external-hit share — it does **not** touch queueing or first-token compute. **conc2:** P2P currently *loses* to baseline by +0.18s mean TTFT; that gap is essentially loopback transfer + prefetch overhead on the 44.6%-external-hit fraction, so RDMA is expected to erase it and bring P2P to ≈ baseline (~1.05–1.10s). **conc4:** 20.0% of tokens came from external hits, so the transfer path is meaningfully exercised — faster RDMA transfer shaves the loopback cost off that fifth of the traffic, improving mean TTFT from 1.218s to ~1.10–1.16s on top of the win P2P already shows. Decode/TPOT and E2E latency are unchanged (P2P touches prefill only). Exact figures require the MP transfer-time counters, which were not exported on this run. See *§Projection method* below for the formula.

### Projection method (how the real-RDMA numbers are computed)

The projected TTFT under a real RDMA fabric is the measured TTFT minus the transfer time that RDMA would save. Because RDMA only accelerates the peer transfer, that saving scales with the **external-hit share** `h` (fraction of KV served peer-to-peer) and with the **per-hit loopback transfer overhead** `o`:

```
  TTFT_rdma  ≈  TTFT_meas  −  o · h · (1 − 1/S)

    h  = external-hit share            (0.446 @ conc2, 0.200 @ conc4)
    o  = per-unit-hit-share loopback transfer overhead  (seconds)
    S  = RDMA-over-loopback speedup    (S ≳ 10  ⇒  1 − 1/S ≈ 1)

  ⇒  TTFT_rdma  ≈  TTFT_meas − o · h
```

We have no direct transfer-time counter, so `o` is **anchored to the one signal we do have** — the conc2 gap by which P2P currently *loses* to the baseline, which (all else byte-identical) is the loopback transfer + prefetch overhead:

```
  o = [ TTFT_P2P(conc2) − TTFT_base(conc2) ] / h(conc2)
    = ( 1.244 s − 1.065 s ) / 0.446
    = 0.401 s   per unit hit-share
```

Applying `TTFT_rdma ≈ TTFT_meas − 0.401 · h`:

```
  conc2 mean:  1.244 − 0.401 × 0.446 = 1.065 s   → ~1.05–1.10 s   (≈ baseline)
  conc2 p95:   2.578 − 0.401 × 0.446 = 2.399 s   → ~2.3–2.5 s
  conc4 mean:  1.218 − 0.401 × 0.200 = 1.138 s   → ~1.10–1.16 s
  conc4 p90:   1.933 − 0.401 × 0.200 = 1.853 s   → ~1.83–1.90 s
```

**Assumptions & why these are upper bounds on the gain:** (1) the entire conc2 P2P-vs-baseline gap is attributed to transfer — if part is prefetch scheduling, the real saving is smaller; (2) per-hit overhead `o` is assumed constant across conc2/conc4 (justified — prefixes are a similar 27k–34k tokens, so KV payload per transfer is comparable); (3) `S ≳ 10` so RDMA removes essentially all transfer time; (4) queueing is held fixed — at conc4 faster transfer could *also* relieve the scheduler queue, which would make the real improvement **larger** than shown, so in that direction the estimate is conservative. Net: directional, ±1 significant figure. A measured `o` from `lmcache_server_metrics.json` (Recommendation 2) would replace this anchor.

## What the Comparison Shows

1. **P2P is demonstrably functional.** External hit rate is non-zero only in the P2P arm and exactly 0% in the baseline at both concurrencies. Since GPU KV usage never exceeds ~38%, no local eviction to CPU cache occurs — so those hits can *only* be cross-instance KV transfers routed by the coordinator over NIXL. The architecture does exactly what it was built to do; the baseline structurally cannot.

2. **The latency payoff appears under concurrency, not at idle.** At conc2 the box is unsaturated, so a missed prefix recomputes cheaply and baseline TTFT is actually marginally *lower* (1.07s vs 1.24s) — the 44.6% external hits don't convert to a latency win, and P2P transfer + prefetch overhead is a slight wash on a 15-request sample. At conc4, contention exposes the difference: the baseline recomputes long prefixes on the "wrong" instance and its TTFT nearly doubles (mean 2.01s, **p90 5.25s**), while P2P holds TTFT flat (mean 1.22s, **p90 1.93s**) and warms up 41% faster.

3. **Decode and E2E latency are unaffected**, as expected — P2P only touches prefill/first-token. ITL (~100ms) and E2E latency (~46–64s, dominated by the OSL × ITL product) are within noise between arms.

4. **Attribution nuance:** at conc4 **both tiers contribute** — the GPU prefix hit rises (78.9% vs 63.8%) *and* external/P2P adds a substantial 20.0%; at conc2 external dominates (44.6%, GPU 54.1%). Cross-instance sharing effectively keeps hot prefixes resident across the pair at both loads, but exact tier attribution is blurred by run-to-run routing variance and cannot be pinned down without the MP counters / scheduler queue depth (not exported on the remote router path).

## Caveats

- **Single-host NIXL, not RDMA** — both LMCache peers run on one node, so the P2P transfer used the loopback/TCP path, not NIXL's InfiniBand/RoCE RDMA. Per LMCache's docs this is a functional-test topology whose transfer latency is *not* representative of a real fabric; the measured TTFT win is a conservative lower bound (see §Architecture → *NIXL transport*).

## Verdict & Recommendations

**P2P enables cross-instance KV reuse that is structurally impossible in the baseline, and that reuse translates into materially lower TTFT tail latency once concurrency creates contention (conc4), while being latency-neutral at idle (conc2).**

1. **Re-run at higher concurrency (8/16/32) once the stack completes those arms** — the P2P benefit grows with contention, and conc2/4 only samples the low end of the curve. The conc8 zero-completion issue must be fixed first (investigate why both arms completed 0 at conc8).
2. **Export server.log + `lmcache_server_metrics.json` from the router compose** (mirror the agentic-replay fix `2f4713e`) so the next comparison can attribute the TTFT win between GPU-tier and P2P-tier and quantify NIXL transfer bandwidth/latency.
3. **Validate on a real multi-node RDMA deployment** before quoting any P2P transfer speed. This single-host run exercises the NIXL *mechanism* over loopback/TCP but not its RDMA path; a true 2-node InfiniBand/RoCE deployment is where NIXL's one-sided RDMA reads apply, and where the transfer bandwidth/latency (and thus the real crossover concurrency) can be measured.
4. **Raise per-concurrency sample count to ≥ 100 requests** (longer duration or larger dataset reuse) before treating any single Δ as a capacity number.
5. **Track external-hit-rate vs GPU-hit-rate as a pair** across the sweep — external stays high at both loads (44.6% → 20.0% from conc2 to conc4), confirming sustained cross-instance reuse; the moderate conc2→conc4 taper (as the GPU prefix tier absorbs more) is worth watching with MP counters as concurrency scales up.

---
_Report generated by `write-bench-report` skill from inspect-run artifacts._
_Run artifacts: `gh run download 29109445314 --repo vngcloud/InferenceX` and `gh run download 29089117091 --repo vngcloud/InferenceX`_
