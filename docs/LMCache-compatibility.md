# LMCache Compatibility Report — vLLM & SGLang × LMCache versions, Full- and Hybrid-Attention Models

**Date:** 2026-06-26
**Test models:** `Qwen/Qwen3-8B` (full attention), `Qwen/Qwen3.5-4B` (hybrid attention)
**Components covered:** vLLM 0.21.0 / 0.23.0 · SGLang 0.5.12 / 0.5.13 · LMCache 0.4.5 / 0.4.6 / 0.5.0 (latest of each as of June 2026)

**Abbreviations:** **FA** = full-attention model (e.g. Qwen3-8B, Llama). **HA** = hybrid-attention model — interleaves linear/Mamba/Gated-DeltaNet layers (a small recurrent *state*) with periodic full-attention layers (a normal KV tensor). Qwen3.5, Qwen3-Next, and Gemma-3/4 are HA. The two layer kinds keep **different-shaped caches**, which is the root of every HA difficulty below.

---

## 1. Executive summary

- **Full-attention models: fully supported on both engines.** LMCache delivers a 3.5–5.5× prefill speedup on repeated prompt prefixes. Use the newest stack on each engine.
- **Hybrid-attention models: supported on vLLM only.** Exactly one stack works: **vLLM 0.23.0 + LMCache 0.5.0 + MP connector + `--mamba-cache-mode align`** (live-verified: 99.4% cache restore on a fresh engine).
- **Hybrid-attention models on SGLang: not supported on any version.** SGLang serves HA models well, but on its *own* native cache — LMCache is never placed in the path. This was re-confirmed on the newest SGLang (0.5.13 + LMCache 0.5.0).
- **The single most important rule for the team:** *the model architecture, not just the version, decides compatibility.* Detect HA models up front and route them to the correct stack (or accept SGLang's native cache).

---

## 2. Compatibility matrices

**Legend:** ✅ works · ⚠️ runs but LMCache does nothing (no benefit) · ❌ does not work · 🔬 live-verified on our hardware · 〰️ inferred from code/prior runs, not separately re-run

### 2.1 Quick-decision matrix (start here)

| You are serving… | on **vLLM** | on **SGLang** |
|---|---|---|
| **Full-attention** model | ✅ Use vLLM 0.21.0+ (LMCache bundled) — **5.5×** | ✅ Use SGLang 0.5.13 + LMCache 0.5.0 — **3.6×** |
| **Hybrid-attention** model | ✅ **Only** vLLM 0.23.0 + LMCache 0.5.0 + MP + `--mamba-cache-mode align` — **99.4% restore** | ❌ **Not possible** — LMCache is bypassed; use SGLang's native cache instead |

> **Bottom line:** For hybrid models, **standardize on vLLM 0.23.0 + LMCache 0.5.0**. SGLang is fine for full-attention LMCache and for serving hybrids *without* LMCache, but it cannot offload a hybrid model's KV through LMCache.

### 2.2 vLLM — detailed matrix

| vLLM | LMCache | Connector / mode | FA model | HA model | Speedup (FA / HA) |
|---|---|---|---|---|---|
| **0.21.0** | 0.4.5 *(bundled)* | `LMCacheConnectorV1` (in-process) | ✅ 🔬 works out-of-the-box | ❌ 🔬 hard crash at startup | 5.5× / — |
| **0.23.0** | 0.4.6 *(bundled)* | `LMCacheMPConnector` (multi-process) | ✅ 〰️ | ❌ 🔬 crash (connector not HMA-capable) | — / — |
| **0.23.0** | **0.5.0** *(pip install)* | `LMCacheMPConnector` + `--mamba-cache-mode align` | ✅ 〰️ | ✅ 🔬 **works** | — / 99.4% restore |

- vLLM 0.21.0 + HA: engine core dies — *"failed to convert the KV cache specs to one unified type"* (it cannot reconcile the mamba-state and full-attention shapes). The required `--mamba-cache-mode` engine flag does not exist before 0.22.
- vLLM 0.23.0 ships LMCache 0.4.6, but **0.4.6 is not enough** for HA — its MP connector is not yet HMA-capable (hybrid memory allocator), so it falls back and crashes the same way. **LMCache 0.5.0 is the gate** for HA.

### 2.3 SGLang — detailed matrix

| SGLang | LMCache | Connector / mode | FA model | HA model | Speedup (FA) |
|---|---|---|---|---|---|
| **0.5.12** | **0.4.5** *(must pin)* | `LMCacheLayerwiseConnector` (in-process) | ✅ 🔬 works | ⚠️ 🔬 serves, but LMCache **silently inert** | 3.7× |
| **0.5.12** | 0.4.6 / 0.5.0 | — | ❌ 🔬 crash at startup | ❌ | — |
| **0.5.13** | **0.5.0** *(pip install)* | `LMCacheMPConnector` (multi-process, **now default**) | ✅ 🔬 works | ❌ 🔬 **bypassed** (routed to native cache) | 3.6× |

- SGLang 0.5.12 had a hard version trap: `pip install lmcache` pulls a version (0.4.6+) that crashes with `TypeError: ...config_file...`. Only **0.4.5** worked. **Resolved in 0.5.13** — LMCache 0.5.0 installs and runs cleanly.
- SGLang 0.5.13 + HA: the server starts and serves, but at cache-selection time SGLang routes the hybrid model to its own `MambaRadixCache` (or `SWARadixCache` for Gemma-style models) **before LMCache is ever considered** — so `--enable-lmcache` has no effect. Two independent code-level barriers confirm this (see §4.4).

### 2.4 LMCache version cheat-sheet

| LMCache | Hybrid-capable? | Pairs with | Note |
|---|---|---|---|
| 0.4.5 | ❌ | vLLM 0.21.0 (bundled), SGLang 0.5.12 (pin) | Single uniform KV shape only; last version SGLang 0.5.12 accepts |
| 0.4.6 | ❌ | bundled in vLLM 0.23.0 | MP connector exists but not HMA-capable → HA still crashes |
| **0.5.0** | ✅ *(vLLM path only)* | vLLM 0.23.0, SGLang 0.5.13 | First HA-capable release; **required** for hybrid offload on vLLM |

---

## 3. How the inspection was done (flow)

The same four-step method was applied to every stack so results are directly comparable.

1. **Build the exact stack.** Start from the official engine image (`vllm/vllm-openai:<ver>`, `lmsysorg/sglang:<ver>`) and install the target LMCache version. At build time we **introspect the connector** — print the LMCache version and the connector constructor signature — so version mismatches surface *before* serving (this is how the SGLang `config_file` trap was caught).
2. **Read the integration source.** For ambiguous cases we inspected the engine's own code that decides whether LMCache is used — specifically SGLang's cache-selection registry and LMCache's connector adapters. This is what proved *why* hybrid models bypass LMCache on SGLang, rather than guessing from behavior.
3. **Serve and check startup.** Launch the model and classify the outcome: clean start, hard crash, or "starts but does nothing." We record which cache implementation the engine actually selected.
4. **Run a controlled cache test (cold → flush → warm).** Send a long shared-prefix prompt twice:
   - **Cold run** — nothing cached; KV is computed and stored into LMCache.
   - **Flush** — empty the GPU's own prefix cache so the next hit *cannot* come from the GPU tier.
   - **Warm run** — send the identical prompt. **Any speedup or cache hit now can only be LMCache.** This isolates LMCache's contribution from the engine's built-in prefix cache.
   For the hybrid "does it work end-to-end" test on vLLM, we went further and sent the warm request to a **brand-new engine instance** with an empty GPU — so a hit proves the KV came purely from LMCache's offload tier.

---

## 4. Observations (what we saw, and why)

### 4.1 Full-attention on vLLM — works, cleanest attribution
LMCache 0.4.5 is bundled in vLLM 0.21.0; no install needed. Cold→flush→warm: **1.46s → 0.26s (5.5×)**. vLLM exposes a dedicated counter for the external (LMCache) tier, so hits are attributable by source with no extra work.

### 4.2 Full-attention on SGLang — works on both versions; newest is better
Both SGLang stacks reuse the flushed prefix from LMCache (post-flush warm ≈ 3.7× / 3.6× faster). The newest stack (0.5.13 + 0.5.0) is preferred because it removes the version pin, uses the modern multi-process connector, and **adds a source-attributed metric** (`cache_source="host"`) that the old stack lacked.

### 4.3 Hybrid-attention on vLLM — the one working path
With **vLLM 0.23.0 + LMCache 0.5.0 + MP connector + `--mamba-cache-mode align`**, a hybrid model's KV was stored on one engine and **restored on a fresh engine at a 99.4% hit rate** — definitive proof of working offload. Requirements: LMCache must be 0.5.0 (0.4.6 is not HMA-capable), the `align` engine flag checkpoints the mamba/GDN state in cache-friendly blocks, and the LMCache daemon + engine must share an IPC namespace (CUDA IPC transfer). Note: vLLM still labels this mode experimental — validate hit rate per model.

### 4.4 Hybrid-attention on SGLang — bypassed by design (both versions)
- **0.5.12:** the server starts and serves the model, but LMCache never engages — post-flush prefix is fully recomputed, no LMCache store/retrieve activity. A silent no-op.
- **0.5.13:** confirmed *structurally*. At startup SGLang logs `Tree cache initialized: impl=MambaRadixCache hybrid_ssm=True` — it selected its **own** mamba cache, not the LMCache cache. The LMCache daemon recorded **zero** store/retrieve activity; post-flush reuse was 0. Two reasons in the code:
  1. SGLang's cache registry returns its native hybrid cache (`MambaRadixCache` / `SWARadixCache`) **before** the `--enable-lmcache` branch is reached — so the flag is dead code for hybrids.
  2. Even if reached, SGLang's LMCache connector hardcodes a single (non-hybrid) KV group and has no concept of the separate mamba-state group a hybrid needs.

  SGLang *does* serve hybrids well and can offload them — but through its **own** hierarchical cache (HiCache / UnifiedTree), not LMCache.

---

## 5. How to view / monitor in production

What to watch depends on the engine, because the two expose LMCache activity differently.

| Engine | Primary "is LMCache working?" signal | How to read it |
|---|---|---|
| **vLLM** | `vllm:external_prefix_cache_hits_total` (and `_queries_total`) | Counter scoped to the *external* (LMCache) tier. Non-zero and growing = LMCache is serving reuse. Optionally the richer `lmcache:*` series on the LMCache metrics port. |
| **SGLang** | `sglang:cached_tokens_total{cache_source="host"}` (v0.5.13) | The `host` source = LMCache (CPU/daemon) tier; `device` = the GPU radix. The generic `lmcache:*` series stay 0 under SGLang — do **not** alert on them. |

Operational rules of thumb:
- **Confirm LMCache is even in the path first.** On SGLang 0.5.13, the startup log line `Tree cache initialized: ... impl=LMCRadixCache` is the precondition for any LMCache metric to move. If it says `MambaRadixCache` / `SWARadixCache` / `UnifiedRadixCache`, LMCache is bypassed (hybrid model) regardless of the flag.
- **Do not trust a hit-*rate* alone.** A 100% hit rate with *zero* lookups means LMCache did nothing. Gate any ratio on the absolute hit/lookup counts being non-zero.
- **LMCache only pays off when reuse overflows GPU memory.** On a small workload that fits in GPU cache, LMCache stays idle and adds nothing — expected. Its value appears with large shared contexts, high concurrency, and long multi-turn sessions (e.g. agentic coding), where prefixes get evicted from GPU but survive in LMCache's larger CPU/disk tier.
- **Detect hybrid models up front.** Read the model config: treat it as hybrid if it lists linear/mamba layer types (or model type ∈ {qwen3_5, qwen3_next, gemma-3/4, …}). Gate LMCache accordingly — enable the vLLM MP+align stack, or accept the engine's native cache.

---

## 6. Recommendations

1. **Full-attention fleet:** vLLM 0.21.0+ (LMCache bundled) or SGLang 0.5.13 + LMCache 0.5.0. Both are production-ready; pick by your existing engine preference.
2. **Hybrid-attention fleet:** standardize on **vLLM 0.23.0 + LMCache 0.5.0** with the MP connector and `--mamba-cache-mode align`. This is the only stack that offloads hybrid KV through LMCache. Validate the hit rate per model (the mode is upstream-experimental).
3. **Do not enable LMCache for hybrid models on SGLang** — it is misleading (the flag is silently ignored). If you need offload for hybrids on SGLang, use SGLang's **native** hierarchical cache instead, and budget for that separately.
4. **Mixed fleet:** run two profiles — full-attention on your preferred engine, hybrids on the vLLM 0.23.0 + LMCache 0.5.0 stack. Don't try to force one connector to cover both.
5. **Standardize monitoring** on the per-engine signals in §5, and add a startup check that asserts the expected cache class is in use before relying on LMCache.
