---
name: inspect-run
description: >
  Full inspection of a completed InferenceX GitHub Actions benchmark run. Downloads
  artifacts, parses performance metrics (TTFT, TPOT, throughput), computes cache hit
  rates (GPU prefix cache, LMCache external), scans server.log for initialization
  anomalies and queue-depth patterns, checks the triggering git commit, and produces
  a structured summary report. Invoke this skill whenever the user asks to inspect,
  review, analyze, or check the results of a benchmark run — even phrased casually
  as "what happened in the run", "the TTFT looks weird", "fetch the results",
  "it finished, can you look at it?", "show me the metrics", or "there's something
  off with the cache hits". Any time a completed CI run needs investigation, use
  this skill.
---

# Inspect Run

Your job is to give the user a complete, accurate picture of a benchmark run in one
pass — what it measured, whether the stack initialized correctly, what the cache hit
rates were, and whether anything looks wrong. The user should not have to dig through
logs or do arithmetic themselves.

## Step 1 — Identify the run

If the user gave a run ID or URL, use it. Otherwise list recent runs:

```bash
gh run list --workflow=e2e-tests.yml --limit 10 \
  --json databaseId,displayTitle,status,conclusion,createdAt,headBranch
```

Confirm with the user if it's ambiguous which run they mean.

## Step 2 — Get the commit context

Before downloading artifacts, understand what changed. This often explains anomalies
before you've even looked at a single metric.

```bash
gh run view <RUN_ID> --json headSha,headBranch,displayTitle,createdAt
git show --stat <SHA>           # which files changed
git log --oneline <SHA>~3..<SHA>  # surrounding commits for context
```

If the changed files include a launch script or master config, read the relevant diff
section — the change is usually the root cause of any regression.

## Step 3 — Download and stage artifacts

```bash
gh run download <RUN_ID> --dir /tmp/infx-dl-<RUN_ID>
```

**Windows MAX_PATH issue:** Artifact subdirectory names embed the full config key and
exceed 260 characters. Python cannot open files at these paths. Always copy the 5 key
files to a short flat directory first — this is non-negotiable on Windows:

```bash
SCRATCH="/c/Users/LAP14714/AppData/Local/Temp/infx_<RUN_ID>"
mkdir -p "$SCRATCH"
find /tmp/infx-dl-<RUN_ID> -name "server_metrics_export.json"   | head -1 | xargs -I{} cp {} "$SCRATCH/server_metrics.json"
find /tmp/infx-dl-<RUN_ID> -name "lmcache_server_metrics.json"  | head -1 | xargs -I{} cp {} "$SCRATCH/lmcache_metrics.json"
find /tmp/infx-dl-<RUN_ID> -name "profile_export_aiperf.json"   | head -1 | xargs -I{} cp {} "$SCRATCH/profile.json"
find /tmp/infx-dl-<RUN_ID> -name "server.log"                   | head -1 | xargs -I{} cp {} "$SCRATCH/server.log"
find /tmp/infx-dl-<RUN_ID> -name "aiperf.log"                   | head -1 | xargs -I{} cp {} "$SCRATCH/aiperf.log"
find /tmp/infx-dl-<RUN_ID> -name "agg_*.json"                   | head -1 | xargs -I{} cp {} "$SCRATCH/agg.json"
```

## Step 4 — Run the parser

The bundled script handles all file parsing and metric computation. Run it against the
staging directory:

```bash
python "d:/projects/InferenceX/.claude/skills/inspect-run/scripts/parse_metrics.py" "$SCRATCH"
```

This outputs a JSON object with sections: `perf`, `profile`, `cache`, `server_log`,
`aiperf`. Use this as the data source for the report — don't re-parse the files manually.

## Step 5 — Interpret the data and flag anomalies

After running the parser, reason about what the numbers mean before writing the report.
Check these things:

### Cache health

The most important cache numbers:
- **`cache.ext_hit_rate_pct`** — LMCache (CPU DRAM) hit rate. This is 0% whenever the
  GPU KV pool never fills enough to evict blocks to LMCache. Low KV usage (< ~15%)
  explains 0% external hits; it's expected, not a bug. If KV usage is high but external
  hits are still 0%, the connector may not be wired.
- **`cache.gpu_hit_rate_pct`** — vLLM GPU prefix cache hit rate. Should be 40–95%
  for multi-turn agentic sessions; below 20% suggests something is wrong with prefix
  caching or the sessions aren't repeating prefixes.
- **`cache.kv_usage_avg_pct` / `kv_usage_max_pct`** — If max < 15%, no eviction
  occurred regardless of concurrency.

### Queue depth (scheduling bottleneck)

From `server_log.runtime`:
- `running.avg` close to 1 with `waiting.avg` > 10 → requests are being processed
  nearly sequentially. Root cause: `--max-num-batched-tokens` is too small relative to
  context length. At 784 tok/step with 30k-token sessions, TTFT will be 100+ seconds.
- Healthy at conc=32: `running.avg` should be 5–20.

### LMCache initialization

From `server_log.init`:
- `block_size_align` — the N in "Setting attention block size to N tokens". Verify it
  matches `LMCACHE_CHUNK_SIZE` in the launch script (784 for Qwen3.5-27B).
- `lmcache_mp_connector: true` — MP connector loaded correctly.
- `lmcache_version` — must be `0.5.0` for hybrid-attention models (0.4.x is not
  SupportsHMA and will crash).
- `heartbeat_running: true` — LMCache fully connected to vLLM EngineCore.
- `hybrid_kv_turned_off: true` — **critical failure**: wrong connector or wrong version.
- `connector_crash: true` — block_size > max_num_batched_tokens at startup.

### TTFT interpretation

TTFT > 10s at concurrency ≥ 8 is almost always a scheduling issue, not a model issue.
The arithmetic: `mean_ttft ≈ (waiting.avg × uncached_tokens_per_req) / prompt_tput`.
If `--max-num-batched-tokens` was set to the block size (e.g., 784), requests queue
behind each other's chunked prefill, resulting in TTFT of 50–200s even on fast hardware.

### aiperf phases

From `aiperf`:
- `timeout_triggered: true` on the profiling phase is **normal** for duration-based runs.
- `warmup_elapsed_s > 120` at low concurrency → TTFT is very high even in warmup.
- `errors > 0` → investigate the aiperf.log for HTTP errors or connection failures.
- `profiling_in_flight_at_end` is normal for high concurrency; those requests complete
  during the grace period and are included in metrics.

## Step 6 — Produce the report

Output this exact structure in markdown:

```
## Run Inspection: <displayTitle>
**Run ID:** <ID> | **Branch:** <branch> | **Status:** <conclusion>
**Commit:** `<SHA>` — <commit message one-liner>

### What changed
<output of git show --stat, condensed to changed files and line counts>

### Performance
| Metric | Value |
|---|---|
| Model / Framework | <model> / <framework> <precision> TP<tp> |
| Hardware | <hw> |
| Concurrency | <conc> |
| Requests completed | <profile.request_count> |
| Benchmark duration | <profile.benchmark_duration>s |
| Mean TTFT | <mean_ttft>s  (p50 <p50_ttft>s · p99 <p99_ttft>s) |
| Mean TPOT | <mean_tpot*1000>ms  (~<mean_intvty> tok/s decode) |
| Total throughput/GPU | <tput_per_gpu> tok/s |
| Input / Output tput | <input_tput_per_gpu> / <output_tput_per_gpu> tok/s |
| Mean E2E latency | <mean_e2el>s |
| Avg ISL / OSL | <isl> / <osl> tokens |
| GPU power | <mean_power_w>W |

### Cache
| Metric | Value |
|---|---|
| GPU prefix cache hit rate | <gpu_hit_rate_pct>%  (<gpu_hits>/<gpu_queries> tokens) |
| External (LMCache) hit rate | <ext_hit_rate_pct>%  (<ext_hits>/<ext_queries> tokens) |
| GPU KV cache usage (avg / max) | <kv_usage_avg_pct>% / <kv_usage_max_pct>% |
| Prompt tokens cached total | <prompt_tokens_cached> |

### Server initialization
- [✓/✗] LMCache version: <lmcache_version>  (need 0.5.0 for hybrid models)
- [✓/✗] Attention block size aligned: <block_size_align> tokens
- [✓/✗] LMCacheMPConnector loaded
- [✓/✗] Heartbeat thread running
- [✓/✗] Hybrid KV manager ON  (✗ if hybrid_kv_turned_off=true)

### Scheduler health (from 10s log samples)
Running avg/max: <running.avg> / <running.max>  |  Waiting avg/max: <waiting.avg> / <waiting.max>

### Anomalies
<bullet list — be specific, not generic. Include the number, not just "TTFT is high".>
  — or "None detected" if everything looks healthy>

### Interpretation
<3–5 sentences. Explain the root cause of any anomaly. Connect the metrics to each
other — e.g., "0% external hit rate is expected here because max KV usage was only
5.6%, meaning no eviction occurred; LMCache is correctly wired but simply never
received any evicted blocks." This section is what the user actually reads.>
```

Use ✓ for passing init checks and ✗ for failing ones. If the run **failed** (conclusion
≠ success), lead the report with the failure cause, then show whatever metrics are
available.

## Anomaly reference

| Symptom | Root cause | Fix |
|---|---|---|
| TTFT > 30s, `running.avg ≈ 1`, `waiting.avg > 15` | `--max-num-batched-tokens` == block_size forces near-sequential prefill | Set to `2 * block_size - 1` (max allowed); LMCache enforces `block_size ≤ mnbt < 2 * block_size` |
| External hit rate 0%, KV usage < 15% | No GPU eviction → LMCache never populated | Expected; need higher concurrency or longer sessions |
| External hit rate 0%, KV usage > 60% | Connector not wired or version mismatch | Check `kv-transfer-config` and lmcache version |
| `hybrid_kv_turned_off: true` | lmcache < 0.5.0 or using V1 connector on hybrid model | `pip install lmcache==0.5.0` + use LMCacheMPConnector |
| `connector_crash: true` | mnbt outside `[block_size, 2*block_size)` | Set `--max-num-batched-tokens $((2 * LMCACHE_CHUNK_SIZE - 1))` |
| SGLang `TypeError: unexpected keyword 'config_file'` | lmcache ≥ 0.4.6 incompatible with SGLang 0.5.12 | Pin `lmcache==0.4.5` |
| `errors > 0` in aiperf | Server overloaded or crashed during run | Check tail of server.log for OOM / crash |

## Step 7 — Offer to persist the report

After delivering the inline report, if `conclusion == "success"`, ask the user:

> "Would you like me to save this as a persisted report to the `dev-lmcache` branch?"

If the user agrees (any affirmative — "yes", "sure", "go ahead", "do it"), invoke the
`write-bench-report` skill. All the data it needs (parser JSON, run metadata, git
context) is already in your conversation — pass it through without re-downloading
artifacts.

If the run **failed** (`conclusion != "success"`), skip this step — failed runs don't
produce complete metric sets worth archiving. You may still offer if the user explicitly
asks to save a partial/failed run report.
