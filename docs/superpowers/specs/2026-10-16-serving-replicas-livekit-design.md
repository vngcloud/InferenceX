# Serving YAML replicas + LiveKit (111868) scenarios: Gemma-4-31B sizing bench — design

Date: 2026-10-16 · Author: Ly Quang Thang · Status: draft, awaiting review
Builds on: `2026-10-12-serving-yaml-dispatch-design.md` (V1), `2026-10-12-custom-agentic-tau2-design.md`
(file datasets), `2026-10-14-serving-router-generic-design.md` (V3 generic router).
Dataset track (parallel): `vmonitor-gateway-datasets/docs/specs/0004-livekit-voice-agent-synth-traces.md`.
Prior art: branch `feat/gemma4-tau2-agentic-arm` (`gemma4tau2_fp8_h200.sh`, 2×TP1 vLLM + router,
recipe path, pre-serving-YAML); handoff `docs/handoffs/2026-09-16-gemma4tau2-inferencex-arm-dispatch.md`.

## Goal

Size Gemma-4-31B serving for customer 111868 (LiveKit voice agent, 300 CCU = 300 concurrent rooms,
TTFT p95 ≤ 0.5 s, negotiable) on `hardware-hcm` 8×H200 through the normal serving-YAML dispatch.
Two serving shapes:

1. **Direct**: 1 GPU, 1 engine, no router.
2. **Prod-like**: 2 GPUs, 2 engines (TP1 each), vLLM router in front, run with **both** policies:
   `cache_aware` (what prod runs) and `consistent_hash` (sticky per session).

Each is replayed with the two datasets of the dataset track (V1 `append-ideal`, V2 `observed`).
The sizing answer per GPU comes from (1); (2) shows what routing costs or saves on top.

## Decisions

| # | Decision |
|---|---|
| D1 | Add `replicas: N` to serving YAML (option A). N engines, `gpus / N` GPUs each; a router is required when N > 1. |
| D2 | Router = vllm-router 0.1.14 installed by `pip` at start, inside the engine image, same as prod (audit 2026-10-15 §2). Compute nodes reach PyPI (probe job 3539: pypi 200, hf 200). |
| D3 | Engine = prod flags on `vllm/vllm-openai:v0.25.1` (tag verified on Docker Hub), **MTP off** (synthetic text would give wrong acceptance). Only deviation from prod. |
| D4 | Two tracks in parallel: this track proves replicas + router on `custom-agentic-tau2-smoke`; the dataset is swapped in when published. |
| D5 | Think-time on = default (`traceIdleGapCapSeconds: 300.0`, as tau2); think-time off = same dataset with `traceIdleGapCapSeconds: 0.0` (a room's pending timers advance as soon as it is idle, `utils/aiperf/src/aiperf/config/dataset/config.py:508-520`). **Not** `ignoreTraceDelays`: only the weka loader reads it (`dataset/loader/weka_trace.py:768`), `mooncake_trace` ignores it. Verified in the smoke; fallback = a third dataset file with `delay` removed. |
| D6 | No `requests` cap in the new scenarios: 300 CCU × 1200 s needs ~55k requests (0.155 req/s per lane measured), the dataset has 12,902 rows; lanes recycle rooms with a fresh cache-bust marker. |

## Facts this design relies on (verified 2026-10-16)

- Serving YAML runs one engine: `serve_generic.sh` starts `SERVE_CMD` once on `$ENGINE_PORT`
  (router) or `$PORT` (direct); `expand_router_placeholders` knows `{{engine_url}}` (singular),
  `benchmarks/benchmark_lib.sh:708-715`. `to_matrix.py:104` sets `tp = gpus`.
- `hardware-hcm-8x_*` runners take serving YAMLs with any `gpus ≤ 8`; the launcher allocates
  `--gres=gpu:<gpus>` (`runners/launch_hardware-hcm.sh:107-109`), so a 1- or 2-GPU job leaves the rest free.
  Today each node has 2 GPUs held by another user (squeue 2026-10-16 ~21:00 ICT).
- AIPerf sends a per-session stable `X-Correlation-ID` (`transports/base_transports.py:126-131`);
  vllm-router 0.1.14 `consistent_hash` reads `x-session-id`, then `x-user-id`, `x-tenant-id`,
  `x-correlation-id` (`src/policies/hash_key.rs:11-18`). Sticky routing works with no extra flag.
- vllm-router 0.1.14 `cache_aware` keys its tree on `session_params.session_id` only for chat
  requests (`src/protocols/spec.rs:538-552`); AIPerf, like LiveKit, sends none, so the bench
  reproduces prod's room hopping.
- Recycled plays get a new marker at the first user message (raw-messages path included,
  `workers/worker.py:324-341`).

## Components

### 1. Schema (`infx/serving/schema.py`)

- `replicas: int = 1`, `ge=1`. Rules (one `ERROR <field>: <fix>` line each):
  `gpus % replicas == 0`; `replicas > 1` needs a `router:` block; not with the Dynamo profile.
- New router placeholder `{{engine_urls}}`: space-separated `http://127.0.0.1:<port>` of every
  engine. `{{engine_url}}` stays = the first engine (back-compat).
- `to_matrix.py`: the matrix row keeps `tp = gpus` (total). `benchmark-tmpl.yml:411` sets
  `GPU_COUNT = TP × PP × PCP` and the launcher allocates `--gres=gpu:$GPU_COUNT`, so `tp` must stay
  the total; per-GPU numbers then divide by the total too. `replicas` travels only in the `serving`
  object (`"replicas": "<N>"`) and `benchmark-tmpl.yml` exports it as `REPLICAS`.

### 2. Launcher (`runners/launch_hardware-hcm.sh`)

When `REPLICAS > 1`: allocate `REPLICAS` free ports, export `ENGINE_PORTS` (space-separated) and
`ENGINE_PORT` = the first. Nothing else changes (router and bench sruns as today).

### 3. `benchmarks/single_node/agentic/serve_generic.sh`

- Engine `r` (0-based) gets `CUDA_VISIBLE_DEVICES` = GPUs `[r·k, (r+1)·k)` of the job's visible
  list (`k = gpus / replicas`), port = `ENGINE_PORTS[r]`, log `server.<r>.log`
  (`server.log` stays for `replicas: 1`). All PIDs are watched; any engine exit fails the run.
- Readiness: every engine's `/health`, then the existing router check (`wait_for_frontend_model`).
- `AIPERF_SERVER_METRICS_URLS` = router metrics + every engine's `/metrics`.
- `server_command.txt` records each engine command.

### 4. `serve_router.sh` / `expand_router_placeholders`

Expand `{{engine_urls}}` from `ENGINE_PORTS`.

### 5. Dataset lock + scenarios

`scenarios/datasets/lock.yaml` gets two keys once the dataset is published:

```yaml
livekit-111868-v1:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: <40-hex>
  file: trace.v1-append-ideal.jsonl
  sha256: <64-hex>
livekit-111868-v2:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: <40-hex>
  file: trace.v2-observed.jsonl
  sha256: <64-hex>
```

Scenarios (copies of `custom-agentic-tau2.yaml`; `entries: 1217`, **no `requests`**,
`duration: 1200.0`, same trajectory ratios, warmup and idle-gap cap):

| Scenario | Dataset | Think time |
|---|---|---|
| `custom-agentic-livekit111868-v1` | `livekit-111868-v1` | on |
| `custom-agentic-livekit111868-v2` | `livekit-111868-v2` | on |
| `custom-agentic-livekit111868-v1-nothink` | `livekit-111868-v1` | off (`traceIdleGapCapSeconds: 0.0`) |
| `custom-agentic-livekit111868-v2-nothink` | `livekit-111868-v2` | off |
| `custom-agentic-livekit111868-smoke` | `livekit-111868-v2` | on, `duration: 180.0` |

`utils/serving/check_scenario_golden.py` covers them (the CLI-equivalence check needs a
reference command; take the tau2 reference with the dataset path and `entries` swapped, and
allow the missing `--request-count`).

### 6. Serving YAMLs (`serving/`)

Common engine `serve` (prod flags minus MTP):

```
vllm serve RedHatAI/gemma-4-31B-it-FP8-block
--served-model-name RedHatAI/gemma-4-31B-it-FP8-block
--tensor-parallel-size 1 --gpu-memory-utilization 0.92 --max-model-len 262144
--max-num-seqs 64 --max-num-batched-tokens 16384
--enable-chunked-prefill --long-prefill-token-threshold 8192
--kv-cache-metrics --enable-prompt-tokens-details
--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

`env: {VLLM_DISABLE_COMPILE_CACHE: "1", NCCL_P2P_LEVEL: NVL}` (prod), image `vllm/vllm-openai:v0.25.1`,
runner `hardware-hcm-8x_01|_02` (free box at dispatch).

| YAML | gpus / replicas | Router | conc (first ladder) |
|---|---|---|---|
| `gemma4-livekit-1gpu-<v>` | 1 / 1 | none | 4, 8, 16, 32, 48, 64 |
| `gemma4-livekit-2rep-cacheaware-<v>` | 2 / 2 | `cache_aware` | 8, 16, 32, 64, 96, 128 |
| `gemma4-livekit-2rep-chash-<v>` | 2 / 2 | `consistent_hash` | 8, 16, 32, 64, 96, 128 |

`<v>` = `v1` / `v2` (one scenario per YAML; nothink YAMLs added when needed). Router:

```yaml
router:
  image: vllm/vllm-openai:v0.25.1
  serve: |
    bash -c "pip install vllm-router==0.1.14 && exec vllm-router --host 0.0.0.0 --port {{router_port}}
    --worker-urls {{engine_urls}} --policy cache_aware --request-timeout-secs 14400 --disable-retries"
```

(`consistent_hash` YAML: same with `--policy consistent_hash`.) Exact quoting is settled by the
validator in the plan.

## Tests

- `utils/test_serving_to_matrix.py`: `replicas` defaults to 1 and old YAMLs render unchanged;
  `gpus % replicas` error; `replicas > 1` without router error; `tp = gpus / replicas`.
- `{{engine_urls}}` expansion test (style of the existing placeholder tests).
- `serve_generic.sh` replica loop: stub engine command (a tiny HTTP server per port) checks
  `CUDA_VISIBLE_DEVICES` slices, ports, per-replica logs, and that one engine dying fails the run.
- Golden check for the five new scenarios.

## Acceptance (behaviour on real hardware)

1. **Flow, before the dataset** (`custom-agentic-tau2-smoke`, conc 8): `gemma4-livekit-1gpu` and
   `gemma4-livekit-2rep-cacheaware` both green; router log shows both workers; `server.0.log`
   and `server.1.log` both serve requests; aiperf scrapes 3 metrics endpoints.
2. **Sticky works**: same smoke with `consistent_hash`. Per-engine `vllm:prefix_cache_hits`
   higher than `cache_aware` at the same conc, and each session's requests land on one engine
   (router log).
3. **Dataset swapped in**: `custom-agentic-livekit111868-smoke` green on `gemma4-livekit-1gpu`;
   aiperf log shows 1,217 unique conversations; run passes 180 s without a requests cap.
4. **Calibration**: `gemma4-livekit-1gpu-v2` at conc 6 (POC peak): prefix hit near the prod
   77.5%; v1 between 77.5% and 86.4%. A large miss means the dataset plan is wrong, not the server.
5. **Sizing ladders**: the six YAMLs of §6 at think time on. Output: max conc per GPU with
   TTFT p95 ≤ 0.5 s for v1 and v2; GPUs for 300 CCU = ⌈300 / that⌉ with the routing factor from (2).

## Out of scope

- MTP on (needs real text); LMCache / KV sharing between replicas (someone else is testing
  `gemma4-lmc-*` on the same nodes); the 1,136 non-LiveKit requests; prod changes.
- The `nothink` YAMLs and ladders beyond §6 until (5) asks for them.
