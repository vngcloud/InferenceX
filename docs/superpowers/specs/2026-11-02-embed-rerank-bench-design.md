# Embedding and reranker serving benchmarks — design

Date: 2026-11-02 · Branch: `feat/embed-rerank-bench` (PR into `vng-benchmark`) · Client: AIPerf (vendored `utils/aiperf`)

## Goal

Benchmark embedding and reranker models through the existing serving pipeline (`python -m infx.launch`,
serving YAMLs, adaptive search, result JSON, `result-report`), with the same two run modes the LLM
`synthetic` scenario has:

- **fixed**: one or more fixed concurrencies (`conc: [...]`), one job per point;
- **search**: AIPerf `adaptive_search` finds the max concurrency under a latency SLO.

Targets: vLLM and SGLang, on RTX PRO 6000 (`runner: rtx6000pro-hcm`), three models:
`BAAI/bge-m3` (embed), `BAAI/bge-reranker-v2-m3` (rerank), `Qwen/Qwen3-Embedding-8B` (embed).

Out of scope: retrieval quality (MTEB), TEI, multimodal embeddings, real-traffic trace replay.

## AIPerf support (verified in `utils/aiperf`)

| Family | AIPerf endpoint type | Path | Payload | Server compatibility |
|---|---|---|---|---|
| embed | `embeddings` | `/v1/embeddings` | `{model, input: [text × batchSize]}` | vLLM, SGLang |
| rerank | `cohere_rankings` | `/v1/rerank` (overridden) | `{model, query, documents}` | vLLM; SGLang to be confirmed by smoke (see Risks) |

`hf_tei_rankings` is not used: it sends `texts`, which vLLM and SGLang reject.

Synthetic datasets exist already: `prompts.isl` + `prompts.batchSize` for embed, and
`rankings.{query_tokens, passages, passage_tokens}` for rerank. There are no token-level
metrics (TTFT, ITL, output tokens). The usable metrics are `request_latency`, `request_throughput` and
`input_token_throughput`.

## Workload grid

The grid follows the NVIDIA NeMo Retriever NIM benchmark matrix, which is also built on AIPerf.

| Family | Axis | Values | Label part |
|---|---|---|---|
| embed | `isl` (tokens per input) | `512`, `2k`, `8k`, `32k` | `isl2k` |
| embed | `batch` (inputs per request) | `1`, `32` | `b32` |
| rerank | `passage` (tokens per passage) | `512`, `2k` | `p512` |
| rerank | `docs` (passages per request) | `10`, `40` | `n40` |
| rerank | query tokens | fixed 32 | — |

The schema accepts `32k` for any model, and each serving YAML picks the values that fit its model:
8k-context models list `isl: [512, 2k, 8k]`, and the Qwen3-Embedding-8B YAMLs add `32k`. For rerank, one request holds query plus
one passage per scored pair, so `passage: 2k` stays far below the 8192 context.

## Serving YAML shape

```yaml
scenario: embed                # or rerank
workload:
  isl: [512, 2k, 8k]           # embed only
  batch: [1, 32]               # embed only
  # passage: [512, 2k]         # rerank only
  # docs: [10, 40]             # rerank only
  search: true                 # false = fixed conc
slo: "request_latency:p90:lt:500"   # optional, search only
conc: 1
```

The `conc` rules stay the same: search needs `conc: 1`, and a grid with more than one point takes exactly
one conc. Labels are `{fixed|sweep}-embed-isl{isl}-b{batch}` and `{fixed|sweep}-rerank-p{passage}-n{docs}`,
plus the usual override suffixes (`-d60`, `-smoke`). `overrides` accepts only `duration`, as `synthetic` does.

## Components

### `inferencex-e2e/benchmarks/single_node/serving/scenarios/embed.yaml`, `rerank.yaml`

These are the base AIPerf configs. Unlike `synthetic.yaml`, they set no `streaming`, no `ignore_eos` and no
`useServerTokenCount`. Each has one synthetic dataset (`entries: 1000`, `randomSeed: 42`) and a profiling
phase with `type: concurrency`, `duration: 300`, `gracePeriod: 30` and `failedRequestThreshold: 0.1`.

- embed: `endpoint: {type: embeddings, path: /v1/embeddings}`
- rerank: `endpoint: {type: cohere_rankings, path: /v1/rerank}`

### `inferencex-e2e/infx/serving/scenarios.py`

- Add `embed` and `rerank` to `FAMILIES`.
- Add `EmbedPoint(isl, batch)` and `RerankPoint(passage, docs)`, frozen dataclasses with a `.label`.
- Add `EMBED_ISL = {"512": 512, "2k": 2048, "8k": 8192, "32k": 32768}`, `EMBED_BATCH = (1, 32)`,
  `RERANK_PASSAGE = {"512": 512, "2k": 2048}`, `RERANK_DOCS = (10, 40)` and `RERANK_QUERY_TOKENS = 32`.
  A value outside these sets raises `ScenarioError` with the usual one-line `<field>: <problem>` message.
- `_embed(point, search, slo)` and `_rerank(point, search, slo)` fill the dataset of the base config.
- Extract the sweep block from `_synthetic` into `_search_block(slo, default_slo, objective)` and share it
  across all three families:
  - synthetic keeps objective `output_token_throughput` with `DEFAULT_SLO`;
  - embed and rerank use objective `request_throughput`;
  - default SLO is `request_latency:p90:lt:500` for embed and `request_latency:p90:lt:1000` for rerank.
- `render()` routes `embed` and `rerank` the same way it routes `synthetic`. It returns `Rendered(...)`
  with `kind="synthetic"`, `dataset="synthetic"`, the label and the duration.

`slo.py` needs no change: it accepts any lowercase AIPerf metric tag, `request_latency` included.

### `inferencex-e2e/infx/serving/schema.py`

- Add `embed` and `rerank` to the `scenario` literal.
- Add an `EmbedWorkload` model (`isl: list[str]`, `batch: list[int]`, `search`) and a `RerankWorkload`
  model (`passage: list[str]`, `docs: list[int]`, `search`). Both set `extra="forbid"`, and each has a
  `points()` method.
- `ServingSpec.workload` becomes `Workload | EmbedWorkload | RerankWorkload | None`. A model validator
  parses the raw mapping with the model that matches `scenario`, so a field from the wrong family fails
  with a clear message rather than a Pydantic union error.
- `variants()` passes the points through unchanged.

### `inferencex-e2e/infx/serving/client.py`

`probe()` runs only for a remote `SERVING_REMOTE_URL`. It sends the body that matches
`SERVING_SCENARIO_FAMILY`:

- embed: `POST /v1/embeddings {model, input: ["OK"]}`
- rerank: `POST /v1/rerank {model, query: "OK", documents: ["OK"]}`

Embed and rerank return no `usage`-based token source. The scenario YAML already leaves
`useServerTokenCount` unset, so `with_token_source` is skipped for them.

### Results: `infx/serving/sweep_result.py` and `infx/results/agentic/`

- `REQUIRED_PROBE_METRICS` becomes per family. For embed and rerank it is `request_throughput`,
  `request_latency` and `input_token_throughput`.
- `probe_metrics()` for embed and rerank returns:
  - `rps`
  - `items_per_s`, which is `rps × batch` (embed) or `rps × docs` (rerank), following the NIM inputs/s and
    passages/s convention
  - `input_tps`
  - `latency_{avg,p50,p90,p99}_ms`
- The nominal cache check (`_nominal_cache_pct`) is skipped when the label has no `cache` part. It already
  is today, so this needs only a confirming test.
- The fixed-concurrency aggregator (`process_agentic_result` and its backends) may assume output tokens.
  The smoke runs show whether it does. Each place that crashes on a missing output metric gets a null
  value; no separate aggregator is written.

### Serving configs (`inferencex-e2e/configs/serving/`)

There are six files, one per model × engine pair, and all use `runner: rtx6000pro-hcm` and `gpus: 1`.
Context is set to the model-card maximum.

| File | Engine flags (beyond model and served name) |
|---|---|
| `bge-m3-vllm-embed.yaml` | `--hf-overrides '{"architectures":["BgeM3EmbeddingModel"]}' --pooler-config '{"task":"embed"}' --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.2 --kv-cache-metrics --enable-prompt-tokens-details` |
| `bge-m3-sglang-embed.yaml` | `--is-embedding --context-length 8192 --mem-fraction-static 0.2 --enable-metrics` |
| `bge-reranker-v2-m3-vllm-rerank.yaml` | `--max-model-len 8192 --gpu-memory-utilization 0.15 --max-num-seqs 64 --kv-cache-metrics --enable-prompt-tokens-details` |
| `bge-reranker-v2-m3-sglang-rerank.yaml` | `--is-embedding --context-length 8192 --mem-fraction-static 0.15 --enable-metrics` |
| `qwen3-embedding-8b-vllm-embed.yaml` | `--runner pooling --dtype bfloat16 --pooler-config '{"seq_pooling_type":"MEAN"}' --max-model-len 32768 --gpu-memory-utilization 0.85 --kv-cache-metrics --enable-prompt-tokens-details` |
| `qwen3-embedding-8b-sglang-embed.yaml` | `--is-embedding --dtype bfloat16 --context-length 32768 --mem-fraction-static 0.85 --enable-metrics` |

If an engine cannot reserve the full context at the stated memory fraction, raise the fraction until it
starts. Never lower the context.

Each config first lands as a smoke variant: one grid point, `conc: 4`, `overrides.duration: 60`. It becomes
the full grid only after the smoke passes.

## Risks, checked by smoke

| Risk | Check | Fix if it fails |
|---|---|---|
| SGLang `/v1/rerank` returns a top-level list, which leaves `cohere_rankings` with no rankings | Error count in the smoke export | One-line patch to `CohereRankingsEndpoint.extract_rankings` in `utils/aiperf` so it accepts a list (`score`/`relevance_score`) |
| SGLang fails to load the `XLMRobertaForSequenceClassification` cross-encoder | Server log at start-up | Drop the SGLang reranker config and note it in the PR |
| The fixed-concurrency aggregator crashes without output tokens | Smoke job exit code and result JSON | Null-tolerant reads at the crash sites |
| vLLM `BgeM3EmbeddingModel` returns sparse/colbert outputs that AIPerf cannot parse | Smoke export | Set `--pooler-config '{"task":"embed"}'` (already set) |

## Testing

Tests follow the AGENTS.md rules: they call the real code with concrete inputs and assert on computed
output.

- `scenarios.render("embed", point=EmbedPoint("2k", 32), search=True)` gives a dataset with
  `isl == 2048` and `batchSize == 32`, a sweep objective of `request_throughput`, an SLA filter of
  `request_latency p90 < 500`, and the label `sweep-embed-isl2k-b32`.
- `scenarios.render("rerank", point=RerankPoint("512", 40))` gives a dataset with `rankings.passages == 40`,
  `passage_tokens == 512` and `query_tokens == 32`, no `sweep` block, and the label `fixed-rerank-p512-n40`.
- Each of these raises a `ScenarioError` naming the field:
  - `isl: 4k`
  - `docs: 7`
  - `slo` set without search
  - an embed workload under `scenario: rerank` (through `load_serving`)
- `probe_metrics()` on a hand-built embed export (`request_throughput.avg = 10`, batch 32) returns
  `items_per_s == 320`.

## Rollout

1. Code and tests on `feat/embed-rerank-bench`, then the local checks (`ruff`, `pytest`) from
   `inferencex-e2e/`.
2. Six smoke dispatches on `rtx6000pro-hcm`. Fix whatever the risk table catches.
3. Full grids, fixed and search. Report through `result-report`.
4. PR into `vng-benchmark`, with a bilingual title and a Chinese body section, after the user confirms the push.
