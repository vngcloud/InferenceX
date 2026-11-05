# Embed/rerank capacity (TPM @ SLO) for PTU sizing — design

Date: 2026-11-03 · Branch: `feat/embed-rerank-ptu` (from `feat/embed-rerank-bench`) · Supersedes the grid part of
`2026-11-02-embed-rerank-bench-design.md`.

## Goal

Product sizes embed/rerank quota the Azure / Together PTU way: one capacity number per model, **input TPM**
(embed and rerank have no output tokens, so normalized TPM = input TPM). The benchmark answers one question
per model and request size:

> How many input tokens per minute does the endpoint take while p90 latency stays under the SLO?

The same rows, re-run after an engine or flag change, are the version-to-version comparison.

Out of scope: aiperf changes, server-side token counting, Qwen endpoints, self-hosted engines, a long
confirmation run (add if search results turn out unstable).

## Request sizes

Three rows per model. The last one matches Product's quota profile (embed ~30k tok/request, rerank ~50k).

| Model | Use case | Items / request | Tokens / item | Input tok / request | p90 SLO | Label |
|---|---|---|---|---|---|---|
| bge-m3 | Query search | 1 | 512 | 512 | 200 ms | `sweep-embed-isl512-b1` |
| bge-m3 | Index tài liệu | 32 | 512 | 16,384 | 1 s | `sweep-embed-isl512-b32` |
| bge-m3 | Index lớn (quota) | 16 | 2,048 | 32,768 | 2 s | `sweep-embed-isl2k-b16` |
| bge-reranker-v2-m3 | Top-10 | 10 | 512 + 32 query | 5,440 | 500 ms | `sweep-rerank-p512-n10` |
| bge-reranker-v2-m3 | Top-40 | 40 | 512 + 32 | 21,760 | 1 s | `sweep-rerank-p512-n40` |
| bge-reranker-v2-m3 | Top-100 (quota) | 100 | 512 + 32 | 54,400 | 2 s | `sweep-rerank-p512-n100` |

The 8k context limit applies per input (embed) and per query+passage pair (rerank), not per request. Checked
against the MaaS gateway on 2026-11-03: 16 × 2k embed → 200, `usage.prompt_tokens` 32,032; 100 × 500-token
rerank → 200, 52,000; one 9k embed input → 400 "at least 8193 input tokens".

Each row runs AIPerf `adaptive_search` (objective `request_throughput`, unchanged). For a fixed request size,
TPM = RPS × tokens/request, so maximizing RPS maximizes TPM. The fixed c1/c5 rows of the 2026-11-03 run are
dropped; when a row has no passing concurrency, the c1 point in `search_history.json` gives the latency floor.

## TPM

- **embed:** `input_token_throughput.avg × 60` from AIPerf (client-side tokenizer count, which matches B × isl).
- **rerank:** AIPerf reports no input tokens for `cohere_rankings`: its payload walker only knows `passages`, and
  Cohere sends `documents` (fix pending upstream in ai-dynamo/aiperf#1516). InferenceX computes
  `rps × docs × (32 + passage) × 60` from the label. The server scores the query with every document, so it
  counts the query per pair as this formula does; it adds a few special tokens per pair. Gateway check: 100 docs
  → formula 51,600 vs server 52,000 (−0.8%). When aiperf#1516 reaches the pinned wheel, keep the formula: the
  upstream count is query-once and lower than the server's.

## Product mapping

One row per request size, one headline column `TPM in`. Product takes a customer's tokens/request, picks the
nearest row at or above it, and divides the customer's TPM by that row's `TPM in` to get the PTU count. RPM is
implied (TPM ÷ tokens/request). `TPM per GPU` stays empty for remote endpoints until the GPU count behind
the endpoint is known.

## Changes

| File | Change |
|---|---|
| `inferencex-e2e/infx/serving/scenarios.py` | `EMBED_BATCH` / `RERANK_DOCS` fixed sets → any positive integer (needed for b16 and n100). `isl` / `passage` keep their named sets. |
| `inferencex-e2e/infx/serving/sweep_result.py` | Pooling metrics gain `tpm`: `input_tps × 60`, or for rerank labels without `input_tps`, `rps × docs × (RERANK_QUERY_TOKENS + passage) × 60`. Label parsing reads the passage size alongside the doc count. |
| `inferencex-e2e/configs/serving/maas-bge-m3-*.yaml`, `maas-bge-reranker-v2-m3-*.yaml` | Replace with one YAML per row (six files): one grid point, `search: true`, the row's `slo`, `overrides.duration: 60`, `conc: 1`. One YAML per row because the SLO differs per row and the schema has one `slo` per file. |
| Qwen `maas-*.yaml` | Untouched. |

Remote endpoints already work end to end (`endpoint: {url, model, api-key-secret, tokenizer, tls-verify}` →
`dispatch.sh`, no engine launch). The GitHub secrets `MAAS_BGE_M3_AUTH` / `MAAS_BGE_RERANKER_V2_M3_AUTH`
(`Basic <b64>`) must be set by the user before the CI dispatch.

## Testing

- `scenarios.render("embed", point=EmbedPoint("2k", 16), search=True)` → dataset `isl 2048`, `batchSize 16`,
  label `sweep-embed-isl2k-b16`. `docs: 100` renders `rankings.passages == 100`. `batch: 0` raises
  `ScenarioError` naming `workload.batch`.
- `probe_metrics` on a hand-built rerank export without `input_token_throughput` (`request_throughput.avg = 2`,
  label `…-rerank-p512-n10`) → `tpm == 2 × 10 × 544 × 60 = 652,800`. On an embed export with
  `input_token_throughput.avg = 1000` → `tpm == 60,000`.
- `to_matrix --check` on the six new YAMLs.

## Rollout

1. Code + tests, `ruff` + `pytest` from `inferencex-e2e/`.
2. Remote runs first: bge-m3 and bge-reranker, the three rows of one endpoint one after another (same endpoint in
   parallel skews latency), the two endpoints in parallel. Local harness if the CI secret is not set yet.
3. Six rows into the summary sheet in `docs/model_adaption/maas_model_capacity.xlsx` with the columns of the
   table above plus `RPM max`, `TPM in`, `p90 đo được`.
4. Ask the user before any push.
