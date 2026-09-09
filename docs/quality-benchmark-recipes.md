# Quality Benchmark Recipes

Use this sheet to choose and report an eight-benchmark quality run. Counts are fixed, representative samples; `full` means the complete official split. Estimates are initial planning ranges and should be replaced with measured model/endpoint rates.

## Categories

| Category | Benchmarks |
| --- | --- |
| General knowledge | GPQA, MMLU-Pro, HLE |
| Coding | LiveCodeBench, SciCode |
| Agentic coding | BFCL, SWE-bench Pro, DeepSWE |

## Tiers

| Tier | Use | General: GPQA / MMLU-Pro / HLE | Coding: LCB / SciCode | Agentic: BFCL / SWE-Pro / DeepSWE |
| --- | --- | --- | --- | --- |
| Smoke | Wiring and artifact validation | 2 / 2 / 2 | 2 / 2 | 4 / 1 / 1 |
| Balanced | Routine comparison and regression tracking | 50 / 100 / 50 | 50 / 8 | 100 / 10 / 10 |
| Extended | Release candidate and high-confidence comparison | full / 500 / 250 | 200 / 30 | 500 / 50 / 50 |
| Full | Announcement or official competition | full / full / full | full / full | full / full / full |

Use deterministic, stratified manifests for Balanced and Extended; do not use the first N rows. Smoke validates execution only and must not be presented as a quality score.

## Concurrency and planning time

Run benchmarks as separate parallel jobs. A category's wall time is approximately its slowest benchmark, not the sum. Start with these per-benchmark concurrencies:

| Tier | General | Coding | Agentic | Endpoint-wide cap | Expected category wall time |
| --- | --- | --- | --- | --- | --- |
| Smoke | 2-4 | LCB 2-4, SciCode 2 | BFCL 2, SWE/DeepSWE 1 | 8 | General 5-15m; Coding 30-45m; Agentic 30-60m |
| Balanced | GPQA/MMLU 8, HLE 4 | LCB 8, SciCode 4 | BFCL 8, SWE/DeepSWE 2 | 16 | General 1.5-3h; Coding 1-2.5h; Agentic 2.5-6h |
| Extended | GPQA/MMLU 8, HLE 6 | LCB 8-12, SciCode 4-6 | BFCL 12, SWE/DeepSWE 4 | 24 | General 6-12h; Coding 4-8h; Agentic 8-18h |
| Full | 8-16 | LCB 12-16, SciCode 6-8 | BFCL 16, SWE/DeepSWE 4-8 | 32 after load test | 12h to several days |

Concurrency improves throughput only until endpoint saturation. Agent steps within one SWE/DeepSWE task remain sequential. Record p50/p95 latency, tokens, retries, and valid samples/hour so future estimates use observed throughput.

## Decision and report

- **Decision:** category, tier, reason, model, endpoint, sample-manifest version, per-benchmark concurrency, endpoint cap.
- **Report:** run URL/ID, commit, samples requested/valid/failed, score, wall time, token usage/cost, retry/error rate, and artifact link.
- **Comparison gate:** same dataset revision, manifest, prompt/harness revision, generation settings, scorer, and tier.
- **Promotion:** Smoke passed -> Balanced; Balanced stable -> Extended; Extended reviewed -> Full.

Full results are the only announcement-grade results. Any partial, retried, or non-comparable run must be labeled explicitly.
