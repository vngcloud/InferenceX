# Quality Benchmark Time Estimates

Estimates are for GLM-5.2 at concurrency 4. API latency, reasoning length,
rate limits, image setup, and sandbox execution can change wall time.

| Benchmark | Full size | Full estimate | 20-sample estimate | Default max generation |
|---|---:|---:|---:|---:|
| GPQA-Diamond | 198 | ~20 min | ~2 min | 8K |
| MMLU-Pro | 12,032 | ~22 h | ~2 min* | 8K |
| HLE | 2,158 | ~4.5 h | ~5 min* | 16K |
| LiveCodeBench | 1,055 | ~8 h | ~15 min | 16K |
| BFCL v4 | ~2,000 | ~1.5 h | <1 min | 8K |
| SciCode | 288 subproblems | ~5.5 h | ~23 min | 16K |
| SWE-bench Pro | 731 | ~45 h | ~30 min | 32K per agent turn |
| DeepSWE | 113 | ~29 h | ~5 h | 32K per agent turn |

\* `LIMIT` applies per subtask for MMLU-Pro and HLE, so `LIMIT=20` evaluates
more than 20 total questions.

These are output ceilings, not target lengths. Natural stop conditions still
end generation early. Override any default with `MAX_GEN_TOKENS`; CI matrix
rows may use `max-gen-tokens`. Streaming avoids idle connection timeouts but
does not remove total request deadlines or the cost of long generations.
