# GLM-5.2 DSpark Agentic Benchmark Report

[English](GLM52_DSPARK_AGENTIC_REPORT.md) | [中文](GLM52_DSPARK_AGENTIC_REPORT_zh.md)

Status: benchmark evidence and serving guidance for the validated H200 agentic recipe.

## Executive conclusion

The validated DSpark block-7 recipe is the strongest measured configuration for the CCU24 agentic workload when the objective is aggregate serving capacity and tail latency. Relative to the same-topology DSpark block-4 run, it delivered:

- 4.58% higher output throughput and 6.28% higher total throughput;
- 5.54% more successful profiled requests;
- 50.15% lower p90 TTFT and 44.75% lower p99 TTFT;
- 12.46% lower p99 end-to-end latency;
- 3.38 average accepted tokens per speculative step and 4.47 at p90.

The 34.14% acceptance-rate figure must not be read in isolation. Block 7 drafts about 7.96 tokens per step, versus 4.97 for block 4. Its accepted length is higher even though the acceptance-rate denominator is larger. The trade-off is 60.18% more draft tokens and 5.03% lower output TPS per active user. For a CCU24 capacity target, keep block 7; for a strict per-session decode-efficiency target, block 4 remains a meaningful fallback.

## 1. Scope and provenance

This report records the production-path experiment, not a synthetic microbenchmark. All three runs use the GLM-5.2 agentic replay workload and the same H200 GreenNode family.

| Item | Value |
|---|---|
| Target model | `zai-org/GLM-5.2-FP8` served from `/models/PhalaCloud/GLM-5.2-W4AFP8` |
| Target quantization | W4AFP8 (`--quantization w4afp8`) |
| DSpark draft | `/models/RedHatAI/GLM-5.2-speculator.dspark` |
| Draft quantization | Explicitly `unquant` (BF16 checkpoint; no target quantization inheritance) |
| Serving image | `thangquang0909/sglang:v0.5.16-dspark-v2-g4` |
| Recipe commit | `310fca9287ce363146cd7718fbf739a8eab63c01` |
| Hardware | H200 GreenNode, 8 GPUs |
| Workload | SemiAnalysis Weka 256k agentic coding replay |
| Concurrency | CCU24 |
| Duration | 3,600-second profiling window plus warmup/teardown |

Runs:

- [EAGLE reference, run 30163138459](https://github.com/vngcloud/InferenceX/actions/runs/30163138459)
- [DSpark block 4, run 30690287203](https://github.com/vngcloud/InferenceX/actions/runs/30690287203)
- [DSpark block 7, run 30693457521](https://github.com/vngcloud/InferenceX/actions/runs/30693457521)

The raw benchmark rows are normalized by the AIPerf metrics extractor. The committed repository contains the recipe and tests; large raw artifacts remain in the benchmark workspace and are referenced by run ID rather than vendored into Git.

## 2. What was fixed before benchmarking

The experiment was preceded by a real serving-path debugging sequence:

1. The initial DSpark path inherited the target W4AFP8 quantization for the draft checkpoint. The draft then produced no useful accepted proposals.
2. The launcher was changed to pass `--speculative-draft-model-quantization unquant`, keeping target quantization and draft quantization independent.
3. The topology was aligned with the reference: TP8/DP8 with DP Attention, DP LM head enabled, and the SGLang cache-aware router.
4. `mem-fraction-static` was reduced from 0.85 to 0.75 to reserve headroom for long agentic prefill and DSpark verification graphs.
5. The launcher requests a 32,768-token chunked-prefill size. Under DP Attention, SGLang adjusts the effective runtime value to 4,096 tokens and records `max_prefill_tokens=16384`; this runtime adjustment is visible in the server log.
6. The recipe test now asserts the generated command, including target `w4afp8`, draft `unquant`, TP8/DP8/DPA, and the selected DSpark block size.

These changes are intentionally in the launcher and its regression test. They do not change generic SGLang quantization inheritance behavior.

## 3. Serving configuration

The material production flags are:

```text
--model-path /models/PhalaCloud/GLM-5.2-W4AFP8
--quantization w4afp8
--tp 8 --dp 8 --enable-dp-attention --enable-dp-lm-head
--chunked-prefill-size 32768
--mem-fraction-static 0.75
--kv-cache-dtype fp8_e4m3
--enable-hierarchical-cache --hicache-ratio 2
--speculative-algorithm DSPARK
--speculative-draft-model-path /models/RedHatAI/GLM-5.2-speculator.dspark
--speculative-draft-model-quantization unquant
--speculative-dspark-block-size 7
--schedule-policy lpm
```

The benchmark uses the router on the public port and the backend metrics endpoint on the adjacent SGLang port. The router is cache-aware and DP-aware. The generated server command confirms the requested launch flags; the startup log confirms the effective TP8/DP8/DPA topology, cache settings, and DSpark initialization. It also records SGLang's DP-Attention adjustment from requested chunked prefill 32,768 to effective 4,096.

### Block-size terminology

In this recipe, `--speculative-dspark-block-size 7` selects DSpark gamma 7. The server arguments show `speculative_num_draft_tokens=8`, and the draft checkpoint declares a native block size of 8. SGLang therefore logs a gamma/configuration mismatch warning (`gamma=7`, draft `block_size=8`) while using the requested gamma-7 serving path. This is expected for the tested configuration, not evidence that DSpark was disabled.

The measured block-7 run drafted 7.957 tokens per step on average. The accepted-length metrics include the verification result and should be interpreted as tokens accepted per speculative step, not as a raw percentage.

## 4. Results

All rows below are CCU24. `ok_with_errors` for block 7 reflects one warmup-only empty-content response; the profiling window itself had zero errors.

| Metric | EAGLE | DSpark block 4 | DSpark block 7 |
|---|---:|---:|---:|
| Successful profiled requests | 2,745 | 2,746 | **2,898** |
| Input TPS | 81,574.88 | 82,506.49 | **87,700.56** |
| Output TPS | 629.49 | 647.48 | **677.11** |
| Total TPS | 82,204.38 | 83,153.96 | **88,377.66** |
| Output TPS/user | **27.50** | 28.92 | 27.46 |
| Request QPS | 0.7541 | 0.7544 | **0.7962** |
| TTFT p50 | **0.908 s** | 1.102 s | 1.023 s |
| TTFT p90 | **3.422 s** | 12.106 s | 6.035 s |
| TTFT p99 | **17.346 s** | 83.871 s | 46.340 s |
| E2E p90 | 83.527 s | 94.704 s | 88.530 s |
| E2E p99 | 294.259 s | 282.372 s | **247.181 s** |
| ITL p90 | 68.089 ms | 69.085 ms | **67.818 ms** |
| Queue time p90 | **3.841 s** | 26.822 s | 21.992 s |
| Queue time p99 | **29.183 s** | 154.686 s | 87.453 s |
| Token usage p90 | 0.73 | 0.97 | 0.97 |
| Response cache hit | 96.364% | 96.246% | **96.398%** |
| Draft tokens/step | 3.976 | 4.968 | **7.957** |
| Accept rate avg | **52.922%** | 51.105% | 34.140% |
| Accept rate p90 | **87.083%** | 65.789% | 49.580% |
| Accept length avg | 2.582 | 3.038 | **3.384** |
| Accept length p90 | 3.613 | 3.632 | **4.471** |
| Prefill forward p90 | 2.939 s | 2.160 s | **1.977 s** |
| GPU utilization avg | 97.024% | 95.171% | 95.030% |
| GPU power avg | 420.14 W | 424.02 W | **417.52 W** |
| GPU memory avg | 145.23 GB | 146.36 GB | **143.84 GB** |
| GPU memory max | 145.78 GB | 149.75 GB | 150.04 GB |

### Block 7 versus block 4

This is the cleanest comparison because topology, image, target, draft, workload, CCU and memory setting are held constant while block size changes.

- Output TPS: **+4.58%**
- Total TPS: **+6.28%**
- Successful requests: **+5.54%**
- TTFT p90/p99: **-50.15% / -44.75%**
- E2E p99: **-12.46%**
- Accept length average/p90: **+11.41% / +23.10%**
- Draft tokens per step: **+60.18%**
- Output TPS/user: **-5.03%**

The result says that block 7 is better at serving the closed-loop CCU24 population, not that every individual decode stream is cheaper. It completes more agentic work and reduces queue/TTFT tails, while paying for longer draft verification.

### Block 7 versus EAGLE

Block 7 has 7.56% higher output TPS and 7.51% higher total TPS than EAGLE, with nearly identical output TPS/user (-0.15%). It also has lower E2E p99 and slightly lower ITL. EAGLE retains a clear TTFT and queue-time advantage in this data.

This comparison is useful but not perfectly memory-matched: EAGLE used `mem-fraction-static=0.85`, while DSpark block 4 and block 7 used 0.75 to reserve prefill/speculation headroom. The EAGLE row should therefore not be used to claim a pure algorithm-only latency win.

## 5. How to read DSpark acceptance

Acceptance has three related but different interpretations:

1. **Accept rate** is the fraction of proposed draft tokens accepted by verification. A longer block increases the denominator, so rate normally falls when the block is extended.
2. **Accept length** is the useful output per speculative step. It is the more direct measure of how many target-model tokens the speculative step saved.
3. **Accept length p90** shows the upper part of the steady-state distribution. Block 7 reaches 4.47 at p90, compared with 3.63 for block 4 and 3.61 for EAGLE.

Therefore, block 7's 34.14% rate is compatible with strong speculation: it proposes almost eight tokens and accepts 3.38 on average. Comparing the percentage alone would penalize the longer block for doing more useful work per verification step.

Direct server metrics across the eight DP ranks show block-7 accept-length p90 values roughly from 3.98 to 4.47 and no zero-accept plateau. The direct profile contains 2,898 successful profiling records; the only error was a warmup response with no actual content.

## 6. Safety and correctness checks

Run 30693457521 passed the workflow and artifact checks:

- GitHub Actions conclusion: `success`.
- Recipe commit: `310fca9287ce363146cd7718fbf739a8eab63c01`.
- CSV extraction: 87 unique rows, zero identity duplicates.
- Block-7 profiling: 2,898 successes and zero profiling errors; the run itself was not cancelled, while 7 in-flight requests were cancelled during timeout cleanup.
- Server metrics and DCGM telemetry: present.
- No server OOM, traceback, CUDA/NCCL fatal error or non-zero GPU XID.
- Aggregate AIPerf output TPS and normalized CSV output TPS agree within 0.304%.

The one warmup `InvalidInferenceResultError` means that one warmup response contained usage/metadata but no content. It did not enter the profiling success set and did not indicate a server crash.

## 7. Reproduction checklist

Use the checked-in recipe and test:

```bash
python -m pytest utils/test_glm52_dspark_recipe.py -v
```

Before dispatching a long run, verify:

1. The target image/model is W4AFP8 and the draft files include `config.json` and `model.safetensors`.
2. The generated command includes target `w4afp8` and draft `unquant` independently.
3. The topology is TP8/DP8 with `--enable-dp-attention` and `--enable-dp-lm-head`.
4. HiCache ratio, FP8 KV cache, memory fraction and chunked prefill match the intended capacity budget; under DP Attention, the effective chunked-prefill size is 4,096.
5. The server log reports DSpark initialization before traffic is sent.
6. Acceptance is read from `sglang:spec_accept_rate` and `sglang:spec_accept_length`, not inferred from throughput alone.
7. Artifacts include AIPerf profile exports, server metrics, GPU telemetry and the exact startup command.

## 8. Recommendation and remaining work

Keep block 7 as the CCU24 production candidate for this image and workload. Keep block 4 as the lower-draft-cost fallback and as the control for future changes. Any final image promotion should preserve the unquantized draft override and the DP LM-head/topology assertions.

The next useful experiment is a repeated block-7/block-4 comparison at CCU8, CCU16, CCU24 and CCU32 with the same memory budget. A single long run establishes the current result, but repeated ladders are needed to quantify variance across lower and higher load.
