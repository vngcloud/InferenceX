# Expert distribution diagnostic

[中文](DISTRIBUTION_zh.md)

Run the existing dispatch workflow on this research branch with
`start-from=distribution_b_ep1`. It schedules two dependent CI jobs on h200-2,
EP1 then EP8. Screening jobs are skipped. Image, model and recorder are pinned
to the existing campaign. Each job has a 60-minute CI ceiling; requests have
five-minute timeouts, the probe has a 20-minute hard ceiling, and it stops
starting new shape cases after 15 minutes. Startup is additional.

This is a synthetic routing/accounting diagnostic, not an AgentX performance
score or a representative production routing distribution. No EAGLE, HiCache,
radix caching, CUDA graphs or EPLB. EP8 explicitly uses DeepEP normal. Small
instrumentation adds forward-mode and local input-tensor length metadata to the
original per-pass recorder; before/after source hashes are uploaded. Per-pass
collection is bounded by the finite request set and timeout, not the stat ring.

## Calibration and workload

1. Single requests of 8, 64 and 512 explicit input tokens, each producing one
   token. Each active MoE layer must have exactly `8 * input_tokens` assignments.
2. Prefill: 512, 8192 and 32768 tokens/request, batch sizes 1 and 8, one output
   token. Prompt token IDs and their hashes are identical across topologies.
3. Decode: contexts of 512 and 8192 tokens, batch sizes 1, 8 and 16, 33 output
   tokens with EOS ignored. Forward-mode metadata separates the prefill from
   the following decode passes. The complete case must account for
   `8 * batch * (input_tokens + output_tokens - 1)` assignments per active layer.

The repeated code/prose seed intentionally fixes content while varying shape.
Identical batch prompts are a correlated-routing stress case. They do not sample
the diversity of concurrent AgentX trajectories. A production conclusion needs
a subsequent trace-derived corpus using the same accounting checks.

Fail immediately on missing rank dumps, unequal EP1 replicas, invalid expert
mapping or failed assignment conservation. Do not divide mismatched counts by
an assumed factor. Retain failure artifacts and stop dependent jobs.

## Metrics and meanings

| Metric | Meaning and interpretation |
|---|---|
| Assignments / routed tokens | With top-k 8, assignments must equal eight times logical routed tokens. First validate the measurement itself. |
| Active experts and empty fraction | Indicates sparsity and fragmentation. One token necessarily leaves at least 248/256 experts empty; this alone is not imbalance. |
| Tokens per active expert (M) | Indicates whether routed GEMMs receive tiny or substantial batches. Router counts exclude kernel padding and do not establish GEMM timing. |
| Hottest-expert share | Fraction of assignments reaching one expert. Compare at matched batch size and phase. |
| Effective experts, exp(entropy) | Number of equally used experts giving the same concentration; approaches 256 for uniform usage. |
| Rank max/mean | Maximum rank assignments divided by mean. 1 is balanced, 2 means one rank receives twice the average. Actual placement is required. |
| Rank load CV | Standard deviation / mean across ranks; 0 is perfectly balanced. |
| Assignment balance efficiency | Mean/max rank load; a count-based balance proxy, not measured GPU utilization or expected speedup. |

Reports preserve every layer and forward mode rather than pooling dense prefill
and sparse decode. Case-level rank loads can hide transient hot ranks: raw
per-pass files are retained, but global pass alignment is not assumed. EP1 rank
loads are projected onto a hypothetical EP8 partition, not actual EP1 compute.
EP8 owners are derived from the recorded physical-to-logical mapping with 32
physical experts/rank and no redundant experts.

After the count gate passes, use representative traces to examine p50/p95/max
load by layer and aligned pass. Only then correlate the busiest rank with GEMM,
dispatch/combine and waiting times in a separate profile. Distribution alone
cannot identify a communication bottleneck or justify enabling EPLB.
