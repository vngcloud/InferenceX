# AgentX EP A/B campaign

[中文](README_zh.md)

Research branch only. Do not merge its replacement dispatch workflow into main.
The existing registered `e2e-tests.yml` dispatches 12 separate, dependency-chained
screening jobs, then two separate recorder diagnostics, exclusively on runner
`h200-greennode_06` (verified h200-2). This runner uses Docker, not the unrelated
four-GPU Slurm partition visible from this machine.

Dispatch this branch with `duration-override=1200`, `agentx-fast=false`.
The configuration order is A, B, E, F, C, D, alternating topology order.
Every screening job has ten warmup requests per lane and 1200 seconds profiling.
Four hours is measurement time only, excluding warmup/startup and diagnostics.
No subsequent job runs after failure. Artifacts upload with `always()`.

Provenance is recovered from original CI run 32627106733. Image, AIPerf and
dataset are pinned. The cached model snapshot is independently recorded.
The user-approved max-running-requests=16 differs from the historical recipe's
2*CCU; historical values are context, not matched baselines for this campaign.
Both topologies use memory fraction 0.80. Cache-aware router behavior remains
identical to the original recipe.

Recorder mode requires startup flags, so diagnostics are separate jobs using
the original stat recorder's 200-slot circular buffer. Recording starts after
normal warmup, lasts at most five minutes, and is excluded from timing results.
Original stat output lacks phase labels, chronological cursor, and verified
cross-rank iteration alignment. Summaries explicitly retain these limitations;
do not interpret pooled slots as synchronized global passes or invent rank load.
Raw counts remain available for further analysis. EPLB stays disabled.

For comparison, use paired output throughput, completed turns, TTFT/ITL p50/p95,
acceptance, workload distributions, queue/batch/cache metrics, and actual KV
capacity. Single runs are screening only. Select confirmation after review;
confirmation is deliberately not auto-dispatched.
