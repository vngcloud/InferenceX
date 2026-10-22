<div align="center">

**English** | [中文](./MIXED_AGENTX_WORKLOAD_DESIGN_zh.md)

</div>

# Mixed AgentX Workload Design

Status: proposed for review.

## Goal

Build a private, reproducible AgentX dataset that measures engine-level serving
capacity under a configurable mix of coding, short multi-turn chat, and RAG
traffic. The first reference profile targets `zai-org/GLM-5.2-FP8`.

The workload measures the cache behavior produced by real prompt structure. It
does not force the aggregate cache-hit rate to match the production observation
of roughly 70%.

## Scope

The benchmark reuses the existing `inferencex-agentx-mvp` scenario, including
its warmup, session-tree concurrency, recycling, first-turn cache busting, and
10-second coding trace idle-gap cap.

This is an engine-level benchmark. The current launcher sends requests directly
to `http://localhost:$PORT/v1/chat/completions`; it does not reproduce
production gateway load balancing or cross-replica cache locality.

An AgentX session is a client-side scheduling and prompt-construction unit.
SGLang receives stateless HTTP requests with cumulative messages. Correlation
IDs are observability metadata and do not create routing affinity.

## Pinned inputs

| Input | Revision | Use |
|---|---|---|
| `semianalysisai/cc-traces-weka-062126` | `23f152f6f0f9399a85901b89a6458def0ef16729` | Coding sessions |
| `allenai/WildChat-1M` | `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | Short multi-turn chat |
| `nvidia/ChatRAG-Bench` | `22ece8bb870ddcf3f7aacfd5b6b0446d112a1e92` | Multi-turn RAG |
| `zai-org/GLM-5.2-FP8` | `70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1` | Reference tokenizer and chat template |

The reference server context limit is 500,000 tokens, matching the existing
GLM-5.2 H200 SGLang launcher. Every request must satisfy:

```text
templated_input_tokens + requested_output_tokens <= 500000
```

Sessions are rejected rather than truncated when any request exceeds the
limit.

## Architecture

Use one thin hybrid loader and one immutable per-model manifest.

- Weka rows are delegated unchanged to the existing Weka loader.
- WildChat and ChatRAG rows are converted to the existing AgentX
  `Conversation`/turn representation.
- The manifest contains ordered logical session instances, source revisions,
  source row IDs, tokenizer/template hashes, weights, timing policy, context
  limit, and build seed.
- A manifest is frozen across framework and hardware comparisons. Changing
  weights produces a new manifest rather than mutating a running workload.

Do not add a new serving scenario, routing abstraction, or dataset framework.

## Workload classes

### Coding

Preserve current Weka reconstruction byte-for-byte, including parent/subagent
trees and existing timing semantics.

Coding selection is stratified by session input-token decile and whether the
session contains subagents.

### Short multi-turn chat

Use complete clean WildChat conversations with:

- at least two assistant responses;
- strict alternating user/assistant roles;
- non-empty string content;
- no toxic or redacted conversation flags;
- monotonic assistant completion timestamps; and
- no request exceeding the target context.

For request `n`, send cumulative recorded history ending at user turn `n`.
Recorded assistant messages construct later prompts; live benchmark responses
are measured and discarded. `max_tokens` is the GLM tokenizer length of the
omitted recorded assistant response.

This class intentionally keeps natural within-conversation prefix reuse.
Sessionless HTTP does not imply cacheless inference: later cumulative prompts
can hit the same engine's prefix cache.

WildChat selection is stratified by round count, language, and source model.

### RAG

Use the ChatRAG source format rather than synthesizing retrieval documents:

- retain the top five `ctxs`, matching the published evaluation default;
- place the RAG instruction and retrieved documents in the structured system
  message;
- retain the latest seven recorded conversation messages, matching the source
  evaluation history window;
- preserve the source dataset-specific answer instruction;
- use `answers[0]` as the omitted target response; and
- set `max_tokens` to the pinned GLM tokenizer length of the non-empty
  `answers[0]`, with no additional output cap; and
- do not apply the source evaluator's 4,096-token truncation. Only the
  reference model's 500,000-token limit applies.

ChatRAG publishes cumulative snapshots without conversation IDs. Reconstruct
parent links when:

```text
child.messages[:-2] == parent.messages
```

Deduplicate exact snapshots first. Create a parent link only when the expression
above identifies exactly one remaining snapshot. A missing or ambiguous parent
starts a partial-history trajectory; its embedded history is still sent intact.
This rule avoids arbitrary stitching when identical message histories have
different retrieved contexts or metadata.

The system/context message, dataset-specific answer instruction, seven-message
window, and GLM chat template are explicit target-model adaptations. Preserve a
canonical copy of the source roles and content before applying them.

The license-reviewed internal default subset pool is:

- Doc2Dial;
- QuAC;
- QReCC;
- DoQA cooking, movies, and travel; and
- ConvFinQA.

| Subset | Source terms | Required handling |
|---|---|---|
| [Doc2Dial](https://huggingface.co/datasets/IBM/doc2dial) | CC BY 3.0 | Preserve attribution and notice |
| [QuAC](https://quac.ai/datasheet.pdf) | MIT; paper citation requested | Preserve notice and citation |
| [QReCC](https://github.com/apple/ml-qrecc#license) | Dataset CC BY-SA 3.0; retrieved web passages retain source rights | Preserve attribution/share-alike notice and provenance |
| [DoQA](https://ixa.eus/node/12931) | CC BY-SA 4.0; derived from Stack Exchange | Preserve attribution/share-alike notice and provenance |
| [ConvFinQA](https://github.com/czyssrs/ConvFinQA) | MIT | Preserve copyright and notice |

Use remains subject to the organization's legal policy. Exclude TopiOCQA by
default due to its non-commercial license. Exclude INSCIT, CoQA, HybriDialogue,
and SQA until their dataset-level reuse terms or source provenance are
approved. The complete download remains available for analysis but is not
selected by the default manifest.

## Timing

### WildChat

Estimate user think time as:

```text
(completion_time(A[i+1]) - completion_time(A[i]))
  - source_service_time_estimate(A[i+1])
```

The source service estimate is p10 grouped by source model and output-token
bucket of `A[i+1]`, using `cl100k_base` for source-model token buckets. The
residual estimates the delay from completion of `A[i]` to submission of the
next user request. Clamp negative residuals to zero and cap the result.

```text
WILDCHAT_THINK_TIME_CAP_SECONDS=10
```

### ChatRAG

ChatRAG has no timestamps. Deterministically sample inter-turn delays from the
accepted capped WildChat residual distribution using the manifest seed. Record
this explicitly as `timing_source: wildchat_proxy`; do not present it as source
timing. Production RAG OTel traces replace this proxy when available.

## Mix

Weights are configurable non-negative input-token shares that sum to 1. The
first reference profile is:

```yaml
mix:
  coding: 0.70
  short_chat: 0.15
  rag: 0.15
```

Select whole trajectories deterministically. The generated estimated
input-token mix must be within 0.5 percentage point of the requested weights.
Also report request share and session share because long RAG prompts can meet a
token target with relatively few requests.

Observed server-token mix is diagnostic. Warn when it differs from target by
more than 3 percentage points, but never rewrite the manifest automatically.

## Expected cache behavior

No extra per-turn salt is added. The AgentX first-turn marker isolates logical
sessions; later reuse comes from actual repeated prefixes.

GLM-5.2 tokenization of ChatRAG shows both desired RAG patterns:

| Subset | Median input tokens | Infinite-cache common-prefix upper bound |
|---|---:|---:|
| ConvFinQA | 943 | 70.25% |
| DoQA variants | 294-301 | 68.12-68.91% |
| Doc2Dial | 1,855 | 15.03% |
| QReCC | 2,679 | 14.29% |
| QuAC | 2,526 | 20.37% |
| INSCIT (excluded by default) | 754 | 9.49% |
| TopiOCQA (excluded by default) | 781 | 10.60% |

For each subset, the last column is:

```text
sum(LCP(parent_prompt, child_prompt) for uniquely linked children)
-----------------------------------------------------------------
sum(input_tokens for all requests)
```

Prompts use top-five contexts, the latest seven messages, the adaptations above,
and the pinned GLM chat template. Missing and ambiguous parents contribute no
LCP numerator. The offline statistic excludes the AgentX first-turn marker,
cache block rounding, eviction, and capacity limits. It is therefore a
reproducible parent-request common-prefix upper bound, not a server cache-hit
prediction. Changing retrieved documents breaks the prefix near the front of
the prompt, while stable document sets retain substantial reuse.

## Metrics

Report combined and per-class:

- requests, sessions, input/output tokens, and TPM;
- TTFT, TPOT, inter-token latency, and end-to-end latency;
- target, estimated, and observed input-token mix; and
- expected prefix reuse.

The authoritative overall cache-hit rate comes from profile-window server
Prometheus counter deltas after warmup has drained:

```text
overall_cache_hit_rate =
  delta(cache_read_input_tokens) / delta(prompt_input_tokens)
```

For SGLang, sum reset-aware deltas of `sglang:cached_tokens` across
`cache_source` labels and divide by the corresponding
`sglang:prompt_tokens` delta. Sum each deduplicated inference worker once; do
not mix router/frontend replicas of the same counter. Capture the start
snapshot immediately before profiling and the end snapshot after profiled
requests drain, excluding warmup and cooldown. Other backends use equivalent
token counters; if none exist, report the token cache-hit rate as unavailable
rather than substituting a whole-run gauge.

Report per-class observed cache hit only when the endpoint returns per-request
cached-token counts. Do not infer per-class cache hit from a global counter.

## Validation

The build fails on source revision mismatch, tokenizer/template hash mismatch,
invalid roles, malformed timing, missing source rows, context overflow, or mix
tolerance failure.

Minimum verification:

- deterministic filtering, timing, and whole-session selection;
- byte-identical Weka reconstruction;
- canonical source role/content equality before adaptation, plus exact tests for
  each allowed context/instruction/window/template transformation;
- exact ChatRAG parent reconstruction for resolvable snapshots;
- context and output token accounting with the pinned GLM tokenizer;
- absence of unneeded privacy/moderation fields;
- AgentX integration replay; and
- aggregate totals equal the sum of per-class totals.

## Delivery

Raw corpora and generated private artifacts stay under the local `datasets/`
tree and are not committed. Only loader code, tests, manifest schema, and
documentation are versioned.

The first implementation stops at the three-class engine-level benchmark.
Gateway routing simulation, automatic production-ratio fitting, retrieval
execution, and public dataset publishing are out of scope.
