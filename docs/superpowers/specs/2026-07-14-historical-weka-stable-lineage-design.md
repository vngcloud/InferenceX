# Historical Weka Stable-Lineage Dataset Design

Date: 2026-07-14

## Status

Approved by the user for implementation on 2026-07-14.

## Objective

Rebuild the GLM-5.2 Claude Code corpus from the 2026-07-09 15:45–16:35 ICT incident window so that AIPerf receives conservative, stable prompt-block identities instead of identities derived directly from the old server's cache-residency outcomes.

The resulting dataset must preserve the successful request population, explicit session/subagent structure, token lengths, output lengths, and recorded timing evidence while allowing each candidate server to produce its own cache eviction and refill behavior.

## Scope

The source corpus and converter are in:

`/Users/lap15120/greennode-code/aiperf-service-docs/benchmarks/20260709_glm5.2-ccu/simulation_20260907`

This design covers:

- correcting `build_weka_trace.py`;
- regenerating its 13 Weka session files;
- correcting the source dataset README;
- copying only the generated session files and a shortened README into InferenceX, as previously agreed;
- configuring the historical replay path to disable flattened-agent inference for this synthetic-hash corpus.

This design does not cover:

- the separate AIPerf hybrid scheduler required to combine absolute root-entry offsets with response-relative think time;
- changes to standard AgentX Weka behavior;
- removal of `--failed-request-threshold 0.05` from replay commands;
- synthetic reconstruction of hidden agents, cross-session shared prefixes, or pre-window cache contents.

## Confirmed Source Facts

The raw dump contains 463 CLI log records across 13 session IDs. Of those:

- 461 are successful upstream requests;
- two are HTTP 429 attempts with no usage data, followed by a successful retry of the same request-size payload;
- 430 successful requests have no explicit agent ID;
- 31 successful requests belong to three explicit `x-claude-code-agent-id` groups;
- 450 successful requests have input lengths that are not divisible by the 64-token Weka block size;
- 50 adjacent request pairs within the current explicit chains overlap in wall-clock time;
- 13 of 16 root/explicit-child chains begin the captured window with nonzero cached-token counts, proving that production cache state existed before the captured chain boundary.

The raw request and response payload fields are both fixed redaction markers. No raw text, token IDs, content hashes, or block hashes survive. Exact content-addressed reconstruction is therefore impossible.

The incident analysis records approximately 97.9 million HiCache tokens evicted during the 50-minute window. `prompt_cached_tokens` is therefore runtime residency evidence, not a stable content-identity signal.

## Root Causes in the Current Converter

### Cache residency is treated as content identity

The converter currently keeps only `ceil(prompt_cached_tokens / 64)` prior IDs and replaces the remaining prompt with new IDs. An original cache eviction is therefore encoded as a permanent content mutation. A candidate with a better cache is forced to repeat the old server's miss.

### Failed attempts become zero-token model turns

The two 429 records have no usage object. The converter defaults their token counts to zero and emits them as model calls. They are failure/retry outcomes of the old server, not logical successful workload turns.

### Partial-block hash counts violate the active reconstructor contract

The converter emits `ceil(input_length / 64)` IDs. The active Weka `ConversationReconstructor` consumes IDs for full blocks and synthesizes the partial tail separately, so non-aligned inputs require `floor(input_length / 64)` IDs.

Walking the existing generated corpus through the current reconstructor produces 435 input-length mismatches, with a maximum absolute error of 122 tokens. Truncating each hash list to `input_length // 64` removes every mismatch.

### Child timestamps use an ambiguous legacy representation

The converter records child-inner `t` relative to the child spawn. The current loader expects absolute root-trace timestamps and uses a heuristic for old relative fixtures. Three child turns are misclassified by that heuristic and land approximately 196 seconds early.

## Design Principles

1. A hash ID represents deterministic block content, not whether that block happened to be resident on the old server.
2. Recorded cached tokens are positive evidence of shared prefix, but a low cached count is not proof that the content changed.
3. Shape-only lineage inference must be conservative. On the canonical SemiAnalysis Weka corpus, top-level predecessor inference based on model, timing, and token-length growth still selected the true prefix only about 76% of the time in the tested full-corpus rule.
4. Only explicit `session_id` and `agent_id` topology is trusted. Synthetic hashes must not be fed back into flattened-agent detection as if they were recorded content hashes.
5. The benchmark starts from a controlled cold server after each candidate restart. It does not attempt to fabricate unknown pre-window cache state.

## Record Selection

`parse_record()` will read and retain `upstream_status` and `request.size` in addition to the existing fields.

- Only 2xx upstream records are converted.
- A successful record missing usage data is a hard conversion error; it must not silently become a zero-token request.
- The two known 429 attempts are counted and reported as excluded retries.
- The shared window origin remains the earliest raw request start. Removing the two later 429 records therefore does not shift the replay clock.

Expected output: 461 model requests in 13 files.

## Stable-Lineage Hash Synthesis

Hash synthesis runs independently for each explicit chain:

- the root chain for a session;
- one chain for each explicit `x-claude-code-agent-id` group.

The numeric ID allocator remains shared across all chains in one trace file so independently minted IDs cannot collide. Root and child chains do not share IDs because the masked data contains no trustworthy parent/child block mapping.

For request `i`:

```text
total_blocks = prompt_tokens // 64
observed_blocks = prompt_cached_tokens // 64

reuse_blocks = min(
    observed_blocks,
    len(previous_hash_ids),
    total_blocks,
)

if request i is the middle request of a strong eviction signature:
    reuse_blocks = min(len(previous_hash_ids), total_blocks)

hash_ids = (
    previous_hash_ids[:reuse_blocks]
    + allocate_fresh_ids(total_blocks - reuse_blocks)
)
```

The zero-to-63-token partial tail is intentionally not represented by a hash ID. AIPerf synthesizes it deterministically. Cache fidelity is therefore block-granular, with a maximum unmodeled prefix tail of 63 tokens per request.

### Strong eviction signature

For three consecutive successful records `A -> B -> C` in the same explicit chain, `B` is marked for stable-prefix repair only when all of the following hold:

- `A`, `B`, and `C` use the same model and streaming mode;
- `A` completes before `B` starts, and `B` completes before `C` starts;
- request byte size is nondecreasing across both edges;
- `B.input >= A.input + A.output`;
- `C.input >= B.input + B.output`;
- `B.cached_tokens < (A.input // 64) * 64`;
- `C.cached_tokens >= (B.input // 64) * 64`.

This pattern says that the logical context grew across both edges, the middle request saw a cache loss, and the following request saw the middle context resident again. The converter treats the low cached count at `B` as an eviction/refill outcome and keeps `A`'s stable block IDs.

The approved raw corpus contains 38 such repair edges. This count is an expected corpus invariant.

All other edges remain conservative: they reuse only the observed cached prefix that can be mapped to the immediately preceding explicit-chain state. The converter does not search arbitrarily far back in history or infer a new DAG from token lengths.

## Timing and Explicit Topology

Root and child normal requests will all use:

```text
t = (request_start_ms - window_start_ms) / 1000
```

Subagent marker `t` remains the first child request's absolute offset from the same window start. Child-inner timestamps are no longer spawn-relative.

`think_time` remains:

```text
max(0, current_start - previous_successful_request_end)
```

computed within each explicit chain after non-2xx filtering.

The converter retains the three explicit subagent entries and their 31 requests. It does not invent child entries for the 50 observed overlap edges because the masked data cannot distinguish title calls, hidden workers, retries, and context edits reliably.

The historical replay command must set:

```bash
AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false
AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1
```

The first setting prevents synthetic LCP patterns from creating false child conversations. The second threads the candidate server's own assistant output into the next turn so the candidate can reuse the KV blocks it actually produced.

## Cold-Start Policy

Each candidate server is restarted before replay. The dataset therefore begins cold even though 13 of 16 captured chains had nonzero cache hits at their first observed request.

No synthetic prewarm requests will be added. The captured data cannot identify the missing pre-window content, and fabricated prewarm traffic would introduce unmeasured load and content sharing. This trades exact production initial state for a controlled, comparable initial condition across candidate configurations.

## Validation

The converter's existing `self_check()` remains the single runnable regression check and will be strengthened rather than adding a new test framework. Any malformed successful record, corpus-count drift, or invariant failure aborts generation with a descriptive assertion or exception; partially validated output must not be presented as usable.

It must verify:

1. Schema and population
   - all 13 files validate as `WekaTrace`;
   - 461 normal/streaming requests are present;
   - exactly three subagent entries and 31 child requests are present;
   - every converted record came from a successful upstream response;
   - exactly two known non-2xx retries were excluded.
2. Block structure
   - `len(hash_ids) == input_length // 64` for every request;
   - every newly allocated ID is unique within its trace namespace;
   - reused IDs form a contiguous prefix;
   - exactly 38 strong-eviction middle edges reuse the full previous block prefix.
3. Reconstruction
   - walk every root and child chain through the current AIPerf `ConversationReconstructor` with deterministic stub callbacks;
   - after every turn, reconstructed token count equals recorded `input_length`;
   - expected mismatch count is zero.
4. Timing
   - timestamps are nonnegative and nondecreasing within each explicit chain;
   - every child-inner timestamp is at or after its subagent marker timestamp;
   - recorded `think_time` matches the successful-record predecessor formula.
5. Loader behavior
   - load with flattened-agent splitting disabled;
   - obtain 13 roots, three children, 461 turns, and no inferred flat-agent conversations.
6. Determinism
   - running the converter twice over the unchanged raw dump produces byte-identical session JSON.

## Documentation Changes

The source README and the shortened InferenceX dataset README will state:

- 461 successful requests, not 463 raw attempts;
- two 429 retries are intentionally excluded;
- hash IDs are conservative logical block identities, not an encoding of the old server's exact per-request cache-hit ratio;
- 38 high-confidence eviction signatures are repaired;
- the run starts cold;
- cross-session prefix sharing and parent/child prefix sharing are not reconstructed;
- hidden-agent topology and 50 within-session overlap edges remain unresolved;
- flattened-agent splitting must be disabled;
- the dataset does not by itself supply the required hybrid timing strategy.

The current claim that the converter reproduces each request's exact observed cache-hit ratio will be removed.

## Files Changed During Implementation

Source dataset repository:

- `benchmarks/20260709_glm5.2-ccu/simulation_20260907/build_weka_trace.py`
- `benchmarks/20260709_glm5.2-ccu/simulation_20260907/README.md`
- 13 regenerated files under `simulation_20260907/sessions/`

InferenceX integration, after regeneration:

- copied historical session files under the agreed agentic dataset location;
- the shortened dataset README;
- historical replay configuration needed to export `AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false`.

No standard AgentX dataset, standard AgentX scenario behavior, or AIPerf hash reconstruction code is changed by this dataset rebuild.

## Acceptance Criteria

The dataset rebuild is accepted when:

- all validation checks above pass;
- the generated corpus contains 461 successful requests and no zero-token 429 turns;
- all reconstructed turns match their recorded input lengths;
- the 38 approved eviction signatures retain stable block identities;
- no synthetic flat-agent topology is created at load time;
- the README states the remaining fidelity limits without claiming exact prompt or cache-state reconstruction.
