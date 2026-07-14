# Remote Historical Weka Fixed-Schedule Design

Date: 2026-07-14

## Status

Approved in conversation for written specification on 2026-07-14. Implementation remains gated on user review of this committed document.

## Objective

Run the validated GLM-5.2 corpus from the 2026-07-09 incident window against the existing GreenNode remote endpoint through InferenceX. The benchmark will use AIPerf's ordinary fixed-schedule trace replay, not the `inferencex-agentx-mvp` scenario.

This is an intentional timing approximation. It preserves recorded absolute request timestamps but does not add the missing response-relative hybrid scheduler.

## Approved Tradeoff

The historical configs will pass `--fixed-schedule` and will not pass `--use-think-time-only`. Under the current AIPerf implementation, fixed schedule prefers each turn's absolute timestamp, so adding the think-time flag would not change scheduling and would misleadingly imply hybrid behavior.

The approximation accepts the current SPAWN/JOIN timing behavior for the three explicit subagents. It does not modify AIPerf scheduling, child dispatch, or join release behavior.

Removing `--scenario inferencex-agentx-mvp` is required because that scenario uses trajectory warmup and recycling rather than replaying the 13 historical sessions once.

## Dataset

Use the InferenceX dataset copy shipped with this integration at:

`/workspace/benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/sessions`

The stable-lineage dataset design and validation remain authoritative in `docs/superpowers/specs/2026-07-14-historical-weka-stable-lineage-design.md`. This integration does not regenerate or redesign the corpus.

The loader must receive:

```bash
AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false
AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1
```

## AIPerf Command Contract

The historical path starts from `aiperf profile` without a scenario and adds:

```text
--url <remote-url>
--endpoint /v1/chat/completions
--endpoint-type chat
--streaming
--model z-ai/glm-5.2
--api-key <remote-secret>
--tokenizer zai-org/GLM-5.2
--tokenizer-trust-remote-code
--input-file <historical-sessions-directory>
--custom-dataset-type weka_trace
--fixed-schedule
--benchmark-duration <duration>
--extra-inputs ignore_eos:true
--random-seed 42
--slice-duration 1
--output-artifact-dir <result-directory>/trace_replay
```

The command must not add:

- `--scenario inferencex-agentx-mvp`;
- `--use-think-time-only`;
- `--concurrency`, because the trace timestamps define arrivals;
- `--use-server-token-count`, because the GreenNode streaming gateway did not emit the required usage event;
- `--failed-request-threshold`;
- trajectory-start, cache-bust, warmup, or unsafe-override flags inherited from the AgentX scenario.

The command does not pass `--benchmark-grace-period`; AIPerf's current default of 30 seconds applies.

## InferenceX Configuration

Add two separate remote-only matrix keys using the existing GreenNode endpoint, model, tokenizer, benchmark-client runner, and `GREENNODE_API_KEY` secret reference:

- Main: duration 3000 seconds, no context cap.
- Smoke: duration 60 seconds, `max-context-length: 100000`.

The smoke cap uses AIPerf's existing whole-session filtering. Prompts are not truncated.

The smallest explicit matrix contract is a fixed-schedule selector plus optional max-context-length propagation. Existing AgentX Weka configs retain their current scenario behavior; only the new historical configs select the scenario-free command branch.

## Failed-Request Threshold Cleanup

Remove `--failed-request-threshold 0.05` from both remote and local AIPerf replay commands, as previously approved. Remove the reporting helper and call sites that exist only to explain threshold-triggered aborts. Normal AIPerf exit status, logs, and result aggregation remain unchanged.

## Data Flow

1. The matrix generator validates the historical config and emits the local dataset path, fixed-schedule selector, duration, and optional context cap.
2. The GitHub workflow forwards those values and resolves `GREENNODE_API_KEY` into the existing remote API-key environment variable without exposing its value.
3. `_remote_replay.sh` validates the local path and exports the two Weka loader settings.
4. `build_replay_cmd()` selects the scenario-free fixed-schedule command branch.
5. AIPerf loads all 13 files once, filters whole sessions only when the smoke context cap applies, and writes artifacts through the existing result pipeline.

## Error Handling

Keep the existing remote endpoint reachability check, missing-dataset-path failure, subprocess timeout, exit-code reporting, and result aggregation. Do not add a replacement failure-rate policy.

Secrets must remain environment references and must be redacted from `benchmark_command.txt` and logs.

## Verification

Implementation verification must include:

1. Focused validation tests for the new fixed-schedule selector and optional context cap.
2. Matrix-generation tests asserting the main and smoke entries, including duration and context-cap differences.
3. Command-generation checks asserting the historical branch includes fixed schedule and `ignore_eos:true`, while excluding scenario, think-time, concurrency, server token count, grace-period override, and failed-request threshold flags.
4. Regression checks that existing AgentX configs still use their current scenario path.
5. `bash -n` for modified shell scripts and the existing `utils/matrix_logic` test suite.
6. A generated-config inspection before any dispatch.

The previously completed dataset self-check is sufficient; the converter and corpus do not need another redesign or regeneration for this integration.

An external smoke dispatch remains a separate, explicit user-authorized action. A successful local command build does not claim that the remote benchmark has run.

## Out of Scope

- Hybrid absolute-entry plus response-relative think-time scheduling.
- Changes to AIPerf fixed schedule, branch orchestration, or SPAWN/JOIN semantics.
- Dataset regeneration or new lineage inference.
- Server-side deployment changes or prewarming.
- Synthetic concurrency or request-rate shaping on top of historical timestamps.
