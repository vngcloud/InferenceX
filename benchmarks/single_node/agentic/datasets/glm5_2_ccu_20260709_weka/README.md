# GLM-5.2 historical Weka corpus (2026-07-09)

Thirteen Claude Code sessions from 2026-07-09 15:45–16:35 ICT, containing
461 successful requests. Two captured HTTP 429 retries are excluded. Three
explicit subagent groups contain 31 of the requests.

The masked vMonitor payload cannot recover prompt tokens or real content
hashes. `hash_ids` are conservative logical 64-token block identities:
observed adjacent prefixes are retained and 38 high-confidence
grow/evict/refill edges are repaired so old-server eviction is not encoded as
content mutation. The run starts cold.

Required environment:

```bash
export AIPERF_DATASET_WEKA_SPLIT_FLATTENED_AGENTS=false
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1
```

Use `sessions/` as `--input-file` with `--custom-dataset-type weka_trace`.

## Canonical fixed-schedule command

Status: not yet executed.

```bash
aiperf profile \
  --url <remote-url> \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model z-ai/glm-5.2 \
  --api-key <remote-secret> \
  --tokenizer zai-org/GLM-5.2 \
  --tokenizer-trust-remote-code \
  --input-file /workspace/benchmarks/single_node/agentic/datasets/glm5_2_ccu_20260709_weka/sessions \
  --custom-dataset-type weka_trace \
  --fixed-schedule \
  --benchmark-duration <duration> \
  --extra-inputs ignore_eos:true \
  --random-seed 42 \
  --slice-duration 1 \
  --output-artifact-dir <result-directory>/trace_replay
```

Only root conversations are initially scheduled; Weka child conversations
spawn through branch metadata. The command uses no scenario, think-time flag,
grace-period override, concurrency flag, or server token count. The main config
runs for 3000 seconds without a context cap. The smoke config runs for 60
seconds and filters whole sessions above `max-context-length: 100000`.

Limits: pre-window cache, prompt text, cross-session/root-child prefix sharing,
hidden-agent topology, and 50 overlapping within-session edges are not
reconstructed. AIPerf also still lacks the hybrid scheduler needed for exact
historical timing (absolute root-entry offsets, then response-relative recorded
think time with SPAWN/JOIN and no recycle), so the dataset alone is not a
faithful historical replay command.
