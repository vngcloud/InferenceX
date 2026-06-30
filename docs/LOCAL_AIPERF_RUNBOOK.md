# Local AIPerf agentic-replay runbook

How to replay the public SemiAnalysis Weka coding traces against a model you serve yourself, using AIPerf on your own machine. This is the local-dev version of the InferenceX CI sweep, so the command shape matches what `benchmark_lib.sh` dispatches on `h200-greennode_01`.

Fill in the placeholders (`<...>`) and run. Everything else is a validated default.

## 1. Serve the model first (required)

AIPerf is a client. It does not start a server. Bring up your model on an OpenAI-compatible endpoint before you run anything below, and keep it running for the whole benchmark.

Two things must be true:

- The endpoint answers `POST /v1/chat/completions` and supports streaming.
- The server exposes a Prometheus `/metrics` endpoint if you want server-side numbers (KV cache, queue depth, running requests). SGLang and vLLM both do by default.

Quick sanity check once the server is up:

```bash
curl -s <ENDPOINT_URL>/v1/models -H "Authorization: Bearer <API_KEY>"
```

If that returns your model id, AIPerf can reach it.

## 2. Turn on DCGM (recommended)

DCGM gives you per-GPU telemetry (utilization, power, memory) alongside the request metrics, which is what makes a run comparable to the CI artifacts. Run the same exporter image the H200 launcher uses:

```bash
docker run -d --rm --gpus all --network host --cap-add SYS_ADMIN \
  --name dcgm-exporter \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04
```

It listens on `localhost:9400/metrics`. Port 9400 must be free, so stop any host-level or k8s dcgm-exporter first. Tear it down with `docker rm -f dcgm-exporter` when you are done.

You can skip this. The run still produces request metrics. You just lose the GPU columns.

## 3. Install AIPerf

The agentic replay needs the `benchtool/agentx-weka` branch of `thangquang09/aiperf`. The PyPI build does not carry the `inferencex-agentx-mvp` scenario or the Weka public-dataset loader.

```bash
git clone -b benchtool/agentx-weka https://github.com/thangquang09/aiperf.git
cd aiperf
uv venv && source .venv/bin/activate
uv pip install -e .
aiperf --version
```

If you are working from this repo, the same branch is already vendored as the `utils/aiperf-mooncake` submodule, and the root workspace `.venv` already has it installed. In that case just `source .venv/bin/activate` from the workspace root and skip the clone.

## 4. Run the benchmark

Default workload: the public SemiAnalysis dataset, passed as the registered alias `semianalysis_cc_traces_weka`. This is the value the CI sweep uses. `--public-dataset` takes an alias name from the loader registry, not a HuggingFace repo path. This alias is a pinned pointer to `semianalysisai/cc-traces-weka-no-subagents-051826` (98 traces, 22.8k requests, no auth). AIPerf fetches it for you, so you do not point at a local file.

If you want to name a HuggingFace repo directly instead of using an alias, use the generic loader: `--public-dataset weka_hf --hf-weka-repo semianalysisai/cc-traces-weka-no-subagents-051826`. The alias path is preferred for reproducibility.

Set your values, then run:

```bash
ENDPOINT_URL="<ENDPOINT_URL>"     # e.g. http://localhost:8000
MODEL="<MODEL_NAME>"              # the id your server reports at /v1/models
API_KEY="<API_KEY>"              # any non-empty string if your server ignores auth
CONCURRENCY=8                     # participants actively coding
DURATION=900                      # seconds; use >=900 for a real run
OUTPUT_DIR="${OUTPUT_DIR:-aiperf-out}"   # where artifacts land

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --model "$MODEL" \
  --url "$ENDPOINT_URL" \
  --api-key "$API_KEY" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --concurrency "$CONCURRENCY" \
  --benchmark-duration "$DURATION" \
  --public-dataset semianalysis_cc_traces_weka \
  --num-dataset-entries 949 \
  --slice-duration 1.0 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --failed-request-threshold 0.05 \
  --use-server-token-count \
  --tokenizer-trust-remote-code \
  --server-metrics "$ENDPOINT_URL/metrics" \
  --gpu-telemetry http://localhost:9400/metrics \
  --output-artifact-dir "$OUTPUT_DIR"
```

Drop `--server-metrics` if your server has no `/metrics`. Drop `--gpu-telemetry` if you skipped DCGM in step 2.

### What the flags mean

| Flag | Why it is there |
|---|---|
| `--scenario inferencex-agentx-mvp` | The closed-loop agentic replay. Each client walks a full coding trajectory with the recorded think-time baked in. |
| `--public-dataset semianalysis_cc_traces_weka` | Registered alias for the public trace corpus. Not a HF repo path; the alias maps to a pinned repo. No `--input-file` needed. |
| `--concurrency` | Number of participants coding at once. This is the capacity knob. |
| `--benchmark-duration` | Run length in seconds. The clients loop over trajectories until the clock runs out. |
| `--use-server-token-count` | Trust the server's token counts instead of re-tokenizing client-side. |
| `--trajectory-start-*-ratio` | Stagger where each client enters its trajectory so they do not all replay in lockstep. |
| `--num-dataset-entries 949` | Trajectory entries the loader expands the corpus into. Separate from the trace count of the source repo. This is the CI default; leave it unless you are deliberately tuning. |

## 5. Smoke test before a real run

For a short shakeout (under 900s), add `--unsafe-override`. AIPerf refuses a short agentic run without it, because trajectories get truncated and the numbers are not representative.

```bash
# 90-second smoke, low concurrency
... --concurrency 4 --benchmark-duration 90 --unsafe-override ...
```

Use a smoke run to confirm the wiring, never to quote performance.

## 6. Read the results

Numbers land in `$OUTPUT_DIR` (default `aiperf-out/`). The source of truth is the raw artifact, not the console summary:

```bash
cat "$OUTPUT_DIR"/profile_export_aiperf.json | jq '{
  total_token_throughput: .total_token_throughput.avg,
  output_token_throughput: .output_token_throughput.avg,
  ttft_p99_ms: .time_to_first_token.p99,
  itl_avg_ms: .inter_token_latency.avg
}'
```

`tok/s/user = 1000 / itl_avg_ms`. The agentic-coding SLA is `tok/s/user >= 20`, which means mean ITL at or under 50 ms.

## Other dataset sources

Pick one source. Keep every other flag the same.

- Another pinned alias: swap the value of `--public-dataset` (e.g. `semianalysis_cc_traces_weka_with_subagents`). Run `aiperf profile --help` to see registered aliases, or check `semianalysis_*` entries in the loader's `plugins.yaml`.
- A HuggingFace repo by name: `--public-dataset weka_hf --hf-weka-repo semianalysisai/cc-traces-weka-<release>`. Use this for a repo that has no pinned alias yet.
- A local Weka-format JSONL: `--input-file <file.jsonl> --custom-dataset-type weka_trace` instead of `--public-dataset`.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `unknown scenario inferencex-agentx-mvp` | PyPI AIPerf, not the fork | Reinstall from `thangquang09/aiperf` `benchtool/agentx-weka` (step 3). |
| Hangs at warmup on huge trajectories | Wrong replay path | Use `--scenario inferencex-agentx-mvp`, not the plain weka path. |
| Run aborts under 900s | Short run without the override | Add `--unsafe-override` for smoke runs (step 5). |
| GPU columns empty | DCGM not reachable | Check `curl localhost:9400/metrics`, confirm port 9400 is free. |
| Connection refused | Server not up or wrong URL | Re-run the step 1 sanity check. |
