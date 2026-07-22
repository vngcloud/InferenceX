# InferenceX agentic project map

## Files

- `configs/nvidia-master.yaml`, `configs/amd-master.yaml`: model metadata, runner pool, scenario, KV offload, and CCU list.
- `configs/runners.yaml`: exact runner nodes under each label. Agentic configs should use a `cluster:*` pool.
- `benchmarks/single_node/agentic/*.sh`: serving command, dataset override, telemetry URLs, model readiness, and AIPerf invocation.
- `runners/launch_<runner-prefix>.sh`: Docker/container mounts, DCGM exporter, and forwarded environment variables.
- `benchmarks/benchmark_lib.sh`: dataset mapping, replay command, warmup, and HiCache validation.
- `.github/workflows/e2e-tests.yml`: manual dispatch, duration override, and optional agentic ingest.

## Dataset mapping

| User choice | Loader override | Hugging Face dataset |
|---|---|---|
| full | `semianalysis_cc_traces_weka_062126` | `semianalysisai/cc-traces-weka-062126` |
| cap 256k | `semianalysis_cc_traces_weka_062126_256k` | `semianalysisai/cc-traces-weka-062126-256k` |

The 256k dataset drops individual requests whose recorded `input + output` exceeds 256,000 tokens. It does not truncate a request at 256k or automatically drop the whole session unless no requests survive.

## Mandatory telemetry

The recipe must export both:

```bash
export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
```

The server command must include `--enable-metrics` and `--enable-cache-report`. HiCache recipes must also include `--enable-hierarchical-cache` and a deliberate `--hicache-size`.

The runner launcher must start `dcgm-exporter`, use host networking, clean it up with a trap, and forward `KV_OFFLOADING`, `KV_OFFLOAD_BACKEND`, and `KV_OFFLOAD_BACKEND_METADATA` into the benchmark container.

## Dispatch pattern

Generate a narrow matrix command using all applicable filters:

```text
full-sweep --config-files <master.yaml> --model-prefix <prefix> --precision <precision> --framework <framework> --runner-type <cluster:pool> --runner-node-filter <exact-node> --scenario-type agentic-coding --min-conc <min> --max-conc <max> --single-node --no-evals
```

Dispatch from the pushed project branch and pass the same branch as the workflow input. Set `duration-override` explicitly, even for the normal 3600-second duration. Set `skip-agentic-ingest=true` by default.

## Preview checklist

Before requesting approval, show:

1. Exact model and container-visible model path.
2. Image, framework, precision, topology, cache, parser, scheduler, and context flags.
3. Dataset, duration, exact runner, and CCUs.
4. Derived `max-running-requests` and CUDA graph batch size per CCU when the recipe scales them.
5. Server metrics/DCGM status and local model preflight.
6. Generated matrix and exact dispatch command.
7. Ingest disabled.
