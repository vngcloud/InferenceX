---
name: agentic-replay-run
description: Run the supported InferenceX Weka coding benchmarks through agentic-replay/AIPerf.
---

# Agentic-replay Weka

Supported path only: `agentic-replay` + `custom-dataset-type: weka_trace` + `benchmark-client: [aiperf]` on `h200-greennode_01`.

Known-good configs:

- `minimaxm2.5-weka-fp8-h200-greennode-sglang-smoke`
- `glm5.2-weka-fp8-h200-greennode-sglang-smoke`

Dataset source:

- Omit `input-file` and `public-dataset` for the default public SemiAnalysis Weka dataset: `semianalysis_cc_traces_weka_with_subagents_060826`.
- Set `public-dataset` for another public Weka dataset.
- Set `input-file` for internal MiniMax Weka-v4/local files.
- Do not use `no-fixed-schedule`, warmup, request-count, think-time, or strip-delay fields.

Required local pieces:

- `utils/aiperf-mooncake` submodule
- `benchmarks/single_node/minimaxm2.5-weka_fp8_h200_sglang.sh`
- `benchmarks/single_node/glm5.2-ep8-deepep_fp8_h200_sglang.sh`
- `runners/launch_h200-greennode.sh`

Validate before dispatch:

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/configs/nvidia-master.yaml'))"
bash -n benchmarks/single_node/minimaxm2.5-weka_fp8_h200_sglang.sh
bash -n benchmarks/single_node/glm5.2-ep8-deepep_fp8_h200_sglang.sh
bash -n runners/launch_h200-greennode.sh
uv run python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys minimaxm2.5-weka-fp8-h200-greennode-sglang-smoke
uv run python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys glm5.2-weka-fp8-h200-greennode-sglang-smoke
```

Successful reference runs:

- MiniMax-M2.5: https://github.com/vngcloud/InferenceX/actions/runs/28376099323
- GLM-5.2-FP8: https://github.com/vngcloud/InferenceX/actions/runs/28279462408
