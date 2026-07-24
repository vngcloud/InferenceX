import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("preflight.py")
SPEC = importlib.util.spec_from_file_location("agentic_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_validate_config_accepts_gpu_resident_kv(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "runners.yaml").write_text(
        "labels:\n  cluster:b300-nv:\n    - b300-netperf_00\n"
    )
    config_path = configs / "nvidia-master.yaml"
    config_path.write_text(
        """smoke:
  image: image
  model: model
  model-prefix: glm5.2
  runner: cluster:b300-nv
  precision: fp4
  framework: sglang
  multinode: false
  scenarios:
    agentic-coding:
    - search-space:
      - { tp: 8, dp-attn: true, kv-offloading: none, conc-list: [40] }
"""
    )
    errors: list[str] = []

    _, spaces = PREFLIGHT.validate_config(
        tmp_path, config_path, "smoke", "b300-netperf_00", [40], errors
    )

    assert spaces[0]["kv-offloading"] == "none"
    assert errors == []


def test_validate_config_accepts_ordered_ccu_subset(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "runners.yaml").write_text(
        "labels:\n  cluster:h200:\n    - h200_01\n"
    )
    config_path = configs / "nvidia-master.yaml"
    config_path.write_text(
        """retry:
  runner: cluster:h200
  multinode: false
  scenarios:
    agentic-coding:
    - search-space:
      - { kv-offloading: dram, kv-offload-backend: { name: hicache }, conc-list: [1, 8, 32] }
"""
    )
    errors: list[str] = []

    PREFLIGHT.validate_config(
        tmp_path, config_path, "retry", "h200_01", [1], errors
    )

    assert errors == []


def test_validate_recipe_accepts_gpu_resident_kv(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.sh"
    recipe.write_text(
        """#!/usr/bin/env bash
require_agentic_kv_offload_none
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AIPERF_SERVER_METRICS_URLS=http://localhost:$PORT/metrics
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
MAX_RUNNING_REQUESTS=$((2 * CONC))
--enable-metrics
--enable-cache-report
run_agentic_replay_and_write_outputs
"""
    )
    errors: list[str] = []

    PREFLIGHT.validate_recipe(recipe, "full", None, {"none"}, errors)

    assert errors == []


def test_validate_recipe_accepts_hicache_ratio(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.sh"
    recipe.write_text(
        """#!/usr/bin/env bash
require_agentic_kv_offload_backend hicache
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
export AIPERF_SERVER_METRICS_URLS=http://localhost:$PORT/metrics
export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics
MAX_RUNNING_REQUESTS=$((2 * CONC))
--enable-metrics
--enable-cache-report
--enable-hierarchical-cache
--hicache-ratio 1.0
run_agentic_replay_and_write_outputs
"""
    )
    errors: list[str] = []

    PREFLIGHT.validate_recipe(recipe, "full", None, {"dram"}, errors)

    assert errors == []


def test_generator_command_targets_exact_config() -> None:
    command = PREFLIGHT.generator_command(
        Path("configs/nvidia-master.yaml"),
        "target-config",
        "b300-netperf_00",
        [8, 32, 48, 64],
    )

    assert command[2:] == [
        "test-config",
        "--config-files",
        "configs/nvidia-master.yaml",
        "--config-keys",
        "target-config",
        "--conc",
        "8",
        "32",
        "48",
        "64",
        "--runner-node-filter",
        "b300-netperf_00",
        "--scenario-type",
        "agentic-coding",
        "--no-evals",
    ]
