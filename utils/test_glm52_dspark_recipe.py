import os
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = (
    REPO_ROOT
    / "benchmarks"
    / "single_node"
    / "agentic"
    / "glm5.2dspark_fp4_h200_sglang.sh"
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_dspark_recipe_loads_bf16_draft_without_target_quantization(tmp_path: Path):
    recipe_dir = tmp_path / "benchmarks" / "single_node" / "agentic"
    recipe_dir.mkdir(parents=True)
    recipe = recipe_dir / RECIPE.name
    shutil.copy2(RECIPE, recipe)

    (tmp_path / "benchmarks" / "benchmark_lib.sh").write_text(
        """
check_env_vars() { :; }
require_agentic_kv_offload_backend() { :; }
require_agentic_kv_offload_none() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
wait_for_server_ready() { :; }
build_replay_cmd() { :; }
run_agentic_replay_and_write_outputs() { :; }
""".strip()
        + "\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "python3", "#!/usr/bin/env bash\nexit 0\n")

    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}\n")
    (draft_dir / "model.safetensors").touch()
    result_dir = tmp_path / "results"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MODEL": "zai-org/GLM-5.2-FP8",
            "TP": "8",
            "EP_SIZE": "8",
            "CONC": "1",
            "KV_OFFLOADING": "none",
            "TOTAL_CPU_DRAM_GB": "1024",
            "RESULT_DIR": str(result_dir),
            "DURATION": "1",
            "DP_ATTENTION": "false",
            "SPEC_DECODING": "mtp",
            "PORT": "18889",
            "DRAFT_MODEL_PATH": str(draft_dir),
        }
    )

    completed = subprocess.run(
        ["bash", str(recipe)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split((result_dir / "sglang_command.txt").read_text())
    assert command[command.index("--quantization") + 1] == "w4afp8"
    assert command[command.index("--speculative-draft-model-path") + 1] == str(
        draft_dir
    )
    assert (
        command[command.index("--speculative-draft-model-quantization") + 1]
        == "unquant"
    )


def test_dspark_dp_attention_matches_reference_topology(tmp_path: Path):
    recipe_dir = tmp_path / "benchmarks" / "single_node" / "agentic"
    recipe_dir.mkdir(parents=True)
    recipe = recipe_dir / RECIPE.name
    shutil.copy2(RECIPE, recipe)

    (tmp_path / "benchmarks" / "benchmark_lib.sh").write_text(
        """
check_env_vars() { :; }
require_agentic_kv_offload_backend() { :; }
require_agentic_kv_offload_none() { :; }
resolve_trace_source() { :; }
install_agentic_deps() { :; }
wait_for_server_ready() { :; }
build_replay_cmd() { :; }
run_agentic_replay_and_write_outputs() { :; }
""".strip()
        + "\n"
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "python3", "#!/usr/bin/env bash\nexit 0\n")

    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text("{}\n")
    (draft_dir / "model.safetensors").touch()
    result_dir = tmp_path / "results"

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MODEL": "zai-org/GLM-5.2-FP8",
            "TP": "8",
            "EP_SIZE": "1",
            "CONC": "24",
            "KV_OFFLOADING": "dram",
            "HICACHE_RATIO": "2",
            "TOTAL_CPU_DRAM_GB": "1024",
            "RESULT_DIR": str(result_dir),
            "DURATION": "1",
            "DP_ATTENTION": "true",
            "SPEC_DECODING": "mtp",
            "PORT": "18889",
            "DRAFT_MODEL_PATH": str(draft_dir),
        }
    )

    completed = subprocess.run(
        ["bash", str(recipe)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    command = shlex.split((result_dir / "sglang_command.txt").read_text())
    assert command[command.index("--tp") + 1] == "8"
    assert command[command.index("--dp") + 1] == "8"
    assert "--ep" not in command
    assert "--enable-dp-attention" in command
    assert "--enable-dp-attention-local-control-broadcast" not in command
    assert "--enable-dp-lm-head" in command
    assert "--numa-node" not in command
    assert "--disable-shared-experts-fusion" in command
    assert command[command.index("--mem-fraction-static") + 1] == "0.75"
    assert command[command.index("--max-running-requests") + 1] == "48"
    assert "--cuda-graph-max-bs" not in command
    assert command[command.index("--schedule-policy") + 1] == "lpm"
