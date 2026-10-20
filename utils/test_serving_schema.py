import textwrap

import pytest

from infx.serving.schema import REPO_ROOT, ServingError, available_scenarios, load_serving

SWEEP_SCENARIO = "sweep-isl8k-osl512-cache50"

GOOD = textwrap.dedent("""\
    name: qwen38-bf16-tp1
    runner: h200-greennode-slurm-1x_01
    image: vllm/vllm-openai:v0.29.0
    model: Qwen/Qwen3.8-27B
    gpus: 1
    serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144
    scenario: agentx-cc-062126-smoke
    conc: [4, 8]
""")


def with_line(old: str, new: str) -> str:
    assert old in GOOD, old
    return GOOD.replace(old, new)


def errors(text: str, repo=REPO_ROOT) -> str:
    with pytest.raises(ServingError) as info:
        load_serving(text, repo)
    return str(info.value)


def test_good_yaml_loads_with_defaults():
    spec = load_serving(GOOD)
    assert spec.name == "qwen38-bf16-tp1"
    assert spec.precision == "bf16"
    assert spec.env == {}
    assert spec.conc == [4, 8]


def test_available_scenarios_lists_both():
    assert {"agentx-cc-062126", "agentx-cc-062126-smoke"} <= set(available_scenarios())


ROUTER = textwrap.dedent("""\
    router:
      image: vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1
      serve: python3 -m dynamo.frontend --http-port=8888 --router-mode=kv
""")


def dynamo(text: str = GOOD) -> str:
    return (text.replace("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_02")
                .replace("gpus: 1", "gpus: 8")
                .replace("serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144",
                         "serve: python3 -m dynamo.sglang --model-path=Qwen/Qwen3.8-27B --tp=8")
              + ROUTER)


def test_router_block_loads():
    spec = load_serving(dynamo())
    assert spec.router.image.endswith(":1.5.0-kvoverlap1")
    assert spec.router.env == {}


def test_router_unknown_key_rejected():
    assert "router.port: unknown key" in errors(dynamo() + "  port: 1\n")


def test_router_needs_hardware_hcm_runner():
    text = dynamo().replace("runner: hardware-hcm-8x_02", "runner: h200-greennode-slurm-1x_01").replace("gpus: 8", "gpus: 1")
    assert "router: needs a hardware-hcm-8x_* runner (only launch_hardware-hcm.sh starts the router)" in errors(text)


@pytest.mark.parametrize("field, text", [
    ("serve", GOOD.replace("--max-model-len 262144", "--max-model-len 262144 --numa-node 0 0 1 1")),
    ("router.serve", dynamo().replace("--router-mode=kv", "--router-mode=kv --numa-node 0")),
])
def test_numa_node_rejected(field, text):
    assert f"{field}: drop --numa-node (benchmark convention)" in errors(text)


def test_router_serve_gets_v1_command_checks():
    text = dynamo().replace("--router-mode=kv", "--router-mode=kv --model-path=/models/x --x=$Y")
    out = errors(text)
    assert "router.serve: local path containing '/models/'" in out
    assert "router.serve: shell variables ($...) are not expanded" in out


def test_router_env_rules():
    text = dynamo() + "  env: {DYN_FILE_KV: /x, 'bad-name': '1'}\n"
    out = errors(text)
    assert "router.env.DYN_FILE_KV: reserved, the benchmark recipe sets it; remove it from env" in out
    assert "router.env.bad-name: not a valid environment variable name" in out


def test_router_image_pattern():
    assert "router.image: only letters, digits and ._/:@- are allowed" in errors(dynamo().replace("1.5.0-kvoverlap1", "x;rm"))


GENERIC_ROUTER_YAML = textwrap.dedent("""\
    router:
      image: sglang-router-patched:v0.5.18-slim-noargcanon
      metrics_port: 18888
      serve: python3 -m sglang_router.launch_router --port={{router_port}} --worker-urls={{engine_url}}
""")


def generic(serve_line: str | None = None) -> str:
    text = dynamo().removesuffix(ROUTER) + GENERIC_ROUTER_YAML
    return text.replace("--worker-urls={{engine_url}}", serve_line) if serve_line else text


def test_generic_router_block_loads():
    spec = load_serving(generic())
    assert spec.router.metrics_port == 18888
    assert "{{engine_url}}" in spec.router.serve


def test_router_unknown_placeholder_rejected():
    out = errors(generic("--worker-urls={{foo}}"))
    assert "router.serve: unknown placeholder {{foo}}; allowed: router_port, engine_url, engine_port" in out
    out = errors(generic() + "  env:\n    X: '{{bar}}'\n")
    assert "router.env.X: unknown placeholder {{bar}}" in out


def test_router_inline_jinja_is_not_a_placeholder():
    load_serving(generic("--chat-template '{{ messages }}'"))


def test_router_metrics_port_range():
    assert "router.metrics_port:" in errors(generic().replace("18888", "70000"))


def test_engine_port_reserved_env():
    out = errors(generic() + "  env:\n    ENGINE_PORT: '1'\n")
    assert "router.env.ENGINE_PORT: reserved, the benchmark recipe sets it; remove it from env" in out


def test_duration_rejected_with_hint():
    assert "the scenario fixes the duration" in errors(GOOD + "duration: 90\n")


def test_unknown_key_names_allowed_keys():
    message = errors(GOOD + "imagee: x\n")
    assert message.startswith("imagee: unknown key")
    assert "image" in message


def test_missing_image_is_reported():
    assert "image: Field required" in errors(with_line("image: vllm/vllm-openai:v0.29.0\n", ""))


def test_name_with_underscore_rejected():
    assert errors(with_line("name: qwen38-bf16-tp1", "name: qwen38_bf16")).startswith("name:")


def test_local_model_path_rejected():
    assert "model: use the HF repo id" in errors(with_line("model: Qwen/Qwen3.8-27B", "model: /models/Qwen3.8-27B"))


def test_local_path_in_serve_rejected():
    text = with_line("serve: vllm serve Qwen/Qwen3.8-27B", "serve: vllm serve /models/PhalaCloud/GLM")
    assert "serve: local path containing '/models/'" in errors(text)


def test_shell_variable_in_serve_rejected():
    assert "serve: shell variables" in errors(with_line("--max-model-len 262144", "--port $PORT"))


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/runners.yaml").write_text(textwrap.dedent("""\
        labels:
          cluster:h200-greennode-slurm: [h200-greennode-slurm-1x_01, h200-greennode-slurm-1x_02]
          cluster:hardware-hcm: [hardware-hcm-8x_01, hardware-hcm-4x_01]
        hardware:
          cluster:h200-greennode-slurm: {gpus-per-node: 1}
          cluster:hardware-hcm: {gpus-per-node: 8}
    """))
    (tmp_path / "scenarios").mkdir()
    (tmp_path / "scenarios/agentx-cc-062126-smoke.yaml").write_text("{}\n")
    return tmp_path


@pytest.mark.parametrize("runner", ["hardware-hcm-4x_01", "nope"])
def test_runner_outside_serving_pool_rejected(repo, runner):
    text = with_line("runner: h200-greennode-slurm-1x_01", f"runner: {runner}").replace("gpus: 1", "gpus: 8")
    with pytest.raises(ServingError) as info:
        load_serving(text, repo)
    assert str(info.value) == (
        f"runner: {runner} cannot run a serving YAML; use one of: "
        "h200-greennode-slurm-1x_01, h200-greennode-slurm-1x_02, hardware-hcm-8x_01"
    )


def test_hardware_hcm_8x_accepts_8_gpus(repo):
    text = with_line("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_01").replace("gpus: 1", "gpus: 8")
    assert load_serving(text, repo).gpus == 8


def test_too_many_gpus_for_runner():
    assert "gpus: 8 > 1 GPUs per node" in errors(with_line("gpus: 1", "gpus: 8"))


def test_missing_scenario_lists_available():
    message = errors(with_line("scenario: agentx-cc-062126-smoke", "scenario: nope"))
    assert "scenario: no scenarios/nope.yaml; available:" in message
    assert "agentx-cc-062126-smoke" in message


def test_scenario_suffix_and_scalar_conc():
    text = with_line("scenario: agentx-cc-062126-smoke", "scenario: scenarios/agentx-cc-062126-smoke.yaml")
    text = text.replace("conc: [4, 8]", "conc: 8")
    spec = load_serving(text)
    assert spec.scenario == "agentx-cc-062126-smoke"
    assert spec.conc == [8]


def test_duplicate_conc_rejected():
    assert "conc: duplicate values" in errors(with_line("conc: [4, 8]", "conc: [8, 8]"))


def test_env_numbers_become_strings():
    spec = load_serving(GOOD + "env:\n  SGLANG_DP_USE_GATHERV: 1\n  NCCL_TIMEOUT: 1.5\n  NCCL_P2P_LEVEL: NVL\n")
    assert spec.env == {"SGLANG_DP_USE_GATHERV": "1", "NCCL_TIMEOUT": "1.5", "NCCL_P2P_LEVEL": "NVL"}


def test_env_bool_rejected():
    assert "quote boolean values" in errors(GOOD + "env:\n  HF_HUB_OFFLINE: true\n")


def test_env_bad_name_rejected():
    assert "env.BAD-NAME: not a valid environment variable name" in errors(GOOD + "env:\n  BAD-NAME: x\n")


@pytest.mark.parametrize("old, new, message", [
    ("image: vllm/vllm-openai:v0.29.0", "image: 'a;$(id)'", "image: only letters, digits and ._/:@- are allowed"),
    ("model: Qwen/Qwen3.8-27B", "model: 'm m'", "model: must be an HF repo id <org>/<name>"),
])
def test_image_and_model_patterns(old, new, message):
    assert errors(with_line(old, new)).startswith(message)


def test_env_reserved_name_rejected():
    message = errors(GOOD + "env:\n  PORT: '9000'\n  HF_HUB_CACHE: /x\n  NCCL_P2P_LEVEL: NVL\n")
    assert message.splitlines() == [
        "env.PORT: reserved, the benchmark recipe sets it; remove it from env",
        "env.HF_HUB_CACHE: reserved, the benchmark recipe sets it; remove it from env",
    ]


def test_not_a_mapping():
    assert errors("- a\n- b\n") == "serving: the top level must be a mapping of keys"


def test_multiple_problems_one_line_each():
    text = with_line("model: Qwen/Qwen3.8-27B", "model: /models/x").replace("gpus: 1", "gpus: 8")
    lines = errors(text).splitlines()
    assert len(lines) == 2


STORAGE = textwrap.dedent("""\
    storage:
      image: lmsysorg/sglang:v0.5.21
      serve: mooncake_master --port={{storage_master_port}} --http_metadata_server_port={{storage_meta_port}}
""")


def with_storage(text: str = GOOD) -> str:
    """GOOD + storage block, pinned to a hardware-hcm runner."""
    return (text.replace("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_01")
                .replace("gpus: 1", "gpus: 8")) + STORAGE


def test_storage_block_loads():
    spec = load_serving(with_storage())
    assert spec.storage is not None
    assert spec.storage.image == "lmsysorg/sglang:v0.5.21"
    assert "{{storage_master_port}}" in spec.storage.serve


def test_storage_unknown_key_rejected():
    text = with_storage().replace("image: lmsysorg/sglang:v0.5.21", "image: lmsysorg/sglang:v0.5.21\n  bogus: 1")
    assert "storage.bogus: unknown key" in errors(text)


def test_storage_needs_hardware_hcm_runner():
    text = GOOD + STORAGE  # h200-greennode runner, not hardware-hcm
    assert "storage: needs a hardware-hcm-8x_* runner" in errors(text)


def test_storage_unknown_placeholder_rejected():
    text = with_storage().replace("{{storage_master_port}}", "{{router_port}}")
    assert "storage.serve: unknown placeholder {{router_port}}" in errors(text)


def test_env_may_reference_storage_ports_only_with_storage_block():
    # with storage block: {{storage_master_port}} is fine in top-level env
    text = with_storage().replace(
        "gpus: 8",
        "gpus: 8\nenv:\n  MOONCAKE_MASTER: '127.0.0.1:{{storage_master_port}}'",
    )
    spec = load_serving(text)
    assert spec.env["MOONCAKE_MASTER"] == "127.0.0.1:{{storage_master_port}}"


def test_env_placeholder_rejected_without_storage_block():
    text = GOOD.replace(
        "gpus: 1",
        "gpus: 1\nenv:\n  MOONCAKE_MASTER: '127.0.0.1:{{storage_master_port}}'",
    )
    assert "env.MOONCAKE_MASTER: unknown placeholder {{storage_master_port}}" in errors(text)


REPLICAS_YAML = textwrap.dedent("""\
    name: gemma4-2rep
    runner: hardware-hcm-8x_01
    image: vllm/vllm-openai:v0.25.1
    model: RedHatAI/gemma-4-31B-it-FP8-block
    gpus: 2
    replicas: 2
    serve: vllm serve RedHatAI/gemma-4-31B-it-FP8-block --tensor-parallel-size 1
    router:
      image: vllm/vllm-openai:v0.25.1
      serve: vllm-router --port {{router_port}} --worker-urls {{engine_urls}} --policy consistent_hash
    scenario: agentx-cc-062126-smoke
    conc: [8]
""")


def test_replicas_default_one_and_router_yaml_loads():
    assert load_serving(GOOD).replicas == 1
    assert load_serving(REPLICAS_YAML).replicas == 2


@pytest.mark.parametrize("text, error", [
    (REPLICAS_YAML.replace("gpus: 2", "gpus: 3"), "replicas: gpus 3 is not a multiple of replicas 2"),
    (REPLICAS_YAML.split("router:")[0] + "scenario: agentx-cc-062126-smoke\nconc: [8]\n",
     "replicas: more than 1 replica needs a router: block"),
    (dynamo().replace("gpus: 8", "gpus: 8\nreplicas: 2"), "replicas: the Dynamo frontend discovers its engines itself"),
    (REPLICAS_YAML.replace("replicas: 2", "replicas: 0"), "replicas: Input should be greater than or equal to 1"),
    (REPLICAS_YAML.replace("gpus: 2\n", "gpus: 2\nenv: {ENGINE_PORTS: x}\n"), "env.ENGINE_PORTS: reserved"),
], ids=["indivisible", "no-router", "dynamo", "zero", "reserved-env"])
def test_replicas_errors(text, error):
    assert error in errors(text)


SCALE_UP_YAML = REPLICAS_YAML.replace("conc: [8]\n", "conc: [8]\nscale_up:\n  replicas: 1\n  after: 60\n")


def test_scale_up_loads():
    spec = load_serving(SCALE_UP_YAML)
    assert (spec.scale_up.replicas, spec.scale_up.after) == (1, 60)


@pytest.mark.parametrize("text, error", [
    (SCALE_UP_YAML.replace("  replicas: 1\n", "  replicas: 2\n"), "scale_up.replicas: 2 must be below replicas 2"),
    (SCALE_UP_YAML.replace("after: 60", "after: 180"), "scale_up.after: 180s is not inside scenario agentx-cc-062126-smoke's 180s"),
    (SCALE_UP_YAML.replace("after: 60", "after: 0"), "scale_up.after: Input should be greater than 0"),
    (SCALE_UP_YAML.replace("after: 60", "after: 60\n  at: 1"), "scale_up.at: unknown key (allowed: replicas, after)"),
    (REPLICAS_YAML.split("router:")[0] + "scenario: agentx-cc-062126-smoke\nconc: [8]\nscale_up: {replicas: 1, after: 60}\n",
     "scale_up: needs a router: block with an /add_worker endpoint"),
], ids=["all-late", "after-past-end", "zero-after", "unknown-key", "no-router"])
def test_scale_up_errors(text, error):
    assert error in errors(text)


def test_scale_up_needs_add_worker_router():
    text = dynamo().replace("gpus: 8", "gpus: 8\nscale_up: {replicas: 1, after: 60}")
    assert "scale_up: needs a router: block with an /add_worker endpoint" in errors(text)


def test_env_rejects_router_placeholders_even_with_storage():
    text = with_storage().replace(
        "gpus: 8",
        "gpus: 8\nenv:\n  X: '{{engine_port}}'",
    )
    assert "env.X: unknown placeholder {{engine_port}}" in errors(text)


@pytest.fixture
def sweep_repo(repo):
    (repo / f"scenarios/{SWEEP_SCENARIO}.yaml").write_text((REPO_ROOT / f"scenarios/{SWEEP_SCENARIO}.yaml").read_text())
    return repo


def test_sweep_scenario_requires_conc_one(sweep_repo):
    text = with_line("scenario: agentx-cc-062126-smoke", f"scenario: {SWEEP_SCENARIO}").replace("conc: [4, 8]", "conc: [8]")
    message = errors(text, sweep_repo)
    assert message.startswith("conc:")
    assert "sweep" in message
    assert load_serving(text.replace("conc: [8]", "conc: 1"), sweep_repo).conc == [1]


def test_non_sweep_scenario_keeps_conc_rule_unchanged(sweep_repo):
    text = with_line("conc: [4, 8]", "conc: [8, 16]")
    spec = load_serving(text, sweep_repo)
    assert spec.conc == [8, 16]
