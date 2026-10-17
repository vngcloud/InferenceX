import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from infx.serving.schema import ServingError, available_scenarios
from infx.serving.to_matrix import build, scenario_facts
from infx.workflows.benchmark_schema import validate_matrix

REPO = Path(__file__).resolve().parents[1]
YAML = textwrap.dedent("""\
    name: qwen38-bf16-tp1
    runner: h200-greennode-slurm-1x_01
    image: vllm/vllm-openai:v0.29.0
    model: Qwen/Qwen3.8-27B
    gpus: 1
    env: {VLLM_X: "1"}
    serve: vllm serve Qwen/Qwen3.8-27B --port 8000 --served-model-name q38
    scenario: agentx-cc-062126-smoke
    conc: [4, 8]
""")
ROW_KEYS = {
    "image", "model", "model-prefix", "precision", "framework", "runner", "tp", "pp", "dcp-size",
    "pcp-size", "ep", "dp-attn", "spec-decoding", "conc", "kv-offloading", "total-cpu-dram-gb",
    "duration", "exp-name", "scenario-type",
}


def test_scenario_facts():
    assert scenario_facts("agentx-cc-062126") == (3600, "semianalysis_cc_traces_weka_062126")
    assert scenario_facts("agentx-cc-062126-smoke") == (180, "semianalysis_cc_traces_weka_062126")


def test_one_row_per_conc_with_generator_shape():
    result = build(YAML)
    rows = result["matrix"]
    assert [row["conc"] for row in rows] == [4, 8]
    for row in rows:
        assert set(row) == ROW_KEYS
    assert rows[1] == {
        "image": "vllm/vllm-openai:v0.29.0", "model": "Qwen/Qwen3.8-27B", "model-prefix": "qwen38-bf16-tp1",
        "precision": "bf16", "framework": "vllm", "runner": "h200-greennode-slurm-1x_01", "tp": 1, "pp": 1,
        "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False, "spec-decoding": "none", "conc": 8,
        "kv-offloading": "none", "total-cpu-dram-gb": 0, "duration": 180, "exp-name": "qwen38-bf16-tp1_c8",
        "scenario-type": "agentic-coding",
    }


def test_rows_pass_the_workflow_schema():
    validate_matrix(build(YAML)["matrix"])  # raises ValueError on any mismatch


def test_serving_object():
    serving = build(YAML)["serving"]
    assert serving == {
        "serve-cmd": "vllm serve Qwen/Qwen3.8-27B --served-model-name q38",
        "serve-env": "VLLM_X=1",
        "served-model-name": "q38",
        "scenario": "agentx-cc-062126-smoke",
        "loader": "semianalysis_cc_traces_weka_062126",
        "yaml": YAML,
        "router-image": "",
        "router-cmd": "",
        "router-env": "",
        "router-profile": "",
        "router-metrics-port": "",
        "replicas": "1",
        "storage-image": "",
        "storage-cmd": "",
        "storage-env": "",
    }


GOLDEN = REPO / "utils/testdata/serving_dynamo_build_golden.json"
NEW_SERVING_FIELDS = (
    "router-profile", "router-metrics-port", "replicas",
    "storage-image", "storage-cmd", "storage-env",
)


@pytest.mark.parametrize("name", sorted(json.loads(GOLDEN.read_text())))
def test_dynamo_yamls_render_as_before(name):
    result = build((REPO / "serving" / f"{name}.yaml").read_text())
    assert result["serving"]["router-profile"] == "dynamo"
    assert result["serving"]["router-metrics-port"] == ""
    for key in NEW_SERVING_FIELDS:
        del result["serving"][key]
    assert result == json.loads(GOLDEN.read_text())[name]


SGLANG_ROUTER = textwrap.dedent("""\
    name: glm53-sglr
    runner: hardware-hcm-8x_02
    image: lmsysorg/sglang:v0.5.18
    model: PhalaCloud/GLM-5.3-W4AFP8
    gpus: 8
    precision: w4afp8
    serve: python3 -m sglang.launch_server --model-path PhalaCloud/GLM-5.3-W4AFP8 --served-model-name zai-org/GLM-5.3 --tp 8 --dp 4 --ep 8 --enable-metrics
    router:
      image: sglang-router-patched:v0.5.18-slim-noargcanon
      metrics_port: 18888
      serve: python3 -m sglang_router.launch_router --port={{router_port}} --prometheus-port=18888 --worker-urls={{engine_url}}
    scenario: agentx-cc-062126
    conc: [8, 24, 32, 48]
""")


def test_generic_router_serving_object_and_rows():
    result = build(SGLANG_ROUTER)
    serving = result["serving"]
    assert serving["router-profile"] == "generic"
    assert serving["router-metrics-port"] == "18888"
    assert "{{engine_url}}" in serving["router-cmd"] and "{{router_port}}" in serving["router-cmd"]
    assert "--port" not in serving["serve-cmd"].split()
    validate_matrix(result["matrix"])
    assert [row["conc"] for row in result["matrix"]] == [8, 24, 32, 48]
    assert {row["framework"] for row in result["matrix"]} == {"sglang"}


DYNAMO = textwrap.dedent("""\
    name: glm53-dyn
    runner: hardware-hcm-8x_02
    image: vcr.vngcloud.vn/108942-public69/dynamo-stack/sglang-runtime:1.5.0
    model: PhalaCloud/GLM-5.3-W4AFP8
    gpus: 8
    precision: w4afp8
    serve: python3 -m dynamo.sglang --model-path=PhalaCloud/GLM-5.3-W4AFP8 --served-model-name=zai-org/GLM-5.3 --tp=8 --enable-metrics
    router:
      image: vcr.vngcloud.vn/60108-backend-worker/dynamo-stack/sglang-runtime:1.5.0-kvoverlap1
      serve: python3 -m dynamo.frontend --http-port=8888 --router-mode=kv
    scenario: agentx-cc-062126-smoke
    conc: [8]
""")


def test_dynamo_serving_object_and_rows():
    result = build(DYNAMO)
    serving = result["serving"]
    assert serving["router-image"].endswith(":1.5.0-kvoverlap1")
    assert serving["router-cmd"] == "python3 -m dynamo.frontend --router-mode=kv"
    assert serving["served-model-name"] == "zai-org/GLM-5.3"
    validate_matrix(result["matrix"])
    assert {row["runner"] for row in result["matrix"]} == {"hardware-hcm-8x_02"}
    assert result["matrix"][0]["tp"] == 8


@pytest.mark.parametrize("path", sorted((REPO / "serving").glob("*.yaml")), ids=lambda p: p.name)
def test_every_checked_in_serving_yaml_builds(path):
    build(path.read_text())


def test_replicas_travel_in_serving_and_tp_stays_total():
    text = (YAML.replace("runner: h200-greennode-slurm-1x_01", "runner: hardware-hcm-8x_01")
                .replace("gpus: 1", "gpus: 2\nreplicas: 2")
            + "router:\n  image: vllm/vllm-openai:v0.25.1\n"
              "  serve: vllm-router --port {{router_port}} --worker-urls {{engine_urls}}\n")
    result = build(text)
    assert result["serving"]["replicas"] == "2"
    assert {row["tp"] for row in result["matrix"]} == {2}
    assert "{{engine_urls}}" in result["serving"]["router-cmd"]


def test_title():
    assert build(YAML)["title"] == "serving qwen38-bf16-tp1 agentx-cc-062126-smoke (180s) c4,8 h200-greennode-slurm-1x_01"


def run_cli(*args):
    return subprocess.run([sys.executable, "-P", "-m", "infx.serving.to_matrix", *args],
                          cwd=REPO, capture_output=True, text=True, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})


def test_cli_json_and_check(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(YAML)
    out = run_cli(str(path))
    assert out.returncode == 0, out.stderr
    assert set(json.loads(out.stdout)) == {"matrix", "serving"}
    check = run_cli("--check", str(path))
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip().startswith("serving qwen38-bf16-tp1 ")


def test_cli_errors_one_per_line(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(YAML.replace("gpus: 1", "gpus: 8").replace("model: Qwen/Qwen3.8-27B", "model: /models/q"))
    out = run_cli(str(path))
    assert out.returncode == 2
    assert [line.split(":")[0] for line in out.stderr.splitlines()] == ["ERROR model", "ERROR gpus"]


def test_cli_missing_file():
    out = run_cli("/nonexistent/serving.yaml")
    assert out.returncode == 2
    assert out.stderr.startswith("ERROR serving: cannot read")


LOCK_OK = f"tau2-cs: {{repo: r/x, revision: {'a' * 40}, file: f, sha256: {'b' * 64}}}\n"


def _scenario_repo(tmp_path, scenario_text, lock_text=LOCK_OK):
    (tmp_path / "scenarios" / "datasets").mkdir(parents=True)
    (tmp_path / "scenarios" / "s.yaml").write_text(scenario_text)
    (tmp_path / "scenarios" / "datasets" / "lock.yaml").write_text(lock_text)
    return tmp_path


FILE_SCENARIO = textwrap.dedent("""\
    benchmark:
      datasets: [{name: NAME, type: file, format: mooncake_trace, path: x}]
      phases: [{name: profiling, kind: profiling, duration: 60}]
""")


def test_file_dataset_must_be_locked(tmp_path):
    repo = _scenario_repo(tmp_path, FILE_SCENARIO.replace("NAME", "nope"))
    with pytest.raises(ServingError, match=r"dataset 'nope' is not in scenarios/datasets/lock.yaml"):
        scenario_facts("s", repo)
    assert scenario_facts("s", _scenario_repo(tmp_path / "ok", FILE_SCENARIO.replace("NAME", "tau2-cs"))) == (60, "tau2-cs")


@pytest.mark.parametrize(
    ("lock_text", "error"),
    [
        (LOCK_OK.replace("a" * 40, "main"), r"tau2-cs.revision 'main' does not match"),
        (LOCK_OK.replace("b" * 64, "b" * 63), r"tau2-cs.sha256 .* does not match"),
        ("tau2-cs: {repo: r/x}\n", r"tau2-cs must have exactly repo, revision, file, sha256"),
        ("{{not yaml\n", r"is not YAML"),
    ],
)
def test_malformed_lock_is_one_error(tmp_path, lock_text, error):
    repo = _scenario_repo(tmp_path, FILE_SCENARIO.replace("NAME", "tau2-cs"), lock_text)
    with pytest.raises(ServingError, match=error):
        scenario_facts("s", repo)


def test_lock_is_not_a_scenario(tmp_path):
    assert available_scenarios(_scenario_repo(tmp_path, FILE_SCENARIO)) == ["s"]


@pytest.mark.parametrize("text", ["benchmark: {}\n", "{{not yaml\n", "benchmark:\n  phases: 3\n  datasets: []\n", "[]\n"])
def test_malformed_scenario_is_one_line_error(tmp_path, text):
    repo = _scenario_repo(tmp_path, text)
    with pytest.raises(ServingError) as info:
        scenario_facts("s", repo)
    message = str(info.value)
    assert message.startswith("scenario: scenarios/s.yaml ")
    assert "\n" not in message
