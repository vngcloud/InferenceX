import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from infx.bench_serving.benchmark_outcome import benchmark_outcome
from infx.bench_serving.speedbench_aiperf import acceptance, convert, parse_prometheus


def _stat(avg: float, **extra: float) -> dict:
    return {"unit": "ms", "avg": avg, "p50": avg, "p90": avg * 1.5, "p99": avg * 2, "std": 1.0,
            **extra}


def _export(**overrides) -> dict:
    export = {
        "schema_version": "1.4",
        "aiperf_version": "0.12.0",
        "was_cancelled": False,
        "error_summary": [],
        "request_count": {"unit": "requests", "avg": 80.0},
        "benchmark_duration": {"unit": "sec", "avg": 100.0},
        "total_isl": {"unit": "tokens", "avg": 655360.0},
        "total_osl": {"unit": "tokens", "avg": 81920.0},
        "request_throughput": {"unit": "requests/sec", "avg": 0.8},
        "output_token_throughput": {"unit": "tokens/sec", "avg": 819.2},
        "total_token_throughput": {"unit": "tokens/sec", "avg": 7372.8},
        "time_to_first_token": _stat(300.0),
        "inter_token_latency": _stat(10.0),
        "inter_chunk_latency": _stat(25.0),
        "request_latency": _stat(10530.0),
        "input_sequence_length": {"unit": "tokens", "avg": 8192.0},
        "output_sequence_length": {"unit": "tokens", "avg": 1024.0},
    }
    export.update(overrides)
    return export


def test_convert_emits_the_keys_fixed_sequence_reads() -> None:
    result = convert(_export(), model="org/model", concurrency=8)

    assert result["model_id"] == "org/model"
    assert result["max_concurrency"] == 8
    assert result["total_token_throughput"] == 7372.8
    assert result["output_throughput"] == 819.2
    assert result["mean_ttft_ms"] == 300.0
    assert result["p99_ttft_ms"] == 600.0
    assert result["median_tpot_ms"] == 10.0
    assert result["mean_itl_ms"] == 25.0
    assert result["p90_e2el_ms"] == 10530.0 * 1.5
    assert result["total_input_tokens"] == 655360
    assert result["num_prompts"] == result["completed"] == 80
    assert result["benchmark_outcome"] == benchmark_outcome(80, 80)


def test_convert_counts_errors_into_the_request_gate() -> None:
    export = _export(error_summary=[{"error_details": {"code": 500}, "count": 10}])

    result = convert(export, model="m", concurrency=1)

    assert result["num_prompts"] == 90
    assert result["completed"] == 80
    assert result["benchmark_outcome"]["status"] == "failed"


def test_convert_refuses_a_cancelled_run() -> None:
    with pytest.raises(ValueError, match="cancelled"):
        convert(_export(was_cancelled=True), model="m", concurrency=1)


def test_convert_result_passes_fixed_sequence_processing() -> None:
    from infx.results.fixed_sequence import build_result

    result = convert(_export(), model="org/model", concurrency=8,
                     metadata={"sb_category": "high_entropy", "sb_extra_inputs": {"ignore_eos": True}})
    env = {"RUNNER_TYPE": "h200", "IMAGE": "img", "MODEL_PREFIX": "p", "FRAMEWORK": "vllm",
           "PRECISION": "fp8", "SPEC_DECODING": "mtp", "ISL": "8192", "OSL": "1024",
           "DISAGG": "false", "IS_MULTINODE": "false", "TP": "1", "EP_SIZE": "1",
           "DP_ATTENTION": "false"}

    data = build_result(result, env)

    assert data["conc"] == 8
    assert data["tput_per_gpu"] == 7372.8
    assert data["mean_ttft"] == pytest.approx(0.3)
    assert data["p99_itl"] == pytest.approx(0.05)
    assert data["mean_intvty"] == pytest.approx(100.0)
    assert data["benchmark_outcome"]["status"] == "passed"


def test_acceptance_uses_counter_deltas() -> None:
    before = {"vllm:spec_decode_num_accepted_tokens_total": 100.0,
              "vllm:spec_decode_num_drafts_total": 50.0,
              "vllm:spec_decode_num_draft_tokens_total": 200.0}
    after = {"vllm:spec_decode_num_accepted_tokens_total": 1100.0,
             "vllm:spec_decode_num_drafts_total": 550.0,
             "vllm:spec_decode_num_draft_tokens_total": 2200.0}

    spec = acceptance(before, after)

    assert spec["spec_accepted_tokens"] == 1000
    assert spec["acceptance_length"] == pytest.approx(3.0)
    assert spec["acceptance_rate"] == pytest.approx(0.5)


def test_acceptance_is_null_without_drafts_and_falls_back_to_sglang_gauge() -> None:
    assert acceptance({}, {})["acceptance_length"] is None
    assert acceptance({}, {"sglang:spec_accept_length": 2.4})["acceptance_length"] == 2.4


def test_parse_prometheus_sums_label_sets_and_skips_comments() -> None:
    text = "\n".join([
        "# HELP vllm:spec_decode_num_drafts_total drafts",
        "# TYPE vllm:spec_decode_num_drafts_total counter",
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 1.5e+03',
        'vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 500',
        "vllm:num_requests_running 3",
    ])

    totals = parse_prometheus(text)

    assert totals["vllm:spec_decode_num_drafts_total"] == 2000.0
    assert totals["vllm:num_requests_running"] == 3.0


_ROWS = [
    {"question_id": f"{i:032d}", "category": cat, "messages": [{"role": "user", "content": "hi"}]}
    for i, cat in enumerate(["high_entropy"] * 3 + ["low_entropy"] * 2)
]


def _fake_aiperf(tmp_path: Path) -> Path:
    export = json.dumps(_export(request_count={"unit": "requests", "avg": 3.0}))
    script = tmp_path / "aiperf"
    script.write_text(f"""#!/usr/bin/env bash
printf '%s\\n' "$@" > {tmp_path}/argv
while [ $# -gt 0 ]; do
  if [ "$1" = --output-artifact-dir ]; then out="$2"; fi
  shift
done
mkdir -p "$out"
cat > "$out/profile_export_aiperf.json" <<'JSON'
{export}
JSON
""")
    script.chmod(0o755)
    return script


def _run_cell(tmp_path: Path, env: str) -> subprocess.CompletedProcess:
    data = tmp_path / "data"
    data.mkdir()
    (data / "throughput_8k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = f"""
        set -e
        export INFMAX_CONTAINER_WORKSPACE={workspace} RESULT_DIR={workspace}/results
        export MODEL=org/model PORT=1 CONC=8 OSL=1024 RESULT_FILENAME=cell
        export SPEEDBENCH_CONFIG=throughput_8k SPEEDBENCH_DIR={data}
        {env}
        source benchmarks/benchmark_lib.sh
        AIPERF_PYTHON={sys.executable}
        AIPERF_CLI={_fake_aiperf(tmp_path)}
        run_speedbench_aiperf
    """
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)


def test_run_speedbench_aiperf_writes_a_fixed_seq_result(tmp_path) -> None:
    result = _run_cell(tmp_path, "export SPEEDBENCH_CATEGORY=high_entropy SPEEDBENCH_META='sb_arm=mtp'")

    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[argv.index("--custom-dataset-type") + 1] == "speed_bench_throughput_8k_high_entropy"
    # CONC*10 = 80, floored at 64, then clamped to the three matching rows.
    assert argv[argv.index("--conversation-num") + 1] == "3"
    assert json.loads(argv[argv.index("--extra-inputs") + 1]) == {"ignore_eos": True}
    out = json.loads((tmp_path / "workspace" / "cell.json").read_text())
    assert out["model_id"] == "org/model"
    assert out["max_concurrency"] == 8
    assert out["sb_category"] == "high_entropy"
    assert out["sb_arm"] == "mtp"
    assert out["acceptance_length"] is None
    assert (tmp_path / "workspace/results/speedbench_aiperf/benchmark_command.txt").is_file()


def test_run_speedbench_aiperf_merges_extra_inputs_and_honours_eos(tmp_path) -> None:
    result = _run_cell(
        tmp_path,
        "export SPEEDBENCH_IGNORE_EOS=0 "
        "SPEEDBENCH_EXTRA_INPUTS='{\"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}}'",
    )

    assert result.returncode == 0, result.stderr
    argv = (tmp_path / "argv").read_text().splitlines()
    assert argv[argv.index("--custom-dataset-type") + 1] == "speed_bench_throughput_8k"
    assert json.loads(argv[argv.index("--extra-inputs") + 1]) == {
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def test_run_speedbench_aiperf_rejects_an_empty_category(tmp_path) -> None:
    result = _run_cell(tmp_path, "export SPEEDBENCH_CATEGORY=mixed")

    assert result.returncode != 0
    assert "matches no rows" in result.stderr
    assert not (tmp_path / "argv").exists()


def test_speedbench_token_reaches_the_greennode_container() -> None:
    script = Path("runners/launch_h200-greennode.sh").read_text()
    run_env = script.split("RUN_ENV=(", 1)[1].split(")", 1)[0]

    assert "SPEEDBENCH_HF_TOKEN" in run_env.split()


def test_speedbench_token_is_wired_only_into_benchmark_tmpl_calls() -> None:
    tmpl = Path(".github/workflows/benchmark-tmpl.yml").read_text()
    assert "      SPEEDBENCH_HF_TOKEN:\n        required: false\n" in tmpl
    assert "  SPEEDBENCH_HF_TOKEN: ${{ secrets.SPEEDBENCH_HF_TOKEN }}\n" in tmpl

    for name in ("e2e-tests", "run-sweep"):
        text = Path(f".github/workflows/{name}.yml").read_text()
        for job in re.split(r"\n    (?=[\w-]+:\n)", text):
            uses = re.search(r"uses: .*workflows/([\w-]+)\.yml", job)
            if not uses or "MODAL_TOKEN_SECRET: ${{" not in job:
                continue
            passes = "SPEEDBENCH_HF_TOKEN: ${{ secrets.SPEEDBENCH_HF_TOKEN }}" in job
            assert passes == (uses.group(1) == "benchmark-tmpl"), (name, uses.group(1))
