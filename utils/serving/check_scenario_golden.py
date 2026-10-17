"""Golden check: scenarios/<name>.yaml + per-run overrides == the reference AgentX CLI.

Run inside the aiperf project env (needs aiperf + cyclopts), from the repo root:

    (cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)

For each scenario it resolves (a) the reference CLI command: the Weka one that
produced the green qwen38 smoke run 37733664587, or the tau2 one of run 37920508184, with --benchmark-duration set to the
scenario's duration, and (b) aiperf profile --config <scenario> + the overrides
that build_replay_cmd_yaml emits (taken by running that bash function), then
compares the two resolved configs field by field. Exit 1 on any difference.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from cyclopts import App

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.flags.resolver import resolve_config

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = {
    "agentx-cc-062126": "weka",
    "agentx-cc-062126-smoke": "weka",
    "custom-agentic-tau2": "tau2",
    "custom-agentic-tau2-smoke": "tau2",
    "custom-agentic-livekit111868-v1": "livekit",
    "custom-agentic-livekit111868-v2": "livekit",
    "custom-agentic-livekit111868-v1-nothink": "livekit",
    "custom-agentic-livekit111868-v2-nothink": "livekit",
    "custom-agentic-livekit111868-smoke": "livekit",
}
# datasets[].name is a label (CLI always says "main"); for file datasets it is the lock key.
ALLOWED = {"weka": set(), "tau2": {".benchmark.datasets[0].name"}, "livekit": {".benchmark.datasets[0].name"}}
# Think-time-off variants cap idle gaps at 0; every other livekit scenario keeps 300.
LIVEKIT_IDLE_CAP = {"custom-agentic-livekit111868-v1-nothink": 0.0, "custom-agentic-livekit111868-v2-nothink": 0.0}
MODEL = "Qwen/Qwen3.8-27B"


def reference_argv(duration: int, artifact_dir: str) -> list[str]:
    return shlex.split(
        "--scenario inferencex-agentx-mvp --url http://localhost:8888 "
        "--endpoint /v1/chat/completions --endpoint-type chat --streaming "
        f"--model {MODEL} --tokenizer {MODEL} --concurrency 8 "
        f"--benchmark-duration {duration} --stats-interval 30 --random-seed 42 "
        "--failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 "
        "--trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 "
        "--trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 "
        "--use-server-token-count --gpu-telemetry http://localhost:9400/metrics "
        "--tokenizer-trust-remote-code --max-context-length 262144 "
        "--num-dataset-entries 393 --slice-duration 1.0 "
        "--server-metrics http://localhost:8888/metrics "
        f"--output-artifact-dir {artifact_dir}/aiperf_artifacts --unsafe-override "
        "--public-dataset semianalysis_cc_traces_weka_062126"
    )


def tau2_reference_argv(duration: int, artifact_dir: str, dataset_file: str, entries: int = 1576,
                        request_count: int | None = 16798, idle_cap: float = 300.0,
                        sampling: str | None = None) -> list[str]:
    """The tau2 command of run 37920508184 (gemma4tau2, c16), model/conc normalized like the Weka one."""
    argv = (
        "--scenario inferencex-agentx-mvp --url http://localhost:8888 "
        "--endpoint /v1/chat/completions --endpoint-type chat --streaming "
        f"--model {MODEL} --tokenizer {MODEL} --concurrency 8 "
        f"--benchmark-duration {duration} --stats-interval 30 --random-seed 42 "
        "--failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 "
        "--trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 "
        f"--trace-idle-gap-cap-seconds {idle_cap:g} --warmup-grace-period 1800 "
        "--use-server-token-count --gpu-telemetry http://localhost:9400/metrics "
        f"--tokenizer-trust-remote-code --num-dataset-entries {entries} --slice-duration 1.0 "
        "--server-metrics http://localhost:8888/metrics "
        f"--output-artifact-dir {artifact_dir}/aiperf_artifacts --unsafe-override "
        f"--custom-dataset-type mooncake_trace --input-file {dataset_file}"
    )
    if request_count is not None:
        argv += f" --request-count {request_count}"
    if sampling is not None:
        argv += f" --dataset-sampling-strategy {sampling}"
    return shlex.split(argv)


def differences(kind: str, reference: list[str], scenario: str, tmp: str) -> list[str]:
    allowed = ALLOWED[kind]
    return [d for d in diff(resolved(reference), resolved(yaml_argv(scenario, tmp))) if d.split(":")[0] not in allowed]


def yaml_argv(scenario: str, artifact_dir: str) -> list[str]:
    """The argv build_replay_cmd_yaml produces, minus the leading `aiperf profile`."""
    script = (
        "set -e\n"
        "export IS_AGENTIC=1 KV_OFFLOADING=none PORT=8888 CONC=8 DURATION=0\n"
        f"export INFMAX_CONTAINER_WORKSPACE={shlex.quote(str(REPO))}\n"
        f"export AIPERF_CLI=aiperf MODEL={MODEL} FRAMEWORK=vllm SERVING_SCENARIO={scenario}\n"
        "export AIPERF_GPU_TELEMETRY_URL=http://localhost:9400/metrics\n"
        "export AIPERF_SERVER_METRICS_URLS=http://localhost:8888/metrics\n"
        "source benchmarks/benchmark_lib.sh\n"
        f"build_replay_cmd_yaml {shlex.quote(artifact_dir)}\n"
        'printf "%s" "$REPLAY_CMD"\n'
    )
    out = subprocess.run(["bash", "-c", script], cwd=REPO, check=True, capture_output=True, text=True).stdout
    argv = shlex.split(out)
    assert argv[:2] == ["aiperf", "profile"], argv[:2]
    return argv[2:]


def parse(argv: list[str]) -> CLIConfig:
    captured: dict[str, CLIConfig] = {}
    app = App(name="golden")

    @app.default
    def _capture(*, cli_config: CLIConfig) -> None:
        captured["cli"] = cli_config

    try:
        app(argv, exit_on_error=False)
    except SystemExit as exc:  # cyclopts exits 0 after running the default command
        if exc.code not in (0, None):
            raise
    return captured["cli"]


def resolved(argv: list[str]) -> dict:
    cli = parse(argv)
    return resolve_config(cli, cli.config_file).model_dump(mode="json", by_alias=True)


def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += diff(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: cli={json.dumps(a)} yaml={json.dumps(b)}"]


def main() -> int:
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        dataset_file = Path(tmp) / "trace.jsonl"
        dataset_file.write_text("")
        os.environ["IX_CUSTOM_DATASET_FILE"] = str(dataset_file)  # substituted by aiperf's YAML loader
        for scenario, kind in SCENARIOS.items():
            data = yaml.safe_load((REPO / "scenarios" / f"{scenario}.yaml").read_text())
            duration = int(data["benchmark"]["phases"][0]["duration"])
            if kind == "weka":
                reference = reference_argv(duration, tmp)
            elif kind == "livekit":
                reference = tau2_reference_argv(duration, tmp, str(dataset_file), entries=1217,
                                                request_count=None,
                                                idle_cap=LIVEKIT_IDLE_CAP.get(scenario, 300.0),
                                                sampling="shuffle")
            else:
                reference = tau2_reference_argv(duration, tmp, str(dataset_file))
            found = differences(kind, reference, scenario, tmp)
            if found:
                failed = True
                print(f"FAIL {scenario}:")
                for line in found:
                    print("  " + line)
            else:
                print(f"OK {scenario} ({duration}s): YAML + overrides == reference CLI")
        # The check must be able to fail: one entry off has to show up.
        negative = differences("tau2", tau2_reference_argv(1200, tmp, str(dataset_file), entries=1575),
                               "custom-agentic-tau2", tmp)
        if negative:
            print(f"OK negative case caught: {negative[0]}")
        else:
            failed = True
            print("FAIL negative case: --num-dataset-entries 1575 was not detected")
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("AIPERF_LOG_LEVEL", "WARNING")
    sys.exit(main())
