#!/usr/bin/env python3
"""Validate a single-node InferenceX agentic benchmark before dispatch."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


DATASETS = {
    "full": "semianalysis_cc_traces_weka_062126",
    "cap-256k": "semianalysis_cc_traces_weka_062126_256k",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--runner-node", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument("--ccu", required=True, help="Comma-separated CCUs")
    parser.add_argument("--branch", help="Pushed feature branch; defaults to the current branch")
    parser.add_argument("--test-name", help="GitHub Actions display name")
    parser.add_argument("--model-container-path")
    parser.add_argument("--model-host-path")
    parser.add_argument("--model-host-root")
    parser.add_argument("--model-container-root")
    parser.add_argument("--ssh-target")
    parser.add_argument("--ssh-port", type=int, default=22)
    return parser.parse_args()


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def require_text(text: str, needle: str, label: str, errors: list[str]) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle!r}", errors)


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def parse_ccus(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --ccu: {exc}") from exc
    if not values or any(value <= 0 for value in values) or len(values) != len(set(values)):
        raise SystemExit("--ccu must contain unique positive integers")
    return values


def resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"expected YAML mapping: {path}")
    return data


def validate_config(
    repo: Path,
    config_path: Path,
    key: str,
    runner_node: str,
    ccus: list[int],
    errors: list[str],
) -> tuple[dict, list[dict]]:
    master = load_yaml(config_path)
    config = master.get(key)
    if not isinstance(config, dict):
        raise SystemExit(f"config key not found: {key}")
    if config.get("multinode") is not False:
        fail("config must set multinode: false", errors)
    scenarios = config.get("scenarios", {})
    if set(scenarios) != {"agentic-coding"}:
        fail("config must contain only scenarios.agentic-coding", errors)
    agentic = scenarios.get("agentic-coding") or []
    spaces = [space for item in agentic for space in item.get("search-space", [])]
    actual_ccus = [value for space in spaces for value in space.get("conc-list", [])]
    if actual_ccus != ccus:
        fail(f"CCU mismatch: expected {ccus}, got {actual_ccus}", errors)
    for space in spaces:
        if space.get("kv-offloading") != "dram":
            fail("agentic search space must set kv-offloading: dram", errors)
        backend = space.get("kv-offload-backend") or {}
        if backend.get("name") != "hicache":
            fail("agentic search space must use hicache backend", errors)
    pool = config.get("runner")
    runners = load_yaml(repo / "configs/runners.yaml").get("labels", {})
    if runner_node not in runners.get(pool, []):
        fail(f"runner {runner_node!r} is not in pool {pool!r}", errors)
    return config, spaces


def validate_recipe(
    recipe_path: Path,
    dataset: str,
    model_container_path: str | None,
    errors: list[str],
) -> str:
    text = recipe_path.read_text()
    checks = {
        "HiCache validation": "require_agentic_kv_offload_backend hicache",
        "server metrics URL": "AIPERF_SERVER_METRICS_URLS=",
        "DCGM telemetry URL": "AIPERF_GPU_TELEMETRY_URL=",
        "server metrics flag": "--enable-metrics",
        "cache report flag": "--enable-cache-report",
        "hierarchical cache flag": "--enable-hierarchical-cache",
        "HiCache size": "--hicache-size",
        "max running requests": "MAX_RUNNING_REQUESTS=$((2 * CONC))",
        "CUDA graph scaling": "CUDA_GRAPH_MAX_BS",
        "agentic replay": "run_agentic_replay_and_write_outputs",
    }
    for label, needle in checks.items():
        require_text(text, needle, label, errors)
    dataset_assignment = rf"^\s*(?:export\s+)?WEKA_LOADER_OVERRIDE={re.escape(DATASETS[dataset])}\s*$"
    if not re.search(dataset_assignment, text, flags=re.MULTILINE):
        fail(f"dataset: missing exact WEKA_LOADER_OVERRIDE={DATASETS[dataset]!s}", errors)
    local_path_match = re.search(
        r'^\s*(?:export\s+)?MODEL_PATH=["\x27]?(/[^"\x27\s]+)["\x27]?\s*$',
        text,
        flags=re.MULTILINE,
    )
    if local_path_match and not model_container_path:
        fail("local MODEL_PATH requires container and remote host model preflight arguments", errors)
    if local_path_match and model_container_path != local_path_match.group(1):
        fail(
            f"container model path mismatch: recipe uses {local_path_match.group(1)!r}, "
            f"preflight received {model_container_path!r}",
            errors,
        )
    elif model_container_path:
        require_text(text, model_container_path, "container model path", errors)
    syntax = run(["bash", "-n", str(recipe_path)], recipe_path.parent)
    if syntax.returncode:
        fail(f"recipe bash syntax failed: {syntax.stderr.strip()}", errors)
    return text


def validate_launcher(
    launcher_path: Path,
    model_host_root: str | None,
    model_container_root: str | None,
    errors: list[str],
) -> None:
    text = launcher_path.read_text()
    checks = {
        "DCGM exporter": "dcgm-exporter",
        "DCGM cleanup": "docker rm -f \"$DCGM_NAME\"",
        "KV offloading env": "KV_OFFLOADING",
        "KV backend env": "KV_OFFLOAD_BACKEND",
        "KV backend metadata env": "KV_OFFLOAD_BACKEND_METADATA",
    }
    for label, needle in checks.items():
        require_text(text, needle, label, errors)
    if model_host_root and model_container_root:
        require_text(text, model_host_root, "host model mount root", errors)
        require_text(text, model_container_root, "container model mount root", errors)
        require_text(text, ":ro", "read-only model mount", errors)
    syntax = run(["bash", "-n", str(launcher_path)], launcher_path.parent)
    if syntax.returncode:
        fail(f"launcher bash syntax failed: {syntax.stderr.strip()}", errors)


def validate_workflow(repo: Path, errors: list[str]) -> None:
    workflow = (repo / ".github/workflows/e2e-tests.yml").read_text()
    require_text(workflow, "skip-agentic-ingest:", "ingest opt-out input", errors)
    require_text(
        workflow,
        "inputs.skip-agentic-ingest != true",
        "ingest opt-out condition",
        errors,
    )


def generator_command(config_path: Path, config: dict, runner_node: str, ccus: list[int]) -> list[str]:
    return [
        "python3",
        "utils/matrix_logic/generate_sweep_configs.py",
        "full-sweep",
        "--config-files",
        str(config_path),
        "--model-prefix",
        str(config["model-prefix"]),
        "--precision",
        str(config["precision"]),
        "--framework",
        str(config["framework"]),
        "--runner-type",
        str(config["runner"]),
        "--runner-node-filter",
        runner_node,
        "--scenario-type",
        "agentic-coding",
        "--min-conc",
        str(min(ccus)),
        "--max-conc",
        str(max(ccus)),
        "--single-node",
        "--no-evals",
    ]


def validate_matrix(repo: Path, command: list[str], ccus: list[int], errors: list[str]) -> list[dict]:
    result = run(command, repo)
    if result.returncode:
        fail(f"matrix generation failed: {result.stderr.strip()}", errors)
        return []
    try:
        matrix = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        fail(f"matrix output is not JSON: {exc}", errors)
        return []
    actual = [entry.get("conc") for entry in matrix]
    if actual != ccus:
        fail(f"generated matrix CCUs differ: expected {ccus}, got {actual}", errors)
    for entry in matrix:
        if entry.get("scenario-type") != "agentic-coding":
            fail("generated matrix contains a non-agentic job", errors)
        if entry.get("run-eval") or entry.get("eval-only"):
            fail("generated matrix contains an eval job", errors)
    return matrix


def validate_remote_model(args: argparse.Namespace, config: dict, errors: list[str]) -> str:
    requested = [
        args.model_container_path,
        args.model_host_path,
        args.model_host_root,
        args.model_container_root,
        args.ssh_target,
    ]
    if not any(requested):
        return "not-applicable"
    if not all(requested):
        fail("remote model validation requires host path/root, container root, and SSH target", errors)
        return "failed"
    host_path = args.model_host_path.rstrip("/")
    container_path = args.model_container_path
    if not container_path:
        fail("remote model validation requires --model-container-path", errors)
        return "failed"
    checks = " && ".join(
        [
            f"test -s {shlex.quote(host_path + '/config.json')}",
            f"test -s {shlex.quote(host_path + '/model.safetensors.index.json')}",
            f"test -z \"$(find {shlex.quote(host_path)} -type f -name '*.incomplete' -print -quit)\"",
        ]
    )
    ssh_base = ["ssh", "-o", "BatchMode=yes", "-p", str(args.ssh_port), args.ssh_target]
    host_check = run([*ssh_base, checks], args.repo)
    if host_check.returncode:
        fail(f"host model check failed: {host_check.stderr.strip()}", errors)
        return "failed"
    image = str(config["image"])
    mount = f"{args.model_host_root}:{args.model_container_root}:ro"
    inside = " && ".join(
        [
            f"test -s {shlex.quote(container_path.rstrip('/') + '/config.json')}",
            f"test -s {shlex.quote(container_path.rstrip('/') + '/model.safetensors.index.json')}",
        ]
    )
    docker_command = " ".join(
        shlex.quote(part)
        for part in ["docker", "run", "--rm", "-v", mount, "--entrypoint", "sh", image, "-lc", inside]
    )
    container_check = run([*ssh_base, docker_command], args.repo)
    if container_check.returncode:
        fail(f"container model check failed: {container_check.stderr.strip()}", errors)
        return "failed"
    return "passed"


def current_branch(repo: Path, requested: str | None, errors: list[str]) -> str:
    if requested:
        branch = requested
    else:
        result = run(["git", "branch", "--show-current"], repo)
        branch = result.stdout.strip() if result.returncode == 0 else ""
    if not branch:
        fail("cannot determine a dispatch branch", errors)
    if branch in {"main", "master"}:
        fail("dispatch from a feature branch, not the default branch", errors)
    return branch


def dispatch_command(
    branch: str,
    generate_cli: str,
    duration: int,
    test_name: str,
) -> list[str]:
    return [
        "gh",
        "workflow",
        "run",
        "e2e-tests.yml",
        "--ref",
        branch,
        "-f",
        f"generate-cli-command={generate_cli}",
        "-f",
        f"test-name={test_name}",
        "-f",
        f"ref={branch}",
        "-f",
        f"duration-override={duration}",
        "-f",
        "skip-agentic-ingest=true",
    ]


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    ccus = parse_ccus(args.ccu)
    config_path = resolve(args.repo, args.config_file)
    recipe_path = resolve(args.repo, args.recipe)
    launcher_path = resolve(args.repo, args.launcher)
    errors: list[str] = []

    config, _ = validate_config(args.repo, config_path, args.config_key, args.runner_node, ccus, errors)
    validate_recipe(recipe_path, args.dataset, args.model_container_path, errors)
    validate_launcher(launcher_path, args.model_host_root, args.model_container_root, errors)
    validate_workflow(args.repo, errors)
    command = generator_command(args.config_file, config, args.runner_node, ccus)
    matrix = validate_matrix(args.repo, command, ccus, errors)
    remote_status = validate_remote_model(args, config, errors)
    branch = current_branch(args.repo, args.branch, errors)
    generate_cli = shlex.join(command[2:])
    test_name = args.test_name or f"{args.config_key} agentic CCU {','.join(map(str, ccus))} {args.duration}s no-ingest"
    dispatch = dispatch_command(branch, generate_cli, args.duration, test_name)

    summary = {
        "branch": branch,
        "config_key": args.config_key,
        "model": config.get("model"),
        "image": config.get("image"),
        "precision": config.get("precision"),
        "framework": config.get("framework"),
        "runner_pool": config.get("runner"),
        "runner_node": args.runner_node,
        "dataset": DATASETS[args.dataset],
        "duration": args.duration,
        "ccu": ccus,
        "server_metrics": "required",
        "dcgm": "required",
        "skip_agentic_ingest": True,
        "remote_model_check": remote_status,
        "matrix_rows": len(matrix),
        "generate_cli_command": generate_cli,
        "dispatch_command": shlex.join(dispatch),
    }
    print(json.dumps(summary, indent=2))
    if errors:
        print("\nPRECHECK FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("\nPRECHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
