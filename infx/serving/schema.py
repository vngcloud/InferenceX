"""Serving YAML (V1+V2): the one file an agent writes to dispatch an AgentX benchmark.

V1 is an engine-only YAML (`vllm serve` / `python3 -m sglang.launch_server`, ServingSpec);
V2 adds an optional `router:` block (RouterSpec); V3 lets it be any router (placeholders, metrics_port).
V4 adds replicas: N engines behind the router (spec 2026-10-16).
Specs: docs/superpowers/specs/2026-10-12-serving-yaml-dispatch-design.md (V1) and
docs/superpowers/specs/2026-10-13-serving-router-dynamo-design.md (V2);
docs/superpowers/specs/2026-10-14-serving-router-generic-design.md (V3).
Every error is one line "<field>: <problem and fix>" so an agent can act on it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
NAME_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
LOCAL_PATH_MARKERS = ("/models/", "/mnt/", "/data/")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# image is spliced into `bash -c` strings on the GPU node; model is word-split
# into the replay command. Keep both to plain registry refs / HF repo ids, or an
# absolute .sqsh path for a squash file already on the runner's node.
IMAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*|/[A-Za-z0-9._/-]+\.sqsh")
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+")
# Variables serve_generic.sh (and the helpers it calls) reads or sets itself;
# exporting them from env: would change the recipe, not the engine.
RESERVED_ENV = (
    "PORT",
    "MODEL",
    "CONC",
    "RESULT_DIR",
    "HF_HUB_CACHE",
    "FRAMEWORK",
    "SERVE_CMD",
    "SERVE_ENV",
    "SERVED_MODEL_NAME",
    "SERVING_SCENARIO",
    "SERVING_LOADER",
    "SERVING_YAML",
    "SERVING_ENGINE_ARGS",
    "WEKA_LOADER_OVERRIDE",
    "AIPERF_HF_CLI",
    "AIPERF_SERVER_METRICS_URLS",
    "AIPERF_REQUIRED_SERVER_METRIC_PREFIX",
    # Dynamo discovery and the router sidecar: set by runners/launch_hardware-hcm.sh.
    "DYN_DISCOVERY_BACKEND",
    "DYN_FILE_KV",
    "DYN_SYSTEM_ENABLED",
    "DYN_SYSTEM_PORT",
    "ROUTER_IMAGE",
    "ROUTER_CMD",
    "ROUTER_ENV",
    "ROUTER_PROFILE",
    "ROUTER_METRICS_PORT",
    "ENGINE_PORT",
    "ENGINE_PORTS",
    "REPLICAS",
    # Storage sidecar (Mooncake master etc.): set by runners/launch_hardware-hcm.sh.
    "STORAGE_IMAGE",
    "STORAGE_CMD",
    "STORAGE_ENV",
    "STORAGE_MASTER_PORT",
    "STORAGE_META_PORT",
)
# Pools whose launcher runs serve_generic.sh on SERVE_CMD, and which runner names in
# each pool do (hardware-hcm: only the full-node 8x identities; smaller tiers share the box).
SERVING_RUNNERS = {
    "cluster:h200-greennode-slurm": re.compile(r"h200-greennode-slurm-1x_\d+"),
    "cluster:hardware-hcm": re.compile(r"hardware-hcm-8x_\d+"),
    "cluster:hcm-rtx6000pro": re.compile(r"hcm-rtx6000pro_\d+"),
}
# Only launch_hardware-hcm.sh and launch_hcm-rtx6000pro.sh start the router / storage sidecars.
ROUTER_RUNNER_PREFIXES = ("hardware-hcm-", "hcm-rtx6000pro_")
STORAGE_RUNNER_PREFIXES = ("hardware-hcm-", "hcm-rtx6000pro_")
# Only launch_hardware-hcm.sh / launch_hcm-rtx6000pro.sh size the salloc from cpus:.
CPUS_RUNNER_PREFIXES = ("hardware-hcm-", "hcm-rtx6000pro_")
# {{name}} (lowercase, no spaces) placeholders expanded at runtime by benchmarks/benchmark_lib.sh.
# {{ x }} (with spaces) is inline Jinja and stays literal.
# router.* can reference the router/engine ports and either storage port if the router
# needs to talk to the storage sidecar (e.g. a health check).
ROUTER_PLACEHOLDERS = (
    "router_port",
    "engine_url",
    "engine_port",
    "engine_urls",
    "storage_master_port",
    "storage_meta_port",
)
# engine_urls = every replica's URL (http://127.0.0.1:<port>, space-separated);
# engine_url stays the first engine's (back-compat, spec 2026-10-16).
# env: (engine) can reference the storage ports — the engine connects to the Mooncake
# master through them. {{router_port}}/{{engine_port}} don't belong here (the engine
# does not know its own HTTP port before the recipe appends --port).
ENV_PLACEHOLDERS = ("storage_master_port", "storage_meta_port")
# storage.serve can reference its own ports (it binds them).
STORAGE_PLACEHOLDERS = ("storage_master_port", "storage_meta_port")
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


class ServingError(ValueError):
    """Validation failure; each line names a field and the fix."""


def _coerce_env(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if isinstance(item, bool):
            raise ValueError(f'{key}: quote boolean values, e.g. "1" or "true"')
        out[key] = str(item) if isinstance(item, (int, float)) else item
    return out


class RouterSpec(BaseModel):
    """A router sidecar in front of the engine (spec 2026-10-14 §1)."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class StorageSpec(BaseModel):
    """A storage sidecar launched before the engine (e.g. Mooncake master for HiCache L3).
    Non-GPU process; the launcher gives it STORAGE_MASTER_PORT / STORAGE_META_PORT and the
    engine reaches it over 127.0.0.1 (pyxis shares the host netns).
    """

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class ServingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    runner: str = Field(min_length=1)
    image: str = Field(min_length=1)
    model: str = Field(min_length=1)
    gpus: int = Field(gt=0)
    cpus: int | None = Field(
        default=None,
        gt=0,
        description="Slurm CPUs for the job's allocation (SERVING_CPUS; unset = the launcher's "
        "per-GPU default), honored by launch_hardware-hcm.sh / launch_hcm-rtx6000pro.sh",
    )
    replicas: int = Field(default=1, ge=1)
    precision: str = Field(default="bf16", pattern=r"^[a-z0-9]+$")
    env: dict[str, str] = Field(default_factory=dict)
    serve: str = Field(min_length=1)
    engine_args: dict[str, Any] | None = Field(
        default=None,
        alias="engine-args",
        description="Extra engine arguments; written next to the run as engine_args.yaml and "
        "passed to the engine as --config (trtllm-serve 1.4.x LlmArgs YAML, trtllm-serve only)",
    )
    router: RouterSpec | None = None
    storage: StorageSpec | None = None
    scenario: str | list[Annotated[str, Field(pattern=NAME_PATTERN)]] = Field(
        min_length=1,
        description="One scenario name, or a list: one benchmark job per scenario in a single run",
    )
    conc: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)

    @field_validator("conc", mode="before")
    @classmethod
    def _scalar_conc(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return [value]
        return value

    @field_validator("scenario", mode="before")
    @classmethod
    def _scenario_name(cls, value: Any) -> Any:
        def strip(name: str) -> str:
            name = name.removeprefix("scenarios/")
            for suffix in (".yaml", ".yml"):
                name = name.removesuffix(suffix)
            return name

        if isinstance(value, str):
            return strip(value)
        if isinstance(value, list):
            return [strip(name) for name in value]
        return value

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


def available_scenarios(repo: Path = REPO_ROOT) -> list[str]:
    return sorted(path.stem for path in (repo / "scenarios").glob("*.yaml"))


def is_local_squash(image: str) -> bool:
    """A squash file already on the runner's node; the launcher starts it as is."""
    return image.startswith("/") and image.endswith(".sqsh")


def _format_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "serving"
        message = item["msg"]
        if item["type"] == "extra_forbidden":
            if item["loc"][:1] == ("router",):
                model = RouterSpec
            elif item["loc"][:1] == ("storage",):
                model = StorageSpec
            else:
                model = ServingSpec
            allowed = (field.alias or name for name, field in model.model_fields.items())
            message = "unknown key (allowed: " + ", ".join(allowed) + ")"
        lines.append(f"{location}: {message}")
    return "\n".join(lines)


def _serving_runners(repo: Path) -> dict[str, int]:
    """Runner name -> gpus-per-node, for every runner that can run a serving YAML."""
    runners = yaml.safe_load((repo / "configs" / "runners.yaml").read_text())
    out: dict[str, int] = {}
    for pool, pattern in SERVING_RUNNERS.items():
        names = [
            name
            for name in (runners.get("labels") or {}).get(pool) or []
            if pattern.fullmatch(name)
        ]
        if not names:
            continue
        gpus = ((runners.get("hardware") or {}).get(pool) or {}).get("gpus-per-node")
        if gpus is None:
            raise ServingError(f"runner: pool {pool} has no gpus-per-node in configs/runners.yaml")
        out.update({name: int(gpus) for name in names})
    return out


def _gpus_per_node(runner: str, repo: Path) -> int:
    allowed = _serving_runners(repo)
    if runner not in allowed:
        raise ServingError(
            f"runner: {runner} cannot run a serving YAML; use one of: {', '.join(allowed)}"
        )
    return allowed[runner]


def load_serving(text: str, repo: Path = REPO_ROOT) -> ServingSpec:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ServingError(f"serving: not valid YAML ({error})") from error
    if not isinstance(raw, dict):
        raise ServingError("serving: the top level must be a mapping of keys")
    if "duration" in raw:
        raise ServingError(
            "duration: not a serving key; the scenario fixes the duration "
            "(pick the -smoke scenario for a short run)"
        )
    try:
        spec = ServingSpec.model_validate(raw)
    except ValidationError as error:
        raise ServingError(_format_validation_error(error)) from error

    problems: list[str] = []
    if not IMAGE_PATTERN.fullmatch(spec.image):
        problems.append(
            "image: only letters, digits and ._/:@- are allowed (e.g. vllm/vllm-openai:v0.29.0)"
        )
    if is_local_squash(spec.image) and not spec.runner.startswith(
        ("hardware-hcm-8x_", "hcm-rtx6000pro_")
    ):
        problems.append(
            "image: a .sqsh file needs a hardware-hcm-8x_* or hcm-rtx6000pro_* runner "
            "(only launch_hardware-hcm.sh / launch_hcm-rtx6000pro.sh start a local squash as is)"
        )
    if spec.model.startswith("/"):
        problems.append("model: use the HF repo id (e.g. Qwen/Qwen3.8-27B), not a local path")
    elif not MODEL_PATTERN.fullmatch(spec.model):
        problems.append(
            "model: must be an HF repo id <org>/<name> using letters, digits and ._- (e.g. Qwen/Qwen3.8-27B)"
        )
    commands = (
        [("serve", spec.serve)]
        + ([("router.serve", spec.router.serve)] if spec.router else [])
        + ([("storage.serve", spec.storage.serve)] if spec.storage else [])
    )
    for field, command in commands:
        for marker in LOCAL_PATH_MARKERS:
            if marker in command:
                problems.append(
                    f"{field}: local path containing {marker!r}; replace it with the HF repo id from model"
                )
                break
        if "$" in command:
            problems.append(
                f"{field}: shell variables ($...) are not expanded; write literal values"
            )
        if "--numa-node" in command:
            problems.append(f"{field}: drop --numa-node (benchmark convention)")
        if field == "serve" and any(
            staged_flag in command
            for staged_flag in ("--extra-engine-args", "--config", "--extra_llm_api_options")
        ):
            problems.append(
                f"{field}: the engine-args YAML flag (--extra-engine-args, --config or "
                "--extra_llm_api_options) is staged from engine-args:; remove the flag"
            )
    envs = (
        [("env", spec.env)]
        + ([("router.env", spec.router.env)] if spec.router else [])
        + ([("storage.env", spec.storage.env)] if spec.storage else [])
    )
    for prefix, env in envs:
        for key, value in env.items():
            if not ENV_NAME.fullmatch(key):
                problems.append(f"{prefix}.{key}: not a valid environment variable name")
            elif key in RESERVED_ENV:
                problems.append(
                    f"{prefix}.{key}: reserved, the benchmark recipe sets it; remove it from env"
                )
            if "\n" in value:
                problems.append(f"{prefix}.{key}: value must be a single line")
    if spec.cpus and not spec.runner.startswith(CPUS_RUNNER_PREFIXES):
        problems.append(
            "cpus: needs a hardware-hcm-8x_* or hcm-rtx6000pro_* runner "
            "(only launch_hardware-hcm.sh / launch_hcm-rtx6000pro.sh size the salloc from cpus:)"
        )
    if spec.router:
        if not IMAGE_PATTERN.fullmatch(spec.router.image):
            problems.append("router.image: only letters, digits and ._/:@- are allowed")
        if is_local_squash(spec.router.image):
            problems.append("router.image: a .sqsh path is engine-image only; use a registry ref")
        if not spec.runner.startswith(ROUTER_RUNNER_PREFIXES):
            problems.append(
                "router: needs a hardware-hcm-8x_* or hcm-rtx6000pro_* runner "
                "(only launch_hardware-hcm.sh / launch_hcm-rtx6000pro.sh start the router)"
            )
        texts = [("router.serve", spec.router.serve)] + [
            (f"router.env.{key}", value) for key, value in spec.router.env.items()
        ]
        for field, body in texts:
            problems.extend(
                f"{field}: unknown placeholder {{{{{name}}}}}; "
                f"allowed: {', '.join(ROUTER_PLACEHOLDERS)}"
                for name in PLACEHOLDER.findall(body)
                if name not in ROUTER_PLACEHOLDERS
            )
    if spec.storage:
        if not IMAGE_PATTERN.fullmatch(spec.storage.image):
            problems.append("storage.image: only letters, digits and ._/:@- are allowed")
        if is_local_squash(spec.storage.image):
            problems.append("storage.image: a .sqsh path is engine-image only; use a registry ref")
        if not spec.runner.startswith(STORAGE_RUNNER_PREFIXES):
            problems.append(
                "storage: needs a hardware-hcm-8x_* or hcm-rtx6000pro_* runner "
                "(only launch_hardware-hcm.sh / launch_hcm-rtx6000pro.sh start the storage sidecar)"
            )
        texts = [("storage.serve", spec.storage.serve)] + [
            (f"storage.env.{key}", value) for key, value in spec.storage.env.items()
        ]
        for field, body in texts:
            problems.extend(
                f"{field}: unknown placeholder {{{{{name}}}}}; "
                f"allowed: {', '.join(STORAGE_PLACEHOLDERS)}"
                for name in PLACEHOLDER.findall(body)
                if name not in STORAGE_PLACEHOLDERS
            )
    # Top-level env: only storage-port placeholders are available (router/engine ports are
    # recipe-owned; engine does not know its own port at env-expansion time).
    allowed_env_placeholders = ENV_PLACEHOLDERS if spec.storage else ()
    for key, value in spec.env.items():
        for name in PLACEHOLDER.findall(value):
            if name not in allowed_env_placeholders:
                hint = (
                    f"allowed: {', '.join(ENV_PLACEHOLDERS)}"
                    if spec.storage
                    else "no placeholders are allowed in env: without a storage block"
                )
                problems.append(f"env.{key}: unknown placeholder {{{{{name}}}}}; {hint}")
    if len(set(spec.conc)) != len(spec.conc):
        problems.append("conc: duplicate values")
    if spec.gpus % spec.replicas:
        problems.append(
            f"replicas: gpus {spec.gpus} is not a multiple of replicas {spec.replicas}; "
            "each replica gets gpus/replicas GPUs"
        )
    if spec.replicas > 1:
        if spec.router is None:
            problems.append(
                "replicas: more than 1 replica needs a router: block whose serve lists {{engine_urls}}"
            )
        elif "dynamo.frontend" in spec.router.serve:
            problems.append(
                "replicas: the Dynamo frontend discovers its engines itself; use replicas: 1"
            )
    from infx.serving.to_matrix import is_sweep  # local import: to_matrix imports this module

    scenarios = [spec.scenario] if isinstance(spec.scenario, str) else list(spec.scenario)
    if not isinstance(spec.scenario, str):
        if len(set(spec.scenario)) != len(spec.scenario):
            problems.append("scenario: duplicate values")
        if len(spec.scenario) > 1 and len(spec.conc) > 1:
            problems.append(
                "scenario: a scenario list runs one job per scenario; set exactly one conc "
                f"(got {len(spec.conc)})"
            )
    for scenario in scenarios:
        if is_sweep(scenario, repo) and spec.conc != [1]:
            problems.append(
                f"conc: sweep scenario {scenario} searches the concurrency itself; "
                f"set conc: 1 (got {spec.conc})"
            )
        if not (repo / "scenarios" / f"{scenario}.yaml").is_file():
            problems.append(
                f"scenario: no scenarios/{scenario}.yaml; available: {', '.join(available_scenarios(repo))}"
            )
    try:
        gpus_per_node = _gpus_per_node(spec.runner, repo)
        if spec.gpus > gpus_per_node:
            problems.append(
                f"gpus: {spec.gpus} > {gpus_per_node} GPUs per node on runner {spec.runner}; pick a bigger runner"
            )
    except ServingError as error:
        problems.append(str(error))
    if problems:
        raise ServingError("\n".join(problems))
    return spec
