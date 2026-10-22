"""Serving YAML -> e2e-tests matrix rows + the rendered serving object.

    python -P -m infx.serving.to_matrix serving.yaml          # {"matrix": [...], "serving": {...}}
    python -P -m infx.serving.to_matrix --check serving.yaml  # the run title only

Rows have exactly the keys the generator emits for single-node agentic rows, so
they pass infx.workflows.benchmark_schema (extra="forbid"). Everything
arm-specific travels in "serving", which e2e-tests.yml hands to benchmark-tmpl.yml.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from infx.serving.render import render
from infx.serving.schema import REPO_ROOT, ServingError, ServingSpec, load_serving

LOCK = Path("scenarios/datasets/lock.yaml")
LOCK_FIELDS = {
    "repo": r".+/.+",
    "revision": r"[0-9a-f]{40}",
    "file": r".+",
    "sha256": r"[0-9a-f]{64}",
}


def load_lock(repo: Path = REPO_ROOT) -> dict[str, dict]:
    """File datasets pinned for scenarios (name -> repo/revision/file/sha256)."""
    path = repo / LOCK
    if not path.is_file():
        return {}
    try:
        lock = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ServingError(f"lock: {LOCK} is not YAML ({exc})".replace("\n", " ")) from None
    if not isinstance(lock, dict):
        raise ServingError(f"lock: {LOCK} must map dataset names to entries")
    for name, entry in lock.items():
        if not isinstance(entry, dict) or set(entry) != set(LOCK_FIELDS):
            raise ServingError(f"lock: {name} must have exactly {', '.join(LOCK_FIELDS)}")
        for field, pattern in LOCK_FIELDS.items():
            if not re.fullmatch(pattern, str(entry[field])):
                raise ServingError(
                    f"lock: {name}.{field} {entry[field]!r} does not match {pattern}"
                )
    return lock


def scenario_facts(scenario: str, repo: Path = REPO_ROOT) -> tuple[int, str]:
    """(profiling duration s, dataset) of scenarios/<scenario>.yaml.

    dataset = the public loader alias, or the lock key of a file dataset.
    """
    where = f"scenarios/{scenario}.yaml"
    try:
        benchmark = yaml.safe_load((repo / where).read_text())["benchmark"]
        phases, datasets = benchmark["phases"], benchmark["datasets"]
        profiling = [p for p in phases if p.get("kind", p.get("name")) == "profiling"]
        duration = int(profiling[0]["duration"]) if len(profiling) == 1 else None
        dataset = datasets[0] if isinstance(datasets, list) and len(datasets) == 1 else None
        kind = dataset.get("type") if isinstance(dataset, dict) else None
    except (
        OSError,
        yaml.YAMLError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        raise ServingError(
            f"scenario: {where} is malformed ({type(exc).__name__}: {exc})".replace("\n", " ")
        ) from None
    if duration is None:
        raise ServingError(f"scenario: {where} must have exactly one profiling phase")
    if dataset is None:
        raise ServingError(f"scenario: {where} must have exactly one dataset")
    if kind == "public":
        return duration, dataset["dataset"]
    if kind == "file":
        name = dataset.get("name")
        if name not in load_lock(repo):
            raise ServingError(
                f"scenario: {where} dataset '{name}' is not in scenarios/datasets/lock.yaml"
            )
        return duration, name
    if kind == "synthetic":
        return duration, "synthetic"
    raise ServingError(
        f"scenario: {where} dataset type must be public, file or synthetic, got {kind!r}"
    )


def is_sweep(scenario: str, repo: Path = REPO_ROOT) -> bool:
    """True when scenarios/<scenario>.yaml carries an adaptive_search sweep (top-level `sweep:`)."""
    try:
        doc = yaml.safe_load((repo / f"scenarios/{scenario}.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return False
    sweep = doc.get("sweep") if isinstance(doc, dict) else None
    return isinstance(sweep, dict) and sweep.get("type") == "adaptive_search"


def _row(spec: ServingSpec, framework: str, conc: int, duration: int, exp_name: str) -> dict:
    return {
        "image": spec.image,
        "model": spec.model,
        "model-prefix": spec.name,
        "precision": spec.precision,
        "framework": framework,
        "runner": spec.runner,
        "tp": spec.gpus,
        "pp": 1,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 1,
        "dp-attn": False,
        "spec-decoding": "none",
        "conc": conc,
        "kv-offloading": "none",
        "total-cpu-dram-gb": 0,
        "duration": duration,
        "exp-name": exp_name,
        "scenario-type": "agentic-coding",
    }


def build(text: str, repo: Path = REPO_ROOT) -> dict:
    spec = load_serving(text, repo)
    rendered = render(spec)
    scenarios = [spec.scenario] if isinstance(spec.scenario, str) else list(spec.scenario)
    multi = len(scenarios) > 1
    rows = []
    # exp-name -> per-job scenario payload: benchmark-tmpl.yml resolves the row's
    # scenario/loader/search from here; flat serving.scenario stays single-scenario only.
    scenario_map: dict[str, dict] = {}
    durations: set[int] = set()
    for scenario in scenarios:
        duration, loader = scenario_facts(scenario, repo)
        durations.add(duration)
        sweep = is_sweep(scenario, repo)
        for conc in spec.conc:
            exp_name = (
                f"{spec.name}-{scenario}" if multi else (f"{spec.name}_search" if sweep else f"{spec.name}_c{conc}")
            )
            rows.append(_row(spec, rendered["framework"], conc, duration, exp_name))
            scenario_map[exp_name] = {"scenario": scenario, "loader": loader, "search": sweep}
    first = scenario_map[rows[0]["exp-name"]]
    return {
        "matrix": rows,
        "serving": {
            "serve-cmd": rendered["serve-cmd"],
            "serve-env": rendered["serve-env"],
            "served-model-name": rendered["served-model-name"],
            # A scenario list must not leak one scenario to every job: the flat
            # fields go empty so a failed scenario-map lookup fails loudly in
            # serve_generic.sh instead of running the wrong scenario 24 times.
            "scenario": "" if multi else first["scenario"],
            "loader": "" if multi else first["loader"],
            "search": False if multi else first["search"],
            "scenario-map": scenario_map,
            "yaml": text,
            "router-image": rendered["router-image"],
            "router-cmd": rendered["router-cmd"],
            "router-env": rendered["router-env"],
            "router-profile": rendered["router-profile"],
            "router-metrics-port": rendered["router-metrics-port"],
            "replicas": str(spec.replicas),
            "storage-image": rendered["storage-image"],
            "storage-cmd": rendered["storage-cmd"],
            "storage-env": rendered["storage-env"],
        },
        "title": (
            f"serving {spec.name} "
            f"{scenarios[0] if not multi else f'{len(scenarios)} scenarios'} "
            f"({min(durations) if len(durations) == 1 else 'mixed'}s) "
            f"c{','.join(str(conc) for conc in spec.conc)} {spec.runner}"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate and print the run title only"
    )
    args = parser.parse_args(argv)
    try:
        try:
            text = args.path.read_text()
        except OSError as error:
            raise ServingError(f"serving: cannot read {args.path} ({error.strerror})") from error
        result = build(text)
    except ServingError as error:
        for line in str(error).splitlines():
            print(f"ERROR {line}", file=sys.stderr)
        return 2
    if args.check:
        print(result["title"])
    else:
        print(json.dumps({"matrix": result["matrix"], "serving": result["serving"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
