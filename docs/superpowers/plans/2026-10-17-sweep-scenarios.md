# Sweep Scenarios (ISL/OSL/cache grid + in-job SLO search) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dispatch a serving YAML against one of 24 synthetic scenarios and get back the highest concurrency that still meets a configurable SLO, found by AIPerf's adaptive search inside a single CI job.

**Architecture:** 24 generated `scenarios/sweep-*.yaml` files (AIPerf synthetic dataset + `adaptive_search` sweep, recipe `max-concurrency-under-sla`, planner `smooth_isotonic`). The serving stack learns that such a scenario has no fixed `conc`, no dataset download, and an adaptive replay command. A new result writer turns `search_history.json` + the winning probe's metrics into one result JSON; a collector builds the 24-row table.

**Tech Stack:** Python 3.12 (Pydantic, PyYAML, pytest), Bash (`benchmark_lib.sh`), AIPerf pinned at `utils/aiperf` @ `faf3f8b3`.

**Spec:** `docs/superpowers/specs/2026-10-17-sweep-scenarios-design.md` (committed `08be1aaa9`).

## Global Constraints

- Grid is exactly ISL {8192, 32768, 65536, 131072} × OSL {512, 1024} × cache {0, 50, 90}% = 24 scenarios; names `sweep-isl{8k,32k,64k,128k}-osl{512,1k}-cache{0,50,90}`.
- Search: concurrency range `[1, 512]`, precision 5%, max 16 iterations, 1 run per probe, planner `smooth_isotonic`, recipe `max-concurrency-under-sla`.
- Probe: duration 300 s, floor 100 completed requests, first-probe warmup 60 s. Seed 42.
- Default SLO: `time_to_first_token:avg:lt:8000` and `inter_token_latency:p90:lt:50`, plus error rate below 1%. SLO tuple format `metric:stat:op:threshold`, stat in {avg,p50,p90,p95,p99}, op in {lt,le,gt,ge}.
- Terminal states are result fields, never a failed job: `ok`, `cliff`, `no_pass_in_range`, `no_failure_in_range`, `max_iterations`.
- Bash (AGENTS.md): no `${VAR:-default}` for caller config, validate with `check_env_vars`, no `set -u`.
- Tests (AGENTS.md "Test quality"): run real code on controlled inputs, expected values worked out by hand; never grep source text or pin checked-in config. A generated scenario file is verified by parsing it and checking values against hand-computed numbers, not by its text.
- Python under `infx/`: typed signatures, Ruff (line length 100). Before pushing Python: commands in `docs/testing.md#python-lint-and-formatting`.
- Commit subjects conventional English; bodies include a Chinese translation. Stage only the files of the task (never `git add -A`).
- Any change affecting benchmark behavior needs an append-only `perf-changelog.yaml` entry (Task 8). Docs ship with `_zh.md`.
- No `git push` or run dispatch without asking the user first (Task 8 hands this back).

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `docs/superpowers/plans/2026-10-17-sweep-scenarios-spike-notes.md` | Create (Task 1) | Verified AIPerf facts that later tasks consume |
| `utils/fixtures/sweep/` | Create (Task 1) | Real captured `search_history.json` + sweep aggregate from the mock run |
| `infx/serving/slo.py` | Create (Task 2) | Parse/validate SLO tuples, emit AIPerf flags |
| `utils/test_serving_slo.py` | Create (Task 2) | SLO tests |
| `utils/serving/gen_sweep_scenarios.py` | Create (Task 3) | The grid, and the writer of the 24 scenario files |
| `scenarios/sweep-*.yaml` | Create (Task 3, generated) | 24 scenarios |
| `utils/test_sweep_scenarios.py` | Create (Task 3) | Generator tests |
| `infx/serving/to_matrix.py` | Modify (Task 4) | Synthetic dataset + sweep scenario facts, `conc` rule, exp-name |
| `infx/serving/schema.py` | Modify (Task 4) | Cross-check `conc` against sweep scenarios |
| `utils/test_serving_to_matrix.py`, `utils/test_serving_schema.py` | Modify (Task 4) | New cases |
| `benchmarks/benchmark_lib.sh` | Modify (Task 5) | `build_replay_cmd_yaml` adaptive branch |
| `benchmarks/single_node/agentic/serve_generic.sh` | Modify (Task 5) | Skip dataset fetch for synthetic, route to sweep writer |
| `utils/test_serving_replay.py` | Modify (Task 5) | Command-shape tests |
| `infx/results/sweep.py` | Create (Task 6) | `search_history.json` → result JSON |
| `utils/test_sweep_result.py` | Create (Task 6) | Writer tests on the captured fixture and canned states |
| `utils/serving/collect_sweep_table.py` | Create (Task 7) | Many results → the PDF-style table |
| `utils/test_sweep_table.py` | Create (Task 7) | Collector tests |
| `utils/serving/sweep_dispatch.sh` | Create (Task 8) | Expand one serving YAML over scenarios |
| `perf-changelog.yaml`, `docs/serving-yaml-procedures.md` + `_zh.md` | Modify (Task 8) | Changelog and docs |

---

### Task 1: Spike — settle AIPerf facts locally against the mock server

This task writes no product code. It answers the spec's open items on a laptop, and its outputs (notes, one verified scenario, fixtures) are consumed by Tasks 3, 4, 5, 6. Everything later that says "per spike notes" refers to this file.

**Files:**
- Create: `docs/superpowers/plans/2026-10-17-sweep-scenarios-spike-notes.md`
- Create: `scenarios/sweep-isl8k-osl512-cache50.yaml` (hand-written, verified)
- Create: `utils/fixtures/sweep/search_history.json`, `utils/fixtures/sweep/profile_export_aiperf_sweep.json` (copied from the mock run)

**Interfaces:**
- Produces: spike notes with these named answers (later tasks quote them): `SWEEP_KEY` (where `sweep:` sits: `benchmark.sweep` or top-level), `ISL_INCLUDES_PREFIX` (true/false), `PREFIX_FIELD` (`prefix_prompts.shared_system_length` or pool form), `SLO_OVERRIDE` (does CLI `--search-sla` replace or add to YAML `sla_filters`), `ARTIFACT_LAYOUT` (paths of `search_history.json` and per-probe metrics), `ENDPOINT` (chat vs completions), `CACHE0_FIELD` (the cache-bust setting that removes prefix sharing), `MIN_REQUESTS_FIELD` (how the 100-request floor is expressed).

- [ ] **Step 1: Start the AIPerf mock server**

Look in `utils/aiperf/tests/aiperf_mock_server/` (its `README`/`__main__`) and start it on a free port, serving an OpenAI-compatible `/v1/chat/completions` and `/metrics`. Record the exact start command in the notes. Run from `utils/aiperf` with `uv run`.

- [ ] **Step 2: Write the reference scenario**

Create `scenarios/sweep-isl8k-osl512-cache50.yaml`, starting from `scenarios/agentx-cc-062126.yaml` as the style reference. Replace the dataset with a synthetic one and add the sweep:

```yaml
schemaVersion: "2.0"
benchmark:
  scenario: inferencex-agentx-mvp   # change only if the spike shows this scenario rejects synthetic data; note the finding
  unsafeOverride: true
  endpoint:
    type: chat
    path: /v1/chat/completions
    streaming: true
    useServerTokenCount: true
  datasets:
  - name: main
    type: synthetic
    entries: 1000
    randomSeed: 42
    isl: 4096             # 50% of 8192 as the unique part, if ISL_INCLUDES_PREFIX is false
    osl: 512
    prefix_prompts:
      shared_system_length: 4096
  phases:
  - name: profiling
    kind: profiling
    type: concurrency
    duration: 300.0
    gracePeriod: 30.0
    failedRequestThreshold: 0.1
  sweep:
    type: adaptive_search
    planner: smooth_isotonic
    search_space:
    - {path: phases.profiling.concurrency, lo: 1, hi: 512, kind: int}
    sla_filters:
    - {metric_tag: time_to_first_token, stat: avg, op: lt, threshold: 8000.0}
    - {metric_tag: inter_token_latency, stat: p90, op: lt, threshold: 50.0}
    max_iterations: 16
    sla_precision: normal
randomSeed: 42
```

Adjust field names until `aiperf profile --config <file> --url http://localhost:<port> --model <mock-model> --tokenizer <small-hf-tokenizer> --no-gpu-telemetry` validates and runs. Each field you had to change from the above is a finding for the notes.

- [ ] **Step 2b: Run it and let the search finish**

Run the scenario against the mock server. Confirm a `search_history.json` is written with a non-null `convergence_reason` and a `boundary_summary`. If the mock server has no capacity limit (everything passes), start it with a latency model or a concurrency cap so the SLO actually breaks (check the mock server's options) — the fixture must contain both a `feasible_max` and an `infeasible_min`.

- [ ] **Step 3: Answer each named item in the notes**

Write the notes file with one section per name from **Interfaces**, each stating the verified answer and how it was verified (command + what you saw). In particular:

- `ISL_INCLUDES_PREFIX`: request with ISL 4096 + shared 4096; read the recorded `input_sequence_length` in the probe's `profile_export.jsonl` (mean). 8192 means the prefix is added on top of `isl`; 4096 means `isl` includes it.
- `CACHE0_FIELD`: run once with a cache-bust target (`benchmark.datasets[0].…` or wherever `CacheBustConfig.target` lives, see `utils/aiperf/src/aiperf/config/dataset/content.py:61`) and check that two requests share no prefix tokens (compare the mock server's received prompts, or its prefix-sharing metric if it has one).
- `SLO_OVERRIDE`: run once more adding `--search-sla "time_to_first_token:avg:lt:1:..."`-style flag on the CLI (a deliberately different threshold) and read `config.sla_filters` in the resulting `search_history.json`.
- `MIN_REQUESTS_FIELD`: find the config field/flag for "at least N completed requests per probe" (`sla_precision` maps to `n_requests_per_probe` per `utils/aiperf/docs/sweeping/bayesian-optimization.md`; decide whether `sla_precision: normal` (1000 requests) or an explicit count meets the spec's floor of 100 within 300 s). If AIPerf has no 100-request floor with a 300 s cap, record the closest behavior and flag it.
- `ARTIFACT_LAYOUT`: `find <artifact_dir> -name '*.json' | sort` after the run; record the paths of `search_history.json`, the sweep aggregate JSON, and each probe's `profile_export_aiperf.json`.

- [ ] **Step 4: Copy fixtures**

```bash
mkdir -p utils/fixtures/sweep
cp <artifact_dir>/search_history.json utils/fixtures/sweep/search_history.json
cp <artifact_dir>/sweep_aggregate/profile_export_aiperf_sweep.json utils/fixtures/sweep/profile_export_aiperf_sweep.json
```
(Use the paths found in `ARTIFACT_LAYOUT`.) Fixtures stay small (<200 KB); if the aggregate is larger, trim per-request detail but keep its key structure.

- [ ] **Step 5: Decide whether to continue**

If the notes show any of these, stop and report to the user before Task 2:
(a) synthetic data is rejected by scenario `inferencex-agentx-mvp` and the alternative needs fork changes in `utils/aiperf`;
(b) the `sweep` block cannot sit in a `--config` YAML alongside `type: concurrency`;
(c) no way to express the request floor.
Otherwise continue.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-10-17-sweep-scenarios-spike-notes.md scenarios/sweep-isl8k-osl512-cache50.yaml utils/fixtures/sweep
git commit -m "docs(sweep): spike notes, verified reference scenario, fixtures

中文: sweep 场景的 AIPerf 事实核查笔记、已验证的参考场景和夹具。"
```

---

### Task 2: SLO parser

**Files:**
- Create: `infx/serving/slo.py`
- Test: `utils/test_serving_slo.py`

**Interfaces:**
- Produces: `class SLOError(ValueError)`; `@dataclass(frozen=True) class SLO` with `metric: str, stat: str, op: str, threshold: float`; `SLO.tuple() -> str`; `parse_slo(text: str) -> list[SLO]`; CLI `python -m infx.serving.slo "<text>"` printing one `metric:stat:op:threshold` per line, exit 2 and `ERROR slo: ...` on stderr for bad input.
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from infx.serving.slo import SLO, SLOError, parse_slo


def test_two_tuples_mixed_stats():
    got = parse_slo("time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50")
    assert got == [
        SLO("time_to_first_token", "avg", "lt", 8000.0),
        SLO("inter_token_latency", "p90", "lt", 50.0),
    ]


def test_tuple_string_round_trips_without_trailing_zeros():
    assert SLO("inter_token_latency", "p90", "lt", 50.0).tuple() == "inter_token_latency:p90:lt:50"
    assert SLO("time_to_first_token", "avg", "le", 0.5).tuple() == "time_to_first_token:avg:le:0.5"


def test_whitespace_and_trailing_separator_tolerated():
    assert len(parse_slo(" time_to_first_token:p99:lt:200 ; ")) == 1


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("", "empty"),
        ("   ;  ", "empty"),
        ("time_to_first_token:avg:lt", "metric:stat:op:threshold"),
        ("time_to_first_token:median:lt:8000", "stat"),
        ("time_to_first_token:avg:<:8000", "op"),
        ("time_to_first_token:avg:lt:fast", "threshold"),
        ("time_to_first_token:avg:lt:-5", "threshold"),
        ("time_to_first_token:avg:lt:nan", "threshold"),
        ("Time To First:avg:lt:8000", "metric"),
    ],
)
def test_rejects_malformed(text, fragment):
    with pytest.raises(SLOError, match=fragment):
        parse_slo(text)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest utils/test_serving_slo.py -v`
Expected: FAIL (`ModuleNotFoundError: infx.serving.slo`).

- [ ] **Step 3: Implement**

```python
"""SLO tuples for sweep scenarios: ``metric:stat:op:threshold`` joined by ``;``.

    python -m infx.serving.slo "time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"

prints one tuple per line (the form AIPerf's ``--search-sla`` takes), or exits 2.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass

STATS = ("avg", "p50", "p90", "p95", "p99")
OPS = ("lt", "le", "gt", "ge")
_METRIC = re.compile(r"[a-z][a-z0-9_]*")


class SLOError(ValueError):
    pass


@dataclass(frozen=True)
class SLO:
    metric: str
    stat: str
    op: str
    threshold: float

    def tuple(self) -> str:
        return f"{self.metric}:{self.stat}:{self.op}:{self.threshold:g}"


def parse_slo(text: str) -> list[SLO]:
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if not parts:
        raise SLOError("slo: empty; give metric:stat:op:threshold[;...]")
    slos = []
    for part in parts:
        fields = part.split(":")
        if len(fields) != 4:
            raise SLOError(f"slo: {part!r} must be metric:stat:op:threshold")
        metric, stat, op, raw = (field.strip() for field in fields)
        if not _METRIC.fullmatch(metric):
            raise SLOError(f"slo: metric {metric!r} must be a lowercase AIPerf metric tag")
        if stat not in STATS:
            raise SLOError(f"slo: stat {stat!r} must be one of {', '.join(STATS)}")
        if op not in OPS:
            raise SLOError(f"slo: op {op!r} must be one of {', '.join(OPS)}")
        try:
            threshold = float(raw)
        except ValueError:
            raise SLOError(f"slo: threshold {raw!r} is not a number") from None
        if not math.isfinite(threshold) or threshold <= 0:
            raise SLOError(f"slo: threshold {raw!r} must be a positive finite number")
        slos.append(SLO(metric, stat, op, threshold))
    return slos


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m infx.serving.slo '<tuples>'", file=sys.stderr)
        return 2
    try:
        slos = parse_slo(argv[0])
    except SLOError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    for slo in slos:
        print(slo.tuple())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest utils/test_serving_slo.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

Run the Ruff commands from `docs/testing.md#python-lint-and-formatting` on `infx/serving/slo.py`.

```bash
git add infx/serving/slo.py utils/test_serving_slo.py
git commit -m "feat(serving): SLO tuple parser for sweep scenarios

中文: 为 sweep 场景新增 SLO 元组解析器。"
```

---

### Task 3: Scenario generator and the 24 files

**Files:**
- Create: `utils/serving/gen_sweep_scenarios.py`
- Create: `scenarios/sweep-*.yaml` (generated; `sweep-isl8k-osl512-cache50.yaml` is regenerated over the Task 1 file and must come out equivalent)
- Test: `utils/test_sweep_scenarios.py`

**Interfaces:**
- Consumes: spike notes `ISL_INCLUDES_PREFIX`, `PREFIX_FIELD`, `CACHE0_FIELD`, `MIN_REQUESTS_FIELD`, `SWEEP_KEY`; `infx.serving.slo.SLO` and `DEFAULT_SLO` strings from the Global Constraints.
- Produces: `GRID_ISL: dict[str,int]`, `GRID_OSL: dict[str,int]`, `GRID_CACHE: tuple[int,...]`; `scenario_name(isl_label, osl_label, cache) -> str`; `shared_prefix_tokens(isl, cache) -> int`; `build_scenario(isl, osl, cache) -> dict` (the YAML mapping); `grid() -> list[tuple[str,int,str,int,int]]` (`isl_label, isl, osl_label, osl, cache`); CLI `python utils/serving/gen_sweep_scenarios.py` writes all 24 into `scenarios/`.

- [ ] **Step 1: Write the failing tests**

Expected numbers are worked out by hand: 65536 × 90% = 58982.4 → 58982; 8192 × 50% = 4096; 131072 × 90% = 117964.8 → 117965; 32768 × 0% = 0.

```python
import yaml

from utils.serving.gen_sweep_scenarios import (
    GRID_CACHE, build_scenario, grid, scenario_name, shared_prefix_tokens,
)


def test_prefix_tokens_rounding():
    assert shared_prefix_tokens(65536, 90) == 58982
    assert shared_prefix_tokens(8192, 50) == 4096
    assert shared_prefix_tokens(131072, 90) == 117965
    assert shared_prefix_tokens(32768, 0) == 0


def test_grid_is_the_24_chosen_points():
    points = grid()
    assert len(points) == 24
    assert {p[0] for p in points} == {"8k", "32k", "64k", "128k"}
    assert {p[2] for p in points} == {"512", "1k"}
    assert {p[4] for p in points} == set(GRID_CACHE) == {0, 50, 90}
    assert len({scenario_name(p[0], p[2], p[4]) for p in points}) == 24


def test_name():
    assert scenario_name("64k", "1k", 90) == "sweep-isl64k-osl1k-cache90"


def test_built_scenario_carries_grid_values_and_search_budget():
    doc = yaml.safe_load(yaml.safe_dump(build_scenario(65536, 1024, 90)))
    benchmark = doc["benchmark"]
    sweep = benchmark["sweep"]
    assert sweep["search_space"][0]["lo"] == 1
    assert sweep["search_space"][0]["hi"] == 512
    assert sweep["max_iterations"] == 16
    assert sweep["planner"] == "smooth_isotonic"
    filters = {(f["metric_tag"], f["stat"], f["op"], f["threshold"]) for f in sweep["sla_filters"]}
    assert filters == {
        ("time_to_first_token", "avg", "lt", 8000.0),
        ("inter_token_latency", "p90", "lt", 50.0),
    }
    dataset = benchmark["datasets"][0]
    assert dataset["osl"] == 1024
    assert dataset["randomSeed"] == 42
    # ISL accounting per spike notes (ISL_INCLUDES_PREFIX): total prompt tokens must equal 65536
    prefix = dataset["prefix_prompts"]["shared_system_length"]
    assert prefix == 58982
    assert prefix + dataset["isl"] == 65536 or dataset["isl"] == 65536  # adapt to ISL_INCLUDES_PREFIX in step 3


def test_cache_zero_has_no_shared_prefix():
    dataset = build_scenario(32768, 512, 0)["benchmark"]["datasets"][0]
    assert "prefix_prompts" not in dataset
```

Note for step 3: the `ISL_INCLUDES_PREFIX` line above is a guard written before the spike answer is known; once Task 1 says which it is, replace the `or` assertion with the single correct equality and add the cache-0 cache-bust assertion per `CACHE0_FIELD` (for cache 0, the dataset or benchmark must carry the cache-bust target from the notes, and for cache 50/90 it must not).

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest utils/test_sweep_scenarios.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

Write `utils/serving/gen_sweep_scenarios.py` so `build_scenario` returns the same structure as the verified file from Task 1, parameterized by the three axes. Skeleton (fill `dataset`/`bust` fields exactly as the spike notes record):

```python
"""Write scenarios/sweep-isl*-osl*-cache*.yaml: the grid lives only in this file.

    python utils/serving/gen_sweep_scenarios.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
GRID_ISL = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072}
GRID_OSL = {"512": 512, "1k": 1024}
GRID_CACHE = (0, 50, 90)
DEFAULT_SLO = (
    {"metric_tag": "time_to_first_token", "stat": "avg", "op": "lt", "threshold": 8000.0},
    {"metric_tag": "inter_token_latency", "stat": "p90", "op": "lt", "threshold": 50.0},
)
HEADER = (
    "# Generated by utils/serving/gen_sweep_scenarios.py - do not edit by hand.\n"
    "# Synthetic ISL/OSL/cache-hit workload; AIPerf searches for the highest concurrency\n"
    "# meeting the SLO (docs/superpowers/specs/2026-10-17-sweep-scenarios-design.md).\n"
)


def scenario_name(isl_label: str, osl_label: str, cache: int) -> str:
    return f"sweep-isl{isl_label}-osl{osl_label}-cache{cache}"


def shared_prefix_tokens(isl: int, cache: int) -> int:
    return round(isl * cache / 100)


def grid() -> list[tuple[str, int, str, int, int]]:
    return [
        (il, iv, ol, ov, cache)
        for il, iv in GRID_ISL.items()
        for ol, ov in GRID_OSL.items()
        for cache in GRID_CACHE
    ]


def build_scenario(isl: int, osl: int, cache: int) -> dict:
    prefix = shared_prefix_tokens(isl, cache)
    dataset: dict = {
        "name": "main",
        "type": "synthetic",
        "entries": 1000,
        "randomSeed": 42,
        "isl": isl - prefix,  # replace per spike ISL_INCLUDES_PREFIX
        "osl": osl,
    }
    if prefix:
        dataset["prefix_prompts"] = {"shared_system_length": prefix}  # per spike PREFIX_FIELD
    # cache == 0: add the cache-bust target from spike CACHE0_FIELD here
    return {
        "schemaVersion": "2.0",
        "benchmark": {
            # ... every key copied from the verified Task 1 scenario, with:
            "datasets": [dataset],
            "sweep": {
                "type": "adaptive_search",
                "planner": "smooth_isotonic",
                "search_space": [
                    {"path": "phases.profiling.concurrency", "lo": 1, "hi": 512, "kind": "int"}
                ],
                "sla_filters": [dict(f) for f in DEFAULT_SLO],
                "max_iterations": 16,
                # request floor per spike MIN_REQUESTS_FIELD
            },
        },
        "randomSeed": 42,
    }


def main() -> None:
    for il, iv, ol, ov, cache in grid():
        path = REPO / "scenarios" / f"{scenario_name(il, ol, cache)}.yaml"
        path.write_text(HEADER + yaml.safe_dump(build_scenario(iv, ov, cache), sort_keys=False))
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
```

The comments marked "per spike" are the only places the spike notes feed code; each must be replaced with the literal value before the tests pass. Copy `schemaVersion`, `endpoint`, `phases`, and the other `benchmark` keys verbatim from the Task 1 scenario.

- [ ] **Step 4: Generate and equivalence-check**

```bash
python utils/serving/gen_sweep_scenarios.py
git diff --stat scenarios/sweep-isl8k-osl512-cache50.yaml
python -m pytest utils/test_sweep_scenarios.py -v
```
Expected: 24 files printed; tests PASS. The regenerated 8k/512/50 file must equal the Task 1 file apart from the header comment; if not, the generator is wrong, not the spike file.

- [ ] **Step 5: Validate every scenario with AIPerf**

Run from `utils/aiperf` for each generated file, against the mock server from Task 1 with a 5 s probe override if available, or with AIPerf's config validation entrypoint (`aiperf profile --config <file> ... --help`-style dry validation if one exists; otherwise a 1-iteration run). Record the command in the spike notes. Goal: all 24 parse; only 128k files are slow.

- [ ] **Step 6: Commit**

```bash
git add utils/serving/gen_sweep_scenarios.py utils/test_sweep_scenarios.py scenarios/sweep-*.yaml
git commit -m "feat(scenarios): 24 synthetic sweep scenarios and their generator

中文: 新增 24 个合成 sweep 场景及其生成器。"
```

---

### Task 4: Serving stack accepts sweep scenarios

**Files:**
- Modify: `infx/serving/to_matrix.py` (`scenario_facts`, `_row`, `build`)
- Modify: `infx/serving/schema.py` (cross-check of `conc`)
- Test: `utils/test_serving_to_matrix.py`, `utils/test_serving_schema.py`

**Interfaces:**
- Consumes: scenario files from Task 3; spike notes `SWEEP_KEY`.
- Produces: `scenario_facts(scenario, repo) -> tuple[int, str]` now returns loader `"synthetic"` for synthetic datasets; new `is_sweep(scenario: str, repo: Path = REPO_ROOT) -> bool`; matrix row `exp-name` is `f"{spec.name}_search"` for sweep scenarios (else unchanged `f"{spec.name}_c{conc}"`); serving YAML for a sweep scenario must set `conc: 1` (validation error otherwise).

Decision recorded here: the schema requires a non-empty `conc`. For sweep scenarios `conc` carries no information, so exactly `[1]` is accepted and anything else is rejected with an explanatory error, instead of silently ignoring user input.

- [ ] **Step 1: Read the current tests for style**

Open `utils/test_serving_to_matrix.py` and `utils/test_serving_schema.py`; reuse their helpers for building a throwaway repo/scenario dir and a minimal serving YAML (do not invent a new fixture pattern).

- [ ] **Step 2: Write the failing tests**

Using the existing helpers, add cases (assert on returned values, expected numbers by hand):

```python
def test_synthetic_scenario_loader_is_synthetic(tmp_repo_with_sweep_scenario):
    duration, loader = scenario_facts("sweep-isl8k-osl512-cache50", tmp_repo_with_sweep_scenario)
    assert (duration, loader) == (300, "synthetic")


def test_is_sweep_distinguishes_scenarios(tmp_repo_with_sweep_scenario):
    assert is_sweep("sweep-isl8k-osl512-cache50", tmp_repo_with_sweep_scenario) is True
    assert is_sweep("agentx-cc-062126-smoke", tmp_repo_with_sweep_scenario) is False


def test_sweep_matrix_row_uses_search_exp_name_and_conc_one(sweep_serving_yaml, tmp_repo_with_sweep_scenario):
    result = build(sweep_serving_yaml, tmp_repo_with_sweep_scenario)
    assert [row["conc"] for row in result["matrix"]] == [1]
    assert result["matrix"][0]["exp-name"] == "demo_search"
    assert result["serving"]["loader"] == "synthetic"
```

and in `test_serving_schema.py`: a sweep scenario with `conc: 8` yields an error line starting `conc:` and mentioning `sweep`; with `conc: 1` no error; a non-sweep scenario with `conc: 8` is unchanged.

`tmp_repo_with_sweep_scenario` should copy one generated scenario and one `agentx-*-smoke` scenario into a temp `scenarios/` with a minimal `lock.yaml`, matching how existing tests build theirs.

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest utils/test_serving_to_matrix.py utils/test_serving_schema.py -v`
Expected: new tests FAIL (`is_sweep` missing / `dataset type must be public or file`).

- [ ] **Step 4: Implement `to_matrix.py`**

In `scenario_facts`, before the final `raise`, add:

```python
    if kind == "synthetic":
        return duration, "synthetic"
```
and change the error text to `dataset type must be public, file or synthetic, got {kind!r}`.

Add:

```python
def is_sweep(scenario: str, repo: Path = REPO_ROOT) -> bool:
    """True when scenarios/<scenario>.yaml carries an adaptive_search sweep (key path per spike SWEEP_KEY)."""
    try:
        benchmark = yaml.safe_load((repo / f"scenarios/{scenario}.yaml").read_text())["benchmark"]
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return False
    sweep = benchmark.get("sweep")
    return isinstance(sweep, dict) and sweep.get("type") == "adaptive_search"
```
(If the spike found `sweep:` at the top level instead of under `benchmark:`, read `doc.get("sweep")` there; this is the single place.)

In `_row`, add a parameter `sweep: bool` and set `"exp-name": f"{spec.name}_search" if sweep else f"{spec.name}_c{conc}"`. In `build`, pass `sweep=is_sweep(spec.scenario, repo)`.

- [ ] **Step 5: Implement the `conc` rule in `schema.py`**

Near the existing `if len(set(spec.conc)) != len(spec.conc):` block in the validation function, add:

```python
    from infx.serving.to_matrix import is_sweep  # local import: to_matrix imports this module

    if is_sweep(spec.scenario, repo) and spec.conc != [1]:
        problems.append(
            "conc: this scenario searches the concurrency itself; set conc: 1 "
            f"(got {spec.conc})"
        )
```
If `to_matrix` already imports `schema` at module level (it does), keep the import inside the function as shown to avoid a cycle.

- [ ] **Step 6: Run tests**

Run: `python -m pytest utils/test_serving_to_matrix.py utils/test_serving_schema.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 7: Commit**

```bash
git add infx/serving/to_matrix.py infx/serving/schema.py utils/test_serving_to_matrix.py utils/test_serving_schema.py
git commit -m "feat(serving): accept synthetic sweep scenarios in matrix and schema

中文: serving 栈支持合成 sweep 场景(矩阵与校验)。"
```

---

### Task 5: Adaptive replay command and launcher routing

**Files:**
- Modify: `benchmarks/benchmark_lib.sh` (`build_replay_cmd_yaml`, around line 3550)
- Modify: `benchmarks/single_node/agentic/serve_generic.sh`
- Test: `utils/test_serving_replay.py`

**Interfaces:**
- Consumes: `python -m infx.serving.slo` (Task 2); spike notes `SLO_OVERRIDE`, `ARTIFACT_LAYOUT`.
- Produces: env contract `SERVING_SEARCH=1` (set by the launcher/workflow from `is_sweep`) and optional `IX_SLO` (tuple string). With `SERVING_SEARCH=1`, `build_replay_cmd_yaml` omits `--concurrency` and appends one `--search-sla <tuple>` per tuple when `IX_SLO` is set. `serve_generic.sh` skips the dataset fetch when `SERVING_LOADER=synthetic`.

- [ ] **Step 1: Find how env reaches the job**

`grep -rn "SERVING_LOADER" .github runners infx utils --include=* -l` and follow `loader` from `infx/serving/to_matrix.py` (`"loader": loader`) to where the workflow exports `SERVING_LOADER`. `SERVING_SEARCH` must be exported the same way: add `"search": is_sweep(...)` to the `serving` dict in `to_matrix.build`, then export it where `SERVING_LOADER` is exported. Add a matching test in `utils/test_serving_to_matrix.py` (`result["serving"]["search"] is True` for the sweep scenario, `False` otherwise) and a test in `utils/test_serving_workflows.py` if that file already checks the exported variables by running the real workflow-expansion code (match its pattern; if it only greps workflow text, skip the test per AGENTS.md).

- [ ] **Step 2: Write the failing command-shape tests**

In `utils/test_serving_replay.py`, using its existing `run()` helper and `ENV`:

```python
def test_search_mode_drops_concurrency_and_adds_slo_flags():
    out = run(r'''
        export SERVING_SEARCH=1
        export IX_SLO="time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    assert " --concurrency " not in out.stdout
    assert " --search-sla time_to_first_token:avg:lt:8000" in out.stdout
    assert " --search-sla inter_token_latency:p90:lt:50" in out.stdout


def test_search_mode_without_slo_override_adds_no_sla_flags():
    out = run(r'''
        export SERVING_SEARCH=1
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert out.returncode == 0, out.stderr
    assert "--search-sla" not in out.stdout
    assert " --concurrency " not in out.stdout


def test_search_mode_rejects_malformed_slo_before_any_run():
    out = run(r'''
        export SERVING_SEARCH=1
        export IX_SLO="time_to_first_token:avg:lt"
        build_replay_cmd_yaml /tmp/r
    ''')
    assert out.returncode != 0
    assert "metric:stat:op:threshold" in out.stderr


def test_non_search_mode_unchanged():
    out = run(r'''
        export SERVING_SEARCH=0
        build_replay_cmd_yaml /tmp/r
        printf '%s' "$REPLAY_CMD"
    ''')
    assert " --concurrency 8" in out.stdout
```

`ENV` in that file sets `CONC=8`; for search mode the function must not require it, so add a test that unsets `CONC` and still succeeds (replace `check_env_vars ... CONC ...` per Step 3).

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest utils/test_serving_replay.py -v`
Expected: new tests FAIL.

- [ ] **Step 4: Implement in `build_replay_cmd_yaml`**

Change the first line from `check_env_vars INFMAX_CONTAINER_WORKSPACE MODEL PORT CONC SERVING_SCENARIO AIPERF_CLI` to validate the common set, then CONC only on the fixed path:

```bash
    check_env_vars INFMAX_CONTAINER_WORKSPACE MODEL PORT SERVING_SCENARIO AIPERF_CLI SERVING_SEARCH
```
and replace the line `REPLAY_CMD+=" --concurrency $CONC"` with:

```bash
    if [ "$SERVING_SEARCH" = "1" ]; then
        # The scenario's adaptive_search drives concurrency; an optional SLO override
        # replaces the scenario default (semantics per spike notes SLO_OVERRIDE).
        if [ -n "${IX_SLO:-}" ]; then
            local slo_tuples
            slo_tuples="$("$AIPERF_PYTHON" -I -m infx.serving.slo "$IX_SLO")" || return 1
            local tuple
            while IFS= read -r tuple; do
                REPLAY_CMD+=" --search-sla $tuple"
            done <<< "$slo_tuples"
        fi
    else
        check_env_vars CONC
        REPLAY_CMD+=" --concurrency $CONC"
    fi
```
The `python -m infx.serving.slo` call must run with the repo root importable: wrap as `(cd "$INFMAX_CONTAINER_WORKSPACE" && "$AIPERF_PYTHON" -m infx.serving.slo "$IX_SLO")` exactly like the other `-m infx...` calls in this file, and drop `-I` (it would hide the repo from `sys.path`). If `SLO_OVERRIDE` in the spike notes says CLI flags append rather than replace, the override must instead be written into the YAML — then implement that in this branch and adjust the tests accordingly.

`SERVING_SEARCH` must be exported by every caller of `build_replay_cmd_yaml` (the other benchmark scripts); set it to `0` where they call it so `check_env_vars` passes: `grep -rn "build_replay_cmd_yaml" benchmarks runners` and add `export SERVING_SEARCH=0` to each non-generic caller. Update the ENV block in `utils/test_serving_replay.py` to export `SERVING_SEARCH=0` as the default.

- [ ] **Step 5: Implement in `serve_generic.sh`**

Add `SERVING_SEARCH` to the `check_env_vars` line. Wrap the dataset block:

```bash
if [[ "$SERVING_LOADER" == synthetic ]]; then
    echo "scenario dataset: synthetic, nothing to fetch"
else
    custom_rc=0
    is_custom_dataset "$SERVING_LOADER" || custom_rc=$?
    case "$custom_rc" in
        # (existing case body unchanged)
    esac
fi
```
Keep the existing `$AIPERF_HF_CLI download "$MODEL"` line outside the `if`. The `CONC` variable is no longer needed for sweep runs, so drop `CONC` from that script's `check_env_vars` only if the workflow still exports it (it does, `1`), otherwise leave it.

- [ ] **Step 6: Run tests**

Run: `python -m pytest utils/test_serving_replay.py utils/test_serving_to_matrix.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/benchmark_lib.sh benchmarks/single_node/agentic/serve_generic.sh utils/test_serving_replay.py infx/serving/to_matrix.py utils/test_serving_to_matrix.py <exported-env files found in step 1>
git commit -m "feat(serving): adaptive replay command and launcher routing for sweep scenarios

中文: sweep 场景的自适应回放命令与启动器路由。"
```

---

### Task 6: Result writer for a search run

**Files:**
- Create: `infx/results/sweep.py`
- Modify: `benchmarks/benchmark_lib.sh` (`run_agentic_replay_and_write_outputs`: when `SERVING_SEARCH=1` call the sweep writer instead of `process_agentic_result`, and tolerate the non-per-request artifact layout)
- Test: `utils/test_sweep_result.py`

**Interfaces:**
- Consumes: fixtures from Task 1 (`utils/fixtures/sweep/*`), spike notes `ARTIFACT_LAYOUT`.
- Produces: `summarize_search(history: dict) -> dict` returning keys `state, max_conc, infeasible_min, binding_constraint, boundary_type, convergence_reason, iterations`; `probe_metrics(sweep_aggregate: dict, conc: int) -> dict` returning `rps, output_tps, tpm, ttft_avg_ms, ttft_p90_ms, itl_avg_ms, itl_p90_ms`; `build_sweep_result(history, sweep_aggregate, env) -> dict`; CLI `python -m infx.results.sweep` reading `RESULT_DIR`, `RESULT_FILENAME`, `AGENTIC_OUTPUT_DIR` like `process_agentic_result` and writing `<AGENTIC_OUTPUT_DIR>/<RESULT_FILENAME>.json`.

- [ ] **Step 1: Write the failing tests**

State tests use small canned dicts shaped like `docs/api/search-history.md`. Probe-metric tests use the real fixture; expected numbers are read by hand from the fixture file and written as literals.

```python
import json
from pathlib import Path

from infx.results.sweep import probe_metrics, summarize_search

FIX = Path(__file__).parent / "fixtures" / "sweep"


def hist(fm, im, reason, boundary_type=None, n=7):
    bs = {"swept_dim_path": "phases.profiling.concurrency", "feasible_max": fm, "infeasible_min": im}
    if boundary_type:
        bs["boundary_type"] = boundary_type
        bs["binding_constraint"] = "time_to_first_token:avg"
    return {"boundary_summary": bs, "convergence_reason": reason, "iterations": [{}] * n}


def test_ok():
    out = summarize_search(hist({"value": 24}, {"value": 25}, "smooth_isotonic_precision_reached", "smooth"))
    assert (out["state"], out["max_conc"], out["infeasible_min"], out["iterations"]) == ("ok", 24, 25, 7)
    assert out["binding_constraint"] == "time_to_first_token:avg"


def test_no_pass_in_range_keeps_row_with_null_conc():
    out = summarize_search(hist(None, {"value": 1}, "smooth_isotonic_no_pass_in_range"))
    assert out["state"] == "no_pass_in_range" and out["max_conc"] is None


def test_no_failure_in_range_means_ceiling_hit():
    out = summarize_search(hist({"value": 512}, None, "smooth_isotonic_no_failure_in_range"))
    assert out["state"] == "no_failure_in_range" and out["max_conc"] == 512


def test_cliff_and_max_iterations():
    assert summarize_search(hist({"value": 40}, {"value": 64}, "smooth_isotonic_cliff_precision_reached", "cliff"))["state"] == "cliff"
    assert summarize_search(hist({"value": 40}, {"value": 44}, "max_iterations", "smooth"))["state"] == "max_iterations"


def test_null_boundary_summary_is_reported_not_raised():
    out = summarize_search({"boundary_summary": None, "convergence_reason": None, "iterations": []})
    assert out["state"] == "no_data" and out["max_conc"] is None


def test_probe_metrics_from_captured_fixture():
    sweep = json.loads((FIX / "profile_export_aiperf_sweep.json").read_text())
    got = probe_metrics(sweep, conc=<winning conc from the fixture>)
    assert got["rps"] == <value read from the fixture by hand>
    assert got["tpm"] == round(got["output_tps"] * 60)
```
Fill the two `<...>` placeholders from the fixture while writing the test (they are data from the real file, not from the implementation); `tpm = output_tps × 60` is the PDF's definition (`Total Throughput (tok/s) × 60`) and is the only derived relation asserted.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest utils/test_sweep_result.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `summarize_search` and `probe_metrics`**

```python
"""Turn an AIPerf adaptive-search run into one InferenceX result JSON."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Key names in the sweep aggregate, confirmed against the Task 1 fixture (spike ARTIFACT_LAYOUT).
# If AIPerf names differ, change only these constants.
COMBOS = "combos"
PARAMS = "parameters"
METRICS = "metrics"


def summarize_search(history: dict) -> dict:
    boundary = history.get("boundary_summary")
    reason = history.get("convergence_reason")
    base = {
        "convergence_reason": reason,
        "iterations": len(history.get("iterations") or []),
        "max_conc": None,
        "infeasible_min": None,
        "binding_constraint": None,
        "boundary_type": None,
    }
    if boundary is None:
        return {**base, "state": "no_data"}
    passed, failed = boundary.get("feasible_max"), boundary.get("infeasible_min")
    base["max_conc"] = passed["value"] if passed else None
    base["infeasible_min"] = failed["value"] if failed else None
    base["binding_constraint"] = boundary.get("binding_constraint")
    base["boundary_type"] = boundary.get("boundary_type")
    if passed is None:
        state = "no_pass_in_range"
    elif failed is None:
        state = "no_failure_in_range"
    elif boundary.get("boundary_type") == "cliff":
        state = "cliff"
    elif reason == "max_iterations":
        state = "max_iterations"
    else:
        state = "ok"
    return {**base, "state": state}


def _stat(metrics: dict, tag: str, stat: str) -> float | None:
    entry = metrics.get(tag) or {}
    value = entry.get(stat)
    return float(value) if value is not None else None


def probe_metrics(sweep_aggregate: dict, conc: int) -> dict:
    """Metrics of the probe that ran at ``conc`` (the search's winning concurrency)."""
    for combo in sweep_aggregate[COMBOS]:
        if combo[PARAMS].get("concurrency") == conc:
            metrics = combo[METRICS]
            output_tps = _stat(metrics, "output_token_throughput", "avg")
            return {
                "rps": _stat(metrics, "request_throughput", "avg"),
                "output_tps": output_tps,
                "tpm": round(output_tps * 60) if output_tps is not None else None,
                "ttft_avg_ms": _stat(metrics, "time_to_first_token", "avg"),
                "ttft_p90_ms": _stat(metrics, "time_to_first_token", "p90"),
                "itl_avg_ms": _stat(metrics, "inter_token_latency", "avg"),
                "itl_p90_ms": _stat(metrics, "inter_token_latency", "p90"),
            }
    raise KeyError(f"no probe at concurrency {conc} in the sweep aggregate")
```
Then `build_sweep_result(history, sweep_aggregate, env)` = `summarize_search(history)` plus, when `max_conc` is not None, `probe_metrics(...)` merged in, plus `scenario`, `exp_name`, `slo` (from `history["config"]["sla_filters"]`) and the usual identity fields copied the same way `build_result` in `infx/results/agentic/__init__.py` copies them from `env` (read that function first and reuse its identity-field helper rather than re-listing fields). `main()` mirrors `process_agentic_result.main()`: locate artifacts via `ARTIFACT_LAYOUT`, load both JSON files, write `<RESULT_FILENAME>.json`, print one summary line, exit 0 for every terminal state (including `no_pass_in_range`), exit 1 only when `search_history.json` is missing or unparseable.

- [ ] **Step 3b: Measured vs nominal cache hit (spec "Verification of the cache-hit axis")**

Add to `utils/test_sweep_result.py`, then to `infx/results/sweep.py`:

```python
def test_cache_mismatch_flag():
    assert cache_check(nominal_pct=90, measured=0.5) == {"nominal_cache_hit": 0.9, "measured_cache_hit": 0.5, "cache_mismatch": True}
    assert cache_check(nominal_pct=90, measured=0.85)["cache_mismatch"] is False
    assert cache_check(nominal_pct=0, measured=None)["cache_mismatch"] is None
```
```python
def cache_check(nominal_pct: int, measured: float | None) -> dict:
    nominal = nominal_pct / 100
    mismatch = None if measured is None else abs(measured - nominal) > 0.10
    return {"nominal_cache_hit": nominal, "measured_cache_hit": measured, "cache_mismatch": mismatch}
```
`measured` comes from the server prefix-cache counters in the probe's `server_metrics_export.json`, via the existing derivation in `infx/results/agentic/server_metrics.py` (`gpu_cache_hit_rate`); `nominal_pct` from the `cacheNN` suffix of `SERVING_SCENARIO`. Merge `cache_check(...)` into `build_sweep_result`. If `ARTIFACT_LAYOUT` shows only a whole-run counter (no per-probe file), record that value and note it in the docs (Task 8 step 5).

- [ ] **Step 4: Run tests**

Run: `python -m pytest utils/test_sweep_result.py -v`
Expected: PASS.

- [ ] **Step 5: Hook into the replay runner**

In `run_agentic_replay_and_write_outputs`, find where it calls `write_agentic_result_json` and `validate_agentic_result`. When `SERVING_SEARCH=1`, call `"$AIPERF_PYTHON" -m infx.results.sweep` (same `cd "$INFMAX_CONTAINER_WORKSPACE"` and env as `write_agentic_result_json`) and skip the per-request validators that assume one concurrency and `profile_export.jsonl` at the top level (`validate_agentic_result`, power adapter, plots) — keep the failed-request gate by relying on `--error-rate-sla` inside the search. Add a test in `utils/test_serving_replay.py` that runs the function with a stubbed `aiperf` on `PATH` (writes the fixture files into `$result_dir/aiperf_artifacts/` per `ARTIFACT_LAYOUT` and exits 0) and asserts the result JSON appears with the expected `state` and `max_conc` — follow the stub-on-PATH pattern AGENTS.md requires.

- [ ] **Step 6: Commit**

```bash
git add infx/results/sweep.py utils/test_sweep_result.py benchmarks/benchmark_lib.sh utils/test_serving_replay.py
git commit -m "feat(results): result writer for adaptive-search sweep runs

中文: 自适应搜索 sweep 运行的结果写入器。"
```

---

### Task 7: Collector for the 24-row table

**Files:**
- Create: `utils/serving/collect_sweep_table.py`
- Test: `utils/test_sweep_table.py`

**Interfaces:**
- Consumes: result JSONs from Task 6 (fields `scenario, state, max_conc, rps, tpm, ttft_avg_ms, itl_p90_ms, binding_constraint`).
- Produces: `parse_scenario(name) -> tuple[int,int,int]` returning `(cache, isl_tokens, osl_tokens)`; `build_table(results: list[dict]) -> list[dict]` sorted by `(cache, isl, osl)`; `to_markdown(rows) -> str`; CLI `python utils/serving/collect_sweep_table.py <results-dir>...` prints markdown and writes `sweep_table.csv` next to the first dir.

- [ ] **Step 1: Write the failing tests**

```python
from utils.serving.collect_sweep_table import build_table, parse_scenario, to_markdown


def test_parse_scenario():
    assert parse_scenario("sweep-isl64k-osl1k-cache90") == (90, 65536, 1024)
    assert parse_scenario("sweep-isl128k-osl512-cache0") == (0, 131072, 512)


def test_rows_sorted_by_cache_then_isl_then_osl():
    rows = build_table([
        {"scenario": "sweep-isl64k-osl1k-cache90", "state": "ok", "max_conc": 20, "tpm": 1},
        {"scenario": "sweep-isl8k-osl512-cache0", "state": "ok", "max_conc": 90, "tpm": 2},
        {"scenario": "sweep-isl8k-osl1k-cache0", "state": "ok", "max_conc": 80, "tpm": 3},
    ])
    assert [r["scenario"] for r in rows] == [
        "sweep-isl8k-osl512-cache0", "sweep-isl8k-osl1k-cache0", "sweep-isl64k-osl1k-cache90",
    ]


def test_non_ok_state_rendered_not_dropped():
    md = to_markdown(build_table([
        {"scenario": "sweep-isl128k-osl512-cache0", "state": "no_pass_in_range", "max_conc": None},
    ]))
    assert "no_pass_in_range" in md
    assert "None" not in md
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest utils/test_sweep_table.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
"""Collect sweep results into the PDF-style table (cache, ISL, OSL -> max CCU, RPS, TPM, ...)."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"sweep-isl(\d+)k-osl(\d+k?)-cache(\d+)")
COLUMNS = ["scenario", "cache", "isl", "osl", "state", "max_conc", "rps", "tpm",
           "ttft_avg_ms", "itl_p90_ms", "binding_constraint"]


def parse_scenario(name: str) -> tuple[int, int, int]:
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError(f"not a sweep scenario name: {name!r}")
    isl, osl, cache = match.groups()
    return int(cache), int(isl) * 1024, int(osl[:-1]) * 1024 if osl.endswith("k") else int(osl)


def build_table(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        cache, isl, osl = parse_scenario(result["scenario"])
        rows.append({**{col: result.get(col) for col in COLUMNS}, "cache": cache, "isl": isl, "osl": osl})
    return sorted(rows, key=lambda row: (row["cache"], row["isl"], row["osl"]))


def to_markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join("-" if row[c] is None else str(row[c]) for c in COLUMNS) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    results = [json.loads(path.read_text()) for d in argv for path in sorted(Path(d).glob("*.json"))]
    rows = build_table([r for r in results if "scenario" in r and r.get("state")])
    print(to_markdown(rows))
    with open(Path(argv[0]) / "sweep_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```
Check: `parse_scenario("sweep-isl8k-osl512-cache0")`: ISL label `8k` → 8 × 1024 = 8192 matches the grid's binary k; OSL `512` → 512; OSL `1k` → 1024.

- [ ] **Step 4: Run tests, commit**

Run: `python -m pytest utils/test_sweep_table.py -v` → PASS.

```bash
git add utils/serving/collect_sweep_table.py utils/test_sweep_table.py
git commit -m "feat(serving): collector for the sweep result table

中文: 新增 sweep 结果表收集器。"
```

---

### Task 8: Dispatch helper, smoke runs, changelog, docs

**Files:**
- Create: `utils/serving/sweep_dispatch.sh`
- Modify: `perf-changelog.yaml` (append only), `docs/serving-yaml-procedures.md`, `docs/serving-yaml-procedures_zh.md`

**Interfaces:**
- Consumes: everything above; `utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]`.
- Produces: `utils/serving/sweep_dispatch.sh <serving.yaml> [--only <glob>]` — for each matching `scenarios/sweep-*.yaml` writes a temp copy of the serving YAML with `scenario:` replaced, `conc: 1`, and `name:` suffixed with the scenario's short name, then runs `dispatch.sh` on it.

- [ ] **Step 1: Write `sweep_dispatch.sh`**

```bash
#!/usr/bin/env bash
set -eo pipefail
# Dispatch one serving YAML over the sweep scenarios (all, or those matching --only).
#   utils/serving/sweep_dispatch.sh serving/glm53-w4afp8-sglang-router.yaml --only 'sweep-isl8k-*'
repo="$(cd "$(dirname "$0")/../.." && pwd)"
src="$1"; shift
only='sweep-*'
if [[ "$1" == --only ]]; then only="$2"; shift 2; fi
[[ -f "$src" ]] || { echo "ERROR no such serving YAML: $src" >&2; exit 2; }
for scenario_file in "$repo"/scenarios/$only.yaml; do
    scenario="$(basename "$scenario_file" .yaml)"
    tmp="$(mktemp --suffix=.yaml)"
    sed -E "s|^scenario:.*|scenario: $scenario|; s|^conc:.*|conc: 1|; s|^name:[[:space:]]*(.*)|name: \1-${scenario#sweep-}|" "$src" > "$tmp"
    echo "== $scenario"
    "$repo/utils/serving/dispatch.sh" "$tmp" "$@"
    rm -f "$tmp"
done
```
Test by running it with `dispatch.sh` stubbed on `PATH`-style: put a stub `dispatch.sh` copy in a temp tree is awkward; instead test the transformation by running the script with a stub repo layout in `tmp_path` (copy the script, a fake `scenarios/` with two files, a fake `utils/serving/dispatch.sh` that `cat`s its argument) and assert the printed YAML has the right `scenario:`, `conc: 1`, and suffixed `name:` for each. Put this test in `utils/test_serving_dispatch.py` following its existing pattern. Check the `name:` pattern allowed by `NAME_PATTERN` in `infx/serving/schema.py` accepts the suffix (length limit); if not, shorten the suffix.

- [ ] **Step 2: Smoke dispatch (needs the user's go-ahead)**

Stop and ask the user before any `git push` or dispatch (Global Constraints). Proposed: push `vng-benchmark`, then `utils/serving/sweep_dispatch.sh serving/qwen38-bf16-tp1.yaml --only 'sweep-isl8k-osl512-cache50'` on the 1-GPU pool. Success = job green, `search_history.json` in artifacts, result JSON with a `state`. Record anything unexpected in the spike notes.

- [ ] **Step 3: 128k scenario and timeout check (needs go-ahead)**

Same with `--only 'sweep-isl128k-osl512-cache0'`. Expected: either an `ok` result or `no_pass_in_range` — both are green. Check the job's wall time against the workflow/Slurm limit and record the limit in the notes; if the limit is below the worst-case estimate (about 2 h), raise it in the workflow in a separate commit.

- [ ] **Step 4: perf-changelog entry**

Append to `perf-changelog.yaml` (append-only, preserve existing bytes; copy the field set of the last entry): description "Add 24 synthetic sweep scenarios (ISL × OSL × cache) with in-job AIPerf SLO search; serving YAML accepts them with `conc: 1`", date 2026-10-17, and the PR link left as the repo's convention requires.

- [ ] **Step 5: Docs**

Add a "Sweep scenarios" section to `docs/serving-yaml-procedures.md`: the grid, `conc: 1`, the default SLO and `IX_SLO` override format, `sweep_dispatch.sh`, how to read the result states (`ok`, `cliff`, `no_pass_in_range`, `no_failure_in_range`, `max_iterations`), the ITL-vs-TPOT caveat. Mirror the same section in `docs/serving-yaml-procedures_zh.md`. Update the scenario list in step 2 of "Write and dispatch a serving YAML".

- [ ] **Step 6: Commit**

```bash
git add utils/serving/sweep_dispatch.sh utils/test_serving_dispatch.py perf-changelog.yaml docs/serving-yaml-procedures.md docs/serving-yaml-procedures_zh.md
git commit -m "feat(serving): sweep dispatch helper, changelog and docs

中文: sweep 派发脚本、变更日志与文档。"
```

- [ ] **Step 7: Full dispatch (needs the user's go-ahead)**

`utils/serving/sweep_dispatch.sh serving/glm53-w4afp8-sglang-router.yaml` (24 jobs). Then download artifacts and run `python utils/serving/collect_sweep_table.py <dir>` for the report table.

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| Grid of 24, naming | 3 |
| Synthetic dataset, ISL/OSL, cache prefix, cache-bust | 1 (verify), 3 |
| SLO as data, mixed stats, `IX_SLO` override, validation | 2, 5 |
| Search params, probe size, planner/recipe | 1 (verify fields), 3 (write) |
| Wrapper support (spike 1) | 1, 5 |
| Result writer, terminal states (spike 2, 3) | 6 |
| 24-row aggregation | 7 |
| Cache-hit verification (measured vs nominal, `cache_mismatch`) | 6 (step 3b) |
| Timeouts (spike 4), 128k synthetic cost (spike 5) | 8 steps 2–3 |
| Rollout, perf-changelog, docs `_zh` | 8 |
| Tests per AGENTS.md | 2, 3, 4, 5, 6, 7, 8 |

Gap found during review (cache-hit verification had no task) and fixed by adding Task 6 step 3b.

**Placeholder scan:** the `<...>` markers in Task 6's fixture test and the "per spike" comments in Task 3's generator are deliberate: they hold values that exist only after Task 1 runs. Everything else is concrete.

**Type consistency:** `scenario_facts` returns `(int, str)` throughout; `is_sweep(scenario, repo)` is used identically in Tasks 4 and 5; `SERVING_SEARCH` is `"0"|"1"` in bash and `bool` in `serving["search"]`; `summarize_search` keys match the collector's `COLUMNS` (`state`, `max_conc`, `binding_constraint`), while `rps`, `tpm`, `ttft_avg_ms`, `itl_p90_ms` come from `probe_metrics`.
