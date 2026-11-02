# Embedding and Reranker Serving Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `embed` and `rerank` scenarios to the serving pipeline, so that embedding and reranker models can be benchmarked with AIPerf. Both fixed-concurrency and SLO-search modes are supported, and the targets are vLLM and SGLang on RTX PRO 6000.

**Architecture:** Two new AIPerf base configs, `embed.yaml` (endpoint `embeddings`) and `rerank.yaml` (endpoint `cohere_rankings` on `/v1/rerank`). `scenarios.py` fills them per grid point, the same way it fills `synthetic`. `schema.py` gets one workload model per family. The remote probe and the sweep result learn the family from the scenario label (`-embed-` / `-rerank-`). Six serving YAMLs drive the runs.

**Tech Stack:** Python 3.10+, Pydantic v2, PyYAML, pytest, AIPerf (vendored in `utils/aiperf`), vLLM and SGLang images.

Spec: `docs/superpowers/specs/2026-11-02-embed-rerank-bench-design.md`

## Global Constraints

- Run every `uv`/`pytest`/`ruff` command from `inferencex-e2e/`.
- Lint: `uvx --exclude-newer PT12H ruff@latest check --fix infx && uvx --exclude-newer PT12H ruff@latest format infx`. Line length is 100, and every Ruff finding must be fixed or marked with an inline `# noqa: CODE`.
- Tests: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving -n 4`.
- Test rules (AGENTS.md): call the real code with concrete inputs and assert on its output. Never grep source, never pin checked-in config, never write tautologies.
- Comments: no header comments, no history. Only a short technical "why", one line.
- Every error message is one line, `<field>: <problem>`.
- Grid values are fixed:
  - embed `isl` ∈ {`512`, `2k`, `8k`, `32k`}, `batch` ∈ {1, 32}
  - rerank `passage` ∈ {`512`, `2k`}, `docs` ∈ {10, 40}, query tokens 32
- Default SLOs: embed `request_latency:p90:lt:500`, rerank `request_latency:p90:lt:1000`. The search objective is `request_throughput`.
- Labels: `{fixed|sweep}-embed-isl{isl}-b{batch}` and `{fixed|sweep}-rerank-p{passage}-n{docs}`.
- Git: work only on branch `feat/embed-rerank-bench`. `git add` the exact files of each task, never `-A`. Author is `-c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai`. Commit subjects are English, and each body carries a `中文：` line. **Never push without asking the user.**

## File Map

| File | Change | Responsibility |
|---|---|---|
| `inferencex-e2e/benchmarks/single_node/serving/scenarios/embed.yaml` | create | AIPerf base config, embeddings endpoint |
| `inferencex-e2e/benchmarks/single_node/serving/scenarios/rerank.yaml` | create | AIPerf base config, rerank endpoint |
| `inferencex-e2e/infx/serving/scenarios.py` | modify | `EmbedPoint`, `RerankPoint`, `_pooling()`, shared `_search_block()`, `render()` routing |
| `inferencex-e2e/infx/serving/schema.py` | modify | `EmbedWorkload`, `RerankWorkload`, scenario literal, per-family error formatting |
| `inferencex-e2e/infx/serving/client.py` | modify | per-family remote probe |
| `inferencex-e2e/infx/serving/sweep_result.py` | modify | pooling probe metrics (`items_per_s`) |
| `inferencex-e2e/infx/tests/serving/test_scenarios.py`, `test_schema.py`, `test_client.py`, `test_sweep_result.py` | modify | tests |
| `inferencex-e2e/configs/serving/{bge-m3,bge-reranker-v2-m3,qwen3-embedding-8b}-{vllm,sglang}-*.yaml` | create | six serving configs |

---

### Task 1: Scenario rendering for `embed` and `rerank`

**Files:**
- Create: `inferencex-e2e/benchmarks/single_node/serving/scenarios/embed.yaml`
- Create: `inferencex-e2e/benchmarks/single_node/serving/scenarios/rerank.yaml`
- Modify: `inferencex-e2e/infx/serving/scenarios.py`
- Test: `inferencex-e2e/infx/tests/serving/test_scenarios.py`

**Interfaces:**
- Produces:
  - `EmbedPoint(isl: str, batch: int)` and `RerankPoint(passage: str, docs: int)`, frozen dataclasses with `.label -> str`
  - constants `EMBED_ISL`, `EMBED_BATCH`, `RERANK_PASSAGE`, `RERANK_DOCS`, `POOLING`, `POOLING_SLO`
  - `render(family, overrides, point: Point | EmbedPoint | RerankPoint | None, search, slo, lock) -> Rendered`
  - `FAMILIES` contains `"embed"` and `"rerank"`

- [ ] **Step 1: Write the failing tests** (append to `test_scenarios.py`; extend its import line)

```python
from infx.serving.scenarios import FAMILIES, EmbedPoint, Point, RerankPoint, ScenarioError, render


def test_embed_search_point_fills_prompts_and_searches_request_throughput():
    rendered = render("embed", {}, EmbedPoint("2k", 32), search=True)
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert dataset["prompts"] == {"isl": 2048, "batchSize": 32}
    assert rendered.doc["benchmark"]["endpoint"] == {"type": "embeddings", "path": "/v1/embeddings"}
    sweep = rendered.doc["sweep"]
    assert sweep["objectives"] == [
        {"metric": "request_throughput", "stat": "avg", "direction": "maximize"}
    ]
    assert sweep["sla_filters"] == [
        {"metric_tag": "request_latency", "stat": "p90", "op": "lt", "threshold": 500.0}
    ]
    assert (rendered.label, rendered.kind, rendered.search) == ("sweep-embed-isl2k-b32", "synthetic", True)


def test_rerank_fixed_point_fills_rankings_without_a_sweep():
    rendered = render("rerank", {"duration": 60}, RerankPoint("512", 40))
    dataset = rendered.doc["benchmark"]["datasets"][0]
    assert dataset["rankings"] == {"queryTokens": 32, "passages": 40, "passageTokens": 512}
    assert rendered.doc["benchmark"]["endpoint"] == {"type": "cohere_rankings", "path": "/v1/rerank"}
    assert "sweep" not in rendered.doc
    assert (rendered.label, rendered.duration) == ("fixed-rerank-p512-n40-d60", 60.0)


def test_rerank_search_uses_its_own_default_slo_and_accepts_an_override():
    default = render("rerank", {}, RerankPoint("2k", 10), search=True)
    assert default.doc["sweep"]["sla_filters"][0]["threshold"] == 1000.0
    custom = render("rerank", {}, RerankPoint("2k", 10), search=True, slo="request_latency:p99:lt:200")
    assert custom.doc["sweep"]["sla_filters"] == [
        {"metric_tag": "request_latency", "stat": "p99", "op": "lt", "threshold": 200.0}
    ]


@pytest.mark.parametrize(
    "family, point, search, slo, message",
    [
        ("embed", EmbedPoint("4k", 1), False, None, "workload.isl: '4k'"),
        ("embed", EmbedPoint("2k", 8), False, None, "workload.batch: 8"),
        ("rerank", RerankPoint("1k", 10), False, None, "workload.passage: '1k'"),
        ("rerank", RerankPoint("512", 7), False, None, "workload.docs: 7"),
        ("embed", EmbedPoint("2k", 1), False, "request_latency:p90:lt:5", "slo: only applies"),
        ("rerank", EmbedPoint("2k", 1), False, None, "workload: rerank takes passage, docs"),
        ("embed", None, False, None, "workload: embed takes isl, batch"),
    ],
)
def test_pooling_rejects_points_outside_the_grid(family, point, search, slo, message):
    with pytest.raises(ScenarioError, match=message.replace("(", r"\(")):
        render(family, {}, point, search, slo)


def test_pooling_takes_only_the_duration_override():
    with pytest.raises(ScenarioError, match="overrides: embed only takes duration"):
        render("embed", {"dataset": "x"}, EmbedPoint("512", 1))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_scenarios.py -q`
Expected: FAIL. Collection raises `ImportError: cannot import name 'EmbedPoint'`.

- [ ] **Step 3: Create the base configs**

`inferencex-e2e/benchmarks/single_node/serving/scenarios/embed.yaml`:

```yaml
schemaVersion: "2.0"
benchmark:
  endpoint:
    type: embeddings
    path: /v1/embeddings
  datasets:
  - name: main
    type: synthetic
    entries: 1000
    randomSeed: 42
    prompts:
      isl: 512
      batchSize: 1
  phases:
  - name: profiling
    kind: profiling
    type: concurrency
    duration: 300.0
    gracePeriod: 30.0
    failedRequestThreshold: 0.1
randomSeed: 42
```

`inferencex-e2e/benchmarks/single_node/serving/scenarios/rerank.yaml`:

```yaml
schemaVersion: "2.0"
benchmark:
  endpoint:
    type: cohere_rankings
    # vLLM and SGLang both serve the Cohere-shaped {query, documents} body here.
    path: /v1/rerank
  datasets:
  - name: main
    type: synthetic
    entries: 1000
    randomSeed: 42
    rankings:
      queryTokens: 32
      passages: 10
      passageTokens: 512
  phases:
  - name: profiling
    kind: profiling
    type: concurrency
    duration: 300.0
    gracePeriod: 30.0
    failedRequestThreshold: 0.1
randomSeed: 42
```

- [ ] **Step 4: Implement in `scenarios.py`**

4a. Add `"embed"` and `"rerank"` to `FAMILIES`. Add the constants below directly under `CACHE = (0, 50, 90)`:

```python
POOLING = ("embed", "rerank")
EMBED_ISL = {"512": 512, "2k": 2048, "8k": 8192, "32k": 32768}
EMBED_BATCH = (1, 32)
RERANK_PASSAGE = {"512": 512, "2k": 2048}
RERANK_DOCS = (10, 40)
RERANK_QUERY_TOKENS = 32
POOLING_SLO = {"embed": "request_latency:p90:lt:500", "rerank": "request_latency:p90:lt:1000"}
```

4b. Add the two point classes after `class Point`:

```python
@dataclass(frozen=True)
class EmbedPoint:
    """One embed grid point: tokens per input and inputs per request."""

    isl: str
    batch: int

    @property
    def label(self) -> str:
        return f"embed-isl{self.isl}-b{self.batch}"


@dataclass(frozen=True)
class RerankPoint:
    """One rerank grid point: tokens per passage and passages per request."""

    passage: str
    docs: int

    @property
    def label(self) -> str:
        return f"rerank-p{self.passage}-n{self.docs}"
```

4c. Move the sweep block out of `_synthetic` into a shared helper. In `_synthetic`, replace everything from `if search:` to the end of the `doc["sweep"] = {...}` dict with `if search:` followed by `doc["sweep"] = _search_block(slo, DEFAULT_SLO, "output_token_throughput")`. Then add:

```python
def _search_block(slo: str | None, default_slo: str, objective: str) -> dict[str, Any]:
    try:
        filters = parse_slo(slo or default_slo)
    except SLOError as error:
        raise ScenarioError(str(error)) from None
    return {
        "type": "adaptive_search",
        "planner": "smooth_isotonic",
        "objectives": [{"metric": objective, "stat": "avg", "direction": "maximize"}],
        "search_space": [
            {"path": "phases.profiling.concurrency", "lo": 1, "hi": 512, "kind": "int"}
        ],
        "sla_filters": [
            {"metric_tag": s.metric, "stat": s.stat, "op": s.op, "threshold": s.threshold}
            for s in filters
        ],
        "max_iterations": 16,
        "sla_precision": "normal",  # 1000 requests/probe cap
    }
```

4d. Add `_pooling` after `_synthetic`:

```python
def _pooling(
    family: str, point: EmbedPoint | RerankPoint | Point | None, search: bool, slo: str | None
) -> dict[str, Any]:
    if slo and not search:
        raise ScenarioError("slo: only applies with workload.search: true")
    doc = load_base(family)
    dataset = doc["benchmark"]["datasets"][0]
    if family == "embed":
        if not isinstance(point, EmbedPoint):
            raise ScenarioError("workload: embed takes isl, batch and search")
        if point.isl not in EMBED_ISL:
            raise ScenarioError(f"workload.isl: {point.isl!r} is not one of {', '.join(EMBED_ISL)}")
        if point.batch not in EMBED_BATCH:
            raise ScenarioError(
                f"workload.batch: {point.batch!r} is not one of {', '.join(map(str, EMBED_BATCH))}"
            )
        dataset["prompts"] = {"isl": EMBED_ISL[point.isl], "batchSize": point.batch}
    else:
        if not isinstance(point, RerankPoint):
            raise ScenarioError("workload: rerank takes passage, docs and search")
        if point.passage not in RERANK_PASSAGE:
            raise ScenarioError(
                f"workload.passage: {point.passage!r} is not one of {', '.join(RERANK_PASSAGE)}"
            )
        if point.docs not in RERANK_DOCS:
            raise ScenarioError(
                f"workload.docs: {point.docs!r} is not one of {', '.join(map(str, RERANK_DOCS))}"
            )
        dataset["rankings"] = {
            "queryTokens": RERANK_QUERY_TOKENS,
            "passages": point.docs,
            "passageTokens": RERANK_PASSAGE[point.passage],
        }
    if search:
        doc["sweep"] = _search_block(slo, POOLING_SLO[family], "request_throughput")
    return doc
```

4e. Change the `point` type of `render()` to `Point | EmbedPoint | RerankPoint | None`. Merge the `synthetic` branch with the new families:

```python
    if family in ("synthetic", *POOLING):
        if family == "synthetic" and point is None:
            raise ScenarioError("workload: synthetic needs isl, osl and cache")
        if extra := sorted(set(overrides) - set(SYNTHETIC_OVERRIDES)):
            raise ScenarioError(
                f"overrides: {family} only takes {', '.join(SYNTHETIC_OVERRIDES)}, not {', '.join(extra)}"
            )
        doc = _pooling(family, point, search, slo) if family in POOLING else _synthetic(point, search, slo)
        suffixes = _apply(doc, family, overrides, lock)
        label = f"{'sweep' if search else 'fixed'}-{point.label}{''.join(suffixes)}"
        return Rendered(doc, label, "synthetic", "synthetic", search, _profiling(doc)["duration"])
```

`point.label` is safe on the pooling path because `_pooling` has already rejected `None`. Also update the module docstring's "six base AIPerf configs" so the count is right: it becomes "ten".

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_scenarios.py -q`
Expected: PASS. All new tests pass, and the existing legacy-fixture parametrization still passes, which proves the synthetic sweep block is byte-identical.

- [ ] **Step 6: Lint, then commit**

```bash
uvx --exclude-newer PT12H ruff@latest check --fix infx && uvx --exclude-newer PT12H ruff@latest format infx
git add benchmarks/single_node/serving/scenarios/embed.yaml benchmarks/single_node/serving/scenarios/rerank.yaml infx/serving/scenarios.py infx/tests/serving/test_scenarios.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): embed and rerank scenarios" -m "中文：新增 embed 与 rerank 场景（AIPerf embeddings / cohere_rankings 端点，固定并发与 SLO 搜索）。"
```

---

### Task 2: Serving YAML schema for embed and rerank workloads

**Files:**
- Modify: `inferencex-e2e/infx/serving/schema.py`
- Test: `inferencex-e2e/infx/tests/serving/test_schema.py`

**Interfaces:**
- Consumes: `scenarios.EmbedPoint`, `scenarios.RerankPoint` and `render()` from Task 1.
- Produces: `EmbedWorkload` and `RerankWorkload` (Pydantic, `extra="forbid"`, `.points()`). `ServingSpec.workload: Workload | EmbedWorkload | RerankWorkload | None`.

- [ ] **Step 1: Write the failing tests**

First look at the top of `test_schema.py` for the helper it uses to build an inventory and a base YAML. Reuse that helper; it is named something like `INVENTORY` or `_inventory()`. Then add the tests below, with `load` standing for `lambda text: load_serving(text, <that inventory>)`:

```python
EMBED_YAML = """
name: bge-m3-embed
runner: rtx6000pro-hcm
image: vllm/vllm-openai:v0.25.1
model: BAAI/bge-m3
gpus: 1
serve: vllm serve BAAI/bge-m3 --max-model-len 8192
scenario: embed
workload: {isl: [512, 2k], batch: [1, 32], search: true}
conc: 1
"""


def test_embed_workload_expands_to_one_search_variant_per_point():
    spec = load(EMBED_YAML)
    assert [v.label for v in variants(spec)] == [
        "sweep-embed-isl512-b1",
        "sweep-embed-isl512-b32",
        "sweep-embed-isl2k-b1",
        "sweep-embed-isl2k-b32",
    ]


def test_rerank_workload_runs_fixed_concurrencies_on_one_point():
    text = EMBED_YAML.replace("scenario: embed", "scenario: rerank").replace(
        "workload: {isl: [512, 2k], batch: [1, 32], search: true}\nconc: 1",
        "workload: {passage: [512], docs: [40]}\nconc: [4, 16]",
    )
    spec = load(text)
    assert [v.label for v in variants(spec)] == ["fixed-rerank-p512-n40"]
    assert spec.conc == [4, 16]


def test_an_embed_workload_under_scenario_rerank_names_the_right_fields():
    text = EMBED_YAML.replace("scenario: embed", "scenario: rerank")
    with pytest.raises(ServingError, match="workload: rerank takes passage, docs and search"):
        load(text)


def test_an_unknown_rerank_workload_key_lists_the_rerank_keys_only():
    text = EMBED_YAML.replace("scenario: embed", "scenario: rerank").replace(
        "{isl: [512, 2k], batch: [1, 32], search: true}", "{passage: [512], docs: [10], isl: [512]}"
    )
    with pytest.raises(ServingError) as error:
        load(text)
    assert str(error.value) == "workload.isl: unknown key (allowed: passage, docs, search)"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_schema.py -q`
Expected: FAIL, because `scenario: embed` is rejected by the literal.

- [ ] **Step 3: Implement in `schema.py`**

3a. Change the import to `from infx.serving.scenarios import EmbedPoint, Point, RerankPoint, Rendered, ScenarioError`.

3b. Add the two models after `class Workload`:

```python
class EmbedWorkload(BaseModel):
    """The embed grid: one job per (isl, batch) point."""

    model_config = ConfigDict(extra="forbid")

    isl: list[str] = Field(min_length=1)
    batch: list[int] = Field(min_length=1)
    search: bool = False

    @field_validator("isl", mode="before")
    @classmethod
    def _labels(cls, value: Any) -> Any:
        return [str(item) for item in _listed(value)]

    @field_validator("batch", mode="before")
    @classmethod
    def _batch(cls, value: Any) -> Any:
        return _listed(value)

    def points(self) -> list[EmbedPoint]:
        return [EmbedPoint(i, b) for i, b in itertools.product(self.isl, self.batch)]


class RerankWorkload(BaseModel):
    """The rerank grid: one job per (passage, docs) point."""

    model_config = ConfigDict(extra="forbid")

    passage: list[str] = Field(min_length=1)
    docs: list[int] = Field(min_length=1)
    search: bool = False

    @field_validator("passage", mode="before")
    @classmethod
    def _labels(cls, value: Any) -> Any:
        return [str(item) for item in _listed(value)]

    @field_validator("docs", mode="before")
    @classmethod
    def _docs(cls, value: Any) -> Any:
        return _listed(value)

    def points(self) -> list[RerankPoint]:
        return [RerankPoint(p, d) for p, d in itertools.product(self.passage, self.docs)]


WORKLOADS: dict[str, type[BaseModel]] = {
    "synthetic": Workload,
    "embed": EmbedWorkload,
    "rerank": RerankWorkload,
}
```

3c. In `ServingSpec`, add `"embed"` and `"rerank"` to the `scenario` `Literal[...]`. Change the field to `workload: Workload | EmbedWorkload | RerankWorkload | None = None`. Pydantic's smart union picks the member whose required keys are present. A member that matches the wrong scenario is rejected later by `scenarios.render`, with the one-line message the tests expect.

3d. Make `_format_validation_error` scenario-aware. A union failure reports one error set per member, located at `("workload", "<ClassName>", ...)`. Keep only the member that belongs to the YAML's scenario, and drop the class name from the location:

```python
def _format_validation_error(error: ValidationError, scenario: Any = None) -> str:
    models = {
        "router": RouterSpec,
        "storage": StorageSpec,
        "workload": WORKLOADS.get(scenario, Workload),
        "endpoint": EndpointSpec,
    }
    member = models["workload"].__name__
    members = {model.__name__ for model in WORKLOADS.values()}
    lines = []
    for item in error.errors():
        loc = list(item["loc"])
        if len(loc) > 1 and loc[0] == "workload" and loc[1] in members:
            if loc[1] != member:
                continue
            del loc[1]
        location = ".".join(str(part) for part in loc) or "serving"
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "extra_forbidden":
            model = models.get(str(loc[0]), ServingSpec) if len(loc) > 1 else ServingSpec
            allowed = (field.alias or name for name, field in model.model_fields.items())
            message = "unknown key (allowed: " + ", ".join(allowed) + ")"
        elif item["loc"] == ("scenario",):
            message = f"must be one of {', '.join(scenarios.FAMILIES)} (variants go in overrides:)"
        lines.append(f"{location}: {message}")
    return "\n".join(lines)
```

In `load_serving`, pass the scenario: `raise ServingError(_format_validation_error(error, raw.get("scenario"))) from error`.

If Pydantic names the union members differently, for example `function-after[...]` instead of the bare class name, print `error.errors()` once in a scratch test. Then match on the actual location value, still keyed by the scenario's model.

- [ ] **Step 4: Run all serving tests and confirm they pass**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving -n 4 -q`
Expected: PASS. Existing synthetic-workload tests and `test_configs.py` must stay green.

- [ ] **Step 5: Lint, then commit**

```bash
uvx --exclude-newer PT12H ruff@latest check --fix infx && uvx --exclude-newer PT12H ruff@latest format infx
git add infx/serving/schema.py infx/tests/serving/test_schema.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): embed and rerank workload schema" -m "中文：serving YAML 支持 embed / rerank 的 workload 网格。"
```

---

### Task 3: Per-family remote probe

**Files:**
- Modify: `inferencex-e2e/infx/serving/client.py` (`probe`, `prepare`)
- Test: `inferencex-e2e/infx/tests/serving/test_client.py`

**Interfaces:**
- Produces: `probe(url, model, api_key, secret="", timeout=60.0, verify=True, family="") -> str`. When `family` is `embed` or `rerank`, it POSTs that family's body and returns `"client"`. Otherwise its behaviour is unchanged.
- Produces: `scenario_family(label: str) -> str`, which returns `"embed"`, `"rerank"` or `""`.

- [ ] **Step 1: Write the failing tests** (append; add `scenario_family` to the import)

```python
@pytest.mark.parametrize(
    "family, path, body",
    [
        ("embed", "/v1/embeddings", {"model": "bge-m3", "input": ["OK"]}),
        ("rerank", "/v1/rerank", {"model": "bge-m3", "query": "OK", "documents": ["OK"]}),
    ],
)
def test_pooling_probe_posts_the_family_body_and_counts_on_the_client(family, path, body):
    url, seen, server = serve(200, b'{"data": []}')
    try:
        assert probe(url, "bge-m3", "sk-1", family=family) == "client"
    finally:
        server.shutdown()
    [(got_path, headers, got_body)] = seen
    assert (got_path, got_body) == (path, body)
    assert headers["Authorization"] == "Bearer sk-1"


def test_pooling_probe_failure_names_the_fix():
    url, _, server = serve(404, b'{"error": {"message": "no such model"}}')
    try:
        with pytest.raises(ProbeError, match="HTTP 404"):
            probe(url, "bge-m3", "", family="rerank")
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "label, family",
    [("sweep-embed-isl2k-b32", "embed"), ("fixed-rerank-p512-n40-d60", "rerank"),
     ("sweep-isl8k-osl1k-cache50", "")],
)
def test_scenario_family_reads_the_label(label, family):
    assert scenario_family(label) == family
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_client.py -q`
Expected: FAIL with `ImportError: cannot import name 'scenario_family'`.

- [ ] **Step 3: Implement**

Add the following below `PROBE_BODY`:

```python
POOLING_PROBES = {
    "embed": ("/v1/embeddings", {"input": ["OK"]}),
    "rerank": ("/v1/rerank", {"query": "OK", "documents": ["OK"]}),
}


def scenario_family(label: str) -> str:
    """"embed" or "rerank" when the scenario label is one of theirs, else ""."""
    return next((family for family in POOLING_PROBES if f"-{family}-" in label), "")
```

Then rewrite `probe`. It keeps its error handling and adds the pooling branch:

```python
def probe(
    url: str,
    model: str,
    api_key: str,
    secret: str = "",
    timeout: float = 60.0,
    verify: bool = True,
    family: str = "",
) -> str:
    """One request shaped like the scenario's: "server" if a chat chunk carries usage, else "client"."""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = authorization(api_key)
    path, extra = POOLING_PROBES.get(family, ("/v1/chat/completions", PROBE_BODY))
    body = json.dumps({"model": model, **extra}).encode()
    request = urllib.request.Request(f"{url}{path}", data=body, headers=headers)  # noqa: S310 -- http(s) only, checked by the schema
    try:
        with urllib.request.urlopen(  # noqa: S310 -- http(s) only, checked by the schema
            request, timeout=timeout, context=_tls_context(verify)
        ) as response:
            if family in POOLING_PROBES:
                return "client"  # no usage to count from; aiperf tokenizes the inputs itself
            for raw in response:
                ...  # the existing SSE loop, unchanged
    except urllib.error.HTTPError as error:
        ...  # unchanged
    except (urllib.error.URLError, OSError) as error:
        ...  # unchanged
    return "client"
```

The `...` lines above mean "keep the existing lines verbatim". Do not type a literal `...` into the code.

In `prepare()`, add `family=scenario_family(env.get("SERVING_SCENARIO_LABEL", "")),` to the `probe(...)` call.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_client.py -q`
Expected: PASS, including the existing chat-probe tests.

- [ ] **Step 5: Lint, then commit**

```bash
uvx --exclude-newer PT12H ruff@latest check --fix infx && uvx --exclude-newer PT12H ruff@latest format infx
git add infx/serving/client.py infx/tests/serving/test_client.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): probe embed and rerank endpoints" -m "中文：远程端点预检按 embed / rerank 发送对应请求体。"
```

---

### Task 4: Sweep result metrics for embed and rerank

**Files:**
- Modify: `inferencex-e2e/infx/serving/sweep_result.py`
- Test: `inferencex-e2e/infx/tests/serving/test_sweep_result.py`

**Interfaces:**
- Produces: `items_per_request(label: str) -> int | None`. It returns the batch for `-embed-...-b{N}`, the docs for `-rerank-...-n{N}`, and `None` for any other label.
- Produces: `probe_metrics(probe_export, conc, items: int | None = None) -> dict`. With `items`, it returns the pooling keys `rps`, `items_per_s`, `input_tps`, `latency_{avg,p50,p90,p99}_ms`. Without it, the LLM keys are unchanged.
- `build_sweep_result` passes `items_per_request(env["SERVING_SCENARIO_LABEL"])`.

- [ ] **Step 1: Write the failing tests** (append; add `items_per_request` to the import)

```python
POOLING_EXPORT = {
    "request_throughput": {"avg": 10.0},
    "input_token_throughput": {"avg": 655360.0},
    "request_latency": {"avg": 80.0, "p50": 75.0, "p90": 120.0, "p99": 200.0},
}


def test_pooling_probe_metrics_count_items_not_output_tokens():
    assert probe_metrics(POOLING_EXPORT, conc=64, items=32) == {
        "rps": 10.0,
        "items_per_s": 320.0,
        "input_tps": 655360.0,
        "latency_avg_ms": 80.0,
        "latency_p50_ms": 75.0,
        "latency_p90_ms": 120.0,
        "latency_p99_ms": 200.0,
    }


def test_pooling_probe_metrics_need_request_latency():
    with pytest.raises(KeyError):
        probe_metrics({"request_throughput": {"avg": 1.0}}, conc=1, items=1)


@pytest.mark.parametrize(
    "label, items",
    [("sweep-embed-isl2k-b32", 32), ("sweep-rerank-p512-n40-d60", 40), ("sweep-isl8k-osl1k-cache50", None)],
)
def test_items_per_request_reads_the_label(label, items):
    assert items_per_request(label) == items


def test_build_sweep_result_for_embed_carries_items_and_no_cache_check():
    result = build_sweep_result(
        _fixture_history(), POOLING_EXPORT, _env(SERVING_SCENARIO_LABEL="sweep-embed-isl2k-b32")
    )
    assert result["items_per_s"] == 320.0
    assert "nominal_cache_hit" not in result
```

If the fixture history in `test_sweep_result.py` has no feasible winner (`max_conc is None`), then `probe_metrics` is not called. In that case the last test must build a history with a winner, using the file's own `hist(...)` helper, the same way `test_build_sweep_result_merges_probe_and_identity` does. Add `import pytest` at the top of the file if it is not there yet.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_sweep_result.py -q`
Expected: FAIL with `ImportError: cannot import name 'items_per_request'`.

- [ ] **Step 3: Implement**

```python
POOLING_PROBE_METRICS = ("request_throughput", "input_token_throughput", "request_latency")


def items_per_request(label: str) -> int | None:
    """Inputs (embed batch) or passages (rerank docs) per request, read off the scenario label."""
    match = re.search(r"-embed-isl[^-]+-b(\d+)|-rerank-p[^-]+-n(\d+)", label)
    return int(match.group(1) or match.group(2)) if match else None
```

In `probe_metrics`, add the `items: int | None = None` parameter and put this branch first:

```python
    if items is not None:
        missing = [tag for tag in POOLING_PROBE_METRICS if tag not in probe_export]
        if missing:
            raise KeyError(f"probe export is missing required metric(s): {missing}")
        rps = _stat(probe_export, "request_throughput", "avg")
        return {
            "rps": rps,
            "items_per_s": rps * items if rps is not None else None,
            "input_tps": _stat(probe_export, "input_token_throughput", "avg"),
            **{
                f"latency_{stat}_ms": _stat(probe_export, "request_latency", stat)
                for stat in ("avg", "p50", "p90", "p99")
            },
        }
```

In `build_sweep_result`, change the call to `probe_metrics(probe_export, result["max_conc"], items_per_request(env.get("SERVING_SCENARIO_LABEL", "")))`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_sweep_result.py -q`
Expected: PASS, including the existing LLM tests.

- [ ] **Step 5: Lint, then commit**

```bash
uvx --exclude-newer PT12H ruff@latest check --fix infx && uvx --exclude-newer PT12H ruff@latest format infx
git add infx/serving/sweep_result.py infx/tests/serving/test_sweep_result.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): items/s and latency in embed/rerank sweep results" -m "中文：embed / rerank 搜索结果输出 items/s、输入 tok/s 与请求延迟分位数。"
```

---

### Task 5: Six smoke serving configs

**Files:**
- Create: `inferencex-e2e/configs/serving/bge-m3-vllm-embed.yaml`, `bge-m3-sglang-embed.yaml`, `bge-reranker-v2-m3-vllm-rerank.yaml`, `bge-reranker-v2-m3-sglang-rerank.yaml`, `qwen3-embedding-8b-vllm-embed.yaml`, `qwen3-embedding-8b-sglang-embed.yaml`
- Test: the existing `infx/tests/serving/test_configs.py`, which validates every file in that directory through `to_matrix.build`

**Interfaces:**
- Consumes: the `scenario: embed|rerank` and `workload:` shapes from Tasks 1–2.

Each config starts as a smoke run: one grid point, fixed `conc: 4`, `overrides: {duration: 60}`. Task 7 widens it to the full grid. Images: `vllm/vllm-openai:v0.25.1` and `lmsysorg/sglang:v0.5.22-cu130`, the versions already in use on this repo's configs. Context is set to the model-card maximum.

- [ ] **Step 1: Write the six files**

`bge-m3-vllm-embed.yaml`:

```yaml
name: bge-m3-vllm-embed
runner: rtx6000pro-hcm
image: vllm/vllm-openai:v0.25.1
model: BAAI/bge-m3
gpus: 1
serve: |
  vllm serve BAAI/bge-m3
  --served-model-name bge-m3
  --hf-overrides '{"architectures":["BgeM3EmbeddingModel"]}'
  --pooler-config '{"task":"embed"}'
  --trust-remote-code
  --max-model-len 8192
  --gpu-memory-utilization 0.2
  --kv-cache-metrics
  --enable-prompt-tokens-details
scenario: embed
workload: {isl: [2k], batch: [32]}
overrides: {duration: 60}
conc: 4
```

`bge-m3-sglang-embed.yaml`:

```yaml
name: bge-m3-sglang-embed
runner: rtx6000pro-hcm
image: lmsysorg/sglang:v0.5.22-cu130
model: BAAI/bge-m3
gpus: 1
serve: |
  python3 -m sglang.launch_server
  --model-path BAAI/bge-m3
  --served-model-name bge-m3
  --is-embedding
  --context-length 8192
  --mem-fraction-static 0.2
  --enable-metrics
scenario: embed
workload: {isl: [2k], batch: [32]}
overrides: {duration: 60}
conc: 4
```

`bge-reranker-v2-m3-vllm-rerank.yaml`:

```yaml
name: bge-reranker-v2-m3-vllm-rerank
runner: rtx6000pro-hcm
image: vllm/vllm-openai:v0.25.1
model: BAAI/bge-reranker-v2-m3
gpus: 1
serve: |
  vllm serve BAAI/bge-reranker-v2-m3
  --served-model-name bge-reranker-v2-m3
  --max-model-len 8192
  --gpu-memory-utilization 0.15
  --max-num-seqs 64
  --kv-cache-metrics
  --enable-prompt-tokens-details
scenario: rerank
workload: {passage: [512], docs: [40]}
overrides: {duration: 60}
conc: 4
```

`bge-reranker-v2-m3-sglang-rerank.yaml`:

```yaml
name: bge-reranker-v2-m3-sglang-rerank
runner: rtx6000pro-hcm
image: lmsysorg/sglang:v0.5.22-cu130
model: BAAI/bge-reranker-v2-m3
gpus: 1
serve: |
  python3 -m sglang.launch_server
  --model-path BAAI/bge-reranker-v2-m3
  --served-model-name bge-reranker-v2-m3
  --is-embedding
  --context-length 8192
  --mem-fraction-static 0.15
  --enable-metrics
scenario: rerank
workload: {passage: [512], docs: [40]}
overrides: {duration: 60}
conc: 4
```

`qwen3-embedding-8b-vllm-embed.yaml`:

```yaml
name: qwen3-embedding-8b-vllm-embed
runner: rtx6000pro-hcm
image: vllm/vllm-openai:v0.25.1
model: Qwen/Qwen3-Embedding-8B
gpus: 1
serve: |
  vllm serve Qwen/Qwen3-Embedding-8B
  --served-model-name qwen3-embedding-8b
  --runner pooling
  --dtype bfloat16
  --pooler-config '{"seq_pooling_type":"MEAN"}'
  --max-model-len 32768
  --gpu-memory-utilization 0.85
  --kv-cache-metrics
  --enable-prompt-tokens-details
scenario: embed
workload: {isl: [2k], batch: [32]}
overrides: {duration: 60}
conc: 4
```

`qwen3-embedding-8b-sglang-embed.yaml`:

```yaml
name: qwen3-embedding-8b-sglang-embed
runner: rtx6000pro-hcm
image: lmsysorg/sglang:v0.5.22-cu130
model: Qwen/Qwen3-Embedding-8B
gpus: 1
serve: |
  python3 -m sglang.launch_server
  --model-path Qwen/Qwen3-Embedding-8B
  --served-model-name qwen3-embedding-8b
  --is-embedding
  --dtype bfloat16
  --context-length 32768
  --mem-fraction-static 0.85
  --enable-metrics
scenario: embed
workload: {isl: [2k], batch: [32]}
overrides: {duration: 60}
conc: 4
```

- [ ] **Step 2: Validate them**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_configs.py -q`
Expected: PASS for all configs, including the six new ones. Any `ERROR <field>: <fix>` line coming from `_engine_problems` is a real rule of the repo, so fix the YAML to satisfy it. One such rule is that a reserved flag the launcher sets, such as `--host` or `--port`, must not appear.

- [ ] **Step 3: Commit**

```bash
git add configs/serving/bge-m3-vllm-embed.yaml configs/serving/bge-m3-sglang-embed.yaml configs/serving/bge-reranker-v2-m3-vllm-rerank.yaml configs/serving/bge-reranker-v2-m3-sglang-rerank.yaml configs/serving/qwen3-embedding-8b-vllm-embed.yaml configs/serving/qwen3-embedding-8b-sglang-embed.yaml
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "serving: embed/rerank smoke configs on RTX PRO 6000 (vLLM + SGLang)" -m "中文：在 RTX PRO 6000 上新增 bge-m3、bge-reranker-v2-m3、Qwen3-Embedding-8B 的 vLLM / SGLang 冒烟配置。"
```

---

### Task 6: Smoke dispatch and risk fixes

This task needs the branch pushed. **Stop and ask the user before pushing.** Then dispatch each config with `inferencex-e2e/utils/serving/dispatch.sh inferencex-e2e/configs/serving/<name>.yaml`, as the `serving-bench` skill describes. Watch the runs with the `live-logs` / `debug-runs` skills. When something fails, use `superpowers:systematic-debugging`.

- [ ] **Step 1: Run the full local suite before pushing**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/ -n 4 -q`
Expected: PASS.

- [ ] **Step 2: Ask the user to confirm the push, then push and dispatch the six smoke configs**

- [ ] **Step 3: Check each run against the risk table**

| Check | Pass condition | Fix if it fails |
|---|---|---|
| Engine starts with the full context | Server log reports the context length at 8192 or 32768 | Raise `--gpu-memory-utilization` / `--mem-fraction-static`; never lower the context |
| AIPerf parses responses | `profile_export_aiperf.json` has `request_latency`, and the error count is about 0 | See the next two rows |
| SGLang `/v1/rerank` response shape | Same as above, on the SGLang rerank run | Patch `utils/aiperf/src/aiperf/endpoints/cohere_rankings.py` `extract_rankings` so it also accepts a top-level list, read as `{"index": r.get("index"), "score": r.get("relevance_score", r.get("score"))}`. Add an aiperf unit test with a list body, then commit |
| SGLang loads the cross-encoder | `bge-reranker-v2-m3-sglang-rerank` server becomes ready | Remove that config and record the reason in the PR description |
| The fixed-concurrency aggregator handles a run with no output tokens | The job is green and writes a result JSON | Find the crash site in `infx/results/agentic/` from the job log. Make that read return null for a missing output metric. Write a test that feeds a pooling-shaped `profile_export_aiperf.json` (as in `POOLING_EXPORT`) through the real function and asserts the null. Commit |

- [ ] **Step 4: Re-dispatch any config that a fix touched, until all six are green, or until SGLang rerank has been dropped with a recorded reason**

---

### Task 7: Full grids and report

- [ ] **Step 1: Widen each smoke config to the full grid, as two files per model × engine**

`<name>.yaml` becomes the search grid: remove `overrides`, set `conc: 1`, and put `search: true` in the workload.
- embed (bge-m3): `workload: {isl: [512, 2k, 8k], batch: [1, 32], search: true}`
- embed (Qwen3-Embedding-8B): `workload: {isl: [512, 2k, 8k, 32k], batch: [1, 32], search: true}`
- rerank: `workload: {passage: [512, 2k], docs: [10, 40], search: true}`

`<name>-ladder.yaml` is the fixed-concurrency ladder. Its `name` gets the `-ladder` suffix, and it uses a single grid point per file because the schema allows only one conc when there are several points:
- embed: `workload: {isl: [2k], batch: [1]}`
- rerank: `workload: {passage: [512], docs: [40]}`
- both: `conc: [1, 4, 16, 64]`

- [ ] **Step 2: Validate**

Run `pytest ... infx/tests/serving/test_configs.py`. Then commit with the usual author flags. The subject is `serving: embed/rerank full grids`, and the body carries a `中文：` line.

- [ ] **Step 3: Ask the user before pushing, then dispatch and report**

Dispatch all configs. Produce the report with the `result-report` skill, and run the `humanizer` skill on the written report first. Then open the PR into `vng-benchmark`. Its title must be bilingual, for example `Embed/rerank serving benchmarks / 嵌入与重排序服务基准测试`. The body is English with a collapsed `中文` section and a collapsed AI model disclosure. Open it only after the user confirms.
