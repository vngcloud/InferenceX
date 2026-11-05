# Embed/rerank TPM @ SLO (PTU sizing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report input TPM at a p90 SLO for six request sizes (bge-m3 embed, bge-reranker-v2-m3 rerank) on the remote MaaS endpoints.

**Architecture:** Two small code changes in `inferencex-e2e/infx/serving/`: the embed batch and rerank doc
count accept any positive integer, and the pooling sweep result gains a `tpm` field (from AIPerf input
tokens for embed, from a label-derived formula for rerank). Six serving YAMLs, one per request size, drive
the existing remote-endpoint path. No AIPerf change.

**Tech Stack:** Python 3.12, pydantic, pytest, AIPerf `adaptive_search`, `utils/serving/dispatch.sh` (GitHub Actions `gn-benchmark.yml`).

Spec: `docs/superpowers/specs/2026-11-03-embed-rerank-ptu-design.md`.

## Global Constraints

- Branch `feat/embed-rerank-ptu`. Do not modify `utils/aiperf/`.
- Rerank tokens per request = `docs × (RERANK_QUERY_TOKENS + passage)`, `RERANK_QUERY_TOKENS = 32`.
- `tpm` = `input_token_throughput.avg × 60` when AIPerf reports it, else `rps × rerank tokens per request × 60`, rounded to int.
- SLOs: embed `isl512-b1` p90 < 200 ms, `isl512-b32` < 1 s, `isl2k-b16` < 2 s; rerank `p512-n10` < 500 ms, `p512-n40` < 1 s, `p512-n100` < 2 s.
- Same-endpoint YAMLs run one after another, never in parallel.
- Ask the user before any `git push` or GitHub dispatch. `git add` named files only.
- Commit as `-c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai`.
- Local checks (from `inferencex-e2e/`):
  `uvx --exclude-newer PT12H ruff@latest check --fix infx`, `uvx --exclude-newer PT12H ruff@latest format infx`,
  `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/ -n 4`.

---

### Task 1: Any positive batch / doc count

**Files:**
- Modify: `inferencex-e2e/infx/serving/scenarios.py:48,50,267-270,279-282`
- Test: `inferencex-e2e/infx/tests/serving/test_scenarios.py:155-196`

**Interfaces:**
- Produces: `EmbedPoint(isl, batch)` / `RerankPoint(passage, docs)` render for any `batch >= 1` / `docs >= 1`; `0` raises `ScenarioError("workload.batch: 0 is not a positive integer")` (same for `workload.docs`). `RERANK_PASSAGE`, `RERANK_QUERY_TOKENS` unchanged (Task 2 imports them).

- [ ] **Step 1: Update the tests**

In `test_scenarios.py`, change the two error cases at lines 191 and 193:

```python
        ("embed", EmbedPoint("2k", 0), None, "workload.batch: 0"),
        ...
        ("rerank", RerankPoint("512", 0), None, "workload.docs: 0"),
```

and add after the rerank render test (line ~178):

```python
def test_pooling_sizes_beyond_the_nim_grid_render():
    embed = render("embed", {}, EmbedPoint("2k", 16), search=True)
    assert embed.doc["benchmark"]["datasets"][0]["prompts"] == {"isl": 2048, "batchSize": 16}
    assert embed.label == "sweep-embed-isl2k-b16"
    rerank = render("rerank", {}, RerankPoint("512", 100), search=True)
    assert rerank.doc["benchmark"]["datasets"][0]["rankings"]["passages"] == 100
    assert rerank.label == "sweep-rerank-p512-n100"
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_scenarios.py -q`
Expected: the new test fails (`workload.batch: 16 is not one of 1, 32`), the `0` cases fail on message text.

- [ ] **Step 3: Implement**

In `scenarios.py` delete `EMBED_BATCH = (1, 32)` and `RERANK_DOCS = (10, 40)`, and replace the two checks:

```python
        if point.batch < 1:
            raise ScenarioError(f"workload.batch: {point.batch!r} is not a positive integer")
```

```python
        if point.docs < 1:
            raise ScenarioError(f"workload.docs: {point.docs!r} is not a positive integer")
```

Run `grep -rn "EMBED_BATCH\|RERANK_DOCS" inferencex-e2e/` and remove any remaining reference.

- [ ] **Step 4: Run, expect PASS**

Run: same pytest command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add inferencex-e2e/infx/serving/scenarios.py inferencex-e2e/infx/tests/serving/test_scenarios.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): any positive embed batch and rerank doc count"
```

---

### Task 2: `tpm` in the pooling sweep result

**Files:**
- Modify: `inferencex-e2e/infx/serving/sweep_result.py:107-152,180-182`
- Test: `inferencex-e2e/infx/tests/serving/test_sweep_result.py:146-186`

**Interfaces:**
- Consumes: `RERANK_PASSAGE: dict[str, int]`, `RERANK_QUERY_TOKENS: int` from `infx.serving.scenarios`.
- Produces: `rerank_request_tokens(label: str) -> int | None`;
  `probe_metrics(probe_export, conc, items=None, request_tokens=None) -> dict`; pooling results carry `tpm: int | None`.

- [ ] **Step 1: Write the failing tests**

In `test_sweep_result.py` import `rerank_request_tokens`, add `"tpm": 39321600,` to the expected dict in
`test_pooling_probe_metrics_count_items_not_output_tokens` (655360 × 60), and add:

```python
@pytest.mark.parametrize(
    "label, tokens",
    [
        ("sweep-rerank-p512-n10", 5440),
        ("sweep-rerank-p512-n100-d60", 54400),
        ("sweep-embed-isl2k-b16", None),
    ],
)
def test_rerank_request_tokens_count_the_query_per_passage(label, tokens):
    assert rerank_request_tokens(label) == tokens


def test_rerank_tpm_comes_from_the_request_size_when_aiperf_counts_no_tokens():
    export = {"request_throughput": {"avg": 2.0}, "request_latency": POOLING_EXPORT["request_latency"]}
    out = build_sweep_result(
        _fixture_history(), export, _env(SERVING_SCENARIO_LABEL="sweep-rerank-p512-n10")
    )
    assert (out["input_tps"], out["tpm"]) == (None, 2 * 10 * 544 * 60)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run --locked --all-extras --group test python -m pytest -c pyproject.toml infx/tests/serving/test_sweep_result.py -q`
Expected: ImportError on `rerank_request_tokens`.

- [ ] **Step 3: Implement**

In `sweep_result.py` add the import and, after `items_per_request`:

```python
from infx.serving.scenarios import RERANK_PASSAGE, RERANK_QUERY_TOKENS


def rerank_request_tokens(label: str) -> int | None:
    """Tokens the server counts for one rerank request: the query is scored with every passage."""
    match = re.search(r"-rerank-p([^-]+)-n(\d+)", label)
    if not match or match.group(1) not in RERANK_PASSAGE:
        return None
    return int(match.group(2)) * (RERANK_QUERY_TOKENS + RERANK_PASSAGE[match.group(1)])
```

Thread `request_tokens` through:

```python
def probe_metrics(
    probe_export: dict,
    conc: int,  # noqa: ARG001
    items: int | None = None,
    request_tokens: int | None = None,
) -> dict:
    ...
    if items is not None:
        return _pooling_probe_metrics(probe_export, items, request_tokens)
```

```python
def _pooling_probe_metrics(probe_export: dict, items: int, request_tokens: int | None) -> dict:
    """Embed/rerank metrics: no output tokens; throughput in items (inputs or passages) per second.

    AIPerf counts no input tokens for cohere_rankings (aiperf#1516), so rerank TPM comes from the
    request size, which counts the query per passage as the server does.
    """
    ...
    rps = _stat(probe_export, "request_throughput", "avg")
    input_tps = _stat(probe_export, "input_token_throughput", "avg")
    tok_s = input_tps if input_tps is not None else (rps * request_tokens if rps and request_tokens else None)
    return {
        "rps": rps,
        "items_per_s": rps * items if rps is not None else None,
        "input_tps": input_tps,
        "tpm": round(tok_s * 60) if tok_s is not None else None,
        ...latency keys unchanged...
    }
```

In `build_sweep_result`:

```python
        label = env.get("SERVING_SCENARIO_LABEL", "")
        result.update(
            probe_metrics(
                probe_export, result["max_conc"], items_per_request(label), rerank_request_tokens(label)
            )
        )
```

If importing `infx.serving.scenarios` from `sweep_result.py` creates an import cycle, copy the two constants
with a `# ponytail:` comment pointing at `scenarios.py` instead.

- [ ] **Step 4: Run, expect PASS**, then the full local checks from Global Constraints.

- [ ] **Step 5: Commit**

```bash
git add inferencex-e2e/infx/serving/sweep_result.py inferencex-e2e/infx/tests/serving/test_sweep_result.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): input TPM in embed/rerank sweep results"
```

---

### Task 3: Six MaaS bge YAMLs, one per request size

**Files:**
- Create: `inferencex-e2e/configs/serving/maas-bge-m3-embed-isl512-b1.yaml`, `…-isl512-b32.yaml`, `…-isl2k-b16.yaml`,
  `maas-bge-reranker-v2-m3-rerank-p512-n10.yaml`, `…-n40.yaml`, `…-n100.yaml`
- Delete (untracked, superseded): `maas-bge-m3-embed-b1.yaml`, `maas-bge-m3-embed-b32.yaml`, `maas-bge-reranker-v2-m3-rerank.yaml`

**Interfaces:**
- Consumes: Task 1 (b16, n100 must validate).

- [ ] **Step 1: Write the files.** Embed template (change `name`, `workload`, `slo` per row):

```yaml
# Remote MaaS endpoint (Basic auth key in GitHub secret MAAS_BGE_M3_AUTH); TPM @ SLO for PTU sizing.
name: maas-bge-m3-embed-isl2k-b16
runner: rtx6000pro-hcm
endpoint:
  url: https://103.196.238.5.nip.io/v3/maas/baai/bge-m3
  model: bge-m3
  api-key-secret: MAAS_BGE_M3_AUTH
  tokenizer: BAAI/bge-m3
  tls-verify: false
scenario: embed
workload: {isl: [2k], batch: [16], search: true}
slo: "request_latency:p90:lt:2000"
overrides: {duration: 60}
conc: 1
```

Rows: `isl512-b1` → `{isl: [512], batch: [1]}`, `lt:200`; `isl512-b32` → `{isl: [512], batch: [32]}`, `lt:1000`.

Rerank template:

```yaml
# Remote MaaS endpoint (Basic auth key in GitHub secret MAAS_BGE_RERANKER_V2_M3_AUTH); TPM @ SLO for PTU sizing.
name: maas-bge-reranker-v2-m3-rerank-p512-n100
runner: rtx6000pro-hcm
endpoint:
  url: https://103.196.238.5.nip.io/v1/maas/baai/bge-reranker-v2-m3
  model: bge-reranker-v2-m3
  api-key-secret: MAAS_BGE_RERANKER_V2_M3_AUTH
  tokenizer: BAAI/bge-reranker-v2-m3
  tls-verify: false
scenario: rerank
workload: {passage: [512], docs: [100], search: true}
slo: "request_latency:p90:lt:2000"
overrides: {duration: 60}
conc: 1
```

Rows: `n10` → `docs: [10]`, `lt:500`; `n40` → `docs: [40]`, `lt:1000`.

- [ ] **Step 2: Validate**

Run from `inferencex-e2e/`:
`for f in configs/serving/maas-bge-*-isl*.yaml configs/serving/maas-bge-*-p512-*.yaml; do python -P -m infx.serving.to_matrix --check $f || echo FAIL $f; done`
Expected: six run titles, no `FAIL`.

- [ ] **Step 3: Delete the three superseded YAMLs, commit the six new ones**

```bash
rm inferencex-e2e/configs/serving/maas-bge-m3-embed-b1.yaml inferencex-e2e/configs/serving/maas-bge-m3-embed-b32.yaml inferencex-e2e/configs/serving/maas-bge-reranker-v2-m3-rerank.yaml
git add inferencex-e2e/configs/serving/maas-bge-m3-embed-isl*.yaml inferencex-e2e/configs/serving/maas-bge-reranker-v2-m3-rerank-p512-*.yaml
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "serving: MaaS bge embed/rerank TPM @ SLO configs"
```

---

### Task 4: Remote runs and the Product table

Needs the user: GitHub secrets set (command in `docs/handoffs/2026-11-03-embed-rerank-maas-benchmark.md` §1, workspace root), and an OK to push the branch and dispatch.

- [ ] **Step 1:** Ask the user to set `MAAS_BGE_M3_AUTH` / `MAAS_BGE_RERANKER_V2_M3_AUTH` and confirm push + dispatch.
- [ ] **Step 2:** `git push -u origin feat/embed-rerank-ptu`, then dispatch the three embed YAMLs one after another
  (`utils/serving/dispatch.sh <yaml> --ref feat/embed-rerank-ptu`, next one after the previous run ends), and the three
  rerank YAMLs the same way in parallel with the embed chain.
- [ ] **Step 3:** For each run read the sweep result JSON (`state`, `max_conc`, `rps`, `tpm`, `latency_p90_ms`). Build the
  six-row table from the spec (`Items / request`, `Tokens / item`, `Input tok / request`, `p90 SLO`, `RPM max = rps × 60`,
  `TPM in = tpm`, `p90 đo được`, `Run` link). Rows with `no_pass_in_range` show the state and the c1 latency from
  `search_history.json`.
- [ ] **Step 4:** Cross-check one rerank row: `tpm ÷ (rps × 60)` must equal 5,440 / 21,760 / 54,400; the gateway count is
  ~0.8% higher (spec §TPM).
- [ ] **Step 5:** Show the table to the user; write it to the `maas_model_capacity.xlsx` summary sheet only on their OK.
