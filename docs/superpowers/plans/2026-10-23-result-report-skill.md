# result-report skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repo skill `result-report` whose stdlib-only script turns InferenceX benchmark runs (GitHub Actions run links/IDs or local artifact/aiperf folders) into one standard `report.md` (+ optional `report.html` with SVG charts) and `data.json`, downloading only the files it needs; plus fixed-CCU synthetic scenarios and one real test run on the HAN 1×H200 node.

**Architecture:** Three modules behind one CLI in `.agents/skills/result-report/scripts/`: `fetch.py` finds configs (one benchmark job = aggregate JSON + raw results tree) and reads files from local dirs or remote artifact zips via HTTP Range; `extract.py` turns each config into a "point" dict (identity, throughput, latency, cache tiers, search summary) and groups points into arms; `render.py` turns the data into blocks, then markdown or HTML. `result_report.py` wires them and writes the output folder.

**Tech Stack:** Python ≥3.10 stdlib only (`zipfile`, `urllib`, `json`, `re`, `argparse`, `unittest`, `http.server` for tests), `gh` CLI for GitHub API + auth token. Fixed-CCU scenarios: existing `utils/serving/gen_sweep_scenarios.py` (pytest via `uv`).

Spec: `docs/superpowers/specs/2026-10-23-result-report-skill-design.md`.

## Global Constraints

- Skill path `.agents/skills/result-report/`; symlink `.claude/skills/result-report -> ../../.agents/skills/result-report`.
- Script is self-contained: stdlib only, imports nothing from `infx/` or `utils/`. Logic copied from the repo carries a `# source: <file>:<line>` comment.
- Never read by default: fixed-mode `server_metrics_export.json`, `profile_export.jsonl`, `inputs.json`, `*timeslices*`. (`--per-request` from the spec is deferred: no consumer yet; noted in the spec update in Task 9.)
- Latency in the report is **ms** with `avg/p50/p90/p95/p99`. Aggregate JSON latency is seconds and has no p99 → ×1000, p99 = `—`.
- Missing value renders as `—` (U+2014), never `0`.
- Cache hit compared nominal vs **overall** (not GPU tier); mismatch when |measured − nominal| > 0.10.
- SGLang: exclude a `cache_source=total` series from the overall sum. vLLM: CPU tier = `—` (not separable from L3).
- Output default `~/bench-reports/<YYYY-MM-DD>-<slug>/`; cache under `<out>/.cache/`. Nothing written inside the repo at runtime.
- Report language English.
- Full-zip fallback only when Range fails; refuse above `--max-full-zip-mb` (default 100) with exit code 3.
- Git: commit locally per task with `-c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai`; `git add` only the task's files; **never push without asking the user first** (user rule).
- Skill tests: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v` (run from repo root).
- Repo tests: `uv run -q --no-project --with pytest --with pyyaml --with pydantic python -m pytest <file> -q`.

## File Structure

| File | Responsibility |
|---|---|
| `utils/serving/gen_sweep_scenarios.py` (modify) | also write `scenarios/fixed-isl*-osl*-cache*.yaml` (no `sweep:` block) |
| `utils/test_sweep_scenarios.py` (modify) | tests for the fixed twins |
| `scenarios/fixed-isl*-osl*-cache*.yaml` (24 new, generated) | synthetic workloads at fixed CCU |
| `serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml`, `...-cache50.yaml` (new) | the HAN 1×H200 test arm |
| `.agents/skills/result-report/SKILL.md` | agent-facing instructions |
| `.agents/skills/result-report/scripts/fetch.py` | sources, local discovery, Range zip reader, remote discovery |
| `.agents/skills/result-report/scripts/extract.py` | point building, cache tiers, search summary, arms, `build_data` |
| `.agents/skills/result-report/scripts/render.py` | blocks, markdown, HTML, SVG charts |
| `.agents/skills/result-report/scripts/result_report.py` | CLI |
| `.agents/skills/result-report/scripts/test_fetch.py`, `test_extract.py`, `test_render.py`, `test_cli.py` | unittest per module |
| `.agents/skills/result-report/scripts/fixtures/` | trimmed real data (built once in Task 3) |
| `.claude/skills/result-report` (symlink) | Claude compatibility |

## Shared data shapes (all tasks)

`Config` (fetch.py): `rf: str` (pairing key), `source: Source`, `agg: dict | None` (aggregate JSON), `files: DirReader | ZipReader | None` (results root; `read(rel) -> bytes | None`, `names() -> list[str]`).

Point (extract.py → render.py):

```text
{
 "rf": str, "source": str, "arm": str, "scenario": str | None,
 "grid": {"kind": "search"|"fixed", "isl": "8k", "osl": "1k", "cache": 50} | None,
 "mode": "fixed"|"search", "conc": int | None,
 "identity": {model, image, framework, precision, tp, ep, dp_attention, spec_decoding, num_gpus, hw},
 "metrics": {"throughput": {rps, input_tps, output_tps, total_tps, output_tps_per_gpu},
             "ttft": {avg,p50,p90,p95,p99}, "itl": {...}, "e2e": {...},
             "tps_per_user", "error_rate_pct", "errors", "client_cache_hit",
             "source": "aiperf"|"aggregate"|"search_aggregate"} | None,
 "cache": {"overall","gpu","cpu","external","router", "tokens_by_tier": {str: float},
           "engine": str|None, "source": "aggregate"|"server_metrics"|"client"} | None,
 "cache_check": {"nominal","measured","mismatch"} | None,
 "search": {"state","convergence_reason","iterations","max_conc","infeasible_min","binding_constraint",
            "boundary_type","sla": [str],"sla_filters": [dict],"binding": str|None,"breach_observed": float|None,
            "best_trials_conc": int|None,
            "probes": [{"iter","conc","feasible","objective","observed": {"<tag>.<stat>": float|None},
                        "output_tps","cache_overall","winner": bool,"best_objective": bool}]} | None,
 "notes": [str]
}
```

Data (`data.json`): `{"title", "generated", "inputs": [str], "baseline": str|None, "arms": [{"label","identity","sources": [str],"config": {filename: text|None}}], "points": [point], "notes": [str], "bytes_fetched": int, "findings": str|None}`.

---

### Task 1: Fixed-CCU twins of the synthetic sweep scenarios

**Files:**
- Modify: `utils/serving/gen_sweep_scenarios.py`
- Modify: `utils/test_sweep_scenarios.py`
- Create (generated): `scenarios/fixed-isl{8k,32k,64k,128k}-osl{512,1k}-cache{0,50,90}.yaml` (24 files)

**Interfaces:**
- Produces: `scenario_name(isl_label, osl_label, cache, kind="sweep") -> str`; `build_scenario(isl, osl, cache, search=True) -> dict`; `_scenario_text(isl, osl, cache, search=True) -> str`; `FIXED_HEADER: str`. Scenario files `scenarios/fixed-isl8k-osl1k-cache0.yaml` and `...-cache50.yaml` used by Task 2.

- [ ] **Step 1: Write the failing tests** — append to `utils/test_sweep_scenarios.py` (and add `FIXED_HEADER, SCENARIOS_DIR` to its import list from `utils.serving.gen_sweep_scenarios`):

```python
def test_fixed_name() -> None:
    assert scenario_name("8k", "1k", 50, kind="fixed") == "fixed-isl8k-osl1k-cache50"


def test_fixed_twin_has_no_sweep_block_and_same_workload() -> None:
    for il, isl, ol, osl, cache in grid():
        fixed = build_scenario(isl, osl, cache, search=False)
        assert "sweep" not in fixed
        assert fixed["benchmark"] == build_scenario(isl, osl, cache)["benchmark"]
        assert yaml.safe_load(_scenario_text(isl, osl, cache, search=False)) == fixed


def test_fixed_files_on_disk_match_generator() -> None:
    for il, isl, ol, osl, cache in grid():
        path = SCENARIOS_DIR / f"{scenario_name(il, ol, cache, kind='fixed')}.yaml"
        assert path.read_text() == FIXED_HEADER + _scenario_text(isl, osl, cache, search=False)


def test_fixed_scenario_is_not_a_search() -> None:
    from infx.serving.to_matrix import is_sweep

    assert not is_sweep("fixed-isl8k-osl1k-cache50")
    assert is_sweep("sweep-isl8k-osl1k-cache50")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run -q --no-project --with pytest --with pyyaml --with pydantic python -m pytest utils/test_sweep_scenarios.py -q`
Expected: FAIL — `ImportError: cannot import name 'FIXED_HEADER'`.

- [ ] **Step 3: Implement** — in `utils/serving/gen_sweep_scenarios.py`:

Update the module docstring first line to `"""Write scenarios/{sweep,fixed}-isl*-osl*-cache*.yaml: the grid lives only in this file.` and add a bullet: `- fixed-*: same workload without the sweep: block; the serving YAML's conc list sets the concurrency (one job per conc).`

Add after `HEADER`:

```python
FIXED_HEADER = (
    "# Generated by utils/serving/gen_sweep_scenarios.py - do not edit by hand.\n"
    "# Synthetic ISL/OSL/cache-hit workload at fixed concurrency: twin of sweep-* without\n"
    "# the sweep: block, so the serving YAML's conc list drives the run.\n"
)
```

Replace `scenario_name`:

```python
def scenario_name(isl_label: str, osl_label: str, cache: int, kind: str = "sweep") -> str:
    return f"{kind}-isl{isl_label}-osl{osl_label}-cache{cache}"
```

`build_scenario(isl, osl, cache, search: bool = True)`: build the dict exactly as today, then before `return`, if not search, drop the key:

```python
    doc = {
        "schemaVersion": "2.0",
        "benchmark": {...unchanged...},
        "sweep": {...unchanged...},
        "randomSeed": 42,
    }
    if not search:
        del doc["sweep"]
    return doc
```

`_scenario_text(isl, osl, cache, search: bool = True)`: split the final `lines += [...]` into the phases part, the sweep part and the trailer:

```python
    lines += [
        "  phases:",
        "  - name: profiling",
        "    kind: profiling",
        "    type: concurrency",
        "    duration: 300.0",
        "    gracePeriod: 30.0",
        "    failedRequestThreshold: 0.1",
    ]
    if search:
        lines += [
            "sweep:",
            "  type: adaptive_search",
            "  planner: smooth_isotonic",
            "  objectives:",
            "  - {metric: output_token_throughput, stat: avg, direction: maximize}",
            "  search_space:",
            "  - {path: phases.profiling.concurrency, lo: 1, hi: 512, kind: int}",
            "  sla_filters:",
            "  - {metric_tag: time_to_first_token, stat: avg, op: lt, threshold: 8000.0}",
            "  - {metric_tag: inter_token_latency, stat: p90, op: lt, threshold: 50.0}",
            "  max_iterations: 16",
            "  sla_precision: normal",
        ]
    lines.append("randomSeed: 42")
    return "\n".join(lines) + "\n"
```

`main()`:

```python
def main() -> None:
    for il, iv, ol, ov, cache in grid():
        for kind, header, search in (("sweep", HEADER, True), ("fixed", FIXED_HEADER, False)):
            path = SCENARIOS_DIR / f"{scenario_name(il, ol, cache, kind)}.yaml"
            path.write_text(header + _scenario_text(iv, ov, cache, search=search))
            print(path.relative_to(REPO))
```

- [ ] **Step 4: Generate the files and verify the sweep files did not change**

Run: `python3 utils/serving/gen_sweep_scenarios.py >/dev/null && git status --short scenarios/ | sort | uniq -c | head; git diff --stat scenarios/`
Expected: 24 new `?? scenarios/fixed-*.yaml`; `git diff --stat scenarios/` prints nothing (sweep files byte-identical).

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run -q --no-project --with pytest --with pyyaml --with pydantic python -m pytest utils/test_sweep_scenarios.py utils/test_serving_dispatch.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add utils/serving/gen_sweep_scenarios.py utils/test_sweep_scenarios.py scenarios/fixed-isl*-osl*-cache*.yaml
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(scenarios): fixed-CCU twins of the synthetic sweep grid (fixed-isl*-osl*-cache*)"
```

---

### Task 2: HAN 1×H200 test run (fixed CCU, HiCache) — runs in the background

**Files:**
- Create: `serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml`
- Create: `serving/gemma4-hicache-1gpu-isl8k-osl1k-cache50.yaml`

**Interfaces:**
- Consumes: Task 1 scenarios `fixed-isl8k-osl1k-cache0`, `fixed-isl8k-osl1k-cache50`.
- Produces: two run URLs (2 jobs each: c8, c32) for acceptance case 7 in Task 10. Record them in the task report.

Why two files: a serving YAML may not combine a scenario list with more than one `conc` (`infx/serving/schema.py:378`). The two files differ only in `name` and `scenario`; result-report strips the `-isl8k-osl1k-cacheN` name suffix and ignores `name/scenario/conc` when grouping, so both runs form one arm.

- [ ] **Step 1: Write `serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml`**

```yaml
# result-report test arm: Gemma-4-31B FP8-block, SGLang v0.5.22, 1xH200 (HAN), HiCache on,
# synthetic ISL 8k / OSL 1k at fixed CCU. Copied from serving/gemma4-sweep-smoke-1xh200.yaml;
# added HiCache, fixed-CCU scenario and a two-step conc ladder. Twin: ...-cache50.yaml.
name: gemma4-hicache-1gpu-isl8k-osl1k-cache0
runner: h200-greennode-slurm-1x_02
image: lmsysorg/sglang:v0.5.22-cu130
model: RedHatAI/gemma-4-31B-it-FP8-block
gpus: 1
precision: fp8
env:
  NCCL_P2P_LEVEL: NVL
serve: |
  python3 -m sglang.launch_server
  --model-path RedHatAI/gemma-4-31B-it-FP8-block
  --tp 1
  --attention-backend fa4
  --chunked-prefill-size 16384
  --max-prefill-tokens 32768
  --schedule-policy shortest-prefill-first
  --swa-full-tokens-ratio 0.15
  --mem-fraction-static 0.90
  --page-size 1
  --context-length 262144
  --enable-hierarchical-cache --hicache-size 40
  --enable-metrics --enable-cache-report
  --reasoning-parser gemma4 --tool-call-parser gemma4
scenario: fixed-isl8k-osl1k-cache0
conc: [8, 32]
```

`--hicache-size 40` (GB host pool) keeps the job under the 100G Slurm `--mem` of `runners/launch_h200-greennode-slurm-1x.sh:71`.

- [ ] **Step 2: Write the cache50 twin** — identical except two lines:

```bash
sed -e 's/^name: .*/name: gemma4-hicache-1gpu-isl8k-osl1k-cache50/' \
    -e 's/^scenario: .*/scenario: fixed-isl8k-osl1k-cache50/' \
    serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml > serving/gemma4-hicache-1gpu-isl8k-osl1k-cache50.yaml
diff serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml serving/gemma4-hicache-1gpu-isl8k-osl1k-cache50.yaml
```

Expected diff: only the `name:` and `scenario:` lines.

- [ ] **Step 3: Validate both locally (same check CI runs)**

Run: `for f in serving/gemma4-hicache-1gpu-isl8k-osl1k-cache{0,50}.yaml; do PYTHONPATH=. uv run -q --no-project --python 3.12 --with pydantic --with pyyaml python -P -m infx.serving.to_matrix --check "$f" || echo FAIL $f; done`
Expected: two title lines, no `FAIL`.

- [ ] **Step 4: Commit**

```bash
git add serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml serving/gemma4-hicache-1gpu-isl8k-osl1k-cache50.yaml
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): Gemma4 1xH200 HiCache arm on fixed-isl8k-osl1k-cache{0,50}, c8/c32 (result-report test run)"
```

- [ ] **Step 5: STOP — ask the user to approve the push.** `dispatch.sh` refuses unless `origin/vng-benchmark` equals local HEAD. Ask: "Push Task 1+2 commits to origin/vng-benchmark so the HAN test run can be dispatched?" Only after an explicit yes:

```bash
git pull --rebase -q origin vng-benchmark && git push -q origin vng-benchmark
utils/serving/dispatch.sh serving/gemma4-hicache-1gpu-isl8k-osl1k-cache0.yaml
utils/serving/dispatch.sh serving/gemma4-hicache-1gpu-isl8k-osl1k-cache50.yaml
```

Expected: each prints `https://github.com/vngcloud/InferenceX/actions/runs/<id>`. Record both URLs. Do not wait for them; continue with Task 3. The 4 jobs queue on one GPU (~1–1.5 h total).

- [ ] **Step 6: Failure fallback (only if a job fails at server start).** Read the job's `server.log` (`gh run view <id> -R vngcloud/InferenceX --log-failed | tail -80`). If SGLang rejects HiCache for this model (hybrid SWA), first drop `--page-size 1`; if it still fails, remove `--enable-hierarchical-cache --hicache-size 40` from both files, commit `fix(serving): drop HiCache from gemma4 result-report test arm (<reason>)`, ask before pushing again, re-dispatch. Acceptance case 7 then checks the CPU column is hidden instead of present.

---

### Task 3: Skill skeleton, fixtures, sources and local discovery

**Files:**
- Create: `.agents/skills/result-report/scripts/fetch.py` (local part)
- Create: `.agents/skills/result-report/scripts/test_fetch.py`
- Create: `.agents/skills/result-report/scripts/fixtures/**` (generated by Step 1)
- Create: symlink `.claude/skills/result-report`

**Interfaces:**
- Produces: `DEFAULT_REPO`, `AGG_PREFIX = "bmk_agentic_"`, `RAW_PREFIX = "agentic_"`, `FETCHED: dict[str, int]`, `Source(kind, repo, run_id, path)` with `.link`, `parse_source(arg, default_repo=DEFAULT_REPO) -> Source`, `DirReader(root)` with `names()`/`read(rel)`, `Config(rf, source, agg, files)`, `local_configs(src) -> list[Config]`, `discover(src, cache_dir, max_full_zip_mb) -> tuple[list[Config], list[str]]` (local branch now; remote branch added in Task 4).
- Fixtures (paths relative to `scripts/fixtures/`):
  - `gh_fixed/bmk_agentic_glm53_c108/glm53_c108.json` — real GLM-5.3 c108 aggregate (sglang, HiCache host 43%, 8 GPUs, 397 validator errors)
  - `gh_fixed/agentic_glm53_c108/{serving.yaml,benchmark_command.txt,server_command.txt,router_command.txt}` and `.../aiperf_artifacts/profile_export_aiperf.json` (trimmed)
  - `aggonly/bmk_agentic_gemma4_c96/gemma4_c96.json` — real Gemma4 c96 aggregate, no raw tree
  - `search_local/ag/{serving.yaml,benchmark_command.txt}`, `search_local/ag/aiperf_artifacts/search_history.json`, `search_local/ag/aiperf_artifacts/search_iter_00NN/profile_runs/run_0001/{profile_export_aiperf.json,server_metrics_export.json}` (trimmed, 12 probes), `search_local/bmk/gemma4_search.json` (real search aggregate)
  - `raw_aiperf/glm5.1_public_ccu16_dur900/profile_export_aiperf.json` (trimmed, keeps `input_config.loadgen.concurrency`)
  - `sglang_server_metrics.json` — handmade (Step 1)

- [ ] **Step 1: Build the fixtures from the real data already on disk**

Run from repo root:

```bash
mkdir -p /tmp/rr-build && cat > /tmp/rr-build/make_fixtures.py <<'EOF'
import json, shutil, sys
from pathlib import Path

F = Path(".agents/skills/result-report/scripts/fixtures")
KEEP = {"time_to_first_token", "inter_token_latency", "request_latency", "request_throughput",
        "input_token_throughput", "output_token_throughput", "total_token_throughput",
        "output_token_throughput_per_user", "error_request_count", "request_error_rate",
        "request_count", "overall_usage_prompt_cache_read_pct"}
SM_PREFIXES = ("vllm:prompt_tokens_by_source", "vllm:prefix_cache_", "vllm:external_prefix_cache_",
               "sglang:prompt_tokens", "sglang:cached_tokens", "smg_cache_aware_hit")

def write(dst, doc):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, indent=1) + "\n")

def trim_export(src, dst):
    doc = json.loads(Path(src).read_text())
    out = {k: v for k, v in doc.items() if k in KEEP}
    loadgen = (doc.get("input_config") or {}).get("loadgen")
    if loadgen and loadgen.get("concurrency") is not None:
        out["input_config"] = {"loadgen": {"concurrency": loadgen["concurrency"]}}
    write(dst, out)

def trim_sm(src, dst):
    metrics = json.loads(Path(src).read_text())["metrics"]
    keep = {k: v for k, v in metrics.items() if k.startswith(SM_PREFIXES)}
    for v in keep.values():
        v.pop("description", None)
        for s in v.get("series", []):
            s.pop("endpoint_url", None)
    write(dst, {"metrics": keep})

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

shutil.rmtree(F, ignore_errors=True)
fill = Path("/tmp/fill-1015/38843679057")
raw = next(fill.glob("agentic_*c108*"))
agg = next(next(fill.glob("bmk_agentic_*c108*")).glob("*.json"))
copy(agg, F / "gh_fixed/bmk_agentic_glm53_c108/glm53_c108.json")
for name in ("serving.yaml", "benchmark_command.txt", "server_command.txt", "router_command.txt"):
    copy(raw / name, F / "gh_fixed/agentic_glm53_c108" / name)
trim_export(raw / "aiperf_artifacts/profile_export_aiperf.json",
            F / "gh_fixed/agentic_glm53_c108/aiperf_artifacts/profile_export_aiperf.json")

only = next(Path("/tmp/run39558375380").glob("bmk_agentic_*/*.json"))
copy(only, F / "aggonly/bmk_agentic_gemma4_c96/gemma4_c96.json")

sweep = Path("/tmp/h200sweep-chk")
for name in ("serving.yaml", "benchmark_command.txt"):
    copy(sweep / "ag" / name, F / "search_local/ag" / name)
copy(sweep / "ag/aiperf_artifacts/search_history.json", F / "search_local/ag/aiperf_artifacts/search_history.json")
for it in sorted((sweep / "ag/aiperf_artifacts").glob("search_iter_*")):
    run = it / "profile_runs/run_0001"
    dst = F / "search_local/ag/aiperf_artifacts" / it.name / "profile_runs/run_0001"
    trim_export(run / "profile_export_aiperf.json", dst / "profile_export_aiperf.json")
    trim_sm(run / "server_metrics_export.json", dst / "server_metrics_export.json")
copy(next(sweep.glob("bmk/*.json")), F / "search_local/bmk/gemma4_search.json")

trim_export("/mnt/poc_tencent/artifacts/glm5.1_public_ccu16_dur900/profile_export_aiperf.json",
            F / "raw_aiperf/glm5.1_public_ccu16_dur900/profile_export_aiperf.json")

series = lambda label, key, total: {"labels": {label: key}, "stats": {"total": total}}
write(F / "sglang_server_metrics.json", {"metrics": {
    "sglang:prompt_tokens": {"type": "counter", "series": [{"labels": {"tp_rank": "0"}, "stats": {"total": 1000.0}}]},
    "sglang:cached_tokens": {"type": "counter", "series": [
        series("cache_source", "device", 500.0), series("cache_source", "host", 200.0),
        series("cache_source", "storage_mooncake", 100.0), series("cache_source", "total", 800.0)]},
    "smg_cache_aware_hit": {"type": "counter", "series": [
        series("result", "hit", 90.0), series("result", "miss", 10.0)]},
}})
print(sum(p.stat().st_size for p in F.rglob("*") if p.is_file()), "bytes")
EOF
python3 -I /tmp/rr-build/make_fixtures.py
grep -rliE 'hf_[A-Za-z0-9]{8}|api_key"?: *"[^<]|password|secret' .agents/skills/result-report/scripts/fixtures && echo "SECRET FOUND - stop" || echo "no secrets"
```

Expected: a byte total under 400000 and `no secrets`.

- [ ] **Step 2: Write the failing tests** — `.agents/skills/result-report/scripts/test_fetch.py`:

```python
import tempfile
import unittest
from pathlib import Path

import fetch

FIX = Path(__file__).resolve().parent / "fixtures"


class ParseSourceTest(unittest.TestCase):
    def test_run_url(self):
        src = fetch.parse_source("https://github.com/vngcloud/InferenceX/actions/runs/39558375380/job/1")
        self.assertEqual((src.kind, src.repo, src.run_id), ("run", "vngcloud/InferenceX", "39558375380"))
        self.assertEqual(src.link, "https://github.com/vngcloud/InferenceX/actions/runs/39558375380")

    def test_bare_id_uses_default_repo(self):
        src = fetch.parse_source("123", default_repo="o/r")
        self.assertEqual((src.kind, src.repo, src.run_id), ("run", "o/r", "123"))

    def test_dir(self):
        src = fetch.parse_source(str(FIX / "gh_fixed"))
        self.assertEqual(src.kind, "dir")
        self.assertEqual(src.path, (FIX / "gh_fixed").resolve())

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            fetch.parse_source("/no/such/dir")


class LocalConfigsTest(unittest.TestCase):
    def configs(self, name):
        return fetch.local_configs(fetch.parse_source(str(FIX / name)))

    def test_gh_layout_pairs_on_rf(self):
        [cfg] = self.configs("gh_fixed")
        self.assertEqual(cfg.rf, "glm53_c108")
        self.assertEqual(cfg.agg["conc"], 108)
        self.assertIn(b"--concurrency 108", cfg.files.read("benchmark_command.txt"))
        self.assertIsNone(cfg.files.read("missing.txt"))

    def test_hand_named_layout_pairs_the_lone_aggregate_with_the_lone_root(self):
        [cfg] = self.configs("search_local")
        self.assertEqual(cfg.agg["state"], "ok")
        self.assertIsNotNone(cfg.files.read("aiperf_artifacts/search_history.json"))

    def test_raw_aiperf_dir_is_a_root_without_aggregate(self):
        [cfg] = self.configs("raw_aiperf")
        self.assertIsNone(cfg.agg)
        self.assertIsNotNone(cfg.files.read("profile_export_aiperf.json"))
        self.assertIn("ccu16", cfg.rf)

    def test_aggregate_only(self):
        [cfg] = self.configs("aggonly")
        self.assertIsNone(cfg.files)
        self.assertEqual(cfg.agg["conc"], 96)

    def test_empty_dir_gives_a_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs, notes = fetch.discover(fetch.parse_source(tmp), Path(tmp) / ".cache", 100)
        self.assertEqual(configs, [])
        self.assertIn("no aggregate JSON or aiperf export", notes[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch'`.

- [ ] **Step 4: Implement** — `.agents/skills/result-report/scripts/fetch.py`:

```python
"""Find benchmark configs in a GitHub Actions run or a local folder, and read their files.

A config is one benchmark job: the small aggregate JSON (artifact ``bmk_agentic_<RF>``)
plus the raw results tree (artifact ``agentic_<RF>``). Remote artifact zips are read
member by member over HTTP Range requests, so only the needed bytes are downloaded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO = "vngcloud/InferenceX"
AGG_PREFIX = "bmk_agentic_"
RAW_PREFIX = "agentic_"
RUN_URL_RE = re.compile(r"github\.com/([^/\s]+/[^/\s]+)/actions/runs/(\d+)")
FETCHED = {"bytes": 0}  # downloaded bytes, reported by the CLI


@dataclass(frozen=True)
class Source:
    kind: str  # "run" or "dir"
    repo: str | None = None
    run_id: str | None = None
    path: Path | None = None

    @property
    def link(self) -> str:
        if self.kind == "run":
            return f"https://github.com/{self.repo}/actions/runs/{self.run_id}"
        return str(self.path)


def parse_source(arg: str, default_repo: str = DEFAULT_REPO) -> Source:
    match = RUN_URL_RE.search(arg)
    if match:
        return Source("run", repo=match.group(1), run_id=match.group(2))
    if arg.isdigit():
        return Source("run", repo=default_repo, run_id=arg)
    path = Path(arg).expanduser()
    if path.is_dir():
        return Source("dir", path=path.resolve())
    raise ValueError(f"not a run URL, run id or directory: {arg}")


class DirReader:
    """Read files under one local results root."""

    def __init__(self, root: Path):
        self.root = root

    def names(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def read(self, rel: str) -> bytes | None:
        path = self.root / rel
        return path.read_bytes() if path.is_file() else None


@dataclass
class Config:
    rf: str  # pairing key: artifact suffix <RF>, or a relative path for hand-named folders
    source: Source
    agg: dict | None  # aggregate JSON
    files: object | None  # DirReader / ZipReader over the results root


def _is_aggregate(doc: object) -> bool:
    return isinstance(doc, dict) and "infmax_model_prefix" in doc and ("request_metrics" in doc or "state" in doc)


def _key(rel: Path, prefix: str) -> str:
    """GH artifact folders pair on their <RF> suffix; anything else keys on its relative path."""
    for part in reversed(rel.parts):
        if part.startswith(prefix):
            return part[len(prefix):]
    return rel.as_posix()


def local_configs(src: Source) -> list[Config]:
    root = src.path
    aggs: dict[str, dict] = {}
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root)
        if any(p.startswith("search_iter_") or p in ("aiperf_artifacts", "aggregate") for p in rel.parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if _is_aggregate(doc):
            aggs[_key(rel, AGG_PREFIX)] = doc
    roots: dict[str, Path] = {}
    for name in ("profile_export_aiperf.json", "search_history.json"):
        for path in sorted(root.rglob(name)):
            rel = path.relative_to(root)
            if any(p.startswith("search_iter_") or p == "aggregate" for p in rel.parts):
                continue
            base = path.parent.parent if path.parent.name == "aiperf_artifacts" else path.parent
            roots.setdefault(_key(base.relative_to(root), RAW_PREFIX), base)
    lone_aggs = [k for k in aggs if k not in roots]
    lone_roots = [k for k in roots if k not in aggs]
    if len(lone_aggs) == 1 and len(lone_roots) == 1:  # e.g. hand-named ag/ + bmk/ copies
        roots[lone_aggs[0]] = roots.pop(lone_roots[0])
    return [
        Config(rf=k, source=src, agg=aggs.get(k), files=DirReader(roots[k]) if k in roots else None)
        for k in sorted(set(aggs) | set(roots))
    ]


def discover(src: Source, cache_dir: Path, max_full_zip_mb: float) -> tuple[list[Config], list[str]]:
    configs = local_configs(src)
    return configs, ([] if configs else [f"{src.link}: no aggregate JSON or aiperf export found"])
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: 10 tests OK.

- [ ] **Step 6: Symlink and commit**

```bash
ln -s ../../.agents/skills/result-report .claude/skills/result-report
git add .agents/skills/result-report/scripts/fetch.py .agents/skills/result-report/scripts/test_fetch.py .agents/skills/result-report/scripts/fixtures .claude/skills/result-report
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): sources, local discovery and trimmed real fixtures"
```

---

### Task 4: Remote runs — Range zip reader, artifact listing, full-zip fallback

**Files:**
- Modify: `.agents/skills/result-report/scripts/fetch.py`
- Modify: `.agents/skills/result-report/scripts/test_fetch.py`

**Interfaces:**
- Consumes: Task 3 `Source`, `Config`, `DirReader`, `FETCHED`, `local_configs`.
- Produces: `RangeFile(url)` (seekable, `.size`, `.fetched`), `blob_url(repo, artifact_id) -> str`, `TooLarge(Exception)`, `ZipReader(repo, artifact: dict, cache_dir: Path, max_full_zip_mb: float, url_for=None)` with `names()`/`read(rel)`, `remote_configs(src, cache_dir, max_full_zip_mb) -> (list[Config], list[str])`; `discover` now dispatches on `src.kind`. Internal `_gh(*args) -> str` is the only subprocess call (tests patch it).

- [ ] **Step 1: Write the failing tests** — append to `test_fetch.py` (add imports `import http.server, io, json, os, re, threading, zipfile` and `from unittest import mock` at the top):

```python
def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data, compress_type=zipfile.ZIP_STORED)
    return buf.getvalue()


class _Handler(http.server.BaseHTTPRequestHandler):
    payloads: dict = {}
    ranges = True

    def do_GET(self):
        body = self.payloads[self.path]
        match = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
        if not (self.ranges and match):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        start, end = int(match[1]), min(int(match[2]), len(body) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(body[start:end + 1])

    def log_message(self, *args):
        pass


class RemoteTest(unittest.TestCase):
    def setUp(self):
        self.big = os.urandom(2_000_000)
        _Handler.ranges = True
        _Handler.payloads = {
            "/raw": _zip_bytes({"server.log": self.big, "aiperf_artifacts/profile_export_aiperf.json": b'{"x": 1}'}),
            "/agg": _zip_bytes({"RF.json": json.dumps({"infmax_model_prefix": "m", "conc": 8, "request_metrics": {}}).encode()}),
        }
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.shutdown()
        self.tmp.cleanup()

    def test_range_read_fetches_a_fraction_of_the_zip(self):
        before = fetch.FETCHED["bytes"]
        zf = zipfile.ZipFile(fetch.RangeFile(self.base + "/raw"))
        self.assertEqual(zf.read("aiperf_artifacts/profile_export_aiperf.json"), b'{"x": 1}')
        self.assertLess(fetch.FETCHED["bytes"] - before, len(_Handler.payloads["/raw"]) // 2)

    def test_zip_reader_caches_members(self):
        art = {"id": 7, "name": "agentic_RF", "size_in_bytes": 10, "workflow_run": {"id": 1}}
        reader = fetch.ZipReader("o/r", art, Path(self.tmp.name), 100, url_for=lambda repo, i: self.base + "/raw")
        self.assertEqual(reader.read("aiperf_artifacts/profile_export_aiperf.json"), b'{"x": 1}')
        self.assertIsNone(reader.read("nope.txt"))
        cached = Path(self.tmp.name) / "7" / "files" / "aiperf_artifacts/profile_export_aiperf.json"
        self.assertTrue(cached.is_file())

    def test_no_range_support_and_big_zip_raises_too_large(self):
        _Handler.ranges = False
        art = {"id": 8, "name": "agentic_RF", "size_in_bytes": 500_000_000, "workflow_run": {"id": 1}}
        reader = fetch.ZipReader("o/r", art, Path(self.tmp.name), 100, url_for=lambda repo, i: self.base + "/raw")
        with self.assertRaises(fetch.TooLarge):
            reader.names()

    def test_remote_configs_pairs_artifacts_and_reports_failures(self):
        arts = [
            {"id": 1, "name": "bmk_agentic_RF", "size_in_bytes": 300, "expired": False, "workflow_run": {"id": 5}},
            {"id": 2, "name": "agentic_RF", "size_in_bytes": 2_000_000, "expired": False, "workflow_run": {"id": 5}},
            {"id": 3, "name": "bmk_agentic_OLD", "size_in_bytes": 300, "expired": True, "workflow_run": {"id": 5}},
        ]
        jobs = [{"name": "bench c8", "conclusion": "success"}, {"name": "bench c32", "conclusion": "failure"}]

        def fake_gh(*args):
            path = args[1]
            if "/jobs" in path:
                return json.dumps({"total_count": 2, "jobs": jobs})
            return json.dumps({"total_count": 3, "artifacts": arts})

        urls = {1: self.base + "/agg", 2: self.base + "/raw"}
        with mock.patch.object(fetch, "_gh", side_effect=fake_gh), \
                mock.patch.object(fetch, "blob_url", side_effect=lambda repo, i: urls[i]):
            configs, notes = fetch.remote_configs(fetch.parse_source("5", "o/r"), Path(self.tmp.name), 100)
        [cfg] = configs
        self.assertEqual((cfg.rf, cfg.agg["conc"]), ("RF", 8))
        self.assertEqual(cfg.files.read("aiperf_artifacts/profile_export_aiperf.json"), b'{"x": 1}')
        self.assertTrue(any("bench c32" in n for n in notes))
        self.assertTrue(any("bmk_agentic_OLD" in n and "expired" in n for n in notes))
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: 4 new errors — `AttributeError: module 'fetch' has no attribute 'RangeFile'`.

- [ ] **Step 3: Implement** — in `fetch.py` add imports `io, subprocess, urllib.error, urllib.request, zipfile`, constants, classes and functions, and replace `discover`:

```python
MIN_FETCH = 256 * 1024  # bytes per Range request; zipfile asks for small chunks
FAILED_CONCLUSIONS = {"failure", "cancelled", "timed_out"}


class TooLarge(Exception):
    """Range reads failed and the full artifact zip is over the --max-full-zip-mb budget."""


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests: enough for zipfile.ZipFile."""

    def __init__(self, url: str):
        self.url, self.pos, self.fetched = url, 0, 0
        self._start, self._buf = 0, b""
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as resp:
            if resp.status != 206:
                raise OSError(f"server ignored Range (HTTP {resp.status})")
            self.size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self._start <= self.pos and self.pos + n <= self._start + len(self._buf)):
            end = min(self.pos + max(n, MIN_FETCH), self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            with urllib.request.urlopen(req) as resp:
                self._buf = resp.read()
            self._start = self.pos
            self.fetched += len(self._buf)
            FETCHED["bytes"] += len(self._buf)
        offset = self.pos - self._start
        data = self._buf[offset:offset + n]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _gh_pages(path: str, key: str) -> list[dict]:
    items, page = [], 1
    while True:
        doc = json.loads(_gh("api", f"{path}?per_page=100&page={page}"))
        items += doc[key]
        if not doc[key] or len(items) >= doc["total_count"]:
            return items
        page += 1


def blob_url(repo: str, artifact_id: int) -> str:
    """The short-lived blob URL the artifact zip endpoint redirects to (it supports Range)."""
    token = _gh("auth", "token").strip()
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact_id}/zip",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        urllib.request.build_opener(_NoRedirect).open(request)
    except urllib.error.HTTPError as err:
        if err.code in (301, 302, 303, 307, 308):
            return err.headers["Location"]
        raise
    raise OSError(f"artifact {artifact_id}: expected a redirect to the zip blob")


class ZipReader:
    """Members of one remote artifact zip, each cached under cache_dir/<artifact id>/."""

    def __init__(self, repo: str, artifact: dict, cache_dir: Path, max_full_zip_mb: float, url_for=None):
        self.repo, self.artifact, self.max_mb = repo, artifact, max_full_zip_mb
        self.cache = cache_dir / str(artifact["id"])
        self._url_for = url_for
        self._zip: zipfile.ZipFile | None = None
        self._dir: DirReader | None = None

    def _backend(self):
        if self._zip is None and self._dir is None:
            try:
                url = (self._url_for or blob_url)(self.repo, self.artifact["id"])
                self._zip = zipfile.ZipFile(RangeFile(url))
            except (OSError, zipfile.BadZipFile) as err:
                self._dir = DirReader(self._download_full(err))
        return self._zip if self._zip is not None else self._dir

    def _download_full(self, reason: Exception) -> Path:
        size_mb = self.artifact["size_in_bytes"] / 1e6
        if size_mb > self.max_mb:
            raise TooLarge(
                f"{self.artifact['name']}: Range read failed ({reason}); the full zip is "
                f"{size_mb:.0f} MB > --max-full-zip-mb {self.max_mb:g}"
            )
        dest = self.cache / "full"
        if not dest.is_dir():
            _gh("run", "download", str(self.artifact["workflow_run"]["id"]), "-R", self.repo,
                "-n", self.artifact["name"], "-D", str(dest))
            FETCHED["bytes"] += self.artifact["size_in_bytes"]
        return dest

    def names(self) -> list[str]:
        listing = self.cache / "names.json"
        if listing.is_file():
            return json.loads(listing.read_text())
        backend = self._backend()
        names = backend.namelist() if isinstance(backend, zipfile.ZipFile) else backend.names()
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(json.dumps(names))
        return names

    def read(self, rel: str) -> bytes | None:
        cached = self.cache / "files" / rel
        if cached.is_file():
            return cached.read_bytes()
        if rel not in self.names():
            return None
        data = self._backend().read(rel)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return data


def remote_configs(src: Source, cache_dir: Path, max_full_zip_mb: float) -> tuple[list[Config], list[str]]:
    notes: list[str] = []
    jobs = _gh_pages(f"repos/{src.repo}/actions/runs/{src.run_id}/jobs", "jobs")
    failed = [job["name"] for job in jobs if job.get("conclusion") in FAILED_CONCLUSIONS]
    if failed:
        notes.append(f"{src.link}: {len(failed)} job(s) did not succeed: " + "; ".join(failed))
    parts: dict[str, dict[str, dict]] = {}
    for art in _gh_pages(f"repos/{src.repo}/actions/runs/{src.run_id}/artifacts", "artifacts"):
        for prefix, kind in ((AGG_PREFIX, "agg"), (RAW_PREFIX, "raw")):
            if not art["name"].startswith(prefix):
                continue
            if art.get("expired"):
                notes.append(f"{src.link}: artifact {art['name']} expired, skipped")
            else:
                parts.setdefault(art["name"][len(prefix):], {})[kind] = art
    configs: list[Config] = []
    for rf, found in sorted(parts.items()):
        raw = ZipReader(src.repo, found["raw"], cache_dir, max_full_zip_mb) if "raw" in found else None
        aggs: list[dict | None] = [None]
        if "agg" in found:
            reader = ZipReader(src.repo, found["agg"], cache_dir, max_full_zip_mb)
            aggs = [json.loads(reader.read(n)) for n in reader.names() if n.endswith(".json")] or [None]
        configs += [Config(rf=rf, source=src, agg=agg, files=raw) for agg in aggs]
    if not configs:
        notes.append(f"{src.link}: no bmk_agentic_* / agentic_* artifacts found")
    return configs, notes


def discover(src: Source, cache_dir: Path, max_full_zip_mb: float) -> tuple[list[Config], list[str]]:
    if src.kind == "run":
        return remote_configs(src, cache_dir, max_full_zip_mb)
    configs = local_configs(src)
    return configs, ([] if configs else [f"{src.link}: no aggregate JSON or aiperf export found"])
```

Note: `"bmk_agentic_X".startswith("agentic_")` is False, so the two prefixes never collide.

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: 14 tests OK.

- [ ] **Step 5: Live smoke against GitHub (read-only)**

Run:

```bash
cd .agents/skills/result-report/scripts && python3 - <<'EOF'
from pathlib import Path
import fetch
cfgs, notes = fetch.discover(fetch.parse_source("39642191354"), Path("/tmp/rr-smoke-cache"), 100)
print(len(cfgs), notes)
print(len(cfgs[0].files.names()), "members; fetched", fetch.FETCHED["bytes"], "bytes")
EOF
cd -
```

Expected: `1 []`, about 84 members, fetched well under 1 MB (the zip is 266 MB).

- [ ] **Step 6: Commit**

```bash
git add .agents/skills/result-report/scripts/fetch.py .agents/skills/result-report/scripts/test_fetch.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): read remote artifact zips over HTTP Range, full-zip fallback with size budget"
```

---

### Task 5: Metric parsers — aiperf export, aggregate fallback, cache tiers

**Files:**
- Create: `.agents/skills/result-report/scripts/extract.py` (parsers only)
- Create: `.agents/skills/result-report/scripts/test_extract.py`

**Interfaces:**
- Consumes: fixtures from Task 3.
- Produces: `PCTS = ("avg","p50","p90","p95","p99")`, `from_aiperf(export) -> metrics`, `from_aggregate(agg) -> metrics` (source `"aggregate"`, p99 None), `from_search_aggregate(agg) -> metrics` (source `"search_aggregate"`), `aggregate_tiers(agg) -> cache | None`, `tiers_from_server_metrics(export) -> cache`, `ms_stats(entry) -> dict`, `tokens(label) -> int`, `parse_grid(scenario) -> dict | None`. Shapes: see "Shared data shapes".

- [ ] **Step 1: Write the failing tests** — `.agents/skills/result-report/scripts/test_extract.py`:

```python
import copy
import json
import unittest
from pathlib import Path

import extract

FIX = Path(__file__).resolve().parent / "fixtures"
load = lambda rel: json.loads((FIX / rel).read_text())
GLM_AGG = "gh_fixed/bmk_agentic_glm53_c108/glm53_c108.json"
GLM_EXPORT = "gh_fixed/agentic_glm53_c108/aiperf_artifacts/profile_export_aiperf.json"
PROBE = "search_local/ag/aiperf_artifacts/search_iter_{:04d}/profile_runs/run_0001/"


class GridTest(unittest.TestCase):
    def test_parse_grid(self):
        self.assertEqual(extract.parse_grid("fixed-isl8k-osl1k-cache50"),
                         {"kind": "fixed", "isl": "8k", "osl": "1k", "cache": 50})
        self.assertEqual(extract.parse_grid("sweep-isl128k-osl512-cache0")["kind"], "search")
        self.assertIsNone(extract.parse_grid("agentx-cc-062126"))
        self.assertIsNone(extract.parse_grid(None))

    def test_tokens(self):
        self.assertEqual((extract.tokens("8k"), extract.tokens("512")), (8192, 512))


class AiperfTest(unittest.TestCase):
    def test_latency_ms_with_p99_and_errors(self):
        m = extract.from_aiperf(load(GLM_EXPORT))
        self.assertAlmostEqual(m["ttft"]["p99"], 723840.0278512397, places=3)
        self.assertAlmostEqual(m["throughput"]["input_tps"], 78369.27687594405, places=3)
        self.assertAlmostEqual(m["error_rate_pct"], 0.5054759898904801, places=6)
        self.assertEqual(m["errors"], 18)
        self.assertAlmostEqual(m["client_cache_hit"], 0.945903879813367, places=6)
        self.assertEqual(m["source"], "aiperf")
        self.assertIsNone(m["throughput"]["output_tps_per_gpu"])

    def test_seconds_are_converted(self):
        m = extract.from_aiperf({"time_to_first_token": {"unit": "sec", "avg": 1.5, "p99": 2.0}})
        self.assertEqual((m["ttft"]["avg"], m["ttft"]["p99"]), (1500.0, 2000.0))

    def test_no_error_metrics_means_zero_errors(self):
        m = extract.from_aiperf({"request_count": {"unit": "requests", "avg": 417.0}})
        self.assertEqual((m["error_rate_pct"], m["errors"]), (0.0, 0))


class AggregateTest(unittest.TestCase):
    def test_fallback_latency_seconds_to_ms_without_p99(self):
        agg = load("aggonly/bmk_agentic_gemma4_c96/gemma4_c96.json")
        m = extract.from_aggregate(agg)
        self.assertAlmostEqual(m["ttft"]["avg"], 357.13, places=2)
        self.assertIsNone(m["ttft"]["p99"])
        self.assertAlmostEqual(m["throughput"]["rps"], 11765 / 1202.89864, places=4)
        self.assertEqual(m["source"], "aggregate")

    def test_sglang_tiers_from_aggregate(self):
        c = extract.aggregate_tiers(load(GLM_AGG))
        self.assertEqual((c["gpu"], c["cpu"], c["overall"]), (0.51417, 0.43174, 0.9459))
        self.assertIsNone(c["external"])
        self.assertIsNone(c["router"])
        self.assertEqual(c["tokens_by_tier"], {"device": 146673664, "host": 123159104})

    def test_total_label_is_not_double_counted(self):
        agg = copy.deepcopy(load(GLM_AGG))
        agg["server_metrics"]["cache"]["cached_tokens_by_source"] = {"device": 100, "host": 50, "total": 150}
        agg["server_metrics"]["tokens"]["prompt_total"] = 300
        c = extract.aggregate_tiers(agg)
        self.assertEqual(c["overall"], 0.5)
        self.assertNotIn("total", c["tokens_by_tier"])

    def test_vllm_cpu_tier_is_not_separable(self):
        agg = copy.deepcopy(load(GLM_AGG))
        agg["server_metrics"]["adapter"] = "vllm"
        self.assertIsNone(extract.aggregate_tiers(agg)["cpu"])

    def test_search_aggregate_metrics(self):
        m = extract.from_search_aggregate(load("search_local/bmk/gemma4_search.json"))
        self.assertAlmostEqual(m["throughput"]["output_tps"], 1334.90684, places=4)
        self.assertAlmostEqual(m["itl"]["p90"], 49.15992, places=4)
        self.assertIsNone(m["itl"]["p99"])


class ServerMetricsTest(unittest.TestCase):
    def test_sglang_tiers_router_and_total_label(self):
        c = extract.tiers_from_server_metrics(load("sglang_server_metrics.json"))
        self.assertEqual((c["gpu"], c["cpu"], c["external"], c["overall"], c["router"]), (0.5, 0.2, 0.1, 0.8, 0.9))
        self.assertNotIn("total", c["tokens_by_tier"])
        self.assertEqual(c["engine"], "sglang")

    def test_vllm_probe(self):
        c = extract.tiers_from_server_metrics(load(PROBE.format(10) + "server_metrics_export.json"))
        self.assertAlmostEqual(c["gpu"], 0.48894, places=4)
        self.assertAlmostEqual(c["overall"], 0.48894, places=4)
        self.assertEqual(c["external"], 0.0)
        self.assertIsNone(c["cpu"])
        self.assertEqual(c["engine"], "vllm")
        self.assertNotIn("local_compute", c["tokens_by_tier"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: `ModuleNotFoundError: No module named 'extract'`.

- [ ] **Step 3: Implement** — `.agents/skills/result-report/scripts/extract.py`:

```python
"""Turn benchmark files into report points: throughput, latency, cache tiers, search summary."""

from __future__ import annotations

import re

PCTS = ("avg", "p50", "p90", "p95", "p99")
GRID_RE = re.compile(r"^(sweep|fixed)-isl(\d+k?)-osl(\d+k?)-cache(\d+)$")
TO_MS = {"ms": 1.0, "sec": 1000.0, "s": 1000.0, "us": 1e-3, "ns": 1e-6}


def tokens(label: str) -> int:
    return int(label[:-1]) * 1024 if label.endswith("k") else int(label)


def parse_grid(scenario: str | None) -> dict | None:
    # source: utils/serving/collect_sweep_table.py:11 (widened to fixed-*; no raise on other names)
    match = GRID_RE.match(scenario or "")
    if not match:
        return None
    kind, isl, osl, cache = match.groups()
    return {"kind": "search" if kind == "sweep" else "fixed", "isl": isl, "osl": osl, "cache": int(cache)}


def ms_stats(entry: object) -> dict:
    if not isinstance(entry, dict):
        return dict.fromkeys(PCTS)
    scale = TO_MS.get(entry.get("unit", "ms"), 1.0)
    return {k: entry[k] * scale if entry.get(k) is not None else None for k in PCTS}


def _avg(export: dict, tag: str) -> float | None:
    entry = export.get(tag)
    return entry.get("avg") if isinstance(entry, dict) else None


def _empty_metrics(source: str) -> dict:
    return {
        "throughput": dict.fromkeys(("rps", "input_tps", "output_tps", "total_tps", "output_tps_per_gpu")),
        "ttft": dict.fromkeys(PCTS), "itl": dict.fromkeys(PCTS), "e2e": dict.fromkeys(PCTS),
        "tps_per_user": None, "error_rate_pct": None, "errors": None, "client_cache_hit": None,
        "source": source,
    }


def from_aiperf(export: dict) -> dict:
    """profile_export_aiperf.json: latency in ms with avg..p99, throughputs carry avg only."""
    m = _empty_metrics("aiperf")
    m["throughput"].update(
        rps=_avg(export, "request_throughput"),
        input_tps=_avg(export, "input_token_throughput"),
        output_tps=_avg(export, "output_token_throughput"),
        total_tps=_avg(export, "total_token_throughput"),
    )
    m["ttft"] = ms_stats(export.get("time_to_first_token"))
    m["itl"] = ms_stats(export.get("inter_token_latency"))
    m["e2e"] = ms_stats(export.get("request_latency"))
    m["tps_per_user"] = _avg(export, "output_token_throughput_per_user")
    rate, errors = _avg(export, "request_error_rate"), _avg(export, "error_request_count")
    if rate is None and errors is None and _avg(export, "request_count") is not None:
        rate, errors = 0.0, 0  # aiperf omits the error metrics when no request failed
    m["error_rate_pct"] = rate
    m["errors"] = int(errors) if errors is not None else None
    hit = _avg(export, "overall_usage_prompt_cache_read_pct")
    m["client_cache_hit"] = hit / 100 if hit is not None else None
    return m


def _ms(seconds: float | None) -> float | None:
    return seconds * 1000.0 if seconds is not None else None


def from_aggregate(agg: dict) -> dict:
    """Aggregate JSON fallback: latency in seconds with mean..p95 (no p99)."""
    m = _empty_metrics("aggregate")
    rm = agg.get("request_metrics") or {}
    latency, thr = rm.get("latency") or {}, rm.get("throughput") or {}
    for key, src in (("ttft", "ttft"), ("itl", "itl"), ("e2e", "e2el")):
        d = latency.get(src) or {}
        m[key] = {"avg": _ms(d.get("mean")), "p50": _ms(d.get("p50")), "p90": _ms(d.get("p90")),
                  "p95": _ms(d.get("p95")), "p99": None}
    duration, ok = thr.get("duration_seconds"), agg.get("num_requests_successful")
    m["throughput"].update(
        rps=ok / duration if ok and duration else None,
        input_tps=(thr.get("input") or {}).get("tokens_per_second"),
        output_tps=(thr.get("output") or {}).get("tokens_per_second"),
        total_tps=(thr.get("total") or {}).get("tokens_per_second"),
    )
    return m


def from_search_aggregate(agg: dict) -> dict:
    """Search-mode aggregate (infx/results/sweep.py): winning probe's rps/tps and ttft/itl avg+p90 in ms."""
    m = _empty_metrics("search_aggregate")
    m["throughput"].update(rps=agg.get("rps"), output_tps=agg.get("output_tps"))
    m["ttft"].update(avg=agg.get("ttft_avg_ms"), p90=agg.get("ttft_p90_ms"))
    m["itl"].update(avg=agg.get("itl_avg_ms"), p90=agg.get("itl_p90_ms"))
    return m


def _cache(source: str, engine: str | None) -> dict:
    return {"overall": None, "gpu": None, "cpu": None, "external": None, "router": None,
            "tokens_by_tier": {}, "engine": engine, "source": source}


def aggregate_tiers(agg: dict) -> dict | None:
    sm = agg.get("server_metrics") or {}
    if not sm.get("present"):
        return None
    cache = sm.get("cache") or {}
    out = _cache("aggregate", sm.get("adapter"))
    tiers = dict(cache.get("cached_tokens_by_source") or {})
    overall = cache.get("overall_cache_hit_rate")
    if "total" in tiers:  # the repo adapter sums a cache_source=total series into overall
        tiers.pop("total")
        prompt_total = (sm.get("tokens") or {}).get("prompt_total")
        overall = sum(tiers.values()) / prompt_total if prompt_total else None
    router = cache.get("router_kv_hit_rate")
    out.update(
        overall=overall,
        gpu=cache.get("gpu_cache_hit_rate"),
        # vLLM copies its external (offload) rate into cpu: the tiers are not separable
        cpu=None if out["engine"] == "vllm" else cache.get("cpu_cache_hit_rate"),
        external=cache.get("external_cache_hit_rate"),
        router=router if router is not None else cache.get("frontend_cache_hit_rate"),
        tokens_by_tier=tiers,
    )
    return out


def _series(metrics: dict, name: str) -> list[dict]:
    entry = metrics.get(name)
    return [s for s in entry.get("series", []) if isinstance(s, dict)] if isinstance(entry, dict) else []


def _value(series: dict) -> float | None:
    # source: infx/results/agentic/common.py:114 (series_stat, preferred total/sum/max/avg)
    stats = series.get("stats") or {}
    for key in ("total", "sum", "max", "avg"):
        if stats.get(key) is not None:
            return float(stats[key])
    return None


def _sum(metrics: dict, name: str) -> float | None:
    values = [v for s in _series(metrics, name) if (v := _value(s)) is not None]
    return sum(values) if values else None


def _by_label(metrics: dict, name: str, label: str) -> dict[str, float]:
    # source: infx/results/agentic/common.py:186 (sum_by_label)
    out: dict[str, float] = {}
    for s in _series(metrics, name):
        key, value = (s.get("labels") or {}).get(label), _value(s)
        if key is not None and value is not None:
            out[str(key)] = out.get(str(key), 0.0) + value
    return out


def _rate(num: float | None, den: float | None) -> float | None:
    return num / den if num is not None and den else None


def tiers_from_server_metrics(export: dict) -> dict:
    """Cache tiers from a (small, per-probe) server_metrics_export.json."""
    metrics = export.get("metrics") if isinstance(export.get("metrics"), dict) else {}
    names = set(metrics)
    if any(n.startswith("sglang:") for n in names):
        # source: infx/results/agentic/backends/sglang.py:53-81, minus the cache_source=total double count
        out = _cache("server_metrics", "sglang")
        prompt = _sum(metrics, "sglang:prompt_tokens")
        by = _by_label(metrics, "sglang:cached_tokens", "cache_source")
        by.pop("total", None)
        storage = [v for k, v in by.items() if k.startswith("storage_")]
        out.update(
            gpu=_rate(by.get("device"), prompt),
            cpu=_rate(by.get("host"), prompt),
            external=_rate(sum(storage), prompt) if storage else None,
            overall=_rate(sum(by.values()), prompt) if by else None,
            tokens_by_tier=by,
        )
    elif any(n.startswith("vllm:") for n in names):
        # source: infx/results/agentic/backends/vllm.py:44-95; cpu stays None (not separable from L3)
        out = _cache("server_metrics", "vllm")
        by = _by_label(metrics, "vllm:prompt_tokens_by_source", "source")
        total = sum(by.values())
        if total > 0:
            local, external = by.get("local_cache_hit"), by.get("external_kv_transfer")
            out.update(
                gpu=_rate(local, total),
                external=_rate(external, total),
                overall=((local or 0.0) + (external or 0.0)) / total,
                tokens_by_tier={k: by[k] for k in ("local_cache_hit", "external_kv_transfer") if k in by},
            )
        else:
            gpu = _rate(_sum(metrics, "vllm:prefix_cache_hits"), _sum(metrics, "vllm:prefix_cache_queries"))
            out.update(gpu=gpu, overall=gpu, external=_rate(
                _sum(metrics, "vllm:external_prefix_cache_hits"), _sum(metrics, "vllm:external_prefix_cache_queries")))
    else:
        out = _cache("server_metrics", None)
    hits = _by_label(metrics, "smg_cache_aware_hit", "result")
    if hits:
        out["router"] = _rate(hits.get("hit"), sum(hits.values()))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: all OK (14 fetch + 14 extract).

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/result-report/scripts/extract.py .agents/skills/result-report/scripts/test_extract.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): aiperf/aggregate metric parsers and tiered cache hit (sglang/vllm/smg)"
```

---

### Task 6: Points, search summary, arms, `build_data`

**Files:**
- Modify: `.agents/skills/result-report/scripts/extract.py`
- Modify: `.agents/skills/result-report/scripts/test_extract.py`

**Interfaces:**
- Consumes: Task 3/4 `Config`, `DirReader`, `Source`, `parse_source`, `local_configs`; Task 5 parsers.
- Produces: `summarize_search(history) -> dict`, `sla_text(filter) -> str`, `build_point(cfg) -> point` (includes a `"config"` dict that `build_data` moves onto the arm), `arm_of(point, cfg) -> (key, label)`, `build_data(configs, notes, baseline=None, title=None, inputs=(), generated="") -> data`. Raises `ValueError` for an unknown `--baseline`.

- [ ] **Step 1: Write the failing tests** — append to `test_extract.py` (add `import tempfile` and `import fetch` at the top):

```python
def _cfg(name):
    [cfg] = fetch.local_configs(fetch.parse_source(str(FIX / name)))
    return cfg


class SearchSummaryTest(unittest.TestCase):
    # source of the cases: infx/results/sweep.py:32 state order
    def test_states(self):
        ok = {"boundary_summary": {"feasible_max": {"value": 60}, "infeasible_min": {"value": 61},
                                   "boundary_type": "smooth"}, "convergence_reason": "precision", "iterations": [1, 2]}
        self.assertEqual(extract.summarize_search(ok)["state"], "ok")
        cliff = copy.deepcopy(ok); cliff["boundary_summary"]["boundary_type"] = "cliff"
        self.assertEqual(extract.summarize_search(cliff)["state"], "cliff")
        none = {"boundary_summary": {"feasible_max": None, "infeasible_min": {"value": 1}}}
        self.assertEqual(extract.summarize_search(none)["state"], "no_pass_in_range")
        self.assertEqual(extract.summarize_search({})["state"], "no_data")

    def test_sla_text(self):
        f = {"metric_tag": "inter_token_latency", "stat": "p90", "op": "lt", "threshold": 50.0}
        self.assertEqual(extract.sla_text(f), "ITL p90 < 50 ms")


class BuildPointTest(unittest.TestCase):
    def test_fixed_gh_layout(self):
        p = extract.build_point(_cfg("gh_fixed"))
        self.assertEqual((p["mode"], p["conc"], p["scenario"]), ("fixed", 108, "agentx-cc-062126"))
        self.assertIsNone(p["grid"])
        self.assertEqual(p["metrics"]["source"], "aiperf")
        self.assertAlmostEqual(p["metrics"]["throughput"]["output_tps_per_gpu"],
                               p["metrics"]["throughput"]["output_tps"] / 8)
        self.assertEqual(p["cache"]["cpu"], 0.43174)
        self.assertTrue(any("397" in n and "18" in n for n in p["notes"]))
        self.assertIn("--concurrency 108", p["config"]["benchmark_command.txt"])

    def test_aggregate_only_notes_missing_p99(self):
        p = extract.build_point(_cfg("aggonly"))
        self.assertEqual((p["metrics"]["source"], p["conc"]), ("aggregate", 96))
        self.assertIsNone(p["metrics"]["ttft"]["p99"])
        self.assertTrue(any("p99 unavailable" in n for n in p["notes"]))

    def test_search(self):
        p = extract.build_point(_cfg("search_local"))
        s = p["search"]
        self.assertEqual((p["mode"], p["conc"], s["state"]), ("search", 60, "ok"))
        self.assertEqual(p["scenario"], "sweep-isl8k-osl1k-cache50")
        self.assertEqual(p["grid"], {"kind": "search", "isl": "8k", "osl": "1k", "cache": 50})
        self.assertEqual(s["binding"], "ITL p90 < 50 ms")
        self.assertAlmostEqual(s["breach_observed"], 50.84856, places=4)
        self.assertEqual((s["infeasible_min"], s["best_trials_conc"]), (61, 48))
        self.assertEqual(len(s["probes"]), 12)
        winner = [pr for pr in s["probes"] if pr["winner"]]
        self.assertEqual([pr["conc"] for pr in winner], [60])
        self.assertAlmostEqual(p["metrics"]["throughput"]["output_tps"], 1334.9068416429348, places=6)
        self.assertAlmostEqual(winner[0]["observed"]["inter_token_latency.p90"], 49.1599168688172, places=6)
        self.assertAlmostEqual(p["cache"]["gpu"], 0.48894, places=4)
        self.assertEqual(p["cache_check"]["nominal"], 0.5)
        self.assertFalse(p["cache_check"]["mismatch"])

    def test_raw_aiperf(self):
        p = extract.build_point(_cfg("raw_aiperf"))
        self.assertEqual((p["mode"], p["conc"], p["scenario"]), ("fixed", 16, None))
        self.assertIsNone(p["cache"])
        self.assertEqual(p["metrics"]["source"], "aiperf")


class ArmsTest(unittest.TestCase):
    def _dir_cfg(self, root, name, scenario, conc):
        d = Path(root) / name
        (d / "aiperf_artifacts").mkdir(parents=True)
        (d / "serving.yaml").write_text(
            f"# comment {name}\nname: gemma4-hicache-1gpu-{scenario.removeprefix('fixed-')}\n"
            f"image: img\nserve: |\n  python3 -m sglang.launch_server --tp 1\nscenario: {scenario}\nconc: [8, 32]\n")
        (d / "benchmark_command.txt").write_text(
            f"aiperf profile --config /workspace/scenarios/{scenario}.yaml --concurrency {conc}\n")
        (d / "aiperf_artifacts/profile_export_aiperf.json").write_text(
            json.dumps({"output_token_throughput": {"unit": "tokens/sec", "avg": 100.0 + conc}}))
        return fetch.Config(rf=name, source=fetch.parse_source(root), agg=None, files=fetch.DirReader(d))

    def test_name_suffix_scenario_and_conc_do_not_split_an_arm(self):
        with tempfile.TemporaryDirectory() as root:
            cfgs = [self._dir_cfg(root, "a", "fixed-isl8k-osl1k-cache0", 8),
                    self._dir_cfg(root, "b", "fixed-isl8k-osl1k-cache50", 32)]
            data = extract.build_data(cfgs, [], inputs=[root], generated="2026-10-23")
        self.assertEqual([a["label"] for a in data["arms"]], ["gemma4-hicache-1gpu"])
        self.assertEqual(data["baseline"], "gemma4-hicache-1gpu")
        self.assertEqual({(p["scenario"], p["conc"]) for p in data["points"]},
                         {("fixed-isl8k-osl1k-cache0", 8), ("fixed-isl8k-osl1k-cache50", 32)})
        self.assertNotIn("config", data["points"][0])
        self.assertIn("serving.yaml", data["arms"][0]["config"])

    def test_unknown_baseline(self):
        with self.assertRaises(ValueError):
            extract.build_data([_cfg("gh_fixed")], [], baseline="nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: `AttributeError: module 'extract' has no attribute 'summarize_search'` (and the others).

- [ ] **Step 3: Implement** — append to `extract.py` (add `import hashlib, json` and `from pathlib import Path` to imports):

```python
SCENARIO_CMD_RE = re.compile(r"--config\s+\S*?scenarios/([\w.-]+)\.yaml")
CONC_CMD_RE = re.compile(r"--concurrency\s+(\d+)")
DIR_CONC_RE = re.compile(r"(?:ccu|conc|_c)(\d+)")
SERVING_SCENARIO_RE = re.compile(r"^scenario:\s*([\w.-]+)\s*$", re.M)
SERVING_NAME_RE = re.compile(r"^name:\s*(\S+)", re.M)
GRID_SUFFIX_RE = re.compile(r"-(?:(?:sweep|fixed)-)?isl\d+k?-osl\d+k?-cache\d+$")
RUN_SUFFIX_RE = re.compile(r"(?:_c\d+|_search)$")
COMMENT_RE = re.compile(r"^\s*#.*\n", re.M)
PER_RUN_KEYS_RE = re.compile(r"^(?:name|scenario|conc):.*\n(?:-\s.*\n)*", re.M)
IDENTITY_KEYS = ("model", "image", "framework", "precision", "tp", "ep", "dp_attention",
                 "spec_decoding", "num_gpus", "hw")
CONFIG_FILES = ("serving.yaml", "server_command.txt", "router_command.txt", "benchmark_command.txt")
SLA_NAMES = {"time_to_first_token": "TTFT", "inter_token_latency": "ITL", "request_latency": "E2E"}
SLA_OPS = {"lt": "<", "le": "≤", "gt": ">", "ge": "≥"}
PROBE_DIR = "aiperf_artifacts/search_iter_{:04d}/profile_runs/run_0001/"


def summarize_search(history: dict) -> dict:
    # source: infx/results/sweep.py:32 (copied: that module imports env-bound identity helpers)
    boundary = history.get("boundary_summary")
    reason = history.get("convergence_reason")
    base = {"convergence_reason": reason, "iterations": len(history.get("iterations") or []),
            "max_conc": None, "infeasible_min": None, "binding_constraint": None, "boundary_type": None}
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


def sla_text(f: dict) -> str:
    tag = f.get("metric_tag", "?")
    unit = " ms" if tag in SLA_NAMES else ""
    return f"{SLA_NAMES.get(tag, tag)} {f.get('stat')} {SLA_OPS.get(f.get('op'), f.get('op'))} {f.get('threshold'):g}{unit}"


def _read_json(files, *paths: str):
    for path in paths:
        data = files.read(path) if files else None
        if data is not None:
            return json.loads(data)
    return None


def _read_text(files, path: str) -> str | None:
    data = files.read(path) if files else None
    return data.decode("utf-8", "replace") if data is not None else None


def _match(pattern: re.Pattern, text: str | None) -> str | None:
    found = pattern.search(text or "")
    return found.group(1) if found else None


def _observed(export: dict | None, f: dict) -> float | None:
    if not export or f.get("stat") not in PCTS:
        return None
    return ms_stats(export.get(f.get("metric_tag")))[f["stat"]]


def search_summary(history: dict, files) -> tuple[dict, dict | None, dict | None]:
    """(search block, winning probe metrics, winning probe cache tiers)."""
    summary = summarize_search(history)
    filters = (history.get("config") or {}).get("sla_filters") or []
    boundary = history.get("boundary_summary") or {}
    winner = (boundary.get("feasible_max") or {}).get("iteration_idx")
    best = ((history.get("best_trials") or [{}])[0] or {}).get("iteration_idx")
    breach = (boundary.get("infeasible_min") or {}).get("first_breach")
    probes, win_metrics, win_cache = [], None, None
    for it in history.get("iterations") or []:
        idx = it["iteration_idx"]
        export = _read_json(files, PROBE_DIR.format(idx) + "profile_export_aiperf.json")
        sm = _read_json(files, PROBE_DIR.format(idx) + "server_metrics_export.json")
        metrics = from_aiperf(export) if export else None
        cache = tiers_from_server_metrics(sm) if sm else None
        if idx == winner:
            win_metrics, win_cache = metrics, cache
        probes.append({
            "iter": idx,
            "conc": next(iter((it.get("variation_values") or {}).values()), None),
            "feasible": it.get("feasible"),
            "objective": (it.get("objective_values") or [None])[0],
            "observed": {f"{f['metric_tag']}.{f['stat']}": _observed(export, f) for f in filters},
            "output_tps": metrics["throughput"]["output_tps"] if metrics else None,
            "cache_overall": cache["overall"] if cache else None,
            "winner": idx == winner,
            "best_objective": idx == best,
        })
    search = {
        **summary,
        "sla": [sla_text(f) for f in filters],
        "sla_filters": filters,
        "binding": sla_text(breach) if breach else None,
        "breach_observed": breach.get("observed") if breach else None,
        "best_trials_conc": next((p["conc"] for p in probes if p["best_objective"]), None),
        "probes": probes,
    }
    return search, win_metrics, win_cache


def build_point(cfg) -> dict:
    agg, files = cfg.agg or {}, cfg.files
    config = {name: _read_text(files, name) for name in CONFIG_FILES}
    serving, bench_cmd = config["serving.yaml"], config["benchmark_command.txt"]
    history = _read_json(files, "aiperf_artifacts/search_history.json", "search_history.json")
    scenario = agg.get("scenario") or _match(SCENARIO_CMD_RE, bench_cmd) or _match(SERVING_SCENARIO_RE, serving)
    notes: list[str] = []
    point = {"rf": cfg.rf, "source": cfg.source.link, "scenario": scenario, "grid": parse_grid(scenario),
             "identity": {k: agg.get(k) for k in IDENTITY_KEYS}, "config": config, "notes": notes,
             "search": None}
    if history:
        search, metrics, cache = search_summary(history, files)
        point.update(mode="search", conc=search["max_conc"], search=search)
        if metrics is None and agg.get("output_tps") is not None:
            metrics = from_search_aggregate(agg)
            notes.append("winning probe export missing: metrics from the search aggregate (avg/p90 only)")
    elif agg.get("state"):  # search aggregate without its raw tree
        point.update(mode="search", conc=agg.get("max_conc"), search=None)
        metrics, cache = from_search_aggregate(agg), None
        notes.append("search raw tree missing: no probe table; metrics from the search aggregate (avg/p90 only)")
    else:
        export = _read_json(files, "aiperf_artifacts/profile_export_aiperf.json", "profile_export_aiperf.json")
        point["mode"] = "fixed"
        if export:
            metrics = from_aiperf(export)
        elif agg.get("request_metrics"):
            metrics = from_aggregate(agg)
            notes.append("p99 unavailable: aggregate-only (no aiperf export); latency converted from seconds")
        else:
            metrics = None
        cache = aggregate_tiers(agg) if agg else None
        conc = (agg.get("conc") or _match(CONC_CMD_RE, bench_cmd)
                or (((export or {}).get("input_config") or {}).get("loadgen") or {}).get("concurrency")
                or _match(DIR_CONC_RE, cfg.rf))
        point["conc"] = int(conc) if conc is not None else None
    if cache is None and metrics and metrics.get("client_cache_hit") is not None:
        cache = {**_cache("client", None), "overall": metrics["client_cache_hit"]}
    gpus = agg.get("num_gpus")
    if metrics and gpus and metrics["throughput"]["output_tps"] is not None:
        metrics["throughput"]["output_tps_per_gpu"] = metrics["throughput"]["output_tps"] / gpus
    validator = (agg.get("request_accounting") or {}).get("records_error_dropped")
    errors = metrics.get("errors") if metrics else None
    if validator is not None and errors is not None and validator - errors > max(10, errors):
        notes.append(f"validator dropped {validator} error records vs {errors} aiperf errors")
    point["metrics"], point["cache"] = metrics, cache
    if point["grid"]:
        nominal = point["grid"]["cache"] / 100
        measured = cache["overall"] if cache else None
        point["cache_check"] = {"nominal": nominal, "measured": measured,
                                "mismatch": None if measured is None else abs(measured - nominal) > 0.10}
    else:
        point["cache_check"] = None
    return point


def _strip_run_suffix(name: str) -> str:
    return GRID_SUFFIX_RE.sub("", RUN_SUFFIX_RE.sub("", name))


def arm_of(point: dict, cfg) -> tuple[str, str]:
    """Arm = same serving config. name/scenario/conc and comments do not split an arm."""
    serving = point["config"]["serving.yaml"]
    if serving:
        body = PER_RUN_KEYS_RE.sub("", COMMENT_RE.sub("", serving))
        return ("yaml:" + hashlib.sha1(body.encode()).hexdigest()[:12],
                _strip_run_suffix(_match(SERVING_NAME_RE, serving) or cfg.rf))
    if cfg.agg:
        return ("id:" + json.dumps(point["identity"], sort_keys=True),
                _strip_run_suffix(cfg.agg.get("infmax_model_prefix") or cfg.rf))
    return "src:" + cfg.source.link, Path(cfg.source.link).name or cfg.source.link


def build_data(configs, notes, baseline=None, title=None, inputs=(), generated="") -> dict:
    arms: dict[str, dict] = {}
    points = []
    for cfg in configs:
        point = build_point(cfg)
        key, label = arm_of(point, cfg)
        config = point.pop("config")
        if key not in arms:
            taken, final, n = {a["label"] for a in arms.values()}, label, 2
            while final in taken:
                final, n = f"{label} #{n}", n + 1
            arms[key] = {"label": final, "identity": point["identity"], "sources": [], "config": config}
        arm = arms[key]
        if point["source"] not in arm["sources"]:
            arm["sources"].append(point["source"])
        point["arm"] = arm["label"]
        points.append(point)
    labels = [a["label"] for a in arms.values()]
    if baseline is not None and baseline not in labels:
        raise ValueError(f"--baseline {baseline!r} is not an arm; arms: {labels}")
    point_notes = [f"{p['arm']} · {p['scenario'] or '-'} · c{p['conc']}: {n}" for p in points for n in p["notes"]]
    return {
        "title": title or (labels[0] if labels else "benchmark report"),
        "generated": generated,
        "inputs": list(inputs),
        "baseline": baseline or (labels[0] if labels else None),
        "arms": list(arms.values()),
        "points": points,
        "notes": list(notes) + point_notes,
        "bytes_fetched": 0,
        "findings": None,
    }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/result-report/scripts/extract.py .agents/skills/result-report/scripts/test_extract.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): report points, search summary with probe table, arm grouping"
```

---

### Task 7: Markdown rendering (tables, deltas, grid matrices)

**Files:**
- Create: `.agents/skills/result-report/scripts/render.py`
- Create: `.agents/skills/result-report/scripts/test_render.py`

**Interfaces:**
- Consumes: Task 6 `build_data` output; `extract.parse_grid`, `extract.tokens`, `extract.sla_text`.
- Produces: `Table(headers, rows)` (row cell = `(text, tone)`, tone `"good"|"bad"|None`), `num(v)`, `pct(v)`, `cell(value, base, fmt, higher_better)`, `build_blocks(data, charts=False) -> list[tuple]`, `to_markdown(blocks) -> str`, `KEY_FINDINGS_PLACEHOLDER`. Block kinds: `("h", level, text)`, `("p", text)`, `("table", Table)`, `("code", lang, text)`, `("details", summary, blocks)`, `("chart", svg)` (HTML only; added in Task 8).

- [ ] **Step 1: Write the failing tests** — `.agents/skills/result-report/scripts/test_render.py`:

```python
import copy
import unittest
from pathlib import Path

import extract
import fetch
import render

FIX = Path(__file__).resolve().parent / "fixtures"


def data_for(*names, baseline=None):
    cfgs = [c for n in names for c in fetch.local_configs(fetch.parse_source(str(FIX / n)))]
    return extract.build_data(cfgs, [], baseline=baseline, inputs=[str(FIX / n) for n in names], generated="2026-10-23")


def grid_point(arm, scenario, conc, out_tps, ttft_p99, state=None, measured=0.5):
    g = extract.parse_grid(scenario)
    p = {"rf": scenario, "source": "s", "arm": arm, "scenario": scenario, "grid": g, "conc": conc,
         "mode": g["kind"], "identity": {}, "notes": [], "cache": {"overall": measured, "source": "aggregate"},
         "cache_check": {"nominal": g["cache"] / 100, "measured": measured,
                         "mismatch": abs(measured - g["cache"] / 100) > 0.10},
         "metrics": extract._empty_metrics("aiperf"), "search": None}
    p["metrics"]["throughput"]["output_tps"] = out_tps
    p["metrics"]["ttft"]["p99"] = ttft_p99
    p["metrics"]["itl"]["p99"] = ttft_p99 / 10 if ttft_p99 is not None else None
    if state:
        p["search"] = {"state": state, "max_conc": conc, "sla": [], "sla_filters": [], "binding": None,
                       "breach_observed": None, "best_trials_conc": None, "probes": [], "infeasible_min": None}
    return p


def grid_data(points, arms):
    return {"title": "t", "generated": "d", "inputs": [], "baseline": arms[0], "notes": [], "bytes_fetched": 0,
            "findings": None, "arms": [{"label": a, "identity": {}, "sources": [], "config": {}} for a in arms],
            "points": points}


class FormatTest(unittest.TestCase):
    def test_num_and_pct(self):
        self.assertEqual((render.num(None), render.num(78369.2), render.num(49.16), render.num(0.357)),
                         ("—", "78,369", "49.2", "0.36"))
        self.assertEqual(render.pct(0.43174), "43.2%")

    def test_delta_tone_follows_direction(self):
        self.assertEqual(render.cell(110.0, 100.0, render.num, True), ("110 (+10.0%)", "good"))
        self.assertEqual(render.cell(110.0, 100.0, render.num, False), ("110 (+10.0%)", "bad"))
        self.assertEqual(render.cell(110.0, None, render.num, True), ("110", None))


class ReportTest(unittest.TestCase):
    def test_fixed_report_hides_empty_cache_columns(self):
        md = render.to_markdown(render.build_blocks(data_for("gh_fixed")))
        self.assertIn("## Key findings", md)
        self.assertIn(render.KEY_FINDINGS_PLACEHOLDER, md)
        self.assertIn("## Scenario: agentx-cc-062126", md)
        self.assertIn("Hit CPU (DRAM)", md)
        self.assertIn("43.2%", md)
        self.assertNotIn("Hit external (L3)", md)
        self.assertIn("### TTFT (ms)", md)
        self.assertIn("723,840", md)
        self.assertIn("validator dropped 397", md)

    def test_aggregate_only_shows_dash_p99_and_note(self):
        md = render.to_markdown(render.build_blocks(data_for("aggonly")))
        self.assertIn("p99 unavailable", md)
        ttft = md.split("### TTFT (ms)")[1].split("###")[0]
        self.assertIn("| — |", ttft)

    def test_raw_folder_has_no_server_metrics(self):
        md = render.to_markdown(render.build_blocks(data_for("raw_aiperf")))
        self.assertIn("No server metrics logged", md)

    def test_search_headline_and_probe_table(self):
        md = render.to_markdown(render.build_blocks(data_for("search_local")))
        self.assertIn("max CCU = **60**", md)
        self.assertIn("binding: ITL p90 < 50 ms (observed 50.85 at c61)", md)
        self.assertIn("highest objective at c48", md)
        self.assertIn("★ max CCU", md)
        self.assertIn("## Grid summary", md)
        self.assertIn("| ISL 8k · OSL 1k | 60 · 1,335 |", md)

    def test_two_arms_get_deltas_against_baseline(self):
        data = data_for("gh_fixed")
        twin = copy.deepcopy(data["points"][0])
        twin["arm"] = "B"
        twin["metrics"]["throughput"]["output_tps"] *= 1.1
        data["points"].append(twin)
        data["arms"].append({"label": "B", "identity": {}, "sources": [], "config": {}})
        md = render.to_markdown(render.build_blocks(data))
        self.assertIn("(+10.0%)", md)
        self.assertIn("(baseline)", md)

    def test_search_grid_matrix_marks(self):
        pts = [grid_point("A", "sweep-isl8k-osl1k-cache0", 64, 2000.0, None, state="ok"),
               grid_point("A", "sweep-isl8k-osl1k-cache50", 0, None, None, state="no_pass_in_range"),
               grid_point("A", "sweep-isl32k-osl1k-cache0", 16, 900.0, None, state="cliff")]
        md = render.to_markdown(render.build_blocks(grid_data(pts, ["A"])))
        self.assertIn("| ISL 8k · OSL 1k | 64 · 2,000 | ✗ |", md)
        self.assertIn("| ISL 32k · OSL 1k | 16* · 900 | — |", md)

    def test_fixed_grid_matrices_and_cache_check(self):
        pts = [grid_point("A", "fixed-isl8k-osl1k-cache0", 8, 500.0, 900.0, measured=0.02),
               grid_point("A", "fixed-isl8k-osl1k-cache0", 32, 1500.0, 2500.0, measured=0.03),
               grid_point("A", "fixed-isl8k-osl1k-cache50", 8, 600.0, 700.0, measured=0.30)]
        md = render.to_markdown(render.build_blocks(grid_data(pts, ["A"])))
        self.assertIn("| Scenario | Arm | c8 | c32 |", md)
        self.assertIn("| fixed-isl8k-osl1k-cache50 | A | 600 | — |", md)
        self.assertIn("⚠", md)
        self.assertIn("<details><summary>Scenario: fixed-isl8k-osl1k-cache0</summary>", md)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: `ModuleNotFoundError: No module named 'render'`.

- [ ] **Step 3: Implement** — `.agents/skills/result-report/scripts/render.py`:

```python
"""Render report data (extract.build_data) as blocks, then markdown or HTML."""

from __future__ import annotations

import re
from dataclasses import dataclass

from extract import PCTS, parse_grid, sla_text, tokens

DASH = "—"
KEY_FINDINGS_PLACEHOLDER = "<!-- KEY_FINDINGS: write 3-6 bullets from data.json, then re-run with --findings (see SKILL.md) -->"
THROUGHPUT = [("RPS", ("rps",)), ("Input tok/s", ("input_tps",)), ("Output tok/s", ("output_tps",)),
              ("Total tok/s", ("total_tps",)), ("Output tok/s/GPU", ("output_tps_per_gpu",))]
CACHE = [("Hit overall", "overall"), ("Hit GPU (HBM)", "gpu"), ("Hit CPU (DRAM)", "cpu"),
         ("Hit external (L3)", "external"), ("Hit router", "router")]
LATENCY = [("TTFT", "ttft"), ("ITL", "itl"), ("E2E latency", "e2e")]
FIXED_MATRICES = [("Output tok/s", ("metrics", "throughput", "output_tps"), True),
                  ("TTFT p99 (ms)", ("metrics", "ttft", "p99"), False),
                  ("ITL p99 (ms)", ("metrics", "itl", "p99"), False)]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[tuple[str, str | None]]]


def num(v: float | None) -> str:
    if v is None:
        return DASH
    a = abs(v)
    return f"{v:,.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def pct(v: float | None) -> str:
    return DASH if v is None else f"{v * 100:.1f}%"


def err_pct(v: float | None) -> str:
    return DASH if v is None else f"{v:.2f}%"


def _tok(v: float) -> str:
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.1f}k" if v >= 1e3 else f"{v:.0f}"


def get(obj, *path):
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _delta(value: float, base: float, higher_better: bool) -> tuple[str, str | None]:
    d = (value - base) / abs(base) * 100
    tone = None if abs(d) < 0.5 else ("good" if (d > 0) == higher_better else "bad")
    return f" ({d:+.1f}%)", tone


def cell(value, base, fmt, higher_better) -> tuple[str, str | None]:
    text = fmt(value)
    if value is None or base is None or base == 0:
        return text, None
    suffix, tone = _delta(value, base, higher_better)
    return text + suffix, tone


def _arm_order(data) -> dict[str, int]:
    return {a["label"]: i for i, a in enumerate(data["arms"])}


def _scenario_sort(s):
    g = parse_grid(s)
    if g:
        return (0, tokens(g["isl"]), tokens(g["osl"]), g["cache"], s)
    return (1 if s else 2, 0, 0, 0, s or "")


def _label(data, arm: str) -> str:
    return arm + (" (baseline)" if arm == data["baseline"] and len(data["arms"]) > 1 else "")


def _conc(p) -> str:
    return DASH if p.get("conc") is None else str(p["conc"])


def _base(data, pts, p):
    if p["arm"] == data["baseline"]:
        return None
    for q in pts:
        if q["arm"] == data["baseline"] and q["scenario"] == p["scenario"] and (
                p["mode"] == "search" or q["conc"] == p["conc"]):
            return q
    return None


def drop_empty(headers, rows, droppable: set[int]) -> Table:
    keep = [i for i in range(len(headers)) if i not in droppable or any(r[i][0] != DASH for r in rows)]
    return Table([headers[i] for i in keep], [[r[i] for i in keep] for r in rows])


def _link(s: str) -> str:
    m = re.search(r"/runs/(\d+)$", s)
    return f"[run {m.group(1)}]({s})" if m else f"`{s}`"


def runs_table(data) -> Table:
    rows = []
    for a in data["arms"]:
        i = a["identity"]
        rows.append([(_label(data, a["label"]), None)] + [
            (str(v) if v is not None else DASH, None) for v in
            (i.get("model"), i.get("hw"), i.get("framework"), i.get("precision"), i.get("num_gpus"))
        ] + [(f"{i.get('tp') or DASH}/{i.get('ep') or DASH}", None), (i.get("image") or DASH, None),
             (", ".join(_link(s) for s in a["sources"]), None)])
    return Table(["Arm", "Model", "HW", "Framework", "Precision", "GPUs", "TP/EP", "Image", "Sources"], rows)


def throughput_table(data, pts) -> list:
    headers = ["Arm", "CCU", *(h for h, _ in THROUGHPUT), *(h for h, _ in CACHE), "Cached tokens by tier"]
    rows = []
    for p in pts:
        b = _base(data, pts, p)
        row = [(_label(data, p["arm"]), None), (_conc(p), None)]
        row += [cell(get(p, "metrics", "throughput", *k), get(b, "metrics", "throughput", *k), num, True)
                for _, k in THROUGHPUT]
        row += [cell(get(p, "cache", k), get(b, "cache", k), pct, True) for _, k in CACHE]
        if get(p, "cache", "source") == "client":
            text, tone = row[2 + len(THROUGHPUT)]
            row[2 + len(THROUGHPUT)] = (text + " (client)", tone)
        tiers = get(p, "cache", "tokens_by_tier") or {}
        row.append((" · ".join(f"{k} {_tok(v)}" for k, v in tiers.items()) or DASH, None))
        rows.append(row)
    table = drop_empty(headers, rows, set(range(2 + len(THROUGHPUT), len(headers))))
    blocks = [("h", 3, "Throughput & cache"), ("table", table)]
    if not any(h in table.headers for h, _ in CACHE):
        blocks.append(("p", "No server metrics logged; cache hit rates unavailable."))
    return blocks


def latency_tables(data, pts) -> list:
    blocks = []
    for title, key in LATENCY:
        headers = ["Arm", "CCU", *PCTS] + (["Out tok/s/user", "Errors"] if key == "e2e" else [])
        rows = []
        for p in pts:
            b = _base(data, pts, p)
            row = [(_label(data, p["arm"]), None), (_conc(p), None)]
            row += [cell(get(p, "metrics", key, s), get(b, "metrics", key, s), num, False) for s in PCTS]
            if key == "e2e":
                row.append(cell(get(p, "metrics", "tps_per_user"), get(b, "metrics", "tps_per_user"), num, True))
                row.append(cell(get(p, "metrics", "error_rate_pct"), get(b, "metrics", "error_rate_pct"), err_pct, False))
            rows.append(row)
        blocks += [("h", 3, f"{title} (ms)"), ("table", Table(headers, rows))]
    warnings = [f"⚠ {p['arm']} c{p['conc']}: {n}" for p in pts for n in p["notes"] if n.startswith("validator dropped")]
    if warnings:  # spec: error-count disagreement goes right under the E2E table
        blocks.append(("p", "\n".join(f"- {w}" for w in warnings)))
    return blocks


def search_blocks(data, pts) -> list:
    blocks = []
    for p in pts:
        s = p.get("search")
        if p["mode"] != "search" or not s:
            continue
        max_conc = s["max_conc"] if s["max_conc"] is not None else DASH
        head = f"**{p['arm']}**: max CCU = **{max_conc}** · state `{s['state']}` · SLA: {', '.join(s['sla']) or DASH}"
        if s["binding"]:
            observed = f"{s['breach_observed']:.2f}" if s["breach_observed"] is not None else DASH
            head += f" · binding: {s['binding']} (observed {observed} at c{s['infeasible_min']})"
        if s["best_trials_conc"] is not None and s["best_trials_conc"] != s["max_conc"]:
            head += f" · highest objective at c{s['best_trials_conc']}"
        rows = []
        for pr in sorted(s["probes"], key=lambda pr: (pr["conc"] is None, pr["conc"] or 0)):
            marks = " ".join(m for m, on in (("★ max CCU", pr["winner"]), ("◆ best objective", pr["best_objective"])) if on)
            rows.append([(str(pr["iter"]), None), (str(pr["conc"]), None),
                         ("✓", "good") if pr["feasible"] else ("✗", "bad"),
                         *[(num(pr["observed"].get(f"{f['metric_tag']}.{f['stat']}")), None) for f in s["sla_filters"]],
                         (num(pr["output_tps"]), None), (pct(pr["cache_overall"]), None), (marks, None)])
        headers = ["Iter", "CCU", "Pass", *s["sla"], "Output tok/s", "Hit overall", "Mark"]
        blocks += [("h", 3, f"Search: {p['arm']}"), ("p", head), ("table", Table(headers, rows))]
    return blocks


def _grid_points(data, kind):
    return [p for p in data["points"] if p.get("grid") and p["grid"]["kind"] == kind]


def search_matrix(data) -> list:
    pts = [p for p in _grid_points(data, "search") if p.get("search")]
    if not pts:
        return []
    workloads = sorted({(p["grid"]["isl"], p["grid"]["osl"]) for p in pts}, key=lambda k: (tokens(k[0]), tokens(k[1])))
    caches = sorted({p["grid"]["cache"] for p in pts})
    arms = [a["label"] for a in data["arms"] if any(p["arm"] == a["label"] for p in pts)]
    find = lambda arm, isl, osl, c: next((p for p in pts if p["arm"] == arm and
                                          (p["grid"]["isl"], p["grid"]["osl"], p["grid"]["cache"]) == (isl, osl, c)), None)
    rows = []
    for isl, osl in workloads:
        row = [(f"ISL {isl} · OSL {osl}", None)]
        for c in caches:
            base, texts, tone = find(data["baseline"], isl, osl, c), [], None
            for arm in arms:
                p = find(arm, isl, osl, c)
                if p is None:
                    texts.append(DASH)
                    continue
                s = p["search"]
                if s["state"] == "no_pass_in_range":
                    texts.append("✗")
                    continue
                text = f"{s['max_conc']}{'*' if s['state'] == 'cliff' else ''} · {num(get(p, 'metrics', 'throughput', 'output_tps'))}"
                if base is not None and p is not base and get(base, "search", "max_conc"):
                    suffix, tone = _delta(s["max_conc"], base["search"]["max_conc"], True)
                    text += suffix
                texts.append(text)
            row.append((" / ".join(texts), tone))
        rows.append(row)
    legend = ("Cell = max CCU under SLA · output tok/s at that CCU"
              + (f" (arms: {' / '.join(arms)})" if len(arms) > 1 else "")
              + ". `✗` no CCU passed the SLA, `*` cliff boundary.")
    return [("h", 3, "Max CCU under SLA"), ("p", legend),
            ("table", Table(["Workload", *(f"cache {c}%" for c in caches)], rows))]


def fixed_matrices(data) -> list:
    pts = _grid_points(data, "fixed")
    if not pts:
        return []
    order = _arm_order(data)
    concs = sorted({p["conc"] for p in pts if p["conc"] is not None})
    keys = sorted({(p["scenario"], p["arm"]) for p in pts}, key=lambda k: (_scenario_sort(k[0]), order[k[1]]))
    find = lambda scen, arm, c: next((q for q in pts if (q["scenario"], q["arm"], q["conc"]) == (scen, arm, c)), None)
    blocks = [("h", 3, "Fixed CCU grid")]
    for title, path, higher_better in FIXED_MATRICES:
        rows = []
        for scen, arm in keys:
            row = [(scen, None), (arm, None)]
            for c in concs:
                p = find(scen, arm, c)
                b = find(scen, data["baseline"], c) if arm != data["baseline"] else None
                row.append(cell(get(p, *path), get(b, *path), num, higher_better) if p else (DASH, None))
            rows.append(row)
        blocks += [("h", 4, title), ("table", Table(["Scenario", "Arm", *(f"c{c}" for c in concs)], rows))]
    return blocks


def cache_check(data) -> list:
    pts = [p for p in data["points"] if p.get("cache_check")]
    if not pts:
        return []
    order = _arm_order(data)
    rows = []
    for p in sorted(pts, key=lambda p: (_scenario_sort(p["scenario"]), order[p["arm"]], p["conc"] or 0)):
        cc = p["cache_check"]
        rows.append([(p["scenario"], None), (p["arm"], None), (_conc(p), None), (pct(cc["nominal"]), None),
                     (pct(cc["measured"]), None),
                     ("⚠ >10 pts off", "bad") if cc["mismatch"] else ("", None)])
    return [("h", 4, "Cache hit: nominal vs measured (overall)"),
            ("table", Table(["Scenario", "Arm", "CCU", "Nominal", "Measured", "Check"], rows))]


def config_blocks(data) -> list:
    blocks = []
    for a in data["arms"]:
        body = []
        for name, text in (a.get("config") or {}).items():
            if text:
                body += [("p", f"`{name}`"), ("code", "yaml" if name.endswith(".yaml") else "bash", text)]
        if body:
            blocks.append(("details", f"Config: {a['label']}", body))
    return [("h", 2, "Configs"), *blocks] if blocks else []


def notes_blocks(data) -> list:
    notes = list(data["notes"])
    points = data["points"]
    if any(get(p, "cache", "engine") == "vllm" for p in points):
        notes.append("vLLM: CPU and L3 hit rates are not separable; offload hits are shown under external (L3).")
    if any(get(p, "cache", "source") == "client" for p in points):
        notes.append("Hit overall marked (client) is aiperf's client-side prompt cache read %, not tiered.")
    if any(get(p, "metrics", "source") in ("aggregate", "search_aggregate") for p in points):
        notes.append("Rows without an aiperf export use the aggregate JSON: latency from seconds, p99 unavailable.")
    if len(data["arms"]) > 1:
        notes.append(f"Deltas are against the baseline arm `{data['baseline']}` at the same scenario and CCU "
                     "(search: same scenario).")
    return [("h", 2, "Notes"), ("p", "\n".join(f"- {n}" for n in notes))] if notes else []


def chart_blocks(data, pts) -> list:
    return []  # Task 8


def build_blocks(data, charts: bool = False) -> list:
    inputs = ", ".join(_link(i) for i in data["inputs"]) or DASH
    blocks = [
        ("h", 1, data["title"]),
        ("p", f"Generated {data['generated']} · inputs: {inputs} · downloaded {data.get('bytes_fetched', 0) / 1e6:.2f} MB"),
        ("h", 2, "Key findings"), ("p", data.get("findings") or KEY_FINDINGS_PLACEHOLDER),
        ("h", 2, "Runs"), ("table", runs_table(data)),
    ]
    grid = search_matrix(data) + fixed_matrices(data) + cache_check(data)
    if grid:
        blocks += [("h", 2, "Grid summary"), *grid]
    order = _arm_order(data)
    for scen in sorted({p["scenario"] for p in data["points"]}, key=_scenario_sort):
        pts = sorted([p for p in data["points"] if p["scenario"] == scen],
                     key=lambda p: (p["conc"] if p["conc"] is not None else -1, order[p["arm"]]))
        body = throughput_table(data, pts) + latency_tables(data, pts) + search_blocks(data, pts)
        if charts:
            body += chart_blocks(data, pts)
        title = f"Scenario: {scen or '(unknown)'}"
        grid_info = parse_grid(scen)
        if grid_info and grid_info["kind"] == "fixed":
            blocks.append(("details", title, body))
        else:
            blocks += [("h", 2, title), *body]
    return blocks + config_blocks(data) + notes_blocks(data)


def _md_table(t: Table) -> str:
    esc = lambda s: s.replace("|", "\\|")
    lines = ["| " + " | ".join(map(esc, t.headers)) + " |", "|" + "---|" * len(t.headers)]
    lines += ["| " + " | ".join(esc(text) for text, _ in row) + " |" for row in t.rows]
    return "\n".join(lines)


def to_markdown(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append("#" * block[1] + " " + block[2])
        elif kind == "p":
            out.append(block[1])
        elif kind == "table":
            out.append(_md_table(block[1]))
        elif kind == "code":
            out.append(f"```{block[1]}\n{block[2].rstrip()}\n```")
        elif kind == "details":
            out.append(f"<details><summary>{block[1]}</summary>\n\n{to_markdown(block[2]).rstrip()}\n\n</details>")
        # "chart" blocks are HTML-only
    return "\n\n".join(out) + "\n"
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: all OK. If an exact-string assertion fails, print the rendered markdown (`print(md)`) and fix the renderer, not the test, unless the test's expected text contradicts the spec layout.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/result-report/scripts/render.py .agents/skills/result-report/scripts/test_render.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): markdown report with deltas, search probe table and grid matrices"
```

---

### Task 8: HTML output with inline SVG charts

**Files:**
- Modify: `.agents/skills/result-report/scripts/render.py`
- Modify: `.agents/skills/result-report/scripts/test_render.py`

**Interfaces:**
- Consumes: Task 7 blocks.
- Produces: `line_chart(title, series: dict[str, list[tuple[float, float]]], x_label, y_label, threshold=None) -> str` (SVG, `""` when no points), real `chart_blocks(data, pts)`, `to_html(blocks, title) -> str`.

Before writing chart code, load the `dataviz` skill (environment rule for any chart) and apply its palette and mark rules to `PALETTE`/strokes below; keep the function signatures.

- [ ] **Step 1: Write the failing tests** — append to `test_render.py`:

```python
class HtmlTest(unittest.TestCase):
    def test_line_chart_has_series_and_threshold(self):
        svg = render.line_chart("ITL", {"A": [(8, 40.0), (32, 55.0)]}, "CCU", "ms", threshold=50.0)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("<polyline", svg)
        self.assertIn('stroke-dasharray="4 3"', svg)
        self.assertEqual(render.line_chart("x", {}, "a", "b"), "")

    def test_html_report_has_tables_tones_and_probe_chart(self):
        data = data_for("search_local")
        page = render.to_html(render.build_blocks(data, charts=True), data["title"])
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("<table>", page)
        self.assertIn('class="good"', page)
        self.assertIn("<svg", page)
        self.assertIn("by probe CCU", page)

    def test_fixed_ladder_gets_three_charts(self):
        pts = [grid_point("A", "fixed-isl8k-osl1k-cache0", 8, 500.0, 900.0),
               grid_point("A", "fixed-isl8k-osl1k-cache0", 32, 1500.0, 2500.0)]
        blocks = render.chart_blocks(grid_data(pts, ["A"]), pts)
        self.assertEqual(len(blocks), 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: `AttributeError: module 'render' has no attribute 'line_chart'`.

- [ ] **Step 3: Implement** — in `render.py` add `import html` and replace the `chart_blocks` stub; add `line_chart`, `to_html`:

```python
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
CSS = """body{font:14px/1.5 system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 16px;color:#111}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}
th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#f6f8fa}
td.good{color:#1a7f37}td.bad{color:#cf222e}
pre{background:#f6f8fa;padding:8px;overflow:auto}details{margin:8px 0}summary{cursor:pointer;font-weight:600}
svg{margin:8px 16px 8px 0}"""


def _tick(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.2g}"


def line_chart(title, series, x_label, y_label, threshold=None, width=560, height=300) -> str:
    points = [xy for values in series.values() for xy in values]
    if not points:
        return ""
    xs = [x for x, _ in points]
    ys = [y for _, y in points] + ([threshold] if threshold is not None else [])
    x0, x1 = min(xs), max(xs)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = max(ys) * 1.1 or 1.0
    left, right, top, bottom = 60, 20, 30, 40
    sx = lambda x: left + (x - x0) / (x1 - x0) * (width - left - right)
    sy = lambda y: height - bottom - y / y1 * (height - top - bottom)
    esc = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" role="img" aria-label="{esc(title)}">',
             f'<text x="{left}" y="18" font-size="13" font-weight="600">{esc(title)}</text>']
    for i in range(5):
        y = y1 * i / 4
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#e5e7eb"/>'
                     f'<text x="{left - 6}" y="{sy(y) + 4:.1f}" font-size="10" text-anchor="end">{_tick(y)}</text>')
    for x in sorted(set(xs)):
        parts.append(f'<text x="{sx(x):.1f}" y="{height - bottom + 14}" font-size="10" text-anchor="middle">{_tick(x)}</text>')
    mid_y = (top + height - bottom) / 2
    parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 6}" font-size="11" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text x="14" y="{mid_y}" font-size="11" text-anchor="middle" transform="rotate(-90 14 {mid_y})">{esc(y_label)}</text>')
    if threshold is not None:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{sy(threshold):.1f}" y2="{sy(threshold):.1f}" '
                     f'stroke="#dc2626" stroke-dasharray="4 3"/>')
    for i, (name, values) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        values = sorted(values)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        parts += [f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{color}"/>' for x, y in values]
        parts.append(f'<text x="{width - right}" y="{top + 14 * i}" font-size="11" text-anchor="end" fill="{color}">{esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def chart_blocks(data, pts) -> list:
    blocks = []
    fixed = [p for p in pts if p["mode"] == "fixed" and p["conc"] is not None and p.get("metrics")]
    arms = list(dict.fromkeys(p["arm"] for p in fixed))

    def series(path, suffix=""):
        return {f"{a}{suffix}": [(p["conc"], get(p, *path)) for p in fixed
                                 if p["arm"] == a and get(p, *path) is not None] for a in arms}

    if len({p["conc"] for p in fixed}) >= 2:
        blocks.append(("chart", line_chart("Output tok/s vs CCU", series(("metrics", "throughput", "output_tps")), "CCU", "tok/s")))
        for name, key in (("TTFT", "ttft"), ("ITL", "itl")):
            both = {**series(("metrics", key, "p50"), " p50"), **series(("metrics", key, "p99"), " p99")}
            blocks.append(("chart", line_chart(f"{name} p50/p99 vs CCU", both, "CCU", "ms")))
    for p in pts:
        s = p.get("search")
        if p["mode"] != "search" or not s or not s["sla_filters"]:
            continue
        f = next((f for f in s["sla_filters"] if sla_text(f) == s["binding"]), s["sla_filters"][0])
        key = f"{f['metric_tag']}.{f['stat']}"
        probe_series = {p["arm"]: [(pr["conc"], pr["observed"][key]) for pr in s["probes"]
                                   if pr["conc"] is not None and pr["observed"].get(key) is not None]}
        blocks.append(("chart", line_chart(f"{p['arm']}: {sla_text(f)} by probe CCU", probe_series, "CCU", "ms",
                                           threshold=f["threshold"])))
    return [b for b in blocks if b[1]]


def _inline(text: str) -> str:
    if text.startswith("<!--"):
        return text
    out = html.escape(text)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)


def _td(text: str, tone: str | None) -> str:
    return f'<td class="{tone}">{_inline(text)}</td>' if tone else f"<td>{_inline(text)}</td>"


def _html_blocks(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append(f"<h{block[1]}>{_inline(block[2])}</h{block[1]}>")
        elif kind == "p":
            lines = block[1].split("\n")
            if all(line.startswith("- ") for line in lines):
                out.append("<ul>" + "".join(f"<li>{_inline(line[2:])}</li>" for line in lines) + "</ul>")
            else:
                out.append(f"<p>{'<br>'.join(_inline(line) for line in lines)}</p>")
        elif kind == "table":
            t = block[1]
            head = "".join(f"<th>{html.escape(h)}</th>" for h in t.headers)
            body = "".join("<tr>" + "".join(_td(text, tone) for text, tone in row) + "</tr>" for row in t.rows)
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif kind == "code":
            out.append(f"<pre><code>{html.escape(block[2])}</code></pre>")
        elif kind == "details":
            out.append(f"<details><summary>{_inline(block[1])}</summary>{_html_blocks(block[2])}</details>")
        elif kind == "chart":
            out.append(block[1])
    return "\n".join(out)


def to_html(blocks, title: str) -> str:
    return (f'<!doctype html>\n<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>\n{_html_blocks(blocks)}\n</body></html>\n")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/result-report/scripts/render.py .agents/skills/result-report/scripts/test_render.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): HTML output with inline SVG charts (ladder and search probes)"
```

---

### Task 9: CLI, SKILL.md, spec touch-up

**Files:**
- Create: `.agents/skills/result-report/scripts/result_report.py`
- Create: `.agents/skills/result-report/scripts/test_cli.py`
- Create: `.agents/skills/result-report/SKILL.md`
- Modify: `docs/superpowers/specs/2026-10-23-result-report-skill-design.md` (Skill files section + `--per-request` deferral + `--findings`)

**Interfaces:**
- Consumes: `fetch.parse_source/discover/TooLarge/FETCHED`, `extract.build_data`, `render.build_blocks/to_markdown/to_html`.
- Produces: `main(argv=None) -> int`. Exit codes: 0 ok, 1 nothing found, 2 bad input/flag, 3 full zip over budget, 4 `gh` missing or failing. Output files `report.md`, `data.json`, optional `report.html`.

- [ ] **Step 1: Write the failing tests** — `.agents/skills/result-report/scripts/test_cli.py`:

```python
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import result_report

FIX = Path(__file__).resolve().parent / "fixtures"


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = result_report.main([str(a) for a in argv])
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def test_writes_report_data_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(FIX / "gh_fixed", FIX / "search_local", "--out", tmp, "--html", "--title", "demo")
            self.assertEqual(code, 0, out)
            md = (Path(tmp) / "report.md").read_text()
            data = json.loads((Path(tmp) / "data.json").read_text())
            page = (Path(tmp) / "report.html").read_text()
        self.assertIn("# demo", md)
        self.assertIn("## Scenario: agentx-cc-062126", md)
        self.assertEqual(len(data["arms"]), 2)
        self.assertIn("<svg", page)
        self.assertIn("report.md", out)

    def test_findings_replace_the_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            findings = Path(tmp) / "findings.md"
            findings.write_text("- GLM c108 reaches 734 output tok/s.\n")
            code, out = run(FIX / "gh_fixed", "--out", tmp, "--findings", findings)
            md = (Path(tmp) / "report.md").read_text()
        self.assertEqual(code, 0, out)
        self.assertIn("- GLM c108 reaches 734 output tok/s.", md)
        self.assertNotIn("KEY_FINDINGS", md)

    def test_bad_input_and_empty_dir(self):
        self.assertEqual(run("/no/such/dir")[0], 2)
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertEqual(run(empty, "--out", Path(tmp) / "o")[0], 1)

    def test_unknown_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run(FIX / "gh_fixed", "--out", tmp, "--baseline", "nope")[0], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: `ModuleNotFoundError: No module named 'result_report'`.

- [ ] **Step 3: Implement** — `.agents/skills/result-report/scripts/result_report.py`:

```python
#!/usr/bin/env python3
"""result-report: InferenceX benchmark run(s) -> report.md + data.json (+ report.html).

    python3 .agents/skills/result-report/scripts/result_report.py <run-url|run-id|dir>... [--html]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extract  # noqa: E402
import fetch  # noqa: E402
import render  # noqa: E402


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "report"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="GitHub Actions run URL, run id, or local folder")
    ap.add_argument("--repo", default=fetch.DEFAULT_REPO, help="repo for bare run ids")
    ap.add_argument("--baseline", help="arm label to compare against (default: first arm)")
    ap.add_argument("--out", type=Path, help="output folder (default ~/bench-reports/<date>-<slug>)")
    ap.add_argument("--title")
    ap.add_argument("--html", action="store_true", help="also write report.html with charts")
    ap.add_argument("--findings", type=Path, help="markdown bullets that replace the Key findings placeholder")
    ap.add_argument("--max-full-zip-mb", type=float, default=100,
                    help="largest artifact zip to download whole when Range reads fail")
    args = ap.parse_args(argv)

    try:
        sources = [fetch.parse_source(a, args.repo) for a in args.inputs]
    except ValueError as err:
        print(f"ERROR {err}", file=sys.stderr)
        return 2
    today = datetime.date.today().isoformat()
    names = [s.run_id or s.path.name for s in sources]
    out = args.out or Path.home() / "bench-reports" / f"{today}-{slugify(args.title or '-'.join(names))}"
    cache = out / ".cache"

    try:
        configs, notes = [], []
        for src in sources:
            found, found_notes = fetch.discover(src, cache, args.max_full_zip_mb)
            configs += found
            notes += found_notes
        if not configs:
            print("ERROR nothing to report:\n" + "\n".join(f"  {n}" for n in notes), file=sys.stderr)
            return 1
        data = extract.build_data(configs, notes, baseline=args.baseline, title=args.title,
                                  inputs=[s.link for s in sources], generated=today)
    except ValueError as err:
        print(f"ERROR {err}", file=sys.stderr)
        return 2
    except fetch.TooLarge as err:
        print(f"ERROR {err}\nAsk the user before downloading it; then re-run with --max-full-zip-mb <MB>.",
              file=sys.stderr)
        return 3
    except FileNotFoundError as err:
        print(f"ERROR {err}: the gh CLI is required for run links (install it and run `gh auth login`)",
              file=sys.stderr)
        return 4
    except subprocess.CalledProcessError as err:
        print(f"ERROR gh failed: {' '.join(err.cmd)}\n{(err.stderr or '').strip()}", file=sys.stderr)
        return 4

    data["bytes_fetched"] = fetch.FETCHED["bytes"]
    if args.findings:
        data["findings"] = args.findings.read_text().strip()
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, indent=1, default=str) + "\n")
    (out / "report.md").write_text(render.to_markdown(render.build_blocks(data)))
    written = ["report.md", "data.json"]
    if args.html:
        (out / "report.html").write_text(render.to_html(render.build_blocks(data, charts=True), data["title"]))
        written.append("report.html")
    print(f"wrote {', '.join(str(out / w) for w in written)}")
    print(f"{len(data['points'])} config(s), {len(data['arms'])} arm(s), baseline {data['baseline']!r}, "
          f"downloaded {data['bytes_fetched'] / 1e6:.2f} MB, {len(data['notes'])} note(s)")
    for note in data["notes"]:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v`
Expected: all OK.

- [ ] **Step 5: Write `.agents/skills/result-report/SKILL.md`**

```markdown
---
name: result-report
description: Turn InferenceX benchmark runs into one standard report. Input is GitHub Actions run links/IDs or downloaded artifact/aiperf folders; output is report.md (+ optional report.html with charts) and data.json with throughput, cache hit by tier (HBM/DRAM/L3/router), TTFT/ITL/E2E avg-p50-p90-p95-p99, max CCU under SLA for search runs, and A/B or grid comparisons. Use when asked to report, summarize, compare or read the results of a benchmark run, an AgentX/tau2/livekit ladder, a sweep-isl*/fixed-isl* grid, or an aiperf output folder.
---

# result-report

One command turns benchmark results into a report the team can paste into Redmine or an MR. Tables
come from the script, so every report has the same columns and units. You write only the Key findings.

## Run it

    python3 .agents/skills/result-report/scripts/result_report.py <input>... [--html] [--title T] [--baseline ARM]

`<input>` is any mix of:
- a run URL `https://github.com/vngcloud/InferenceX/actions/runs/<id>` or a bare run id (repo `--repo`, default vngcloud/InferenceX);
- a folder of downloaded artifacts (`bmk_agentic_*`, `agentic_*`) or hand-named copies;
- a raw aiperf output folder (contains `profile_export_aiperf.json`).

Run links need the `gh` CLI, logged in. The script reads single files out of the artifact zips with
HTTP Range requests: a 266 MB search artifact costs about 1 MB. It never opens the heavy files
(`server_metrics_export.json` of fixed runs, `profile_export.jsonl`, `inputs.json`, timeslices).

Output goes to `~/bench-reports/<date>-<slug>/` (or `--out DIR`): `report.md`, `data.json`,
`report.html` with `--html`. Downloads are cached in `<out>/.cache/`, so a re-run is fast.

## Several inputs: arms and grids

Inputs are grouped automatically:
- **Arm** = one serving config. Runs whose `serving.yaml` differ only in `name`, `scenario` or `conc`
  are the same arm (so one config dispatched once per scenario stays one arm).
- **Scenario** = the workload. `sweep-isl*-osl*-cache*` / `fixed-isl*-osl*-cache*` names become grid axes.

Same arm + many scenarios → a grid summary. Several arms + the same scenario → a comparison: every
non-baseline cell shows `value (Δ%)` against the baseline arm (first arm, or `--baseline <label>`).

## Write the Key findings

1. Run the command once.
2. Read `data.json` (`points[]`: one per config; `arms[]`; `notes[]`) and the tables in `report.md`.
3. Write 3–6 bullets to a file, in English, then re-run the same command with `--findings <file>`
   (cached, takes seconds). Rules:
   - Lead with the decision: the best arm per scenario, or the max CCU under SLA.
   - Every bullet names the arm, scenario and CCU and quotes a number from the tables.
   - Quote deltas only when |Δ| ≥ 5 %.
   - For search runs, name the binding SLA and where it broke (`ITL p90 < 50 ms, 50.85 ms at c61`).
   - Repeat any warning from Notes that changes the conclusion: errors, `⚠` cache mismatch,
     aggregate-only rows (no p99), failed or expired jobs.
   - No adjectives without a number.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | report written | — |
| 1 | no configs found | check the run has `bmk_agentic_*`/`agentic_*` artifacts, or the folder layout |
| 2 | bad input or unknown `--baseline` | fix the argument; the error lists the arm labels |
| 3 | Range reads failed and the full zip is over `--max-full-zip-mb` (100) | ask the user, then re-run with a bigger `--max-full-zip-mb` |
| 4 | `gh` missing or failing | `gh auth status`; install / log in |

## Reading the tables

- Latency is ms; columns avg/p50/p90/p95/p99. `—` means the value does not exist, never zero.
- Rows built from the aggregate JSON only (no aiperf export) have no p99; Notes says so.
- Cache columns: overall, GPU (HBM), CPU (DRAM), external (L3), router. Columns empty for every row are hidden.
  vLLM cannot split CPU from L3: its offload hits show as external.
- Grid tables compare the nominal cache % in the scenario name against the measured **overall** hit; `⚠` when they differ by more than 10 points.
- Search runs: the headline CCU is the largest CCU that passed every SLA (`feasible_max`), not the highest-throughput probe; the probe table marks both.

## Tests

    python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v
```

- [ ] **Step 6: Update the spec** — in `docs/superpowers/specs/2026-10-23-result-report-skill-design.md`:
  - Replace the `## Skill files` code block with the file list of this plan's File Structure (four modules + four test files + fixtures + SKILL.md + symlink).
  - Replace the CLI line with: `result_report.py <input>... [--repo owner/name] [--baseline ARM] [--out DIR] [--title T] [--html] [--findings FILE] [--max-full-zip-mb 100]`. Add one sentence: "`--findings` re-renders with the agent's Key findings; `--per-request` is deferred until someone needs per-request charts."
  - In Testing item 1, change "one stdlib unittest file" to "stdlib unittest files per module".

- [ ] **Step 7: Run the full suite and a real local report**

Run:

```bash
python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v 2>&1 | tail -3
python3 .agents/skills/result-report/scripts/result_report.py /tmp/fill-1015/38843679057 --out /tmp/rr-local --html
head -40 /tmp/rr-local/report.md
```

Expected: `OK`; the report has two GLM rows (c108, c120 — the c120 row aggregate-only), CPU (DRAM) column present, the c120 row noted as aggregate-only.

- [ ] **Step 8: Commit**

```bash
git add .agents/skills/result-report/SKILL.md .agents/skills/result-report/scripts/result_report.py .agents/skills/result-report/scripts/test_cli.py docs/superpowers/specs/2026-10-23-result-report-skill-design.md
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(result-report): CLI, SKILL.md, spec touch-up"
```

---

### Task 10: Acceptance — fresh subagents per case

**Files:**
- Modify only if a case fails: the module at fault + its test file (add a regression test first).

**Interfaces:**
- Consumes: the finished skill; Task 2 run URLs.

The point of this task: future agents only see `SKILL.md`. Each case runs in a fresh subagent with no context from this session.

- [ ] **Step 1: Wait for the Task 2 runs** — `gh run view <id> -R vngcloud/InferenceX --json status,conclusion` for both; continue when both are `completed` (any conclusion; failed jobs are a valid input for the report).

- [ ] **Step 2: Dispatch one fresh `general-purpose` subagent per case, in parallel**, prompt template:

```text
You are testing a skill. Read /home/thanglq5/greennode-code/InferenceOptimization/InferenceX/.agents/skills/result-report/SKILL.md
and follow it to produce a report for: <INPUT>. Use --out /tmp/rr-accept/<CASE> --html.
Write Key findings as SKILL.md says. Then report back, in this format:
1. The exact commands you ran and their exit codes.
2. The "downloaded X MB" line.
3. Three numbers from report.md, each with the file and JSON path you checked it against
   (aiperf profile_export_aiperf.json or the aggregate JSON, inside <out>/.cache or the input folder), and whether they match.
4. Every place where SKILL.md was unclear or wrong, quoted.
5. These case checks: <CHECKS>.
Do not modify any file under the repo.
```

| Case | `<INPUT>` | `<CHECKS>` |
|---|---|---|
| 1 | `https://github.com/vngcloud/InferenceX/actions/runs/39558375380` | p99 present (raw tree read over Range); CPU column hidden; downloaded < 2 MB |
| 1b | `/tmp/run39558375380` | Notes says p99 unavailable (aggregate-only); TTFT p99 cells are `—` |
| 2 | `https://github.com/vngcloud/InferenceX/actions/runs/39642191354` | downloaded ≪ 266 MB; headline max CCU and binding SLA; probe table marks ★ and ◆ |
| 2b | `39235947774 39235964228` | Grid summary "Max CCU under SLA" matrix with `✗`/`*` where states say so; how many arms and why |
| 3 | `/tmp/fill-1015/38843679057` | CPU (DRAM) tier present (~43 %); validator-error note on c108 |
| 4 | `/mnt/poc_tencent/artifacts` | "No server metrics logged"; CCU from the folder/aiperf config |
| 5 | `https://github.com/vngcloud/InferenceX/actions/runs/39618356748 https://github.com/vngcloud/InferenceX/actions/runs/39642191354` | two arms, baseline marked, Δ% present in the HTML with good/bad colours |
| 6 | `https://github.com/vngcloud/InferenceX/actions/runs/39583087151` | failed jobs listed in Notes; exit code 0 or 1 with a clear message, no traceback |
| 7 | the two Task 2 run URLs | one arm; Fixed CCU grid matrices with c8/c32; nominal 0 %/50 % vs measured overall; CPU column present (or hidden if Task 2 dropped HiCache) |

- [ ] **Step 3: Triage** — for each case report: any mismatch between a reported number and its source, any traceback, any SKILL.md confusion is a defect. For each defect: write a failing unittest that reproduces it (fixture or synthetic dict), fix, run the full suite, commit `fix(result-report): <defect>`. Clarity defects → edit SKILL.md. Re-run only the failed cases with fresh subagents.

- [ ] **Step 4: Final verification (superpowers:verification-before-completion)**

Run:

```bash
python3 -m unittest discover -s .agents/skills/result-report/scripts -p 'test_*.py' -v 2>&1 | tail -3
uv run -q --no-project --with pytest --with pyyaml --with pydantic python -m pytest utils/test_sweep_scenarios.py utils/test_serving_dispatch.py -q
git status --short
git log --oneline origin/vng-benchmark..HEAD
```

Expected: both suites pass; `git status` shows only pre-existing untracked files; the log lists this plan's commits. Report case results (pass/fail per case + downloaded MB) to the user and ask before pushing.
