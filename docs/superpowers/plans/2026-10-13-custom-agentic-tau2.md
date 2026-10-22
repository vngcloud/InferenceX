# custom-agentic tau2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scenario: custom-agentic-tau2` (or `-smoke`) in a serving YAML runs the tau2 customer-service trace benchmark on the 1×H200 runner exactly like the Weka AgentX scenario, with a seed-, revision- and sha256-pinned dataset fetched from public `Green-Lab/customer-service-agent-traces`, and `GREENLAB_HF_TOKEN` wired for future private datasets.

**Architecture:** Builds on V1 (serving-YAML dispatch). A lock file `scenarios/datasets/lock.yaml` pins file datasets (HF repo, revision, file, sha256). Scenario YAMLs name a locked dataset (`datasets[0].name`) and take its path from `${IX_CUSTOM_DATASET_FILE}`. `to_matrix.scenario_facts` accepts such `type: file` scenarios; `serve_generic.sh` fetches + verifies the file via `utils/serving/fetch_custom_dataset.py` instead of `resolve_trace_source`, then the unchanged `build_replay_cmd_yaml` runs aiperf `--config`.

**Tech Stack:** bash (benchmark recipes), Python 3.12 (pydantic, pyyaml, huggingface_hub), aiperf fork `utils/aiperf` @ `faf3f8b3`, GitHub Actions (unchanged), pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-custom-agentic-tau2-design.md` (read its "Amendments from the V1 review" section: it overrides the body where they differ).

## How to start (fresh session)

```bash
cd /home/thanglq5/greennode-code/InferenceOptimization/InferenceX-tau2   # worktree, branch feat/custom-agentic-tau2
git status                                   # clean; HEAD is the spec/plan commits on top of V1 3ba14d43e
git submodule update --init utils/aiperf     # must be faf3f8b3
```

- Another session owns `/home/thanglq5/greennode-code/InferenceOptimization/InferenceX` (branch `feat/serving-config-dispatch`). Never cd there, never checkout/commit that branch.
- PYTEST (serving tests only; whole-suite collection breaks once `utils/aiperf` is initialised):
  `uv run -q --python 3.12 --with pytest --with pyyaml --with pydantic --no-project python -m pytest -c utils/pytest.ini --rootdir utils utils/test_serving_*.py -q`
- GOLDEN: `(cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)`
- Commits: `git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit ...`; `git add` only the task's files; never `git add -A`/`.`; never commit `.serena/`.
- **Never push.** Never ssh to nodes. Never print/paste a token (HF, GitHub) into files, commits or output.

## Global Constraints

- Scenario family, not a new scenario-type: results keep `scenario-type: agentic-coding`; do not touch `infx/matrix`, `infx/results`, `infx/klaud`, workflows' ingest.
- Durations: `custom-agentic-tau2` = 1200 s, `custom-agentic-tau2-smoke` = 180 s.
- Dataset: 1576 entries, `requests: 16798`, `randomSeed: 42`, format `mooncake_trace`; sampling params not set (model default).
- Lock path: `scenarios/datasets/lock.yaml` (subdirectory: V1's `scenarios/*.yaml` glob must never see it). Lock key for tau2: `tau2-cs`.
- Dataset repo: `Green-Lab/customer-service-agent-traces` (public), file `trace.tau2.jsonl`, content byte-identical to `thangquang09/customer-service-agent-traces@8deff81a5135f7af8ebea721e4cc8ef30e218fa4` (md5 `1d6073c485afec5e53495043101852c5`, 1,126,364,854 B).
- `GREENLAB_HF_TOKEN`: read only inside Python (`os.environ`); bash never expands it; used whenever set; never named `HF_TOKEN`.
- Error style (V1): one line `ERROR <what>: <fix>` on stderr, exit 1, no traceback for expected failures.
- AGENTS.md rule followed by V1: no tests that grep source text; test behaviour.

## Review Focus

1. Token leakage via xtrace: `serve_generic.sh` runs under `set -x`; `GREENLAB_HF_TOKEN=hf_dummySECRET` must never appear in any output → Task 4 `test_bash_wrapper_never_traces_token`.
2. A scenario file that is malformed (missing `benchmark`, `phases`, bad YAML) must yield `ServingError` one-liner, not a traceback → Task 3 `test_malformed_scenario_is_one_line_error`.
3. A Weka scenario must still go through `resolve_trace_source` (no regression) and a non-locked file dataset must be rejected at validation time, not on the GPU → Task 3 `test_file_dataset_must_be_locked`, Task 4 `test_check_flag`.
4. A partially downloaded / swapped dataset must fail loudly with both hashes → Task 4 `test_sha_mismatch_fails_with_both_hashes`.
5. `scenario: datasets` or `scenario: lock` must not be offered/accepted as a scenario → Task 3 `test_lock_is_not_a_scenario`.

---

### Task 1: V1 review leftovers (serving.yaml newline, dispatch wording, GPU metrics CSV)

**Files:**
- Modify: `benchmarks/single_node/agentic/serve_generic.sh` (the `printf '%s\n' "${SERVING_YAML:-}"` line)
- Modify: `utils/serving/dispatch.sh` (the "did not show up" error line, ~line 65)
- Create: `benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv`

**Interfaces:** Consumes nothing new. Produces nothing later tasks call.

- [x] **Step 1: serving.yaml without extra newline**

In `serve_generic.sh` replace
```bash
printf '%s\n' "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"
```
with
```bash
printf '%s' "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"
```
(get-jobs already appends the file's final newline; the artifact must equal the user's file byte for byte.)

- [x] **Step 2: dispatch.sh must tell the agent not to redispatch**

Replace the line
```bash
echo "ERROR dispatched, but the run did not show up within 2 minutes; check: gh run list --repo $GH_REPO -w e2e-tests.yml -L 5" >&2
```
with
```bash
echo "ERROR dispatched, but the run did not show up within 2 minutes; do not redispatch, check: gh run list --repo $GH_REPO -w e2e-tests.yml -L 5" >&2
```

- [x] **Step 3: ship the DCGM fieldset for serve_generic**

The launcher mounts `${BENCH_SCRIPT%.sh}.gpu_metrics.csv` when it exists; baseline run 37733664587 used the qwen38 recipe's CSV.
```bash
git show 295131eb9:benchmarks/single_node/agentic/qwen38_bf16_h200.gpu_metrics.csv > benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv
wc -l benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv   # expect 66
```

- [x] **Step 4: Run tests**

Run: `bash -n benchmarks/single_node/agentic/serve_generic.sh && bash -n utils/serving/dispatch.sh && <PYTEST>`
Expected: all pass (V1 tests unchanged).

- [x] **Step 5: Commit**

```bash
git add benchmarks/single_node/agentic/serve_generic.sh utils/serving/dispatch.sh benchmarks/single_node/agentic/serve_generic.gpu_metrics.csv
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "fix(serving): exact serving.yaml artifact, no-redispatch hint, DCGM fieldset for serve_generic"
```

---

### Task 2: Publish the dataset to Green-Lab and write the lock

**Files:**
- Create: `utils/serving/publish_tau2_dataset.py`
- Create: `scenarios/datasets/lock.yaml`

**Interfaces:**
- Produces: `scenarios/datasets/lock.yaml` with exactly one key `tau2-cs: {repo, revision, file, sha256}` (revision 40-hex, sha256 64-hex). Tasks 3–6 read it.

The user approved publishing exactly `thangquang09/customer-service-agent-traces` to Green-Lab as public. The token is the workstation's `~/.cache/huggingface/token` (HF user `thangquang09`, admin of Green-Lab); huggingface_hub reads it itself; never print it.

- [x] **Step 1: Write the script**

`utils/serving/publish_tau2_dataset.py`:
```python
"""Publish the tau2 customer-service traces to Green-Lab (public); print the lock entry.

    uv run -q --python 3.12 --with 'huggingface_hub>=0.34' --with pyyaml --no-project \
        python utils/serving/publish_tau2_dataset.py --local <trace.tau2.jsonl> [--dry-run]

Checks that the local file is byte-identical to the private source snapshot
(md5 + the LFS sha256 HF stores), rewrites the card's provenance (the data is the
tau2 slice of the public Exgentic/agent-llm-traces-v2, not gateway traffic),
uploads card + plot + data in one commit and verifies the uploaded sha256.
Token: huggingface_hub default (~/.cache/huggingface/token).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

SRC = "thangquang09/customer-service-agent-traces"
SRC_REV = "8deff81a5135f7af8ebea721e4cc8ef30e218fa4"
DST = "Green-Lab/customer-service-agent-traces"
FILE = "trace.tau2.jsonl"
PLOT = "plots/distributions.png"
MD5 = "1d6073c485afec5e53495043101852c5"
PROVENANCE = (
    "> **Source.** Derived from the public "
    "[`Exgentic/agent-llm-traces-v2`](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2): "
    "the three tau2 domains (airline, retail, telecom) of agent runs recorded through a LiteLLM proxy, "
    "converted to aiperf `mooncake_trace` lines (messages and tools verbatim). Customers are tau2's "
    "synthetic users (`@example.com`). The upstream dataset declares no license (its v1 was "
    "CDLA-Permissive-2.0); the upstream terms apply.\n"
)


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fix_readme(text: str) -> str:
    text = re.sub(r"^extra_gated_prompt:.*\n", "", text, flags=re.M)
    text, n = re.subn(r"^> ⚠️ \*\*Private and sensitive\.\*\*.*?\n\n", PROVENANCE + "\n", text, flags=re.M | re.S)
    if n != 1:
        sys.exit("ERROR README: expected exactly one 'Private and sensitive' block to replace")
    for phrase in ("internal gateway", "real production", "Internal GreenNode use only"):
        if phrase.lower() in text.lower():
            sys.exit(f"ERROR README: still says {phrase!r}; fix fix_readme()")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if digest(args.local, "md5") != MD5:
        sys.exit(f"ERROR {args.local}: md5 is not {MD5}")
    sha = digest(args.local, "sha256")
    api = HfApi()
    source = api.get_paths_info(SRC, [FILE], revision=SRC_REV, repo_type="dataset")[0]
    if source.lfs is None or source.lfs.sha256 != sha:
        sys.exit(f"ERROR {args.local}: sha256 {sha} differs from {SRC}@{SRC_REV}")
    readme = fix_readme(Path(hf_hub_download(SRC, "README.md", revision=SRC_REV, repo_type="dataset")).read_text())
    plot = hf_hub_download(SRC, PLOT, revision=SRC_REV, repo_type="dataset")
    if args.dry_run:
        print(readme)
        print(f"# dry run: would upload {FILE} sha256={sha} to {DST} (public)")
        return 0

    api.create_repo(DST, repo_type="dataset", private=False, exist_ok=True)
    commit = api.create_commit(
        DST,
        repo_type="dataset",
        operations=[
            CommitOperationAdd("README.md", readme.encode()),
            CommitOperationAdd(PLOT, plot),
            CommitOperationAdd(FILE, str(args.local)),
        ],
        commit_message=f"tau2 customer-service traces (from {SRC}@{SRC_REV[:8]})",
    )
    revision = commit.oid
    uploaded = api.get_paths_info(DST, [FILE], revision=revision, repo_type="dataset")[0]
    if uploaded.lfs is None or uploaded.lfs.sha256 != sha:
        sys.exit(f"ERROR {DST}@{revision}: uploaded sha256 does not match {sha}")
    if api.dataset_info(DST).private:
        sys.exit(f"ERROR {DST} is private; expected public")
    print(yaml.safe_dump({"tau2-cs": {"repo": DST, "revision": revision, "file": FILE, "sha256": sha}}, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 2: Dry run**

```bash
uv run -q --python 3.12 --with 'huggingface_hub>=0.34' --with pyyaml --no-project \
  python utils/serving/publish_tau2_dataset.py --dry-run \
  --local /home/thanglq5/greennode-code/InferenceOptimization/vmonitor-gateway-datasets/out/agent_traces_v2/trace.tau2.jsonl
```
Expected: the fixed README (frontmatter without `extra_gated_prompt`, the "Source." paragraph where the warning was) and `# dry run: would upload trace.tau2.jsonl sha256=<64-hex> ...`. If the regex in `fix_readme` does not match the real card, adjust only `fix_readme` until it does; the phrase check must stay.

- [x] **Step 3: Publish (approved by the user) and write the lock**

```bash
uv run -q --python 3.12 --with 'huggingface_hub>=0.34' --with pyyaml --no-project \
  python utils/serving/publish_tau2_dataset.py \
  --local /home/thanglq5/greennode-code/InferenceOptimization/vmonitor-gateway-datasets/out/agent_traces_v2/trace.tau2.jsonl \
  > /tmp/tau2-lock-entry.yaml
cat /tmp/tau2-lock-entry.yaml
```
Expected: 4 lines, `tau2-cs:` with `repo: Green-Lab/customer-service-agent-traces`, 40-hex `revision`, `file: trace.tau2.jsonl`, 64-hex `sha256`. Upload of 1.1 GB takes minutes. If the subagent environment cannot reach HF with write access, stop and report BLOCKED with the exact error (the controller runs this step).

Write `scenarios/datasets/lock.yaml`:
```bash
mkdir -p scenarios/datasets
{ printf '%s\n' \
  '# File datasets for scenarios/*.yaml, keyed by the scenario'"'"'s datasets[0].name.' \
  '# Each entry pins one file at one HF commit; utils/serving/fetch_custom_dataset.py' \
  '# downloads it (token: GREENLAB_HF_TOKEN if set) and verifies sha256.'; cat /tmp/tau2-lock-entry.yaml; } > scenarios/datasets/lock.yaml
cat scenarios/datasets/lock.yaml
```

- [x] **Step 4: Verify anonymously (public)**

```bash
uv run -q --python 3.12 --with 'huggingface_hub>=0.34' --no-project python -c "
from huggingface_hub import HfApi; import yaml
e = yaml.safe_load(open('scenarios/datasets/lock.yaml'))['tau2-cs']
i = HfApi(token=False).get_paths_info(e['repo'], [e['file']], revision=e['revision'], repo_type='dataset')[0]
assert i.lfs.sha256 == e['sha256'], 'sha mismatch'; print('public OK', i.size)"
```
Expected: `public OK 1126364854`.

- [x] **Step 5: Commit**

```bash
git add utils/serving/publish_tau2_dataset.py scenarios/datasets/lock.yaml
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): publish tau2 traces to Green-Lab; dataset lock"
```

---

### Task 3: tau2 scenarios, golden check, and file datasets in `scenario_facts`

**Files:**
- Create: `scenarios/custom-agentic-tau2.yaml`, `scenarios/custom-agentic-tau2-smoke.yaml`
- Modify: `utils/serving/check_scenario_golden.py`
- Modify: `infx/serving/to_matrix.py` (`scenario_facts`, new `load_lock`)
- Test: `utils/test_serving_to_matrix.py`, create `utils/test_serving_lock.py`

**Interfaces:**
- Consumes: `scenarios/datasets/lock.yaml` (Task 2).
- Produces: `infx.serving.to_matrix.load_lock(repo: Path = REPO_ROOT) -> dict[str, dict]`; `scenario_facts(scenario, repo) -> tuple[int, str]` returns `(1200, "tau2-cs")` for `custom-agentic-tau2`; the `serving.loader` value for tau2 runs is `tau2-cs` (Task 4 keys off it).

- [x] **Step 1: Write the scenarios**

`scenarios/custom-agentic-tau2.yaml`:
```yaml
# custom-agentic family: file-backed agentic traces. tau2 customer-service traces
# (airline/retail/telecom from Exgentic/agent-llm-traces-v2), 1576 sessions /
# 16798 turns, pinned in scenarios/datasets/lock.yaml (key tau2-cs).
# serve_generic.sh downloads + sha256-checks the file and exports IX_CUSTOM_DATASET_FILE.
# Matches past tau2 runs (e.g. 37920508184): 1200 s, request cap 16798, seed 42.
# Per-run values (url, model, tokenizer, concurrency, metrics/telemetry URLs,
# artifact dir) are CLI overrides added by build_replay_cmd_yaml.
# Equivalence with the CLI form: utils/serving/check_scenario_golden.py.
schemaVersion: "2.0"
benchmark:
  scenario: inferencex-agentx-mvp
  unsafeOverride: true
  endpoint:
    type: chat
    path: /v1/chat/completions
    streaming: true
    useServerTokenCount: true
  datasets:
  - name: tau2-cs
    type: file
    format: mooncake_trace
    path: ${IX_CUSTOM_DATASET_FILE}
    entries: 1576
    randomSeed: 42
    traceIdleGapCapSeconds: 300.0
  phases:
  - name: profiling
    kind: profiling
    type: concurrency
    duration: 1200.0
    requests: 16798
    gracePeriod: 30.0
    failedRequestThreshold: 0.1
    trajectoryStartMinRatio: 0.25
    trajectoryStartMaxRatio: 0.75
    warmupRequestsPerLane: 10
    agenticWarmupGracePeriod: 1800.0
  artifacts:
    sliceDuration: 1.0
  tokenizer:
    trustRemoteCode: true
  runtime:
    statsInterval: 30.0
randomSeed: 42
```
`scenarios/custom-agentic-tau2-smoke.yaml`: identical except the first comment line becomes `# Smoke variant of custom-agentic-tau2.yaml: 180 s profiling, for bring-up only.` (keep the remaining comment lines) and `duration: 180.0`.

- [x] **Step 2: Extend the golden check**

In `utils/serving/check_scenario_golden.py`:
- replace `SCENARIOS = ("agentx-cc-062126", "agentx-cc-062126-smoke")` with
```python
SCENARIOS = {
    "agentx-cc-062126": "weka",
    "agentx-cc-062126-smoke": "weka",
    "custom-agentic-tau2": "tau2",
    "custom-agentic-tau2-smoke": "tau2",
}
# datasets[].name is a label (CLI always says "main"); for file datasets it is the lock key.
ALLOWED = {"weka": set(), "tau2": {".benchmark.datasets[0].name"}}
```
- add after `reference_argv`:
```python
def tau2_reference_argv(duration: int, artifact_dir: str, dataset_file: str, entries: int = 1576) -> list[str]:
    """The tau2 command of run 37920508184 (gemma4tau2, c16), model/conc normalized like the Weka one."""
    return shlex.split(
        "--scenario inferencex-agentx-mvp --url http://localhost:8888 "
        "--endpoint /v1/chat/completions --endpoint-type chat --streaming "
        f"--model {MODEL} --tokenizer {MODEL} --concurrency 8 "
        f"--benchmark-duration {duration} --stats-interval 30 --random-seed 42 "
        "--failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 "
        "--trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 "
        "--trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 "
        "--use-server-token-count --gpu-telemetry http://localhost:9400/metrics "
        f"--tokenizer-trust-remote-code --num-dataset-entries {entries} --slice-duration 1.0 "
        "--server-metrics http://localhost:8888/metrics "
        f"--output-artifact-dir {artifact_dir}/aiperf_artifacts --unsafe-override "
        f"--custom-dataset-type mooncake_trace --input-file {dataset_file} --request-count 16798"
    )


def differences(kind: str, reference: list[str], scenario: str, tmp: str) -> list[str]:
    allowed = ALLOWED[kind]
    return [d for d in diff(resolved(reference), resolved(yaml_argv(scenario, tmp))) if d.split(":")[0] not in allowed]
```
- replace `main()` with:
```python
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
```
Also update the module docstring's first paragraph to say it covers the Weka reference (run 37733664587) and the tau2 reference (run 37920508184).

- [x] **Step 3: Run the golden check**

Run: `(cd utils/aiperf && uv run -q --python 3.12 python ../serving/check_scenario_golden.py)`
Expected:
```
OK agentx-cc-062126 (3600s): YAML + overrides == reference CLI
OK agentx-cc-062126-smoke (180s): YAML + overrides == reference CLI
OK custom-agentic-tau2 (1200s): YAML + overrides == reference CLI
OK custom-agentic-tau2-smoke (180s): YAML + overrides == reference CLI
OK negative case caught: .benchmark.datasets[0].entries: cli=1575 yaml=1576
```

- [x] **Step 4: Write failing tests for `scenario_facts`**

Append to `utils/test_serving_to_matrix.py`:
```python
def test_scenario_facts_file_dataset():
    assert scenario_facts("custom-agentic-tau2") == (1200, "tau2-cs")
    assert scenario_facts("custom-agentic-tau2-smoke") == (180, "tau2-cs")


def _scenario_repo(tmp_path, scenario_text, lock_text="tau2-cs: {repo: r/x, revision: a, file: f, sha256: b}\n"):
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


@pytest.mark.parametrize("text", ["benchmark: {}\n", "{{not yaml\n", "benchmark:\n  phases: 3\n  datasets: []\n", "[]\n"])
def test_malformed_scenario_is_one_line_error(tmp_path, text):
    repo = _scenario_repo(tmp_path, text)
    with pytest.raises(ServingError) as info:
        scenario_facts("s", repo)
    message = str(info.value)
    assert message.startswith("scenario: scenarios/s.yaml ")
    assert "\n" not in message
```
Add `from infx.serving.schema import ServingError` to the imports if missing. If an existing test asserts the old text "must have exactly one public dataset", update it to "must have exactly one dataset".

Create `utils/test_serving_lock.py`:
```python
import re

from infx.serving.schema import available_scenarios
from infx.serving.to_matrix import load_lock, scenario_facts


def test_lock_entries_are_pinned():
    lock = load_lock()
    assert "tau2-cs" in lock
    for name, entry in lock.items():
        assert set(entry) == {"repo", "revision", "file", "sha256"}, name
        assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), name
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name


def test_every_scenario_resolves():
    for scenario in available_scenarios():
        duration, loader = scenario_facts(scenario)
        assert duration > 0 and loader


def test_lock_is_not_a_scenario():
    names = available_scenarios()
    assert {"custom-agentic-tau2", "custom-agentic-tau2-smoke"} <= set(names)
    assert not {"lock", "datasets"} & set(names)
```

- [x] **Step 5: Run to see them fail**

Run: `<PYTEST>`
Expected: FAIL — `ImportError: cannot import name 'load_lock'` (test_serving_lock) and `ServingError ... must have exactly one public dataset` / tracebacks (KeyError) in the new to_matrix tests.

- [x] **Step 6: Implement**

In `infx/serving/to_matrix.py`, add below the imports:
```python
LOCK = Path("scenarios/datasets/lock.yaml")


def load_lock(repo: Path = REPO_ROOT) -> dict[str, dict]:
    """File datasets pinned for scenarios (name -> repo/revision/file/sha256)."""
    path = repo / LOCK
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}
```
and replace `scenario_facts` with:
```python
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
    except (OSError, yaml.YAMLError, KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ServingError(f"scenario: {where} is malformed ({type(exc).__name__}: {exc})".replace("\n", " ")) from None
    if duration is None:
        raise ServingError(f"scenario: {where} must have exactly one profiling phase")
    if dataset is None:
        raise ServingError(f"scenario: {where} must have exactly one dataset")
    if kind == "public":
        return duration, dataset["dataset"]
    if kind == "file":
        name = dataset.get("name")
        if name not in load_lock(repo):
            raise ServingError(f"scenario: {where} dataset '{name}' is not in scenarios/datasets/lock.yaml")
        return duration, name
    raise ServingError(f"scenario: {where} dataset type must be public or file, got {kind!r}")
```

- [x] **Step 7: Run tests**

Run: `<PYTEST>`
Expected: all pass.

- [x] **Step 8: Commit**

```bash
git add scenarios/custom-agentic-tau2.yaml scenarios/custom-agentic-tau2-smoke.yaml utils/serving/check_scenario_golden.py infx/serving/to_matrix.py utils/test_serving_to_matrix.py utils/test_serving_lock.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): custom-agentic-tau2 scenarios; scenario_facts accepts lock-pinned file datasets"
```

---

### Task 4: Fetch + verify the locked dataset on the runner

**Files:**
- Create: `utils/serving/fetch_custom_dataset.py`
- Modify: `benchmarks/benchmark_lib.sh` (add `is_custom_dataset`, `fetch_custom_dataset` right after `build_replay_cmd_yaml`)
- Modify: `benchmarks/single_node/agentic/serve_generic.sh` (dataset branch)
- Test: create `utils/test_serving_fetch_dataset.py`

**Interfaces:**
- Consumes: lock (Task 2), `SERVING_LOADER` = lock key for file scenarios (Task 3).
- Produces: `python utils/serving/fetch_custom_dataset.py <name>` → stdout = local path, exit 0; `--check <name>` → exit 0 iff locked. Bash: `is_custom_dataset <name>`, `fetch_custom_dataset <name>` (exports `IX_CUSTOM_DATASET_FILE`, writes `$RESULT_DIR/dataset.txt`).

- [x] **Step 1: Write the failing tests**

`utils/test_serving_fetch_dataset.py`:
```python
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SECRET = "hf_dummySECRET"
FAKE_HUB = textwrap.dedent('''\
    import json, os
    from . import errors
    def hf_hub_download(repo_id, filename, revision=None, repo_type=None, token=None):
        with open(os.environ["FAKE_HF_LOG"], "a") as f:
            f.write(json.dumps({"repo": repo_id, "file": filename, "revision": revision,
                                "repo_type": repo_type, "token": token}) + "\\n")
        if os.environ.get("FAKE_HF_RAISE") == "notfound":
            raise errors.RepositoryNotFoundError("404 for " + repo_id)
        return os.environ["FAKE_HF_FILE"]
''')
FAKE_ERRORS = textwrap.dedent('''\
    class HfHubHTTPError(Exception):
        response = None
    class RepositoryNotFoundError(HfHubHTTPError): pass
    class GatedRepoError(RepositoryNotFoundError): pass
''')
CONTENT = b'{"session_id": "s1", "messages": []}\n'


@pytest.fixture()
def world(tmp_path):
    """A mini repo (script + lock) and a fake huggingface_hub on PYTHONPATH."""
    import hashlib
    sha = hashlib.sha256(CONTENT).hexdigest()
    repo = tmp_path / "repo"
    (repo / "utils/serving").mkdir(parents=True)
    (repo / "scenarios/datasets").mkdir(parents=True)
    shutil.copy(REPO / "utils/serving/fetch_custom_dataset.py", repo / "utils/serving/")
    (repo / "scenarios/datasets/lock.yaml").write_text(
        f"tau2-cs: {{repo: Green-Lab/x, revision: {'a' * 40}, file: trace.jsonl, sha256: {sha}}}\n")
    fake = tmp_path / "fake/huggingface_hub"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(FAKE_HUB)
    (fake / "errors.py").write_text(FAKE_ERRORS)
    data = tmp_path / "trace.jsonl"
    data.write_bytes(CONTENT)
    results = tmp_path / "results"
    results.mkdir()
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "fake"), "FAKE_HF_LOG": str(tmp_path / "calls"),
           "FAKE_HF_FILE": str(data), "RESULT_DIR": str(results), "RUNNER_NAME": "r1"}
    env.pop("GREENLAB_HF_TOKEN", None)
    return {"repo": repo, "env": env, "data": data, "results": results, "calls": tmp_path / "calls", "sha": sha}


def fetch(world, *args, **env):
    return subprocess.run([sys.executable, str(world["repo"] / "utils/serving/fetch_custom_dataset.py"), *args],
                          env={**world["env"], **env}, capture_output=True, text=True)


def test_token_passed_path_printed_provenance_written(world):
    out = fetch(world, "tau2-cs", GREENLAB_HF_TOKEN=SECRET)
    assert out.returncode == 0, out.stderr
    assert out.stdout == f"{world['data']}\n"
    call = world["calls"].read_text()
    assert f'"token": "{SECRET}"' in call and f'"revision": "{"a" * 40}"' in call and '"repo_type": "dataset"' in call
    assert "GREENLAB_HF_TOKEN: set" in out.stderr
    assert SECRET not in out.stdout + out.stderr
    provenance = (world["results"] / "dataset.txt").read_text()
    assert provenance.startswith(f"tau2-cs Green-Lab/x@{'a' * 40} ") and world["sha"] in provenance


def test_no_token_means_anonymous(world):
    out = fetch(world, "tau2-cs")
    assert out.returncode == 0, out.stderr
    assert '"token": null' in world["calls"].read_text()
    assert "GREENLAB_HF_TOKEN: unset" in out.stderr


def test_sha_mismatch_fails_with_both_hashes(world):
    world["data"].write_bytes(CONTENT + b"x")
    out = fetch(world, "tau2-cs")
    assert out.returncode == 1
    assert out.stderr.splitlines()[-1].startswith("ERROR dataset tau2-cs: sha256 mismatch")
    assert world["sha"] in out.stderr and "got " in out.stderr
    assert "Traceback" not in out.stderr


def test_unreadable_repo_names_token_state_not_value(world):
    out = fetch(world, "tau2-cs", GREENLAB_HF_TOKEN=SECRET, FAKE_HF_RAISE="notfound")
    assert out.returncode == 1
    assert out.stderr.splitlines()[-1] == (
        f"ERROR dataset tau2-cs: Green-Lab/x@{'a' * 40} not readable; GREENLAB_HF_TOKEN is set on runner r1")
    assert SECRET not in out.stdout + out.stderr


def test_unknown_name(world):
    out = fetch(world, "nope")
    assert out.returncode == 1
    assert "ERROR dataset nope: not in scenarios/datasets/lock.yaml" in out.stderr


def test_check_flag(world):
    assert fetch(world, "--check", "tau2-cs").returncode == 0
    assert fetch(world, "--check", "semianalysis_cc_traces_weka_062126").returncode == 1
    assert not world["calls"].exists()


def test_bash_wrapper_never_traces_token(world):
    script = textwrap.dedent(f'''\
        set -e
        export IS_AGENTIC=1 KV_OFFLOADING=none PORT=8888 CONC=8 DURATION=180 MODEL=m FRAMEWORK=vllm
        export INFMAX_CONTAINER_WORKSPACE={world["repo"]}
        source {REPO}/benchmarks/benchmark_lib.sh
        AIPERF_PYTHON={sys.executable}
        set -x
        is_custom_dataset tau2-cs
        fetch_custom_dataset tau2-cs
        [[ $- == *x* ]] && echo XTRACE_RESTORED
        echo "FILE=$IX_CUSTOM_DATASET_FILE"
    ''')
    out = subprocess.run(["bash", "-c", script], env={**world["env"], "GREENLAB_HF_TOKEN": SECRET},
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert f"FILE={world['data']}" in out.stdout
    assert "XTRACE_RESTORED" in out.stdout
    assert SECRET not in out.stdout + out.stderr
```

- [x] **Step 2: Run to see them fail**

Run: `<PYTEST> -k fetch_dataset` (or the full PYTEST)
Expected: FAIL — `fetch_custom_dataset.py` missing (`FileNotFoundError` in `shutil.copy`).

- [x] **Step 3: Write `utils/serving/fetch_custom_dataset.py`**

```python
"""Fetch one lock-pinned file dataset into the HF cache; print its local path.

    python utils/serving/fetch_custom_dataset.py <name>          # path on stdout
    python utils/serving/fetch_custom_dataset.py --check <name>  # exit 0 iff <name> is locked

<name> is a key of scenarios/datasets/lock.yaml (repo, revision, file, sha256).
Token: GREENLAB_HF_TOKEN when set (public repos too, so private ones need no code
change), else anonymous. Read only here, never echoed. Status and errors go to
stderr; failures are one `ERROR ...` line and exit 1. Writes
$RESULT_DIR/dataset.txt (name repo@revision file sha256 path) when RESULT_DIR is set.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import yaml

LOCK = Path(__file__).resolve().parents[2] / "scenarios" / "datasets" / "lock.yaml"


def fail(message: str) -> None:
    print(f"ERROR {message}", file=sys.stderr)
    sys.exit(1)


def load_lock() -> dict:
    return (yaml.safe_load(LOCK.read_text()) or {}) if LOCK.is_file() else {}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> str:
    entry = load_lock().get(name)
    if not entry:
        fail(f"dataset {name}: not in scenarios/datasets/lock.yaml")
    token = os.environ.get("GREENLAB_HF_TOKEN") or None
    state = "set" if token else "unset"
    print(f"GREENLAB_HF_TOKEN: {state}", file=sys.stderr)

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    where = f"{entry['repo']}@{entry['revision']}"
    unreadable = (f"dataset {name}: {where} not readable; GREENLAB_HF_TOKEN is {state} "
                  f"on runner {os.environ.get('RUNNER_NAME', '?')}")
    try:
        path = hf_hub_download(entry["repo"], entry["file"], revision=entry["revision"],
                               repo_type="dataset", token=token)
    except RepositoryNotFoundError:  # includes GatedRepoError
        fail(unreadable)
    except HfHubHTTPError as exc:
        if getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
            fail(unreadable)
        raise
    actual = sha256(path)
    if actual != entry["sha256"]:
        fail(f"dataset {name}: sha256 mismatch for {where}/{entry['file']}: expected {entry['sha256']}, got {actual}")
    if os.environ.get("RESULT_DIR"):
        Path(os.environ["RESULT_DIR"], "dataset.txt").write_text(
            f"{name} {where} file={entry['file']} sha256={actual} path={path}\n")
    return path


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--check":
        return 0 if argv[1] in load_lock() else 1
    if len(argv) != 1:
        fail("usage: fetch_custom_dataset.py [--check] <name>")
    print(fetch(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [x] **Step 4: Add the bash helpers to `benchmarks/benchmark_lib.sh`** (directly after the closing `}` of `build_replay_cmd_yaml`)

```bash
# True when <name> is a lock-pinned file dataset (scenarios/datasets/lock.yaml).
is_custom_dataset() {
    check_env_vars INFMAX_CONTAINER_WORKSPACE AIPERF_PYTHON
    "$AIPERF_PYTHON" "$INFMAX_CONTAINER_WORKSPACE/utils/serving/fetch_custom_dataset.py" --check "$1"
}

# Download + sha256-verify a lock-pinned file dataset and export
# IX_CUSTOM_DATASET_FILE (the scenario YAML's dataset path). GREENLAB_HF_TOKEN
# is read only inside the Python helper; xtrace is paused so nothing
# token-shaped can reach the log, and restored afterwards.
fetch_custom_dataset() {
    check_env_vars INFMAX_CONTAINER_WORKSPACE AIPERF_PYTHON
    local name="$1" path restore=""
    [[ $- == *x* ]] && restore=1
    { set +x; } 2>/dev/null
    if ! path=$("$AIPERF_PYTHON" "$INFMAX_CONTAINER_WORKSPACE/utils/serving/fetch_custom_dataset.py" "$name"); then
        [[ -n "$restore" ]] && set -x
        return 1
    fi
    export IX_CUSTOM_DATASET_FILE="$path"
    echo "dataset $name: $IX_CUSTOM_DATASET_FILE"
    [[ -n "$restore" ]] && set -x
    return 0
}
```

- [x] **Step 5: Branch in `serve_generic.sh`**

Replace
```bash
# Validates the loader name and pre-downloads the trace dataset; the
# --public-dataset flag it builds is unused (the scenario YAML names the dataset).
export WEKA_LOADER_OVERRIDE="$SERVING_LOADER"
resolve_trace_source
install_agentic_deps
```
with
```bash
install_agentic_deps
# Scenario dataset: a lock-pinned file (custom-agentic-*: download, sha256 check,
# IX_CUSTOM_DATASET_FILE for the YAML) or a public Weka loader (resolve_trace_source
# validates and pre-downloads it; the --public-dataset flag it builds is unused,
# the scenario YAML names the dataset).
if is_custom_dataset "$SERVING_LOADER"; then
    fetch_custom_dataset "$SERVING_LOADER"
else
    export WEKA_LOADER_OVERRIDE="$SERVING_LOADER"
    resolve_trace_source
fi
```
and in the header comment change `#   SERVING_LOADER     public trace dataset of that scenario` to
`#   SERVING_LOADER     dataset of that scenario: public loader alias or scenarios/datasets/lock.yaml key`.

- [x] **Step 6: Run tests**

Run: `bash -n benchmarks/benchmark_lib.sh && bash -n benchmarks/single_node/agentic/serve_generic.sh && <PYTEST>`
Expected: all pass, including the 7 new fetch tests.

- [x] **Step 7: Commit**

```bash
git add utils/serving/fetch_custom_dataset.py benchmarks/benchmark_lib.sh benchmarks/single_node/agentic/serve_generic.sh utils/test_serving_fetch_dataset.py
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "feat(serving): fetch + sha256-verify lock-pinned datasets; GREENLAB_HF_TOKEN read only in Python"
```

---

### Task 5: `GREENLAB_HF_TOKEN` reaches the container (launcher test)

**Files:**
- Modify: `utils/test_serving_launcher.py`
- Modify (comment only): `runners/launch_h200-greennode-slurm-1x.sh` (above the `export -p | grep -vE ...` line)

**Interfaces:** Consumes `run_launcher` (existing test helper). Produces nothing.

- [x] **Step 1: Let `run_launcher` take custom stubs**

Change its signature to `def run_launcher(tmp_path: Path, extra_env: dict[str, str], stubs: dict[str, str] = STUBS) -> list[list[str]]:` and its loop to `for name, body in stubs.items():`.

- [x] **Step 2: Write the test**

```python
def test_greenlab_token_from_runner_env_reaches_container_env(tmp_path):
    """Runner .env -> job env -> launcher `export -p` -> .ix_env (sourced in the container)."""
    stdin_log = tmp_path / "srun_stdin"
    run_launcher(tmp_path, {"SERVE_CMD": "vllm serve m", "GREENLAB_HF_TOKEN": "hf_dummy"},
                 stubs={**STUBS, "srun": f"cat >> {stdin_log}"})
    assert b'declare -x GREENLAB_HF_TOKEN="hf_dummy"' in stdin_log.read_bytes()
```

- [x] **Step 3: Run it**

Run: `<PYTEST> -k greenlab`
Expected: PASS without launcher code changes (V1's deny-list does not match `GREENLAB_`). If it fails, the deny-list regex changed: add `GREENLAB_HF_TOKEN` back by narrowing the regex, never by echoing the value.

- [x] **Step 4: Document the knob in the launcher**

Above the `export -p | grep -vE ...` line add:
```bash
# Runner .env keys (e.g. GREENLAB_HF_TOKEN for lock-pinned private datasets) are in
# the job env and travel to the container through this file (mode 600, deleted after
# `source`). Never echo it; it is not a GitHub secret, so Actions would not mask it.
```

- [x] **Step 5: Run all tests and commit**

Run: `bash -n runners/launch_h200-greennode-slurm-1x.sh && <PYTEST>` → all pass.
```bash
git add utils/test_serving_launcher.py runners/launch_h200-greennode-slurm-1x.sh
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "test(serving): GREENLAB_HF_TOKEN from the runner env reaches the container"
```

---

### Task 6: Skill, example serving YAML, infra handoff

**Files:**
- Create: `serving/qwen38-bf16-tp1-tau2.yaml`
- Modify: `.agents/skills/serving-bench/SKILL.md`
- Modify: `utils/test_serving_dispatch.py` (`test_example_serving_yaml_is_valid` covers both examples)
- Create: `docs/superpowers/handoffs/2026-10-13-custom-agentic-tau2-infra.md`

**Interfaces:** Consumes everything above. Produces the stateless-agent surface.

- [x] **Step 1: Example YAML**

```bash
sed 's/^scenario: .*/scenario: custom-agentic-tau2-smoke/; s/^name: qwen38-bf16-tp1$/name: qwen38-bf16-tp1-tau2/' \
  serving/qwen38-bf16-tp1.yaml > serving/qwen38-bf16-tp1-tau2.yaml
diff serving/qwen38-bf16-tp1.yaml serving/qwen38-bf16-tp1-tau2.yaml   # exactly the name and scenario lines differ
```
If the first file's `name:` line differs from `name: qwen38-bf16-tp1`, edit the new file's `name:` to `qwen38-bf16-tp1-tau2` by hand.

- [x] **Step 2: Test both examples**

In `utils/test_serving_dispatch.py`, parametrize `test_example_serving_yaml_is_valid` over `["serving/qwen38-bf16-tp1.yaml", "serving/qwen38-bf16-tp1-tau2.yaml"]` (keep its body; replace the hard-coded path with the parameter). Run `<PYTEST>` → pass.

- [x] **Step 3: Skill**

In `.agents/skills/serving-bench/SKILL.md`:
- In section 2, replace the scenario bullet with:
  `- the scenario is not given → run \`ls scenarios/*.yaml\`, list the names and ask which: \`agentx-cc-062126\` (Weka coding traces, 3600 s), \`agentx-cc-062126-smoke\` (180 s), \`custom-agentic-tau2\` (tau2 customer-service agent traces, 1200 s), \`custom-agentic-tau2-smoke\` (180 s). \`scenarios/datasets/\` is not a scenario;`
- Change the context-length bullet's opening from "both scenarios replay" to "all scenarios replay".
- After the `env:` bullet ("**Env vars** ... go under `env:`") add: `- No secrets in \`env:\` (the YAML is shown in the run inputs and saved as an artifact). Dataset access uses the runner's own \`GREENLAB_HF_TOKEN\`; never set it in the YAML.`
Keep the file under ~70 lines.

- [x] **Step 4: Infra handoff (for the person with node access)**

`docs/superpowers/handoffs/2026-10-13-custom-agentic-tau2-infra.md`:
````markdown
# custom-agentic tau2: infra steps (needs node access)

Code is done on `feat/custom-agentic-tau2` (stacked on V1 `feat/serving-config-dispatch`).
These steps need a person with access to han-3 / slurm and the user's OK to push.

1. Runner token (runner `h200-greennode-slurm-1x_01`, node `hoanq3-h200-1x-han-3`, runner dir `/data/thanglq5/runner-1x_01`):
   append `GREENLAB_HF_TOKEN=<HF read token of thangquang09 or a Green-Lab read token>` to `/data/thanglq5/runner-1x_01/.env`
   (keep the file mode 600; never paste the token in chat, logs or git). From the workstation, without echoing it:
   ```bash
   ssh slurm-client 'ssh hoanq3-h200-1x-han-3 "umask 077; cat >> /data/thanglq5/runner-1x_01/.env"' < <(printf 'GREENLAB_HF_TOKEN=%s\n' "$(cat ~/.cache/huggingface/token)")
   ```
2. Restart the runner so it re-reads `.env` (runner = sbatch daemon job, script `/shared/thanglq5/daemon-1x01.sh`):
   on `ssh slurm-client`: `squeue -u thanglq5` → `scancel <daemon job id>` → `sbatch /shared/thanglq5/daemon-1x01.sh`;
   then `gh api repos/vngcloud/InferenceX/actions/runners --paginate -q '.runners[]|select(.name=="h200-greennode-slurm-1x_01")|[.status,.busy]'` → `["online",false]`.
3. Push (user OK): `git push -u origin feat/custom-agentic-tau2` (includes V1 commits).
4. Smoke both scenarios from the pushed branch:
   ```bash
   utils/serving/dispatch.sh serving/qwen38-bf16-tp1.yaml
   utils/serving/dispatch.sh serving/qwen38-bf16-tp1-tau2.yaml
   ```
   Pass = both runs green; tau2 `benchmark.log` shows `unique conversations: 1576`; `results/dataset.txt` names
   `Green-Lab/customer-service-agent-traces@<lock revision>`; the recipe log shows `GREENLAB_HF_TOKEN: set`.
   A second tau2 run must not re-download the 1.1 GB file (HF cache hit).
5. Private-path check (proves a future private dataset needs only a lock entry), inside any container on `_01`, with only the runner env:
   `python -c "import os; from huggingface_hub import hf_hub_download as d; print(d('Green-Lab/speed-bench-prepared','README.md',repo_type='dataset',token=os.environ['GREENLAB_HF_TOKEN']))"`
6. Stateless-agent acceptance: a fresh agent (e.g. GreenNode GLM) with only the `serving-bench` skill, prompt:
   "benchmark agentx tau2 smoke cho config: image vllm/vllm-openai:v0.29.0, `vllm serve Qwen/Qwen3.8-27B --max-model-len 262144`, conc 8, runner h200-greennode-slurm-1x_01".
   Pass = it writes one serving YAML with `scenario: custom-agentic-tau2-smoke`, runs `dispatch.sh` once, and the run is green.
````

- [x] **Step 5: Run everything and commit**

Run: `<PYTEST>` and `GOLDEN` → all pass / all OK.
```bash
git add serving/qwen38-bf16-tp1-tau2.yaml .agents/skills/serving-bench/SKILL.md utils/test_serving_dispatch.py docs/superpowers/handoffs/2026-10-13-custom-agentic-tau2-infra.md
git -c user.name="Ly Quang Thang" -c user.email=thanglq5@greennode.ai commit -m "docs(serving): tau2 in the serving-bench skill, example YAML, infra handoff"
```

---

### Task 7: Final verification (controller)

- [x] `<PYTEST>` → all serving tests pass; GOLDEN → 4× OK + negative caught.
- [x] `bash -n` on `benchmarks/benchmark_lib.sh`, `benchmarks/single_node/agentic/serve_generic.sh`, `runners/launch_h200-greennode-slurm-1x.sh`, `utils/serving/dispatch.sh`.
- [x] `python -P -m infx.serving.to_matrix --check serving/qwen38-bf16-tp1-tau2.yaml` (with the PYTEST uv env) prints a title containing `custom-agentic-tau2-smoke (180s)`.
- [x] Render the full get-jobs output for the tau2 example: `python -P -m infx.serving.to_matrix serving/qwen38-bf16-tp1-tau2.yaml` → `serving.loader == "tau2-cs"`, rows have `duration: 180`.
- [x] `git grep -n "hf_[A-Za-z0-9]\{20,\}"` → no tokens committed.
- [x] Whole-branch review (`git diff 3ba14d43e..HEAD`) against the spec + amendments.
- [x] User-level handoff note in `/home/thanglq5/greennode-code/InferenceOptimization/docs/handoffs/2026-10-13-custom-agentic-tau2.md`: branch, commits, what is done, the infra checklist path, open items.
