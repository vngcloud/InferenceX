#!/usr/bin/env python3
"""Standalone throughput test: an aiperf concurrency sweep against a live,
already-deployed endpoint using a real coding-session trace dataset. See
design/throughput-test.md for the full design.

This is deliberately NOT part of smoke-test.yml/run_smoke_test.py -- smoke
test and throughput test are two unrelated tests. Throughput is a heavier
check against a shared production endpoint (real trace-derived prompts, not
tiny synthetic isl/osl padding) and gets its own workflow, cadence, dataset,
and ingest schema.

Usage:
    python3 utils/throughput_test/run_throughput_test.py \
        --matrix-entry '<json matrix entry>'
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from discover import fetch_version  # noqa: E402
from gpu_metrics import fetch_gpu_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
AIPERF_ADAPTER = REPO_ROOT / "utils" / "bench_serving" / "aiperf_adapter.py"

# Real Claude Code coding-session traces with full subagent fan-out
# (391 traces), public HF dataset, no auth required -- the "current" variant
# per the aiperf fork's own plugins.yaml ("Default corpus for DSv4 agentic
# recipes"). Requires the thangquang09/aiperf fork (benchtool/agentx-weka
# branch, see .gitmodules) -- not present in the previous vngcloud/aiperf pin.
DEFAULT_DATASET = "semianalysis_cc_traces_weka_with_subagents_060826"

# Matches the internal agentic-replay recipe (benchmark-tmpl.yml /
# run-sweep.yml) rather than a lightweight synthetic ping -- this dataset's
# conversations are real multi-turn coding sessions, not one-shot prompts.
DEFAULT_SCENARIO = "inferencex-agentx-mvp"

# Caps reconstruction cost per sweep point. Override per-stack via
# throughput-tests.yaml's num-dataset-entries for a fuller run.
DEFAULT_NUM_DATASET_ENTRIES = 64

# Real agentic sessions need real wall-clock time to reach steady state;
# 30s (the old lightweight-ping default) undercounted completions. Matches
# the internal recipe's full-duration sweep.
DEFAULT_BENCHMARK_DURATION_S = 600

RANDOM_SEED = 42

# AIPerf auto-scales worker processes with CPU count (up to 32), and each
# worker appears to hold its own copy of the reconstructed dataset in memory.
# Confirmed live: on a 16-core, 31GB-RAM client host, a 100-entry
# semianalysis_cc_traces_weka sweep grew to ~31.6GB RSS and got OOM-killed by
# the kernel, twice, taking the whole runner service down with it. Cap
# workers hard for this lightweight live-check sweep -- it doesn't need
# aiperf's full auto-scaled worker pool to sustain conc up to 32.
MAX_AIPERF_WORKERS = 4

# Real semianalysis_cc_traces_weka* conversations can run very long (every
# trace has >=20 main-agent turns, cumulative context grows turn over turn).
# Confirmed live: at the deployment's old 32768-token max context, requests
# over the limit got rejected outright with HTTP 400 ("input longer than the
# model's context length"), which is why every sweep point came back with
# empty stats regardless of duration/entries/worker tuning -- no amount of
# tuning those fixes a request-rejection problem. The deployment's model
# context has since been raised to 128K; this stays comfortably under that
# to leave headroom for completion tokens. Drop any oversized conversation
# before it ever hits the endpoint rather than let it 400 mid-run.
DEFAULT_MAX_CONTEXT_LENGTH = 120000

# Matches the internal agentic-replay recipe's --slice-duration -- windows
# the trace replay's timing reconstruction to 1s slices.
DEFAULT_SLICE_DURATION_S = 1.0

# Buffer added on top of --benchmark-duration for aiperf's own setup
# (tokenizer/dataset download + reconstruction, observed to take several
# minutes even for a 100-trace subset) and in-flight request drain at the
# end of the window. A stuck subprocess raises after this instead of
# hanging indefinitely.
SUBPROCESS_TIMEOUT_BUFFER_S = 600


def endpoint_type_for(endpoint: str) -> str:
    """Derive aiperf's --endpoint-type from the discovered endpoint path.

    Must be derived from the path shape, not the stack's framework name --
    see design/throughput-test.md. Fail loudly on anything we haven't
    validated rather than guess and silently mis-shape requests.
    """
    if endpoint.endswith("chat/completions"):
        return "chat"
    if endpoint.endswith("completions"):
        return "completions"
    raise ValueError(
        f"Don't know how to derive --endpoint-type from endpoint path {endpoint!r}"
    )


def run_one_concurrency(
    entry: dict,
    conc: int,
    dataset: str,
    num_dataset_entries: int,
    duration: int,
    result_dir: Path,
    scenario: str | None,
    max_context_length: int | None,
    slice_duration: float | None,
    use_server_token_count: bool,
    unsafe_override: bool,
) -> dict:
    result_filename = f"throughput_{entry['name']}_conc{conc}"
    cmd = [
        "python3",
        str(AIPERF_ADAPTER),
        "--model",
        entry["model"],
        "--url",
        entry["base_url"],
        "--endpoint",
        entry["endpoint"],
        "--endpoint-type",
        endpoint_type_for(entry["endpoint"]),
        "--concurrency",
        str(conc),
        "--benchmark-duration",
        str(duration),
        "--public-dataset",
        dataset,
        "--num-dataset-entries",
        str(num_dataset_entries),
        "--tokenizer-trust-remote-code",
        "--random-seed",
        str(RANDOM_SEED),
        "--max-workers",
        str(MAX_AIPERF_WORKERS),
        "--result-filename",
        result_filename,
        "--result-dir",
        str(result_dir),
    ]
    if scenario:
        cmd.extend(["--scenario", scenario])
    if max_context_length is not None:
        cmd.extend(["--max-context-length", str(max_context_length)])
    if slice_duration is not None:
        cmd.extend(["--slice-duration", str(slice_duration)])
    if use_server_token_count:
        cmd.append("--use-server-token-count")
    if unsafe_override:
        cmd.append("--unsafe-override")
    if entry.get("gpu_metrics_url"):
        cmd.extend(["--gpu-telemetry-url", entry["gpu_metrics_url"]])

    # Bounded, not indefinite: dataset load/tokenizer setup can reasonably
    # take a few minutes (see SUBPROCESS_TIMEOUT_BUFFER_S), but this must
    # eventually fail loud rather than hang -- a GH Actions cancel doesn't
    # reliably interrupt this subprocess (observed: a cancelled run left the
    # job stuck ~30min with logs never persisted), so a stuck run needs to
    # surface as a clear timeout error, not an unobservable hang.
    timeout_s = duration + SUBPROCESS_TIMEOUT_BUFFER_S
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"aiperf_adapter.py timed out after {timeout_s}s for conc={conc} "
            f"(benchmark-duration={duration}s + {SUBPROCESS_TIMEOUT_BUFFER_S}s setup/drain buffer):\n"
            f"--- stdout ---\n{(exc.stdout or '')[-2000:]}\n"
            f"--- stderr ---\n{(exc.stderr or '')[-2000:]}"
        ) from exc
    if proc.returncode != 0:
        # Surface both streams -- aiperf's own failure text can land on
        # either, and swallowing one silently hides the real root cause
        # (e.g. an HF Hub rate-limit warning followed by a stdout traceback).
        raise RuntimeError(
            f"aiperf_adapter.py failed for conc={conc}:\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )

    # aiperf can exit 0 while every request errored (e.g. hit the
    # deployment's context-length limit) -- that produces an all-empty
    # sweep point with no indication why. Surface error_summary from its
    # own raw export whenever it's non-empty, so a future silent-failure
    # mode doesn't require re-deriving this diagnostic from scratch (see
    # design/throughput-test.md's context-length-limit history).
    raw_export_path = result_dir / f"{result_filename}_aiperf" / "profile_export_aiperf.json"
    if raw_export_path.exists():
        raw_export = json.loads(raw_export_path.read_text())
        error_summary = raw_export.get("error_summary")
        if error_summary:
            print(
                f"::warning::[conc={conc}] aiperf reported request errors:\n"
                f"{json.dumps(error_summary, indent=2)}",
                file=sys.stderr,
            )
    result_path = result_dir / f"{result_filename}.json"
    with open(result_path) as f:
        return json.load(f)


def _config_snapshot(entry: dict, version_payload: dict) -> dict:
    """framework/precision/tp from /discover (already on the matrix entry,
    no extra call), plus disaggregation from /version when present (mirrors
    metadata.data's convention -- only pd-disaggregation stacks report it).

    InferenceX-app needs these to resolve a throughput sweep point to a
    `configs` row: their natural key needs framework/precision/tp/hardware,
    and throughput-test/smoke-test are two fully independent workflows with
    no shared run ID or guaranteed-same timestamp, so InferenceX-app can't
    safely join them by (stack, latest-date) on their side -- a redeploy
    between the two runs would silently attribute throughput numbers to the
    wrong config. Snapshotting into this artifact avoids that join entirely.
    """
    snapshot = {
        "framework": entry.get("framework"),
        "precision": entry.get("precision"),
        "tp": entry.get("tp"),
    }
    if "disaggregation" in version_payload:
        snapshot["disaggregation"] = version_payload["disaggregation"]
    return snapshot


def run(entry: dict, throughput_config: dict) -> dict:
    dataset = throughput_config.get("dataset", DEFAULT_DATASET)
    num_dataset_entries = throughput_config.get(
        "num-dataset-entries", DEFAULT_NUM_DATASET_ENTRIES
    )
    duration = throughput_config.get("benchmark-duration-s", DEFAULT_BENCHMARK_DURATION_S)
    conc_list = throughput_config["conc-list"]
    scenario = throughput_config.get("scenario", DEFAULT_SCENARIO)
    max_context_length = throughput_config.get(
        "max-context-length", DEFAULT_MAX_CONTEXT_LENGTH
    )
    slice_duration = throughput_config.get("slice-duration-s", DEFAULT_SLICE_DURATION_S)
    use_server_token_count = throughput_config.get("use-server-token-count", True)
    unsafe_override = throughput_config.get("unsafe-override", True)

    version_before = fetch_version(entry["version_url"])
    config_snapshot = _config_snapshot(entry, version_before)

    sweep = []
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = Path(tmp)
        for conc in conc_list:
            try:
                point = run_one_concurrency(
                    entry,
                    conc,
                    dataset,
                    num_dataset_entries,
                    duration,
                    result_dir,
                    scenario,
                    max_context_length,
                    slice_duration,
                    use_server_token_count,
                    unsafe_override,
                )
            except Exception as exc:  # noqa: BLE001 -- surface any failure as a run failure
                return {
                    "ok": False,
                    "detail": f"throughput sweep failed at conc={conc}: {exc}",
                    "data": {"dataset": dataset, **config_snapshot, "completed": sweep},
                }
            sweep.append({"conc": conc, **point})

    # Best-effort: a heavy sweep can apparently make the stack's own
    # /version endpoint transiently 503 right after the sweep finishes
    # (observed repeatedly on sglang-vanilla) -- don't discard an already-
    # completed sweep over that. redeployed_mid_run becomes unconfirmable
    # (not the same as confirmed-false) rather than crashing the whole run.
    try:
        version_after = fetch_version(entry["version_url"])
        redeployed = version_before != version_after
    except Exception as exc:  # noqa: BLE001 -- best-effort check, not the sweep itself
        print(
            f"::warning::[{entry['name']}] couldn't re-check /version after "
            f"the sweep, redeployed_mid_run is unconfirmed: {exc}",
            file=sys.stderr,
        )
        redeployed = None

    try:
        gpu_model = fetch_gpu_model(entry.get("gpu_metrics_url"))
    except Exception as exc:  # noqa: BLE001 -- enrichment, not the check itself -- never fail the sweep over this
        print(f"::warning::[{entry['name']}] gpu_model lookup failed: {exc}", file=sys.stderr)
        gpu_model = None

    data = {
        "dataset": dataset,
        "num_dataset_entries": num_dataset_entries,
        # Snapshotted at test time, not looked up at ingest time -- see the
        # gpu-metrics discussion with InferenceX-app: gpu_metrics_url reports
        # live pod state, which may have moved/rescheduled by the time
        # ingest runs.
        "gpu_model": gpu_model,
        **config_snapshot,
        "sweep": sweep,
        "redeployed_mid_run": redeployed,
    }
    if redeployed:
        return {
            "ok": False,
            "detail": (
                "stack redeployed mid-run (version_url changed between the "
                "start and end of the sweep) -- throughput numbers may mix two "
                "deployments, not trusting them"
            ),
            "data": data,
        }

    return {
        "ok": True,
        "detail": f"completed sweep at conc={conc_list}",
        "data": data,
    }


THROUGHPUT_COLUMNS = [
    ("conc", "conc"),
    ("total_token_throughput", "total tok/s"),
    ("output_throughput", "output tok/s"),
    ("mean_ttft_ms", "TTFT (ms)"),
    ("mean_tpot_ms", "ITL (ms)"),
    ("mean_e2el_ms", "e2e latency (ms)"),
]


def render_summary(entry: dict, result: dict) -> str:
    icon = "✅" if result["ok"] else "❌"
    lines = [
        f"## Throughput test: `{entry['name']}`",
        "",
        f"{icon} {result['detail']}",
    ]

    sweep = result["data"].get("sweep")
    if sweep:
        headers = [label for _, label in THROUGHPUT_COLUMNS]
        lines += [
            "",
            f"**dataset:** `{result['data']['dataset']}`",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        for point in sweep:
            row = [
                f"{point.get(key, ''):.2f}" if isinstance(point.get(key), float) else str(point.get(key, ""))
                for key, _ in THROUGHPUT_COLUMNS
            ]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-entry", required=True, help="One JSON matrix entry (see generate_matrix.py)"
    )
    parser.add_argument("--summary-file", default=None, help="Path to append the Markdown summary to")
    parser.add_argument(
        "--results-file",
        default=None,
        help="Path to write the full raw results as JSON, for upload as a build artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = json.loads(args.matrix_entry)

    result = run(entry, entry["throughput"])
    summary = render_summary(entry, result)

    print(summary)
    if args.summary_file:
        with open(args.summary_file, "a") as f:
            f.write(summary)

    if args.results_file:
        raw = {
            "stack": entry["name"],
            "test_type": "throughput",
            # Lets InferenceX-app file these into its own "live-check" tab,
            # separate from full sweep runs -- see design/throughput-test.md.
            "run_type": "live-check",
            "ok": result["ok"],
            "detail": result["detail"],
            "data": result["data"],
        }
        with open(args.results_file, "w") as f:
            json.dump(raw, f, indent=2)

    if not result["ok"]:
        print(f"::error::[{entry['name']}] throughput test failed: {result['detail']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
