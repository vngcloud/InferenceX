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

# Real Claude Code coding-session traces (949 traces, 136k requests), public
# HF dataset, no auth required. Prompt text is synthesized from a coding
# corpus against preserved per-request hash_ids/timing -- richer and more
# representative than smoke-test's flat isl/osl padding. See
# utils/aiperf/src/aiperf/dataset/loader/semianalysis_cc_traces_weka.py.
DEFAULT_DATASET = "semianalysis_cc_traces_weka"

# Caps reconstruction cost per sweep point -- the full corpus is 949 traces /
# 136k requests. Override per-stack via throughput-tests.yaml's
# num-dataset-entries if a fuller run is wanted.
DEFAULT_NUM_DATASET_ENTRIES = 100

RANDOM_SEED = 42

# AIPerf auto-scales worker processes with CPU count (up to 32), and each
# worker appears to hold its own copy of the reconstructed dataset in memory.
# Confirmed live: on a 16-core, 31GB-RAM client host, a 100-entry
# semianalysis_cc_traces_weka sweep grew to ~31.6GB RSS and got OOM-killed by
# the kernel, twice, taking the whole runner service down with it. Cap
# workers hard for this lightweight live-check sweep -- it doesn't need
# aiperf's full auto-scaled worker pool to sustain conc up to 32.
MAX_AIPERF_WORKERS = 4

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

    # TEMP diagnostic: aiperf can exit 0 while producing empty per-point
    # stats (dataset exhausted before benchmark-duration elapses, or every
    # request silently erroring) -- surface its own stdout/stderr even on
    # success so we can see request counts/errors instead of guessing.
    print(f"--- aiperf stdout (conc={conc}) ---\n{proc.stdout[-4000:]}", file=sys.stderr)
    print(f"--- aiperf stderr (conc={conc}) ---\n{proc.stderr[-4000:]}", file=sys.stderr)

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
    duration = throughput_config["benchmark-duration-s"]
    conc_list = throughput_config["conc-list"]

    version_before = fetch_version(entry["version_url"])
    config_snapshot = _config_snapshot(entry, version_before)

    sweep = []
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = Path(tmp)
        for conc in conc_list:
            try:
                point = run_one_concurrency(
                    entry, conc, dataset, num_dataset_entries, duration, result_dir
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
