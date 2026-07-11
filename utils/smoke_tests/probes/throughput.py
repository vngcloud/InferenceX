#!/usr/bin/env python3
"""Throughput probe: a short aiperf-based concurrency sweep against a live,
already-deployed endpoint. See design/throughput-test.md for the full
design -- this deliberately reuses utils/bench_serving/aiperf_adapter.py
(the team's standard aiperf wrapper) rather than any plain HTTP client.

Everything about *where* to send requests and *what* is being served comes
from the live /discover entry passed in as `entry`; only concurrency/
duration/isl/osl (the `throughput_config` block from smoke-tests.yaml) is
InferenceX's own input.

Usage (standalone):
    python3 utils/smoke_tests/probes/throughput.py \
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
from result import ProbeResult  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
AIPERF_ADAPTER = REPO_ROOT / "utils" / "bench_serving" / "aiperf_adapter.py"


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


def run_one_concurrency(entry: dict, conc: int, isl: int, osl: int, duration: int, result_dir: Path) -> dict:
    result_filename = f"smoke_{entry['name']}_conc{conc}"
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
        "--isl",
        str(isl),
        "--osl",
        str(osl),
        "--result-filename",
        result_filename,
        "--result-dir",
        str(result_dir),
    ]
    if entry.get("gpu_metrics_url"):
        cmd.extend(["--gpu-telemetry-url", entry["gpu_metrics_url"]])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"aiperf_adapter.py failed for conc={conc}:\n{proc.stderr[-4000:]}"
        )

    result_path = result_dir / f"{result_filename}.json"
    with open(result_path) as f:
        return json.load(f)


def run(entry: dict, throughput_config: dict) -> ProbeResult:
    isl = throughput_config["isl"]
    osl = throughput_config["osl"]
    duration = throughput_config["benchmark-duration-s"]
    conc_list = throughput_config["conc-list"]

    version_before = fetch_version(entry["version_url"])

    sweep = []
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = Path(tmp)
        for conc in conc_list:
            try:
                point = run_one_concurrency(entry, conc, isl, osl, duration, result_dir)
            except Exception as exc:  # noqa: BLE001 -- surface any failure as a probe failure
                return ProbeResult(
                    ok=False,
                    detail=f"throughput sweep failed at conc={conc}: {exc}",
                    data={"completed": sweep},
                )
            sweep.append({"conc": conc, **point})

    version_after = fetch_version(entry["version_url"])
    redeployed = version_before != version_after

    data = {"sweep": sweep, "redeployed_mid_run": redeployed}
    if redeployed:
        return ProbeResult(
            ok=False,
            detail=(
                "stack redeployed mid-probe (version_url changed between the "
                "start and end of the sweep) -- throughput numbers may mix two "
                "deployments, not trusting them"
            ),
            data=data,
        )

    return ProbeResult(
        ok=True,
        detail=f"completed sweep at conc={conc_list}",
        data=data,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-entry", required=True, help="One JSON matrix entry (see generate_matrix.py)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = json.loads(args.matrix_entry)
    result = run(entry, entry["throughput"])

    print(json.dumps(result.data, indent=2))
    if not result.ok:
        print(f"::error::{result.detail}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
