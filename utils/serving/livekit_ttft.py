"""TTFT of a LiveKit (custom-agentic-livekit111868-*) run, split by a room's first turn vs later turns.

    python utils/serving/livekit_ttft.py <artifact-dir | profile_export.jsonl> ... [--plan <plan.csv>]

A directory is searched for every aiperf_artifacts/profile_export.jsonl (one per conc); each is read
with the plan of the dataset its run used: dataset.txt names it (e.g. livekit-111868-v1), and the
lock entry <name>-plan in scenarios/datasets/lock.yaml pins the plan CSV that ships with the trace
(fetched like the trace: GREENLAB_HF_TOKEN). --plan uses a local CSV instead.

First turns (no parent in the plan) carry the room's whole system prompt uncached, so they set the
tail; "5s turns" are the requests whose logged client timeout is 5 s. Only the profiling phase counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

UNCACHED_BUCKETS = ((0, 5000), (5000, 11000), (11000, 20000), (20000, None))


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else float("nan")


def load_plan(path: str | Path) -> dict[str, list[dict]]:
    plan: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            plan[row["session_id"]].append(row)
    for turns in plan.values():
        turns.sort(key=lambda r: int(r["i"]))
    return plan


def load_requests(profile_export: Path, plan: dict[str, list[dict]]) -> list[dict]:
    """One dict per profiling request with a TTFT: ttft (s), first, timeout_5s, uncached (tokens)."""
    rows = []
    with open(profile_export) as f:
        for line in f:
            record = json.loads(line)
            meta, metrics = record["metadata"], record["metrics"]
            if meta.get("benchmark_phase") != "profiling" or "time_to_first_token" not in metrics:
                continue
            turn = plan[meta["conversation_id"]][meta["turn_index"]]
            rows.append({
                "ttft": metrics["time_to_first_token"]["value"] / 1e3,
                "first": turn["parent"] == "",
                "timeout_5s": float(turn["timeout"]) == 5.0,
                "uncached": int(turn["isl"]) - int(turn["shared"]),
            })
    return rows


def describe(name: str, rows: list[dict]) -> list[str]:
    ttft = [r["ttft"] for r in rows]
    p95 = quantile(ttft, 0.95)
    over = [r for r in rows if r["ttft"] > p95]
    first = [r for r in rows if r["first"]]
    lines = [(f"{name}: n={len(rows)} p95={p95:.2f}s  first-turn share={len(first) / len(rows):.1%}  "
              f"first-turn share of >p95: {sum(r['first'] for r in over) / max(len(over), 1):.0%}")]
    for label, sel in (("first", first), ("later", [r for r in rows if not r["first"]])):
        x = [r["ttft"] for r in sel]
        if not x:
            continue
        lines.append(f"  {label}: n={len(x):5d} p50={quantile(x, .5):.2f} p90={quantile(x, .9):.2f} "
                     f"p95={quantile(x, .95):.2f} p99={quantile(x, .99):.2f} "
                     f">1s={sum(v > 1 for v in x) / len(x):.0%}  "
                     f"median uncached={quantile([r['uncached'] for r in sel], .5)}")
    return lines


def report(rows: list[dict]) -> str:
    if not rows:
        return "no profiling requests with a TTFT"
    lines = describe("all", rows)
    lines += describe("5s turns", [r for r in rows if r["timeout_5s"]]) if any(r["timeout_5s"] for r in rows) else []
    first = [r for r in rows if r["first"]]
    for lo, hi in UNCACHED_BUCKETS:
        x = [r["ttft"] for r in first if r["uncached"] >= lo and (hi is None or r["uncached"] < hi)]
        if x:
            lines.append(f"  first turns uncached {lo}-{hi or ''}: n={len(x)} p50={quantile(x, .5):.2f} "
                         f"p95={quantile(x, .95):.2f} >1s={sum(v > 1 for v in x) / len(x):.0%}")
    return "\n".join(lines)


def find_exports(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("profile_export.jsonl"))


def dataset_name(profile_export: Path) -> str:
    """The dataset name from the run's dataset.txt, next to aiperf_artifacts/."""
    for parent in profile_export.parents:
        txt = parent / "dataset.txt"
        if txt.is_file():
            return txt.read_text().split()[0]
    raise SystemExit(f"ERROR {profile_export}: no dataset.txt above it; pass --plan")


def plan_path(name: str) -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fetch_custom_dataset import fetch, load_lock

    if f"{name}-plan" not in load_lock():
        raise SystemExit(f"ERROR dataset {name}: no {name}-plan in scenarios/datasets/lock.yaml; pass --plan")
    return fetch(f"{name}-plan")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="artifact dirs or profile_export.jsonl files")
    parser.add_argument("--plan", help="plan CSV to use instead of the locked one")
    args = parser.parse_args(argv)
    exports = [e for p in args.paths for e in find_exports(p)]
    if not exports:
        raise SystemExit("ERROR no profile_export.jsonl under " + " ".join(map(str, args.paths)))
    plans: dict[str, dict] = {}
    for export in exports:
        key = args.plan or dataset_name(export)
        if key not in plans:
            plans[key] = load_plan(args.plan or plan_path(key))
        print(f"== {export}")
        print(report(load_requests(export, plans[key])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
