"""TTFT of a LiveKit (custom-agentic-livekit111868-*) run, split by a room's first turn vs later turns.

    python utils/serving/livekit_ttft.py <artifact-dir | profile_export.jsonl> ... [--plan <plan.csv>]

A directory is searched for every aiperf_artifacts/profile_export.jsonl (one per conc); each is read
with the plan of the dataset its run used: dataset.txt names it (e.g. livekit-111868-v1), and the
lock entry <name>-plan in scenarios/datasets/lock.yaml pins the plan CSV that ships with the trace
(fetched like the trace: GREENLAB_HF_TOKEN). --plan uses a local CSV instead.

First turns (no parent in the plan) carry the room's whole system prompt uncached, so they set the
tail; "5s turns" are the requests whose logged client timeout is 5 s. Only the profiling phase counts.
Dial warm-up rows (warm=1 in the plan of a -warm dataset: the first prompt minus the user's utterance,
max_tokens 1, sent before the room's first turn) are left out of every view and get their own line:
they are not on the user's path, but must finish before the first turn arrives.

Open-loop runs (custom-agentic-livekit111868-*-openloop): replayed calls are `<room>~k` (the plan row is the
room's); only calls whose first request started inside the scoring window of the run's openloop.json are kept
(the ramp and the tail are left out); "5s turns" also get the time to the first sentence (TTFS: TTFT plus the
next TTFS_TOKENS - 1 chunks, about what the TTS needs to start speaking).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

UNCACHED_BUCKETS = ((0, 5000), (5000, 11000), (11000, 20000), (20000, None))
TTFS_TOKENS = 12
PLAY_RE = re.compile(r"~\d+$")   # open-loop replays of a room: <room>~k


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


def ttfs(metrics: dict, ttft_s: float) -> float | None:
    """TTFT plus the first TTFS_TOKENS - 1 inter-chunk gaps (s); None for a reply shorter than that."""
    gaps = (metrics.get("inter_chunk_latency") or {}).get("value") or []
    if len(gaps) < TTFS_TOKENS - 1:
        return None
    return ttft_s + sum(gaps[: TTFS_TOKENS - 1]) / 1e3


def load_requests(profile_export: Path, plan: dict[str, list[dict]]) -> list[dict]:
    """One dict per profiling request with a TTFT: ttft (s), ttfs (s or None), first, timeout_5s,
    uncached (tokens), warm, call (conversation id, ~k kept), start (ns or None)."""
    rows = []
    with open(profile_export) as f:
        for line in f:
            record = json.loads(line)
            meta, metrics = record["metadata"], record["metrics"]
            if meta.get("benchmark_phase") != "profiling" or "time_to_first_token" not in metrics:
                continue
            turn = plan[PLAY_RE.sub("", meta["conversation_id"])][meta["turn_index"]]
            ttft_s = metrics["time_to_first_token"]["value"] / 1e3
            rows.append({
                "ttft": ttft_s,
                "ttfs": ttfs(metrics, ttft_s),
                "call": meta["conversation_id"],
                "start": meta.get("request_start_ns"),
                "first": turn["parent"] == "",
                "timeout_5s": float(turn["timeout"]) == 5.0,
                "uncached": int(turn["isl"]) - int(turn["shared"]),
                "warm": turn.get("warm") == "1",
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


def score_window(profile_export: Path) -> tuple[float, float] | None:
    """[t0, t1) s after the first request, from the run's openloop.json (next to aiperf_artifacts/); None if none."""
    for parent in profile_export.parents:
        sidecar = parent / "openloop.json"
        if sidecar.is_file():
            t0, t1 = json.loads(sidecar.read_text())["score_window_s"]
            return float(t0), float(t1)
    return None


def in_window(rows: list[dict], window: tuple[float, float]) -> list[dict]:
    """Rows of the calls whose first request started inside the window (all of a kept call's requests stay)."""
    starts = [r["start"] for r in rows if r["start"] is not None]
    if not starts:
        raise SystemExit("ERROR open-loop run without request_start_ns in profile_export.jsonl")
    zero = min(starts)
    opened: dict[str, float] = {}
    for r in rows:
        if r["start"] is not None:
            opened[r["call"]] = min(opened.get(r["call"], float("inf")), (r["start"] - zero) / 1e9)
    return [r for r in rows if window[0] <= opened.get(r["call"], -1.0) < window[1]]


def describe_ttfs(rows: list[dict]) -> list[str]:
    x = [r["ttfs"] for r in rows if r["ttfs"] is not None]
    if not x:
        return []
    return [f"  TTFS ({TTFS_TOKENS} tokens): n={len(x)} p50={quantile(x, .5):.2f} p90={quantile(x, .9):.2f} "
            f"p95={quantile(x, .95):.2f} p99={quantile(x, .99):.2f}"]


def describe_warm(rows: list[dict]) -> list[str]:
    x = [r["ttft"] for r in rows]
    return [(f"warm-ups (excluded above): n={len(x)} p50={quantile(x, .5):.2f} p90={quantile(x, .9):.2f} "
             f"p95={quantile(x, .95):.2f} p99={quantile(x, .99):.2f} max={max(x):.2f}  done "
             + " ".join(f"<={t:g}s={sum(v <= t for v in x) / len(x):.1%}" for t in (0.5, 1, 1.5, 2)))]


def report(rows: list[dict]) -> str:
    warm = [r for r in rows if r["warm"]]
    rows = [r for r in rows if not r["warm"]]
    if not rows:
        return "no profiling requests with a TTFT"
    lines = describe("all", rows)
    turns = [r for r in rows if r["timeout_5s"]]
    lines += describe("5s turns", turns) + describe_ttfs(turns) if turns else []
    first = [r for r in rows if r["first"]]
    for lo, hi in UNCACHED_BUCKETS:
        x = [r["ttft"] for r in first if r["uncached"] >= lo and (hi is None or r["uncached"] < hi)]
        if x:
            lines.append(f"  first turns uncached {lo}-{hi or ''}: n={len(x)} p50={quantile(x, .5):.2f} "
                         f"p95={quantile(x, .95):.2f} >1s={sum(v > 1 for v in x) / len(x):.0%}")
    lines += describe_warm(warm) if warm else []
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
        rows = load_requests(export, plans[key])
        window = score_window(export)
        print(f"== {export}")
        if window is not None:
            rows = in_window(rows, window)
            print(f"open loop: calls opened in [{window[0]:.0f}, {window[1]:.0f}) s")
        print(report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
