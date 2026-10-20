"""TTFT around a serving-YAML scale_up: did p95 hold when the late engine joined?

Reads one job's results directory (the `agentic_<name>_c<conc>_...` artifact):
  aiperf_artifacts/profile_export.jsonl  per-request records (profiling phase only)
  scale_up.jsonl                         when serve_generic.sh added each late engine
  router.log                             optional: vllm-router consistent_hash decisions, to tell
                                         which engine served each request

Prints TTFT p50/p95/p99 per window (before the join, then 0-60 s, 60-300 s and 300 s+ after it),
the after-join requests split by engine (stayed / moved, and a moved room's first turn on the new
engine, which is the one that loads its prefix from Mooncake or recomputes it), and a p95 timeline.

  python3 utils/serving/scale_up_ttft.py <results dir> [--slo-ms 1000] [--bucket 30]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROUTE = re.compile(r"Consistent hash routing: key='header:x-correlation-id:([^']+)' -> worker='([^']+)'")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
WINDOWS = ((None, 0), (0, 60), (60, 300), (300, None))  # seconds relative to the join


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation (numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def load_records(results: Path) -> list[dict]:
    records = []
    with (results / "aiperf_artifacts/profile_export.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            meta = record["metadata"]
            ttft = (record.get("metrics") or {}).get("time_to_first_token")
            records.append({
                "phase": meta["benchmark_phase"],
                "start_ns": meta["request_start_ns"],
                "corr": meta["x_correlation_id"],
                "room": meta["conversation_id"],
                "turn": meta["turn_index"],
                "ttft_ms": ttft["value"] if ttft else None,
                "error": bool(record.get("error")),
            })
    return records


def load_joins(results: Path) -> list[dict]:
    path = results / "scale_up.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path}: missing; was this a scale_up run, and did profiling reach the join?")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assign_engines(records: list[dict], router_log: Path) -> int:
    """Set record["engine"] from router.log, zipping each session's routing lines with its requests
    in start order. Returns how many sessions could not be matched (count mismatch)."""
    routes: dict[str, list[str]] = defaultdict(list)
    with router_log.open(errors="replace") as handle:
        for line in handle:
            match = ROUTE.search(ANSI.sub("", line))
            if match:
                routes[match.group(1)].append(match.group(2))
    by_session: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_session[record["corr"]].append(record)
    unmatched = 0
    for corr, session in by_session.items():
        workers = routes.get(corr, [])
        if len(workers) != len(session):
            unmatched += 1
            continue
        for record, worker in zip(sorted(session, key=lambda r: r["start_ns"]), workers):
            record["engine"] = worker
    return unmatched


def row(label: str, records: list[dict], slo_ms: float | None) -> str:
    ttfts = [r["ttft_ms"] for r in records if r["ttft_ms"] is not None and not r["error"]]
    errors = sum(r["error"] for r in records)
    p50, p95, p99 = (percentile(ttfts, q) for q in (50, 95, 99))
    fmt = lambda v: "     -" if v is None else f"{v / 1000:6.2f}"
    verdict = ""
    if slo_ms is not None and p95 is not None:
        verdict = "  ok" if p95 <= slo_ms else "  OVER SLO"
    return f"{label:<34}{len(records):>7}{errors:>6}  {fmt(p50)}  {fmt(p95)}  {fmt(p99)}{verdict}"


def report(results: Path, slo_ms: float | None, bucket: int) -> str:
    records = load_records(results)
    joins = load_joins(results)
    join_ns = min(j["added_ns"] for j in joins)
    late_urls = {j["url"] for j in joins}
    router_log = results / "router.log"
    unmatched = assign_engines(records, router_log) if router_log.is_file() else None
    profiling = [r for r in records if r["phase"] == "profiling"]
    if not profiling:
        raise SystemExit("no profiling records")
    rel = lambda r: (r["start_ns"] - join_ns) / 1e9

    header = f"{'':<34}{'reqs':>7}{'err':>6}  {'p50 s':>6}  {'p95 s':>6}  {'p99 s':>6}"
    into = (join_ns - min(r["start_ns"] for r in profiling)) / 1e9
    out = [f"join: {', '.join(sorted(late_urls))} at {into:.0f} s into profiling", "",
           "TTFT by window (request start relative to the join)", header]
    for low, high in WINDOWS:
        label = ("before join" if low is None
                 else f"+{low}..{'end' if high is None else f'+{high}'} s")
        window = [r for r in profiling
                  if (low is None or rel(r) >= low) and (high is None or rel(r) < high)]
        out.append(row(label, window, slo_ms))

    if unmatched is not None:
        after = [r for r in profiling if rel(r) >= 0 and "engine" in r]
        late = [r for r in after if r["engine"] in late_urls]
        stayed = [r for r in after if r["engine"] not in late_urls]
        # A room's first request on the late engine after the join: its prefix is not in that
        # engine's GPU cache (Mooncake load or recompute). Rooms that started after the join
        # have nothing to move, so only rooms seen before the join count as moved.
        seen_before = {r["corr"] for r in records if r["start_ns"] < join_ns}
        first_on_late: dict[str, dict] = {}
        for record in sorted(late, key=lambda r: r["start_ns"]):
            if record["corr"] in seen_before:
                first_on_late.setdefault(record["corr"], record)
        moved_first = list(first_on_late.values())
        moved_first_ids = {id(r) for r in moved_first}
        moved_later = [r for r in late if r["corr"] in first_on_late and id(r) not in moved_first_ids]
        new_rooms = [r for r in late if r["corr"] not in seen_before]
        out += ["", "After the join, by engine (router.log)", header,
                row("stayed on the first engine(s)", stayed, slo_ms),
                row("late engine, all", late, slo_ms),
                row("  moved room, first turn there", moved_first, slo_ms),
                row("  moved room, later turns", moved_later, slo_ms),
                row("  room started after the join", new_rooms, slo_ms)]
        if unmatched:
            out.append(f"({unmatched} sessions skipped: router.log and aiperf disagree on their request count)")

    out += ["", f"TTFT p95 per {bucket} s bucket (bucket start relative to the join)"]
    buckets: dict[int, list[dict]] = defaultdict(list)
    for record in profiling:
        buckets[int(rel(record) // bucket) * bucket].append(record)
    for start in sorted(buckets):
        ttfts = [r["ttft_ms"] for r in buckets[start] if r["ttft_ms"] is not None and not r["error"]]
        p95 = percentile(ttfts, 95)
        bar = "" if p95 is None else "#" * min(60, round(p95 / 50))
        over = " <- over SLO" if slo_ms is not None and p95 is not None and p95 > slo_ms else ""
        p95_text = "     -" if p95 is None else f"{p95 / 1000:6.2f}"
        out.append(f"{start:+6d} s {len(buckets[start]):>5} req  p95 {p95_text} s  {bar}{over}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="the job's results directory (agentic_* artifact)")
    parser.add_argument("--slo-ms", type=float, help="TTFT p95 SLO in ms; marks windows above it")
    parser.add_argument("--bucket", type=int, default=30, help="timeline bucket in seconds (default 30)")
    args = parser.parse_args(argv)
    print(report(args.results, args.slo_ms, args.bucket))
    return 0


if __name__ == "__main__":
    sys.exit(main())
