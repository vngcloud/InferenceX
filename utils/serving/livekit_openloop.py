"""Open-loop LiveKit trace: calls arrive by the clock, not when a lane frees up.

    python utils/serving/livekit_openloop.py <trace.jsonl> <out.jsonl> <openloop.json> --ccu N
        [--call-duration-s 127] [--duration-s T | --scenario scenarios/<name>.yaml] [--ramp-s 120] [--tail-s 60]
        [--seed 42]

The lane replay (custom-agentic-livekit111868-*) starts a lane's next call the moment its previous one ends, and
each turn the logged gap after the previous reply ends, so a faster server shortens every call and receives more
calls per hour at the same concurrency. Here the load is a parameter: calls open as a Poisson process at rate
CCU / call_duration_s per second (call_duration_s = the trace's mean call length at production reply speed), ramping
linearly from zero over the first ramp_s seconds, then steady until duration_s - tail_s. Every server sees the same
arrivals for the same seed.

Each arrival takes the next call of a seeded shuffle of the trace. Its first request gets the arrival `timestamp`
(aiperf fixed_schedule); later requests keep their logged `delay` (sent that long after the previous reply ends).
When more arrivals than calls are needed the shuffle cycles: the k-th play of a call (k >= 2) gets session id
`<room>~k` and a `play<k> ` marker at the start of its first user message, so the call's own context is cold again
while the shared system prompt still hits, as for a new caller. livekit_ttft.py strips the suffix to find the plan.

openloop.json records the arrivals and the scoring window: calls that open inside [ramp_s, duration_s - tail_s)
are scored; earlier ones are the ramp, later ones only keep the server loaded while the scored calls finish.
Standard library only (PyYAML for --scenario); two streaming passes over the trace (1.4 GB for V1.1-warm).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys

SID_RE = re.compile(r'"session_id":\s*"([^"]*)"')


def arrivals(rng: random.Random, rate: float, ramp_s: float, horizon_s: float) -> list[float]:
    """Poisson arrival times (s) in [0, horizon_s) at rate * min(1, t / ramp_s), by thinning."""
    out: list[float] = []
    if rate <= 0:
        return out
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= horizon_s:
            return out
        ramp = min(1.0, t / ramp_s) if ramp_s > 0 else 1.0
        if rng.random() < ramp:
            out.append(t)


def scan_rooms(path: str) -> list[str]:
    rooms: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            m = SID_RE.search(line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                rooms.append(m.group(1))
    return rooms


def mark(row: dict, play: int) -> dict:
    """Play marker at the start of the first user message: the call's own text goes cold, the system prompt does not."""
    msgs = [dict(m) for m in row.get("messages") or []]
    for m in msgs:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = f"play{play} " + m["content"]
            break
    return {**row, "messages": msgs}


def scenario_duration(path: str) -> float:
    import yaml

    phases = yaml.safe_load(open(path))["benchmark"]["phases"]
    return float(next(p for p in phases if p.get("kind", p.get("name")) == "profiling")["duration"])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("sidecar")
    ap.add_argument("--ccu", type=float, required=True, help="target mean concurrent calls")
    ap.add_argument("--call-duration-s", type=float, default=127.0, help="mean call length the rate is set from")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--duration-s", type=float, help="whole profiling phase: ramp + scored + tail")
    group.add_argument("--scenario", help="read the profiling phase duration from this scenario YAML")
    ap.add_argument("--ramp-s", type=float, default=120.0)
    ap.add_argument("--tail-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    duration = a.duration_s if a.duration_s is not None else scenario_duration(a.scenario)
    horizon = duration - a.tail_s
    if horizon <= a.ramp_s:
        sys.exit(f"duration {duration:g} s leaves no scored window after ramp {a.ramp_s:g} s and tail {a.tail_s:g} s")
    rate = a.ccu / a.call_duration_s
    rng = random.Random(a.seed)
    rooms = scan_rooms(a.src)
    if not rooms:
        sys.exit("no rooms in trace")
    order = rooms[:]
    rng.shuffle(order)
    times = arrivals(rng, rate, a.ramp_s, horizon)
    # arrival i -> (room, play): the shuffle cycles once every room has been opened
    plays: dict[str, list[tuple[int, int]]] = {}
    for i, t in enumerate(times):
        plays.setdefault(order[i % len(order)], []).append((i // len(order) + 1, int(round(t * 1000))))

    first_seen: set[str] = set()
    written = 0
    with open(a.src) as fin, open(a.dst, "w") as fout:
        for line in fin:
            m = SID_RE.search(line) if line.strip() else None
            sid = m.group(1) if m else None
            if sid is None or sid not in plays:
                continue
            row = json.loads(line)
            opening = sid not in first_seen
            first_seen.add(sid)
            for play, t_ms in plays[sid]:
                out = dict(row) if play == 1 else mark(row, play)
                if play > 1:
                    out["session_id"] = f"{sid}~{play}"
                if opening:
                    out.pop("delay", None)
                    out["timestamp"] = t_ms
                fout.write(json.dumps(out) + "\n")
                written += 1

    t0 = times[0] if times else 0.0
    sidecar = {
        "ccu": a.ccu, "call_duration_s": a.call_duration_s, "calls_per_s": rate, "ramp_s": a.ramp_s,
        "duration_s": duration, "tail_s": a.tail_s, "seed": a.seed, "arrivals": len(times),
        "rooms_in_trace": len(rooms), "max_play": max((p for v in plays.values() for p, _ in v), default=0),
        "requests": written, "first_arrival_s": t0,
        # aiperf's autoOffset and the scorer both take the first arrival as t = 0
        "score_window_s": [a.ramp_s - t0, horizon - t0],
    }
    with open(a.sidecar, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"open loop: {len(times)} calls at {rate * 60:.1f}/min (ccu {a.ccu:g} / {a.call_duration_s:g} s), "
          f"max play {sidecar['max_play']}, {written} requests -> {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
