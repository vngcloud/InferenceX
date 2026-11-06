"""Speculative-decoding acceptance of one serving point, shared by every scenario that measures it.

The client snapshots the engines' Prometheus counters before and after the replay and files the
delta under ``spec_decode`` in the aggregate result:

    acceptance_length = 1 + accepted tokens / draft events     (vLLM counters; independent of depth)
    acceptance_rate   = accepted tokens / drafted tokens
    draft_depth       = drafted tokens / draft events           (num_speculative_tokens)

SGLang counts verify steps per request (``sglang:spec_verify_calls``) and generated tokens
(``sglang:generation_tokens``); each verify step emits the accepted drafts plus the bonus token, so

    acceptance_length = generated tokens / verify calls         (SGLang counters, over the run)

Older SGLang without those counters falls back to the running-average gauge
``sglang:spec_accept_length``, read after the run. Being an average, the gauge is not summed like
the counters: with several series (data-parallel ranks, replicas) it is the mean of the nonzero
ones. Summing it reported 4x the true length on a DP4 server. A point without speculative decoding
(no vLLM spec counters, no SGLang verify calls, gauge 0.0) gets None from ``summarize`` and the
aggregate stays untouched.
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VLLM_COUNTERS = {
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "draft_events": "vllm:spec_decode_num_drafts_total",
    "drafted_tokens": "vllm:spec_decode_num_draft_tokens_total",
}
SGLANG_ACCEPT_LENGTH = "sglang:spec_accept_length"
SGLANG_VERIFY_CALLS = "sglang:spec_verify_calls"
SGLANG_GENERATION_TOKENS = "sglang:generation_tokens"
GAUGES = {SGLANG_ACCEPT_LENGTH}
WANTED = {*VLLM_COUNTERS.values(), SGLANG_VERIFY_CALLS, SGLANG_GENERATION_TOKENS, *GAUGES}
_SAMPLE = re.compile(r"^(?P<name>[A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def _mean_nonzero(values: list[float]) -> float:
    nonzero = [v for v in values if v != 0]
    return sum(nonzero) / len(nonzero) if nonzero else 0.0


def _combine(samples: Iterable[tuple[str, float]]) -> dict[str, float]:
    """Counters summed over their samples; a gauge in ``GAUGES`` is the mean of its nonzero samples."""
    sums: dict[str, float] = {}
    gauges: dict[str, list[float]] = {}
    for name, value in samples:
        if name in GAUGES:
            gauges.setdefault(name, []).append(value)
        else:
            sums[name] = sums.get(name, 0.0) + value
    return {**sums, **{name: _mean_nonzero(values) for name, values in gauges.items()}}


def parse_prometheus(text: str) -> dict[str, float]:
    """Each metric combined across label sets (e.g. per-engine series), as ``_combine`` does."""
    samples: list[tuple[str, float]] = []
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        if math.isfinite(value):
            samples.append((match["name"], value))
    return _combine(samples)


def snapshot(urls: Iterable[str], timeout: float = 10.0) -> dict[str, float]:
    """The speculative-decoding series combined over ``urls``; an unreachable URL or a missing series is skipped."""
    samples: list[tuple[str, float]] = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 -- the job's own engine ports
                totals = parse_prometheus(response.read().decode())
        except (OSError, ValueError) as error:
            print(f"WARNING spec-decode metrics: {url} unreadable ({error})", file=sys.stderr)
            continue
        samples += [(name, value) for name, value in totals.items() if name in WANTED]
    return _combine(samples)


def acceptance(before: dict[str, float], after: dict[str, float]) -> dict[str, Any] | None:
    """Acceptance over the run from two snapshots, or None when the engine reports no speculative decoding."""
    # SGLang exports the gauge on every engine, 0.0 when the engine does not speculate.
    sglang_length = after.get(SGLANG_ACCEPT_LENGTH, 0.0)
    verify_calls = round(after.get(SGLANG_VERIFY_CALLS, 0.0) - before.get(SGLANG_VERIFY_CALLS, 0.0))
    generated = round(
        after.get(SGLANG_GENERATION_TOKENS, 0.0) - before.get(SGLANG_GENERATION_TOKENS, 0.0)
    )
    if (
        not {*before, *after} & set(VLLM_COUNTERS.values())
        and verify_calls <= 0
        and sglang_length <= 0
    ):
        return None
    delta = {
        key: round(after.get(name, 0.0) - before.get(name, 0.0))
        for key, name in VLLM_COUNTERS.items()
    }
    result: dict[str, Any] = {f"spec_{key}": value for key, value in delta.items()}
    result["acceptance_length"] = (
        1 + delta["accepted_tokens"] / delta["draft_events"] if delta["draft_events"] > 0 else None
    )
    result["draft_depth"] = (
        round(delta["drafted_tokens"] / delta["draft_events"])
        if delta["draft_events"] > 0
        else None
    )
    result["acceptance_rate"] = (
        delta["accepted_tokens"] / delta["drafted_tokens"] if delta["drafted_tokens"] > 0 else None
    )
    if verify_calls > 0:
        result["spec_verify_calls"] = verify_calls
        result["spec_generated_tokens"] = generated
    if result["acceptance_length"] is None and verify_calls > 0 and generated > 0:
        result["acceptance_length"] = generated / verify_calls
    elif result["acceptance_length"] is None and sglang_length > 0:
        result["acceptance_length"] = sglang_length
    return result


def save(path: Path, snap: dict[str, float]) -> None:
    path.write_text(json.dumps(snap, indent=2))


def summarize(before_path: Path, after_path: Path) -> dict[str, Any] | None:
    """``acceptance`` of the two snapshot files the client wrote; None when either is missing."""
    if not (before_path.is_file() and after_path.is_file()):
        return None
    return acceptance(json.loads(before_path.read_text()), json.loads(after_path.read_text()))


def merge(aggregate_path: Path, spec: dict[str, Any]) -> None:
    """Add ``spec`` as ``spec_decode`` to the aggregate result, if the aggregate exists."""
    if not aggregate_path.is_file():
        return
    result = json.loads(aggregate_path.read_text())
    result["spec_decode"] = spec
    aggregate_path.write_text(json.dumps(result, indent=2))
