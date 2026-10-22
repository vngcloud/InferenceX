"""Scenario YAML copy whose ``sweep.sla_filters`` carry the given SLO tuples.

    python -m infx.serving.slo_override SCENARIO.yaml COPY.yaml '<tuples>'

AIPerf silently ignores CLI ``--search-sla`` when ``--config`` carries a
``sweep:`` block (docs/superpowers/plans/2026-10-17-sweep-scenarios-spike-notes.md,
SLO_OVERRIDE), so an IX_SLO override must be written into the YAML: this writes
a copy of SCENARIO.yaml with ``sweep.sla_filters`` replaced by the tuples (one
``metric:stat:op:threshold`` per line, the output of ``python -m infx.serving.slo``).
"""

from __future__ import annotations

import sys

import yaml


def filters(tuples: str) -> list[dict]:
    out = []
    for line in tuples.splitlines():
        metric, stat, op, threshold = line.split(":")
        out.append({"metric_tag": metric, "stat": stat, "op": op, "threshold": float(threshold)})
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: python -m infx.serving.slo_override SCENARIO.yaml COPY.yaml '<tuples>'",
            file=sys.stderr,
        )
        return 2
    src, dst, tuples = argv
    try:
        with open(src) as handle:
            doc = yaml.safe_load(handle)
    except OSError as error:
        print(f"ERROR slo_override: cannot read {src} ({error.strerror})", file=sys.stderr)
        return 2
    sweep = doc.get("sweep") if isinstance(doc, dict) else None
    if not isinstance(sweep, dict) or sweep.get("type") != "adaptive_search":
        print(
            f"ERROR slo_override: {src} has no adaptive_search sweep: block; IX_SLO cannot apply",
            file=sys.stderr,
        )
        return 2
    try:
        sweep["sla_filters"] = filters(tuples)
    except (ValueError, AttributeError):
        print("ERROR slo_override: SLO tuples must be metric:stat:op:threshold", file=sys.stderr)
        return 2
    with open(dst, "w") as handle:
        yaml.safe_dump(doc, handle, sort_keys=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
