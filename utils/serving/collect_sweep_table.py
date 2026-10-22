"""Collect sweep results into the PDF-style table (cache, ISL, OSL -> max CCU, RPS, TPM, ...)."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"sweep-isl(\d+)k-osl(\d+k?)-cache(\d+)")
COLUMNS = [
    "scenario",
    "cache",
    "isl",
    "osl",
    "state",
    "max_conc",
    "rps",
    "tpm",
    "ttft_avg_ms",
    "itl_p90_ms",
    "binding_constraint",
]


def parse_scenario(name: str) -> tuple[int, int, int]:
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError(f"not a sweep scenario name: {name!r}")
    isl, osl, cache = match.groups()
    return int(cache), int(isl) * 1024, int(osl[:-1]) * 1024 if osl.endswith("k") else int(osl)


def build_table(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        cache, isl, osl = parse_scenario(result["scenario"])
        rows.append(
            {**{col: result.get(col) for col in COLUMNS}, "cache": cache, "isl": isl, "osl": osl}
        )
    return sorted(rows, key=lambda row: (row["cache"], row["isl"], row["osl"]))


def to_markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append(
            "| " + " | ".join("-" if row[c] is None else str(row[c]) for c in COLUMNS) + " |"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    results = [
        json.loads(path.read_text()) for d in argv for path in sorted(Path(d).glob("*.json"))
    ]
    rows = build_table([r for r in results if "scenario" in r and r.get("state")])
    print(to_markdown(rows))
    with open(Path(argv[0]) / "sweep_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
